#!/usr/bin/env python3

import logging.config
import os
import re
from pathlib import Path
from typing import Optional, Union, Dict, List

import boto3
import click
import dask
import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, LocalCluster
from geopy.distance import distance

import hotspot_utils as util

__s3_pattern__ = r"^s3://" r"(?P<bucket>[^/]+)/" r"(?P<keyname>.*)"


def _distance(df: pd.DataFrame) -> List:
    """Helper method to compute distance.

    :param df: The DataFrame to compute distance.

    :returns:
        List of distance computed from DataFrame.
    """
    dist = [
        distance(
            (x["latitude"], x["longitude"]),
            (x["2_latitude"], x["2_longitude"]),
        ).meters
        for _, x in df.iterrows()
    ]
    return dist


def _prepare_pivot_dataframe(
    df: dd.DataFrame,
) -> dd.DataFrame:
    """Helper method to convert datetime columns to datetime and subset using distance threhold.

    :param df: The dask DataFrame.

    """
    df["datetime"] = dd.to_datetime(df["datetime"])
    df["solar_day"] = dd.to_datetime(df["solar_day"])
    df["2_datetime"] = dd.to_datetime(df["2_datetime"])
    df["timedelta"] = abs(df["datetime"] - df["2_datetime"])
    df["count"] = 1
    df = df.astype({"dist_m": float})
    df = df.set_index("solar_day")
    return df


def _to_timedelta(df: pd.DataFrame) -> pd.Series:
    """Helper method to calculate time difference.

    :param df: The DataFrame to compute distance

    :returns:
        The pandas Series object with computed time difference.
    """

    return abs(df["datetime"] - df["2_datetime"])


def _pandas_pivot_table(
    df: pd.DataFrame,
    index: List[str],
    columns: List[str],
    values: List[str],
    aggfunc: dict,
) -> pd.DataFrame:
    """Helper method to perform pandas pivot operations.

    :param df: The DataFrame to compute pivot operation.
    :param index: The names of the columns to make new DataFrame's index.
    :param columns: The names of columns to make new DataFrame's columns.
    :param values: The columns to use for populating new frame's value.
    :param aggfunc: The function to apply in pivot operation.

    :returns:
        The reshaped pandas DataFrame
    """
    _df = pd.pivot_table(
        df,
        values=values,
        index=index,
        aggfunc=aggfunc,
        columns=columns,
    )
    return _df


def _dask_pivot_table(
    df: pd.DataFrame,
    index: str,
    column: str,
    values: str,
    aggfunc: str,
) -> dd.DataFrame:
    """Helper method to perform dask dataframe pivot operations.

    :param df: The DataFrame to compute pivot operation.
    :param index: column to be index.
    :param column: column to be columns.
    :param values: column to aggregate.
    :param aggfunc: The function to apply in pivot operation.

    :returns:
        The reshaped dask DataFrame
    """
    df = df.categorize(columns=[index, column])
    _df = dd.reshape.pivot_table(
        df, index=index, columns=column, values=values, aggfunc=aggfunc
    )
    return _df


def process_cooccurence(
    nearest_csv_files_dict: Dict,
    dist_threshold: Optional[float] = 5000.0,
    name_prefix: Optional[str] = "",
    outdir: Optional[Union[Path, str]] = Path(os.getcwd()),
) -> None:
    """Process to compute the co-occurrence matrix between different product types.

    :param nearest_csv_files_dict: The dictionary with key as satellite sensor product type
                                   and value as file path to .csv file.
    :param dist_threshold: The distance threhold to subset the data.
    :param name_prefix: The name prefix that will be added to output file.
    :param outdir: The name of the output directory to save outputs.
    """
    df_list = [
        util.load_csv(fp)
        for _, fp in nearest_csv_files_dict.items()
        if Path(fp).exists()
    ]
    df = dd.concat(df_list)
    df["dist_m"] = df.map_partitions(_distance)
    df = _prepare_pivot_dataframe(df)

    # compute numerator
    numerator = _dask_pivot_table(
        df[df["dist_m"] < dist_threshold],
        index="2_satellite_sensor_product",
        column="satellite_sensor_product",
        values="count",
        aggfunc="count",
    ).compute()
    numerator.to_csv(
        outdir.joinpath(
            f"{name_prefix}_{int(dist_threshold)}m_numerator_pivot_table.csv"
        ).as_posix()
    )

    # compute denominator
    denominator = _dask_pivot_table(
        df,
        index="2_satellite_sensor_product",
        column="satellite_sensor_product",
        values="count",
        aggfunc="count",
    ).compute()
    denominator.to_csv(
        outdir.joinpath(
            f"{name_prefix}_{int(dist_threshold)}m_denominator_pivot_table.csv"
        ).as_posix()
    )

    # compute difference
    difference = denominator - numerator
    difference.to_csv(
        outdir.joinpath(
            f"{name_prefix}_{int(dist_threshold)}m_difference_pivot_table.csv"
        ).as_posix()
    )

    # compute percentage
    percentage = numerator / denominator
    percentage.to_csv(
        outdir.joinpath(
            f"{name_prefix}_{int(dist_threshold)}m_percentage.csv"
        ).as_posix()
    )

    # compute mean time
    timemean = _dask_pivot_table(
        df[df["dist_m"] < dist_threshold],
        index="2_satellite_sensor_product",
        column="satellite_sensor_product",
        values="timedelta",
        aggfunc="mean",
    ).compute()
    timemean.to_csv(
        outdir.joinpath(
            f"{name_prefix}_{int(dist_threshold)}m_timemean.csv"
        ).as_posix()
    )

    # compute average_distance
    averagedist = _dask_pivot_table(
        df[df["dist_m"] < dist_threshold],
        index="2_satellite_sensor_product",
        column="satellite_sensor_product",
        values="dist_m",
        aggfunc="mean",
    ).compute()
    numerator.to_csv(
        outdir.joinpath(
            f"{name_prefix}_{int(dist_threshold)}m_averagedist.csv"
        ).as_posix()
    )


def process_nearest_points(
    hotspots_files: Dict,
    lon_west: float,
    lat_south: float,
    lon_east: float,
    lat_north: float,
    start_date: str,
    end_date: str,
    start_time: str,
    end_time: str,
    chunks: Optional[int] = 100,
    outdir: Optional[Union[Path, str]] = Path(os.getcwd()),
) -> Dict:
    """Processing of nearest points for different products in hotpots.

    This process will temporally and spatially subset the data and merge all the hotspots GeoDataFrame
    to have consistent fields among different hotspots products provider: dea, landgate, esa, eumetsat and nasa.
    The individual merged hotspots product from different provider and sensor types are compared with other
    hotspots products to find the spatially nearesthotspots from different products on same solar_day or
    solar_night datetime. The final results is the individual hotspots product with nearest hotspots from
    other products. The results are saved as a .csv files using product type in the filename.

    :param hotspots_files: The dictionary with key as data provider and value as file path.
    :param lon_west: The western longitude of the extent to subset.
    :param lat_south: The southern latitude of the extent to subset.
    :param lon_east: The eastern longitude of the extent to subset.
    :param lat_north: The northern latitide of the extent to subset.
    :param start_time: The start time to subset the data.
    :param end_time: The end time to subset the data.
    :param start_date: The start date to subset the data.
    :param end_date: The end date to subset the data
    :param chunks: The number of blocks to be sub-divided into.
    :param outdir: The output directory to save the output files.

    :returns:
        None.
        All the .csv output files containing the nearest hotspots details are written into outdir.
    """

    _LOG.info("Running fire hotspots analysis...")
    process_kwargs = {
        "bbox": (lon_west, lat_south, lon_east, lat_north),
        "start_date": start_date,
        "end_date": end_date,
        "start_time": start_time,
        "end_time": end_time,
        "num_chunks": chunks,
    }

    _LOG.info(
        "Reading spatial and temporal subsets of all hotspots dataframes..."
    )
    all_hotspots_tasks = util.get_all_hotspots_tasks(
        hotspots_files, **process_kwargs
    )
    attrs_normalization_tasks = [
        dask.delayed(util.normalize_features)(df) for df in all_hotspots_tasks
    ]

    _LOG.info("Merging hotspots dataframe...")
    hotspots_gdf = pd.concat(
        [df for df in dask.compute(*attrs_normalization_tasks)]
    )
    print(hotspots_gdf.count())
    _LOG.info(f"The merged hotspots DataFrame sample:\n {hotspots_gdf}")
    _LOG.info("The spatial and temporal extents of merged hotspots.")
    _LOG.info(
        f"minimum datetime: {hotspots_gdf['datetime'].min()}, maximum datetime: {hotspots_gdf['datetime'].max()}"
    )
    _LOG.info(
        f"longitude range: {hotspots_gdf['longitude'].min()}  to {hotspots_gdf['longitude'].max()}"
    )
    _LOG.info(
        f"latitude range: {hotspots_gdf['latitude'].min()}  to {hotspots_gdf['latitude'].max()}"
    )

    solar_start_dt = hotspots_gdf['solar_day'].min()
    solar_end_dt = hotspots_gdf['solar_day'].max()

    _LOG.info(f"Generating satellite swaths from {solar_start_dt.date()} to {solar_end_dt.date()}")
    swath_directory = Path(outdir).joinpath(f"swaths_{int(lon_east)}_{int(lon_west)}")
    if not swath_directory.exists():
        swath_directory.mkdir(exist_ok=True, parents=True)

    swath_generation_tasks = util.swath_generation_tasks(
        solar_start_dt,
        solar_end_dt,
        lon_east,
        lon_west,
        swath_directory=swath_directory,
        config_file="s3vtconfig.yaml"
    )
    _ = dask.compute(*swath_generation_tasks)


    _LOG.info(f"Generating neareast hotspots...")
    unique_products = [
        p for p in hotspots_gdf["satellite_sensor_product"].unique()
    ]
    nearest_hotspots_product_files = dict()
    for product_a in unique_products:
        hotspots_compare_tasks = []
        for product_b in unique_products:
            geosat_flag = False
            if ("AHI" in [product_a, product_b]) or ("INS1" in [product_a, product_b]):
                geosat_flag = True

            gdf_a = hotspots_gdf[
                hotspots_gdf["satellite_sensor_product"] == product_a
            ]
            gdf_b = hotspots_gdf[
                hotspots_gdf["satellite_sensor_product"] == product_b
            ]
            hotspots_compare_tasks.append(
                dask.delayed(util.hotspots_compare)(
                    gdf_a,
                    gdf_b,
                    lon_east,
                    lon_west,
                    "solar_day",
                    geosat_flag,
                )
            )
        outfile = Path(outdir).joinpath(f"nearest_points.{product_a}.csv")
        product_a_nearest_gdfs = dask.compute(*hotspots_compare_tasks)
        product_a_nearest_gdfs_merged = pd.concat(
            [df for df in product_a_nearest_gdfs if df is not None],
            ignore_index=True
        )
        product_a_nearest_gdfs_merged.reset_index(inplace=True, drop=True)
        product_a_nearest_gdfs_merged.to_csv(outfile.as_posix())
        nearest_hotspots_product_files[product_a] = outfile
    return nearest_hotspots_product_files


@click.command(
    "s3vt-frp-analysis", help="Processing of fire hotspots analysis."
)
@click.option(
    "--nasa-frp",
    type=click.STRING,
    help="NASA FRP geojson file path.",
    default="s3://s3vtaustralia/nasa_hotspots_gdf.geojson",
    show_default=True,
)
@click.option(
    "--esa-frp",
    type=click.STRING,
    help="ESA FRP geojson file path",
    default="s3://s3vtaustralia/s3vt_hotspots.geojson",
    show_default=True,
)
@click.option(
    "--eumetsat-frp",
    type=click.STRING,
    help="EUMETSAT FRP geojson file path",
    default="s3://s3vtaustralia/s3vt_eumetsat_hotspots.geojson",
    show_default=True,
)
@click.option(
    "--landgate-frp",
    type=click.STRING,
    help="LANDGATE FRP geojson file path",
    default="s3://s3vtaustralia/landgate_hotspots_gdf.geojson",
    show_default=True,
)
@click.option(
    "--dea-frp",
    type=click.STRING,
    help="DEA FRP geojson file path",
    default=None,
    show_default=True,
)
@click.option(
    "--lon-west",
    type=click.FLOAT,
    help="western longitude to form bounding box",
    default=147.0,
    show_default=True,
)
@click.option(
    "--lon-east",
    type=click.FLOAT,
    help="eastern longitude to form bounding box",
    default=154.0,
    show_default=True,
)
@click.option(
    "--lat-south",
    type=click.FLOAT,
    help="southern latitude to form bounding box",
    default=-38.0,
    show_default=True,
)
@click.option(
    "--lat-north",
    type=click.FLOAT,
    help="northern latitude to form bounding box",
    default=-27.0,
    show_default=True,
)
@click.option(
    "--start-date",
    type=click.STRING,
    help="start date to form temporal subset [YYYY-MM-DD]",
    default="2019-11-01",
    show_default=True,
)
@click.option(
    "--end-date",
    type=click.STRING,
    help="end date to form temporal subset [YYYY-MM-DD]",
    default="2020-10-08",
    show_default=True,
)
@click.option(
    "--start-time",
    type=click.STRING,
    help="start-time to form temporal subset [HH:MM]",
    default="21:00",
    show_default=True,
)
@click.option(
    "--end-time",
    type=click.STRING,
    help="end-time to form temporal subset [HH:MM]",
    default="03:00",
    show_default=True,
)
@click.option(
    "--outdir",
    type=click.STRING,
    help="The working directory where all the output files will be saved.",
    default=Path(os.getcwd()).joinpath("workdir").as_posix(),
    show_default=True,
)
@click.option(
    "--chunks",
    type=click.INT,
    help="Number of chunks to block geojson files used in multi-processing.",
    default=100,
    show_default=True,
)
def main(
    nasa_frp: click.STRING,
    esa_frp: click.STRING,
    eumetsat_frp: click.STRING,
    landgate_frp: click.STRING,
    dea_frp: click.STRING,
    lon_west: click.FLOAT,
    lon_east: click.FLOAT,
    lat_south: click.FLOAT,
    lat_north: click.FLOAT,
    start_date: click.STRING,
    end_date: click.STRING,
    start_time: click.STRING,
    end_time: click.STRING,
    outdir: click.STRING,
    chunks: click.INT,
):
    _LOG.info(f"Processing Active Fire Analysis...")
    # make a work directory if it does not exist.
    if not Path(outdir).exists():
        Path(outdir).mkdir(exist_ok=True)

    # this sections is to download frp geojson from location provided is in s3
    hotspots_files = {
        "nasa": nasa_frp,
        "esa": esa_frp,
        "eumetsat": eumetsat_frp,
        "landgate": landgate_frp,
        "dea": dea_frp,
    }
    aws_session = boto3.Session(profile_name="s3vt")
    s3_client = aws_session.client("s3")

    # check if any files need to be download from s3
    for provider, fp in hotspots_files.items():
        if fp is None:
            _LOG.info(f"{provider} frp is None. excluding this frp product.")
            continue
        if re.match(__s3_pattern__, fp):
            _, bucket, _key, _ = re.split(__s3_pattern__, fp)
            outfile = Path(outdir).joinpath(Path(fp).name)
            hotspots_files[provider] = outfile
            if outfile.exists():
                _LOG.info(f"{fp} exists: skipped download")
                continue
            _LOG.info(f"downloading {fp} to {outfile.as_posix()}")
            s3_client.download_file(bucket, _key, outfile.as_posix())

    nearest_hotspots_csv_files = process_nearest_points(
        hotspots_files,
        lon_west,
        lat_south,
        lon_east,
        lat_north,
        start_date,
        end_date,
        start_time,
        end_time,
        chunks=chunks,
        outdir=outdir,
    )
    """
    nearest_hotspots_csv_files = {
        "AQUA_MODIS_LANDGATE": "nearest_points.AQUA_MODIS_LANDGATE.csv",
        "AQUA_MODIS_NASA6.03": "nearest_points.AQUA_MODIS_NASA6.03.csv",
        "NOAA 20_VIIRS_LANDGATE": "nearest_points.NOAA 20_VIIRS_LANDGATE.csv",
        "NOAA 20_VIIRS_NASA2.0NRT": "nearest_points.NOAA 20_VIIRS_NASA2.0NRT.csv",
        "SENTINEL_3A_SLSTR_ESA": "nearest_points.SENTINEL_3A_SLSTR_ESA.csv",
        "SENTINEL_3A_SLSTR_EUMETSAT": "nearest_points.SENTINEL_3A_SLSTR_EUMETSAT.csv",
        "SENTINEL_3B_SLSTR_ESA": "nearest_points.SENTINEL_3B_SLSTR_ESA.csv",
        "SENTINEL_3B_SLSTR_EUMETSAT": "nearest_points.SENTINEL_3B_SLSTR_EUMETSAT.csv",
        "SUOMI NPP_VIIRS_LANDGATE": "nearest_points.SUOMI NPP_VIIRS_LANDGATE.csv",
        "SUOMI NPP_VIIRS_NASA1": "nearest_points.SUOMI NPP_VIIRS_NASA1.csv",
        "TERRA_MODIS_LANDGATE": "nearest_points.TERRA_MODIS_LANDGATE.csv",
        "TERRA_MODIS_NASA6.03": "nearest_points.TERRA_MODIS_NASA6.03.csv",
    }
    """
    process_cooccurence(
        nearest_hotspots_csv_files,
        dist_threshold=5000.0,
        name_prefix=f"{start_date.replace('-', '')}_{end_date.replace('-', '')}",
    )


if __name__ == "__main__":
    # Configure log here for now, move it to __init__ at top level once
    # code is configured to run as module
    client = Client(asynchronous=True)
    LOG_CONFIG = Path(__file__).parent.joinpath("logging.cfg")
    logging.config.fileConfig(LOG_CONFIG.as_posix())
    _LOG = logging.getLogger(__name__)
    main()
    client.close()
