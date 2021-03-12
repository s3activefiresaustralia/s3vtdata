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

_LOG = logging.getLogger(__name__)
    
def process_nearest_points(
    nasa_frp: Union[Path, str],
    esa_frp: Union[Path, str],
    eumetsat_frp: Union[Path, str],
    landgate_frp: Union[Path, str],
    dea_frp: Union[Path, str],
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
    compare_field: Optional[str] = "solar_day",
    swath_config_file: Optional[Union[Path, str]] = "s3vtconfig.yaml"
) -> List:
    """Processing of nearest points for different products in hotpots.

    This process will temporally and spatially subset the data and merge all the hotspots GeoDataFrame
    to have consistent fields among different hotspots products provider: dea, landgate, esa, eumetsat and nasa.
    The individual merged hotspots product from different provider and sensor types are compared with other
    hotspots products to find the spatially nearesthotspots from different products on same solar_day or
    solar_night datetime. The final results is the individual hotspots product with nearest hotspots from
    other products. The results are saved as a .csv files using product type in the filename.

    :param nasa_frp: The path to NASA FRP .geojson file.
    :param esa_frp: The path to ESA FRP .geojson file.
    :param eumetsat_frp: The path to EUMETSAT .geojson file.
    :param landgate_frp: The path to LANDGATE .geojson file.
    :param dea_frp: The path to DEA .geojson file.
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
    :param compare_field: The field (column) used in finding the nearest hotspots.
    :param swath_config_file: The config file used in swath generation.
    
    :returns:
        All the .csv output files path containing the nearest hotspots.
    """
    
    _LOG.info(f"Processing Neareast Hotspots...")
    # make a work directory if it does not exist.
    if not Path(outdir).exists():
        Path(outdir).mkdir(exist_ok=True)

    # this sections is to download frp geojson from location provided is in s3
    hotspots_files_dict = {
        "nasa": nasa_frp,
        "esa": esa_frp,
        "eumetsat": eumetsat_frp,
        "landgate": landgate_frp,
        "dea": dea_frp,
    }
    
    aws_session = boto3.Session()
    s3_client = aws_session.client("s3")
    
    hotspots_files = util.fetch_hotspots_files(
        hotspots_files_dict,
        s3_client,
        outdir
    )
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
    _LOG.info(
        f"The merged hotspots dataframe head:\n {hotspots_gdf.head()}"
    )
    _LOG.info(
        f"The merged hotspots dataframe count:\n {hotspots_gdf.count()}"
        
    )
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
        config_file=swath_config_file
    )
    _ = dask.compute(*swath_generation_tasks)

    _LOG.info(f"Generating neareast hotspots...")
    unique_products = [
        p for p in hotspots_gdf["satellite_sensor_product"].unique()
    ]
    nearest_hotspots_product_files = []
    for product_a in unique_products:
        outfile = Path(outdir).joinpath(f"nearest_points_{product_a}_{compare_field}.csv")
        if outfile.exists():
            _LOG.info(
                f"{outfile.as_posix()} exists. skipped nearest"
                f" hotspots processing for product {product_a}."
                " Same file will be used in analysis."
            )
        else:
            gdf_a = hotspots_gdf[
                hotspots_gdf["satellite_sensor_product"] == product_a
            ]
            nearest_hotspots_dfs = []
            for product_b in unique_products:
                _LOG.info(f"Comparing Hotspots for {product_a} and {product_b}")
                geosat_flag = False
                if ("AHI" in [product_a, product_b]) or ("INS1" in [product_a, product_b]):
                    geosat_flag = True
                gdf_b = hotspots_gdf[
                    hotspots_gdf["satellite_sensor_product"] == product_b
                ]
                product_a_df = util.hotspots_compare(
                    gdf_a,
                    gdf_b,
                    compare_field,
                    geosat_flag,
                    swath_directory
                )
                if product_a_df is not None:
                    nearest_hotspots_dfs.append(product_a_df)
            nearest_hotspots_dfs = pd.concat(nearest_hotspots_dfs, ignore_index=True)
            nearest_hotspots_dfs.reset_index(inplace=True, drop=True)
            nearest_hotspots_dfs.to_csv(outfile.as_posix())
        nearest_hotspots_product_files.append(outfile)

    return nearest_hotspots_product_files


@click.command(
    "process-nearest-hotspots", help="Processing of the nearest hotspots."
)
@click.option(
    "--nasa_frp",
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
@click.option(
    "--compare-field",
    type=click.STRING,
    help="The column used in finding the nearest hotspots.",
    default="solar_day",
    show_default=True,
)
def main(
    nasa_frp: click.STRING,
    esa_frp: click.STRING,
    eumetsat_frp: click.STRING,
    landgate_frp: click.STRING,
    dea_frp: click.STRING,
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
    compare_field: Optional[str] = "solar_day"
) -> List:
    
    processing_parameters = {
        "nasa_frp": nasa_frp,
        "esa_frp": esa_frp,
        "eumetsat_frp": eumetsat_frp,
        "landgate_frp": landgate_frp,
        "dea_frp": dea_frp,
        "lon_west": lon_west,
        "lat_south": lat_south,
        "lon_east": lon_east,
        "lat_north": lat_north,
        "start_date": start_date,
        "end_date": end_date,
        "start_time": start_time,
        "end_time": end_time,
        "chunks": chunks,
        "outdir": outdir,
        "compare_field": compare_field
    }
    
    return process_nearest_points(**processing_parameters)

    
if __name__ == "__main__":
    # Configure log here for now, move it to __init__ at top level once
    # code is configured to run as module
    # client = Client(asynchronous=True)
    LOG_CONFIG = Path(__file__).parent.joinpath("logging.cfg")
    logging.config.fileConfig(LOG_CONFIG.as_posix())
    _LOG = logging.getLogger(__name__)
    main()
    # client.close()
