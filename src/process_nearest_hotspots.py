#!/usr/bin/env python3

import logging.config
import os
import re
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple

import boto3
import click
import dask
import dask.dataframe as dd
import pandas as pd
import geopandas as gpd

from dask.distributed import Client, LocalCluster
from geopy.distance import distance

import src.hotspot_utils as util

__s3_pattern__ = r"^s3://" r"(?P<bucket>[^/]+)/" r"(?P<keyname>.*)"

_LOG = logging.getLogger(__name__)


def unique_product_hotspots(
    product_a: str,
    all_hotspots_gdf: gpd.GeoDataFrame,
    compare_field: str,
    swath_directory: Union[Path, str],
    outfile: Union[Path, str],
) -> Union[Path, None]:
    """Helper method to compute nearest hotspots for a unique product_a.
    
    :param product_a: The product_a to compare against all the products.
    :param all_hotspots_gdf: The GeoDataFrame with all the hotspots datasets.
    :param compare_field: The solar_day or solar_night.
    :param swath_directory: The directory where swath data are stored.
    :param outfile: The csv outfile to save the nearest hotspots result for product_a
    
    :returns:
        None if no nearest hotspots are present.
        The output file path if nearest hotspots are present.
    """
    unique_products = [p for p in all_hotspots_gdf["satellite_sensor_product"].unique()]
    nearest_hotspots_dfs = []
    gdf_a = all_hotspots_gdf[
        all_hotspots_gdf["satellite_sensor_product"] == product_a
    ]
    for product_b in unique_products:
        _LOG.info(f"Comparing Hotspots for {product_a} and {product_b}")
        geosat_flag = False
        if ("AHI" in [product_a, product_b]) or ("INS1" in [product_a, product_b]):
            geosat_flag = True
        gdf_b = all_hotspots_gdf[
            all_hotspots_gdf["satellite_sensor_product"] == product_b
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
    if not nearest_hotspots_dfs:
        return None
    nearest_hotspots_dfs = pd.concat(nearest_hotspots_dfs, ignore_index=True)
    nearest_hotspots_dfs.reset_index(inplace=True, drop=True)
    nearest_hotspots_dfs.to_csv(outfile.as_posix())
    return outfile


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
    swath_config_file: Optional[Union[Path, str]] = None,
) -> List:
    """Processing of nearest points for different products in hotpots.

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
    
    if not Path(outdir).exists():
        Path(outdir).mkdir(exist_ok=True)
    
    if swath_config_file is None:
        swath_config_file = Path(__file__).parent.parent.joinpath("configs", "s3vtconfig.yaml")
    
    _LOG.info(f"Processing FRP Hotspots Datasets")
    hotspots_gdf = util.process_hotspots_gdf(
        nasa_frp=nasa_frp,
        esa_frp=esa_frp,
        eumetsat_frp=eumetsat_frp,
        landgate_frp=landgate_frp,
        dea_frp=dea_frp,
        start_date=start_date,
        end_date=end_date,
        start_time=start_time,
        end_time=end_time,
        bbox=(lon_west, lat_south, lon_east, lat_north),
        chunks=chunks,
        outdir=outdir,
    )

    _LOG.info(f"Processing Neareast Hotspots...")
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
    
    # Compute swath genration tasks sequentially.
    # for swath_task in swath_generation_tasks:
    #     swath_task.compute()
    _ = dask.compute(*swath_generation_tasks)

    _LOG.info(f"Generating neareast hotspots...")
    unique_products = [
        p for p in hotspots_gdf["satellite_sensor_product"].unique()
    ]
    all_product_tasks = []
    for product_a in unique_products:
        outfile = Path(outdir).joinpath(f"nearest_points_{product_a}_{compare_field}.csv")
        if outfile.exists():
            _LOG.info(
                f"{outfile.as_posix()} exists. skipped nearest"
                f" hotspots processing for product {product_a}."
                " Same file will be used in analysis."
            )
            continue
        all_product_tasks.append(dask.delayed(unique_product_hotspots)(product_a, hotspots_gdf, compare_field, swath_directory, outfile))
    outfiles = [fid for fid in dask.compute(*all_product_tasks) if fid is not None]
    return outfiles
    

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
    compare_field: Optional[str] = "solar_day",
    
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
    main()
    # client.close()
