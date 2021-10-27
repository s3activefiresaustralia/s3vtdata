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
    swath_gdf: gpd.GeoDataFrame,
    outfile: Union[Path, str],
    start_time: str,
    end_time: str,
    test: Optional[bool] = False
) -> Union[Path, None]:
    """Helper method to compute nearest hotspots for a unique product_a.
    
    :param product_a: The product_a to compare against all the products.
    :param all_hotspots_gdf: The GeoDataFrame with all the hotspots datasets.
    :param compare_field: The solar_day or solar_night.
    :param swath_gdf: The concatenated GeoDataFrame from the daily swath geojson files.
    :param outfile: The csv outfile to save the nearest hotspots result for product_a
    :param start_time: The start time to subset the data.
    :param end_time: The end time to subset the data.
    :param test: Flag to run in test/debug mode.
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
            swath_gdf,
            start_time,
            end_time,
            test=test
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
    sentinel3_swath_geojson: Union[Path, str],
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
    test: Optional[bool] = True
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
    :param sentinel3_swath_geojson: The full path to Sentine3 swath geodataframe geojson file.
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
    :param test: The flag to indicate if it is a test process.
    :returns:
        All the .csv output files path containing the nearest hotspots.
    """
    
    if not Path(outdir).exists():
        Path(outdir).mkdir(exist_ok=True)
    
    if swath_config_file is None:
        swath_config_file = Path(__file__).parent.parent.joinpath("configs", "s3vtconfig.yaml")
    
    hotspots_pkl_file = Path(outdir).joinpath(
        f"all_hotspots_{int(lon_east)}_{int(lon_west)}_{start_date.replace('-','')}_{end_date.replace('-','')}_{start_time.replace(':', '')}_{end_time.replace(':','')}.pkl"
    )
    if hotspots_pkl_file.exists():
        _LOG.info(f"{hotspots_pkl_file.as_posix()} exists. Reading hotspots GeoDataFrame from the file.")
        hotspots_gdf = pd.read_pickle(hotspots_pkl_file)
    else:
        _LOG.info(f"Processing FRP Hotspots from GeoJSON files")
        hotspots_gdf = util.process_hotspots_gdf(
            nasa_frp=nasa_frp,
            esa_frp=esa_frp,
            eumetsat_frp=eumetsat_frp,
            landgate_frp=landgate_frp,
            dea_frp=dea_frp,
            start_date=start_date,
            end_date=end_date,
            bbox=(lon_west, lat_south, lon_east, lat_north),
            chunks=chunks,
            outdir=outdir,
        )
        _LOG.info(f"Saving spatial and temporal subset hotspots dataframe to pickle file {hotspots_pkl_file.as_posix()}")
        hotspots_gdf.to_pickle(hotspots_pkl_file)
    
    if test:
        _LOG.info(f"Saving spatial and temporal hotspots dataframe to GeoJSON file {hotspots_pkl_file.with_suffix('.geojson').as_posix()}")
        hotspots_gdf.to_file(hotspots_pkl_file.with_suffix(".geojson"),
            driver="GeoJSON"
        )
    
    start_time_utc, end_time_utc = util.convert_solar_time_to_utc(lon_east, lon_west, start_time, end_time)    
    _LOG.info(f"Processing subset between {start_time_utc} and {end_time_utc} utc time")    
    hotspots_gdf = hotspots_gdf.set_index("datetime", drop=False)
    hotspots_gdf = hotspots_gdf.between_time(start_time_utc, end_time_utc)
    print(hotspots_gdf.head())

    _LOG.info(f"Processing Neareast Hotspots...")
    solar_start_dt = hotspots_gdf['solar_day'].min()
    solar_end_dt = hotspots_gdf['solar_day'].max()
    
    swath_directory = Path(outdir).joinpath(
        f"swaths_{int(lon_east)}_{int(lon_west)}_{solar_start_dt.strftime('%Y%m%d')}_{solar_end_dt.strftime('%Y%m%d')}"
    )
    swath_pkl_file = swath_directory.with_suffix(".pkl")
    
    if swath_pkl_file.exists():
        _LOG.info(f"{swath_pkl_file.as_posix()} exists. Reading Swath GeoDataFrame from the file.")
        swath_gdf = pd.read_pickle(swath_pkl_file)
    else:
        if not swath_directory.exists():
            swath_directory.mkdir(exist_ok=True, parents=True)
        _LOG.info(f"Generating satellite swaths from {solar_start_dt.date()} to {solar_end_dt.date()}")
        swath_generation_tasks = util.swath_generation_tasks(
            solar_start_dt,
            solar_end_dt,
            lon_east,
            lon_west,
            swath_directory=swath_directory,
            config_file=swath_config_file
        )

        # Compute swath genration tasks sequentially.
        for swath_task in swath_generation_tasks:
            swath_task.compute()

        # _ = dask.compute(*swath_generation_tasks)
            # Create concatenated swath GeoDataFrame from the daily swath geometry.
        _LOG.info("Generating satellite swath concatenated GeoDataFrame..")
        
        sentinel3_swath_geojson = util.fetch_sentinel3_swath_files(sentinel3_swath_geojson, outdir)
        s3_swath_gdf = gpd.read_file(sentinel3_swath_geojson)
        s3_swath_gdf = s3_swath_gdf[(s3_swath_gdf['AcquisitionOfSignalUTC'] >= start_date) & (s3_swath_gdf['AcquisitionOfSignalUTC'] <= end_date)]
        swath_gdf = util.concat_swath_gdf(
            swath_directory,
            s3_swath_gdf,
            archive=True,
            delete=True
        )
        swath_gdf.to_pickle(swath_pkl_file)
    
    _LOG.info("Subsetting only valid swath geometry...")
    swath_gdf = swath_gdf[swath_gdf['geometry'].is_valid == True]
    
    _LOG.info(f"Generating neareast hotspots...")
    unique_products = [
        p for p in hotspots_gdf["satellite_sensor_product"].unique()
    ]
   
    all_product_tasks = []
    for product_a in unique_products:
        outfile = Path(outdir).joinpath(f"nearest_points_{product_a}_{start_time.replace(':','')}_{end_time.replace(':','')}.csv")
        if outfile.exists():
            _LOG.info(
                f"{outfile.as_posix()} exists. skipped nearest"
                f" hotspots processing for product {product_a}."
                " Same file will be used in analysis."
            )
            continue
        all_product_tasks.append(dask.delayed(unique_product_hotspots)(product_a, hotspots_gdf, compare_field, swath_gdf, outfile, start_time_utc, end_time_utc, test=test))
    outfiles = [fid for fid in dask.compute(*all_product_tasks) if fid is not None]
    return outfiles
