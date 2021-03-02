#!/usr/bin/env python3

import logging
import logging.config

import os
from typing import Optional, Tuple, Union, Dict
from pathlib import Path

from datetime import datetime

import pandas as pd
import geopandas as gpd
import click

import dask
import dask.dataframe as dd
from dask.distributed import Client

import hotspot_utils as util


_LOG = logging.getLogger(__name__)



def main(
    hotspots_files: Dict,
    lon_west,
    lat_south,
    lon_east,
    lat_north,
    start_date: str,
    end_date: str,
    start_time: str,
    end_time: str,
    chunks: Optional[int] = 100
):
    """Main processing for s3vt fire hotspots analysis."""
    _LOG.info("Running fire hotspots analysis...")
    process_kwargs = {
        "bbox": (lon_west, lat_south, lon_east, lat_north),
        "start_date": start_date,
        "end_date": end_date,
        "start_time": start_time,
        "end_time": end_time,
        "num_chunks": chunks
    }

    _LOG.info("Reading spatial and temporal subsets of all hotspots dataframes...")
    all_hotspots_tasks = util.get_all_hotspots_tasks(hotspots_files, **process_kwargs)
    attrs_normalization_tasks = [dask.delayed(util.normalize_features)(df) for df in all_hotspots_tasks]

    _LOG.info("Merging hotspots dataframe...")
    hotspots_gdf = pd.concat([df for df in dask.compute(*attrs_normalization_tasks)])
    print(hotspots_gdf)

    _LOG.info("The spatial and temporal extents of merged hotspots.")
    _LOG.info(f"minimum datetime: {hotspots_gdf['solar_day'].min()}, maximum datetime: {hotspots_gdf['solar_day'].max()}")
    _LOG.info(f"longitude range: {hotspots_gdf['longitude'].min()}  to {hotspots_gdf['longitude'].max()}")
    _LOG.info(f"latitude range: {hotspots_gdf['latitude'].min()}  to {hotspots_gdf['latitude'].max()}")

    unique_products = [p for p in hotspots_gdf["satellite_sensor_product"].unique()]

    for productA in unique_products:
        hotspots_compasre_tasks = []
        for productB in unique_products:
            geosat_flag = False

            if ('AHI' in [productA, productB]) | ('INS1' in [productA, productB]):
                geosat_flag = True

            gdfA = hotspots_gdf[hotspots_gdf["satellite_sensor_product"] == productA]
            gdfB = hotspots_gdf[hotspots_gdf["satellite_sensor_product"] == productB]
            hotspots_compasre_tasks.append(
                dask.delayed(util.hotspots_compare)(
                    gdfA,
                    gdfB,
                    lon_east,
                    lon_west,
                    "solar_day",
                    "s3vtconfig.yaml",
                    geosat_flag
                )
            )
        productA_nearest_gdfs = dask.compute(*hotspots_compasre_tasks)
        productA_nearest_gdfs_merged = pd.concat([df for df in productA_nearest_gdfs if df is not None])
        productA_nearest_gdfs_merged.reset_index(inplace=True, drop=True)
        productA_nearest_gdfs_merged.to_csv(f"nearest_points.{productA}.csv")



if __name__ == "__main__":

    # Configure log here for now, move it to __init__ at top level once
    # code is configured to run as module

    LOG_CONFIG = Path(__file__).parent.joinpath("logging.cfg")
    logging.config.fileConfig(LOG_CONFIG.as_posix())

    hotspots_files = {
        "nasa": "nasa_hotspots_gdf.geojson",
        "esa": "s3vt_hotspots.geojson",
        "eumetsat": "s3vt_eumetsat_hotspots.geojson",
        "landgate": "landgate_hotspots_gdf.geojson",
        "dea": None
    }
    lon_west = 147.0
    lon_east = 154.0
    lat_south = -38.0
    lat_north = -27.0
    start_date = "2019-11-01"
    end_date = "2020-10-8"
    start_time = "21:00"
    end_time = "3:00"
    chunks = 100
    main(
        hotspots_files,
        lon_west,
        lat_south,
        lon_east,
        lat_north,
        start_date,
        end_date,
        start_time,
        end_time,
        chunks=chunks
    )

