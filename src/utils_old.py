#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple, Union, Optional, List, Dict

import gc
import matplotlib.pyplot as plt
import seaborn
import boto3
from botocore.exceptions import ClientError
from netCDF4 import Dataset
import datetime as dt
import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import shapely.speedups
shapely.speedups.enable()
#import xmltodict
import yaml
import os
import subprocess
from datetime import date
from datetime import datetime
from datetime import timedelta
import logging
logger = logging.getLogger()
import os
import netCDF4
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString
from geopy.distance import distance
from pathlib import Path
import shutil
import requests
from requests.auth import HTTPBasicAuth
import folium
import json
import joblib
import time
import geopandas as gpd
import pandas as pd
import shapely.speedups

import subprocess
from datetime import datetime
import logging as logger

import os
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
import dask
import dask.dataframe as dd
from dask import delayed

logger.basicConfig(format="%(levelname)s:%(message)s", level=logger.INFO)

__features__ = [
    "latitude",
    "longitude",
    "satellite",
    "sensor",
    "confidence",
    "power",
    "datetime",
    "solar_day",
    "satellite_sensor_product",
    "geometry",
]

__viirs_confidence_maps__ = {"h": 100.0, "l": 10.0, "n": 50.0}

__slstr_ignore_attrs__ = [
    'FRP_MWIR',
    'FRP_SWIR',
    'FRP_uncertainty_MWIR',
    'FLAG_SWIR_SAA',
    'FRP_uncertainty_SWIR',
    'Glint_angle',
    'IFOV_area',
    'Radiance_window',
    'S7_Fire_pixel_radiance',
    'TCWV',
    'classification', 
    'i',
    'j',
    'n_SWIR_fire',
    'n_cloud',
    'n_water',
    'n_window',
    'time',
    'transmittance_MWIR',
    'transmittance_SWIR',
    'used_channel'
]


def get_satellite_swaths(configuration, start, period, solar_day):
    """
    Function to determine the common imaging footprint of a pair of sensors

    Returns the imaging footprints of a pair of sensors for a period from a starting datetime
    """
    output = Path("output")
    dirpath = Path.joinpath(output, solar_day)

    if dirpath.exists():
        logger.info(str(solar_day) + " exists - skipping swath generation")
        success = True
    else:
        dirpath.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(
                "Generating swaths "
                + str(
                    [
                        "python",
                        "swathpredict.py",
                        "--configuration",
                        configuration,
                        "--start",
                        start,
                        "--period",
                        period,
                        "--output_path",
                        str(dirpath),
                    ]
                )
            )
            subprocess.call(
                [
                    "python",
                    "swathpredict.py",
                    "--configuration",
                    configuration,
                    "--start",
                    start,
                    "--period",
                    period,
                    "--output_path",
                    str(dirpath),
                ]
            )

            success = True
        except:
            success = False
            logger.info("Swath generation failed")

    return success


def pairwise_swath_intersect(satsensorsA, satsensorsB, solar_day):
    """
    Function intersect geometries of sensors for a given solar day

    Returns intersection geometry
    """
    logger.info(
        "Running intersection for " + str(satsensorsA) + " " + str(satsensorsB)
    )
    satsensorsA = [w.replace(" ", "_") for w in satsensorsA]
    satsensorsB = [w.replace(" ", "_") for w in satsensorsB]

    filesA = []
    filesB = []

    output = Path("output")
    dirpath = Path.joinpath(output, solar_day)

    # Add a time interval for the intersection based on start_time end_time

    for sat in satsensorsA:

        filesA.extend(
            [
                f
                for f in os.listdir(str(dirpath))
                if (sat in f) and ("swath.geojson" in f)
            ]
        )

    for sat in satsensorsB:

        filesB.extend(
            [f for f in os.listdir(str(dirpath)) if sat in f and "swath.geojson" in f]
        )

    gpdlistA = []
    for file in filesA:
        df = gpd.read_file(Path.joinpath(dirpath, file))
        gpdlistA.append(df)
    gpdlistB = []
    for file in filesB:
        df = gpd.read_file(Path.joinpath(dirpath, file))
        gpdlistB.append(df)
    return (pd.concat(gpdlistA), pd.concat(gpdlistB))


def compare_hotspots(
    productA: str,
    hotspots_gdf: gpd.GeoDataFrame,
    eastlon: float,
    westlon: float,
):
    """
    Function compares sensor swaths of productA to those of sensors in hotspots_gdf -subsets hotspots to
    common imaged area and determines points nearest to productA from hotspots_gdf. Computes distance between points.

    Returns dataframe containing matched pairs between the two sets of hotspots as w
    """
    appended_dataframe = []
    nearest_points_list = []
    satellite_sensor_product_intersections = {}
    for productB in set(hotspots_gdf["satellite_sensor_product"]):
        gdfA = hotspots_gdf[(hotspots_gdf["satellite_sensor_product"] == productA)]
        gdfB = hotspots_gdf[(hotspots_gdf["satellite_sensor_product"] == productB)]

        # For each solar day group in gdfA
        for Aname, Agroup in gdfA.resample("D", on="solar_day"):

            minutctime, maxutctime, deltautctime = solar_day_start_stop_period(
                eastlon, westlon, Aname
            )

            # For each solar day group in gdfB
            for Bname, Bgroup in gdfB.resample("D", on="solar_day"):

                # Do where the solar days are the same in gdfA and B
                if Aname == Bname:
                    logger.info(productA + " " + productB)

                    logger.info(
                        str(Aname)
                        + " "
                        + str(minutctime)
                        + " "
                        + str(minutctime)
                        + " "
                        + str(deltautctime)
                    )

                    # Generate the GeoJSON for each satellite in s3vtconfig.yaml

                    get_satellite_swaths(
                        "s3vtconfig.yaml",
                        minutctime.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        str(int(deltautctime.total_seconds() / 60)),
                        str(Aname.date()),
                    )

                    # Geostationary satellites need an exception
                    if not (
                        ("AHI" in [productA, productB])
                        or ("INS1" in [productA, productB])
                    ):

                        # Include a try except to counteract failures where swath intersect fails
                        try:

                            # Get geometries for satellite sensors in gpdA and gpdB
                            gpd1, gpd2 = pairwise_swath_intersect(
                                set(Agroup["satellite"]),
                                set(Bgroup["satellite"]),
                                str(Aname.date()),
                            )

                            # Union before intersect
                            gpd1 = gpd1.unary_union
                            gpd2 = gpd2.unary_union

                            # Intersect geometries
                            intersection = gpd1.intersection(gpd2)
                            logger.info(str(intersection))

                            if intersection == None:
                                logger.info("Intersection is None")
                            else:
                                logger.info("Intersection successful")
                            # Use intersection results to subset points (compare common imaged area)

                            logger.info(
                                "Before intersection "
                                + str(Aname)
                                + " "
                                + str(Agroup["satellite_sensor_product"].count())
                                + " "
                                + str(Bgroup["satellite_sensor_product"].count())
                            )

                            pip_mask = Agroup.within(intersection)
                            Agroup = Agroup.loc[pip_mask]
                            Agroup.reset_index(drop=True, inplace=True)

                            pip_mask = Bgroup.within(intersection)
                            Bgroup = Bgroup.loc[pip_mask]
                            Bgroup.reset_index(drop=True, inplace=True)
                            logger.info(
                                "After intersection "
                                + str(Aname)
                                + " "
                                + str(Agroup["satellite_sensor_product"].count())
                                + " "
                                + str(Bgroup["satellite_sensor_product"].count())
                            )

                            if (Agroup["solar_day"].count() == 0) or (
                                Bgroup["solar_day"].count() == 0
                            ):
                                logger.info("Nothing to input to ckdnearest")

                            per_solarday_nearest_hotspots = ckdnearest(Agroup, Bgroup)

                            appended_dataframe.append(per_solarday_nearest_hotspots)

                        except:
                            logger.info("Skipping")
                    else:
                        # Himawari AHI or INS1 geostationary case
                        # A better approach here is to check if either has a swath available
                        # If not - defer to the intersection of the one with a geometry
                        # TODO - improve for Himawari
                        try:
                            per_solarday_nearest_hotspots = ckdnearest(
                                Agroup.reset_index(drop=True, inplace=True),
                                Bgroup.reset_index(drop=True, inplace=True),
                            )
                            print(len(appended_dataframe))
                            appended_dataframe.append(per_solarday_nearest_hotspots)
                            # per_solarday_nearest_hotspots.to_file(productA+'.geojson')
                        except:
                            logger.info("Skipping")
        try:

            nearest_points = gpd.GeoDataFrame(
                pd.concat(appended_dataframe, ignore_index=True)
            )
            nearest_points.reset_index(inplace=True, drop=True)

            outputfile = "nearest_points." + productA + ".csv"
            nearest_points.to_csv(outputfile)
        except Exception:
            logger.info("Nothing to concatenate")

    return productA


def solar_day_start_stop_period(longitude_east, longitude_west, solar_day):
    """
    Function solar day start time from longitude and solar day in utc

    Returns datetime start stop in utc and period between in minutes
    """
    # Solar day time relative to UTC and local longitude
    SECONDS_PER_DEGREE = 240
    # Offset for eastern limb
    offset_seconds_east = int(longitude_east * SECONDS_PER_DEGREE)
    offset_seconds_east = np.timedelta64(offset_seconds_east, "s")
    # offset for wester limb
    offset_seconds_west = int(longitude_west * SECONDS_PER_DEGREE)
    offset_seconds_west = np.timedelta64(offset_seconds_west, "s")
    # time between two limbs
    offset_day = np.timedelta64(1440, "m") + abs(
        offset_seconds_east - offset_seconds_west
    )
    # ten_am_crossing_adjustment = np.timedelta64(120, 'm')
    # Solar day start at eastern limb
    solar_day_start_utc = (np.datetime64(solar_day) - offset_seconds_east).astype(
        datetime
    )
    # Solar day finish at western limb
    solar_day_finish_utc = (
        (np.datetime64(solar_day) + offset_day) - offset_seconds_east
    ).astype(datetime)
    # Duration of solar day
    solar_day_duration = np.timedelta64(
        (solar_day_finish_utc - solar_day_start_utc), "m"
    )

    return (
        solar_day_start_utc,
        solar_day_finish_utc,
        solar_day_duration.astype(datetime),
    )


def solar_day(utc, longitude):
    """
    Function solar day for a given UTC time and longitude input

    Returns datetime object representing solar day
    """
    SECONDS_PER_DEGREE = 240
    offset_seconds = int(longitude * SECONDS_PER_DEGREE)
    offset = np.timedelta64(offset_seconds, "s")
    return (np.datetime64(utc) + offset).astype(datetime)


def ckdnearest(gdA, gdB):
    """
    Function to find points in "B" nearest to "A" geopandas dataframe

    Returns geopandas dataframe with records representing matches
    """
    nA = np.array(list(zip(gdA.geometry.x, gdA.geometry.y)))
    nB = np.array(list(zip(gdB.geometry.x, gdB.geometry.y)))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdf = gpd.GeoDataFrame(
        pd.concat(
            [
                gdA.reset_index(drop=True),
                gdB.loc[idx].reset_index(drop=True).add_prefix("2_"),
                pd.Series(dist, name="dist"),
            ],
            axis=1,
        )
    )
    return gdf


def load_csv(
    csv_file: Union[Path, str],
    lazy_load: Optional[bool] = True,
    dtype: Optional[dict] = None,
    skiprows: Optional[int] = 0,
):
    """Method to lazy load csv file using pandas dataframe or dask dataframe.

    :param csv_file: Full path to a csv file.
    :param lazy_load: The flag to indicate whether to lazy load using dask?
    """
    # if not lazy loading then use pandas dataframe to read.
    if not lazy_load:
        return pd.read_csv(csv_file, dtype=dtype, skiprows=skiprows)

    # if lazy load then use dask.dataframe to lazy load
    return dd.read_csv(csv_file, dtype=dtype, skiprows=skiprows)


def load_geojson(
    geojson_file: Union[Path, str],
    bbox: Optional[Tuple] = None,
    ignore_fields: Optional[List] = None
) -> gpd.GeoDataFrame:
    """Method to read a geojson features spatially subsetted by a bbox.

    :param geojson_file: Full path to a geojson file to read.
    :param bbox: The bounding box to subset features of geojson file.
    :param ignore_fields: The fields to ignore from when reading geojson file.
    
    :returns:
        The GeoDataFrame loaded from geojson file.
    """
    if ignore_fields is None:
        return gpd.read_file(geojson_file, bbox=bbox)
    return gpd.read_file(geojson_file, bbox=bbox, ignore_fields=ignore_fields)


def temporal_subset_df(
    df: gpd.GeoDataFrame,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> gpd.GeoDataFrame:
    """Method to temporally subset the data
    
    :param df: The data to be subsetted.
    :param start_time: The start time to subset the data.
    :param end_time: The end time to subset the data.
    :param start_date: The start date to subset the data.
    :param end_date: The end date to subset the data.
    
    :returns:
        The temporally subsetted GeoDataFrame
    """
    # create solar day as a the datetime index
    df = df.set_index(pd.DatetimeIndex(df.solar_day.values))
    
    if (start_date is not None) & (end_date is not None):
        df = df.loc[start_date:end_date]
    if (start_time is not None) & (end_time is not None):
        df = df.between_time(start_time, end_time)
    return df


def chunk_gpd(
    df: gpd.GeoDataFrame,
    num_chunks: Optional[int] = 1
) -> List:
    """Method to divide df into multiple chunks.
    
    :param df: The GeoDataFrame to be sub-divided into chunks.
    :param num_chunks: The number of blocks to be sub-divided into.
    
    :returns:
        The list of data sub-divided.
    """
    return np.array_split(df, num_chunks)


def normalize_features(
    df: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Method to prepare eumetsat data equivalent.

    :param df: GeoPandas dataframe object.
    
    :returns:
        The GeoDataFrame.
    """
    # drop cols not in __features__
    cols_drop = set(df.columns) - set(__features__)
    gdf = df.drop(cols_drop, axis=1)
    gdf["confidence"] = gdf["confidence"].map(__viirs_confidence_maps__)
    gdf["confidence"] = gdf["confidence"].fillna(-1.0)
    return gdf


def modis_viirs_temporal_subset_normalize(
    gdf: gpd.GeoDataFrame,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Method to temporally subset and normalize features attributes for MODIS and VIIRS product.

    :param gdf: The GeoDataFrame dataset.
    :param start_time: The start time to subset the data.
    :param end_time: The end time to subset the data.
    :param start_date: The start date to subset the data.
    :param end_date: The end date to subset the data
    
    :returns:
        The subsetted GeoDataFrame.
    """
    gdf["datetime"] = pd.to_datetime(gdf["datetime"])
    gdf["solar_day"] = pd.to_datetime(gdf["solar_day"])
    gdf = temporal_subset_df(gdf, start_time, end_time, start_date, end_date)
    return gdf


def slstr_temporal_subset_normalize(
    gdf: gpd.GeoDataFrame,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    provider: Optional[str] = None
) -> gpd.GeoDataFrame:
    """Method to temporally subset and normalize features attributes for SLSTR product.

    :param gdf: The GeoDataFrame dataset.
    :param start_time: The start time to subset the data.
    :param end_time: The end time to subset the data.
    :param start_date: The start date to subset the data.
    :param end_date: The end date to subset the data
    
    :returns:
        The subsetted GeoDataFrame.
    """
    gdf["datetime"] = pd.to_datetime(gdf["date"])
    gdf["solar_day"] = pd.to_datetime(gdf["solar_day"])
    
    if provider == "eumetsat":
        #TODO fix this while write geojson files.
        gdf = gdf.replace("SENTINEL_S3A", "SENTINEL_3A")
        gdf = gdf.replace("SENTINEL_S3B", "SENTINEL_3B")
        gdf["satellite_sensor_product"] = (
            gdf["satellite"] + "_" + gdf["sensor"] + "_EUMETSAT"
        )
    elif provider == "esa":
        gdf["satellite_sensor_product"] = (
            gdf["satellite"] + "_" + gdf["sensor"] + "_ESA"
        )
    else:
        raise ValueError(f"Provider must be esa or eumetsat: not {provider}")

    gdf.rename(columns={"F1_Fire_pixel_radiance": "power"}, inplace=True)
    gdf = temporal_subset_df(gdf, start_time, end_time, start_date, end_date)
    return gdf


def process_nasa_hotspots(
    geojson_file: Union[Path, str],
    bbox: Optional[Tuple] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    num_chunks: Optional[int] = 1
) -> List[dask.delayed]:
    """Method to load, subset and normalize feature attributes for NASA product.

    :param geojson_file: Full path to a geojson file to read.
    :param bbox: The bounding box to subset features of geojson file.
    :param start_time: The start time to subset the data.
    :param end_time: The end time to subset the data.
    :param start_date: The start date to subset the data.
    :param end_date: The end date to subset the data
    :param num_chunks: The number of blocks to be sub-divided into.

    :returns:
        List of dask delayed tasks that would return subsetted GeoDataFrame and normalize features.
    """
    gdf = load_geojson(geojson_file, bbox=bbox)
    gdf_chunks = chunk_gpd(gdf, num_chunks) 
    
    gdf_tasks = []
    for df in gdf_chunks:
        task = delayed(modis_viirs_temporal_subset_normalize)(df, start_time, end_time, start_date, end_date)
        gdf_tasks.append(task)
    return gdf_tasks


def process_dea_hotspots(
    geojson_file: Union[Path, str],
    bbox: Optional[Tuple] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    num_chunks: Optional[int] = 1
) -> List[dask.delayed]:
    """Method to load, subset and normalize features attributes for DEA product.

    :param geojson_file: Full path to a geojson file to read.
    :param bbox: The bounding box to subset features of geojson file.
    :param start_time: The start time to subset the data.
    :param end_time: The end time to subset the data.
    :param start_date: The start date to subset the data.
    :param end_date: The end date to subset the data
    :param num_chunks: The number of blocks to be sub-divided into.

    :returns:
        List of dask delayed tasks that would return subsetted GeoDataFrame and normalize features.    """
    gdf = load_geojson(geojson_file, bbox=bbox)
    gdf_chunks = chunk_gpd(gdf, num_chunks) 
    
    gdf_tasks = []
    for df in gdf_chunks:
        task = delayed(modis_viirs_temporal_subset_normalize)(df, start_time, end_time, start_date, end_date)
        gdf_tasks.append(task)
    return gdf_tasks


def process_landgate_hotspots(
    geojson_file: Union[Path, str],
    bbox: Optional[Tuple] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    num_chunks: Optional[int] = 1
) -> List[dask.delayed]:
    """Method load and normalize features attributes for Landgate product.

    :param geojson_file: Fupp path to a geojson file to read.
    :param bbox: The bounding box to subset features of geojson file.
    :param start_time: The start time to subset the data.
    :param end_time: The end time to subset the data.
    :param start_date: The start date to subset the data.
    :param end_date: The end date to subset the data
    :param num_chunks: The number of blocks to be sub-divided into.

    :returns:
        List of dask delayed tasks that would return subsetted GeoDataFrame and normalize features.
    """
    gdf = load_geojson(geojson_file, bbox=bbox)
    gdf_chunks = chunk_gpd(gdf, num_chunks) 
    
    gdf_tasks = []
    for df in gdf_chunks:
        task = delayed(modis_viirs_temporal_subset_normalize)(df, start_time, end_time, start_date, end_date)
        gdf_tasks.append(task)
    return gdf_tasks


def process_eumetsat_hotspots(
    geojson_file: Union[Path, str],
    bbox: Optional[Tuple] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    num_chunks: Optional[int] = 1
) -> List[dask.delayed]:
    """Method load, subset and normalize features attributes for EUMETSAT product.

    :param geojson_file: Fupp path to a geojson file to read.
    :param bbox: The bounding box to subset features of geojson file.
    :param start_time: The start time to subset the data.
    :param end_time: The end time to subset the data.
    :param start_date: The start date to subset the data.
    :param end_date: The end date to subset the data
    :param num_chunks: The number of blocks to be sub-divided into.

    :returns:
        List of dask delayed tasks that would return subsetted GeoDataFrame and normalize features.
    """
    gdf = load_geojson(geojson_file, bbox=bbox, ignore_fields=__slstr_ignore_attrs__)
    gdf_chunks = chunk_gpd(gdf, num_chunks) 
    
    gdf_tasks = []
    for df in gdf_chunks:
        task = delayed(slstr_temporal_subset_normalize)(df, start_time, end_time, start_date, end_date, "eumetsat")
        gdf_tasks.append(task)
    return gdf_tasks


def process_esa_hotspots(
    geojson_file: Union[Path, str],
    bbox: Optional[Tuple] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    num_chunks: Optional[int] = 1
) -> List[dask.delayed]:
    """Method load, subset and normalize features attributes for ESA product.

    :param geojson_file: Fupp path to a geojson file to read.
    :param bbox: The bounding box to subset features of geojson file.
    :param start_time: The start time to subset the data.
    :param end_time: The end time to subset the data.
    :param start_date: The start date to subset the data.
    :param end_date: The end date to subset the data
    :param num_chunks: The number of blocks to be sub-divided into.

    :returns:
        List of dask delayed tasks that would return subsetted GeoDataFrame and normalize features.
    """
    gdf = load_geojson(geojson_file, bbox=bbox, ignore_fields=__slstr_ignore_attrs__)
    gdf_chunks = chunk_gpd(gdf, num_chunks) 
    
    gdf_tasks = []
    for df in gdf_chunks:
        task = delayed(slstr_temporal_subset_normalize)(df, start_time, end_time, start_date, end_date, "esa")
        gdf_tasks.append(task)
    return gdf_tasks


def get_all_hotspots_tasks(
    input_files_dict: Dict,
    bbox: Optional[Tuple] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    num_chunks: Optional[int] = 1
) -> List[dask.delayed]:
    """Method to prepare hotspots geo dataframes from input geojson files.
        
    The data from geojson files are spatially subsetted to bbox extent and
    temporally subsetted to hours between start and end time, and dates between
    start and end date.
    
    :param geojson_file: Fupp path to a geojson file to read.
    :param bbox: The bounding box to subset features of geojson file.
    :param start_time: The start time to subset the data.
    :param end_time: The end time to subset the data.
    :param start_date: The start date to subset the data.
    :param end_date: The end date to subset the data
    :param num_chunks: The number of blocks to be sub-divided into.

    :returns:
        The List of all dask delayed tasks that would return GeoDataFrame
    """
    hotspots_tasks = []
    for name, fid in input_files_dict.items():
        if fid is None:
            continue
        logger.info(f"reading and subsetting GeoDataFrame for {name}: {fid}")
        kwargs = {
            "bbox": bbox,
            "start_time": start_time,
            "end_time": end_time,
            "start_date": start_date,
            "end_date": end_date,
            "num_chunks": num_chunks
        }

        if name == "nasa":
            tasks = process_nasa_hotspots(fid, **kwargs)
        elif name == "esa":
            tasks = process_esa_hotspots(fid, **kwargs)
        elif name == "eumetsat":
            tasks = process_eumetsat_hotspots(fid, **kwargs)
        elif name == "landgate":
            tasks = process_landgate_hotspots(fid, **kwargs)
        elif name == "dea":
            tasks = process_dea_hotpots(fid, **kwargs)
        hotspots_tasks += tasks
    return hotspots_tasks
    
    
def delta_datetime(df):
    return None


if __name__ == "__main__":
    os.environ["NUMEXPR_MAX_THREADS"] = "8"
    
    kwargs = {
        "lon_west": 147.0,
        "lon_east": 154.0,
        "lat_south": -38.0,
        "lat_north": -27.0,
        "start_date": "2019-11-01",
        "end_date": "2020-10-8",
        "start_time": "21:00",
        "end_time": "3:00"
    }

    files_dict = {
        "nasa": "nasa_hotspots_gdf.geojson",
        "esa": "s3vt_hotspots.geojson",
        "eumetsat": "s3vt_eumetsat_hotspots.geojson",
        "landgate": "landgate_hotspots_gdf.geojson",
        "dea": None
    }
    hotspots_df = get_all_hotspots_tasks(
        files_dict,
        start_time=start_time,
        end_time=end_time,
        start_date=start_date,
        end_date=end_date,
        num_chunks=num_chunks
    )
