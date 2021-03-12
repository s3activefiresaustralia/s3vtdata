#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import re
import subprocess
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Tuple, Union, Optional, List, Dict

import boto3
import dask
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
from dask import delayed
from fiona.errors import DriverError
from geopy.distance import distance
from scipy.spatial import cKDTree

_LOG = logging.getLogger(__name__)


__features__ = [
    "latitude",
    "longitude",
    "satellite",
    "sensor",
    "confidence",
    "power",
    "datetime",
    "solar_day",
    "solar_night",
    "satellite_sensor_product",
    "geometry",
]

__viirs_confidence_maps__ = {"h": 100.0, "l": 10.0, "n": 50.0}

__slstr_ignore_attrs__ = [
    "FRP_MWIR",
    "FRP_SWIR",
    "FRP_uncertainty_MWIR",
    "FLAG_SWIR_SAA",
    "FRP_uncertainty_SWIR",
    "Glint_angle",
    "IFOV_area",
    "Radiance_window",
    "S7_Fire_pixel_radiance",
    "TCWV",
    "classification",
    "i",
    "j",
    "n_SWIR_fire",
    "n_cloud",
    "n_water",
    "n_window",
    "time",
    "transmittance_MWIR",
    "transmittance_SWIR",
    "used_channel",
]

__s3_pattern__ = r"^s3://" r"(?P<bucket>[^/]+)/" r"(?P<keyname>.*)"


def get_satellite_swaths(
    configuration: Union[Path, str],
    solar_dt: np.datetime64,
    lon_east: float,
    lon_west: float,
    swath_directory: Optional[Union[Path, str]] = None
) -> bool:
    """
    Function to determine the common imaging footprint of a pair of sensors

    Returns the imaging footprints of a pair of sensors for a period from a starting datetime
    """
    if swath_directory is None:
        swath_directory = Path(os.getcwd()).joinpath(f"output_{int(lon_east)}_{int(lon_west)}")
        swath_directory.mkdir(parents=True, exist_ok=True)
    solar_date = str(solar_dt.date())

    solar_date_swath_dir = swath_directory.joinpath(solar_date)
    if solar_date_swath_dir.exists():
        _LOG.debug(str(solar_date) + " exists - skipping swath generation")
        return True

    solar_date_swath_dir.mkdir(parents=True)
    min_dt, _, delta_dt = solar_day_start_stop_period(
            lon_east, lon_west, solar_dt
    )
    start = min_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    period = str(int(delta_dt.total_seconds() / 60))
    _LOG.debug(
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
                solar_date_swath_dir.as_posix(),
            ]
        )
    )
    try:
        subprocess.check_call(
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
                solar_date_swath_dir.as_posix(),
            ]
        )
        return True
    except Exception as err:
        _LOG.debug(f"Swath generation failed with err: {err}")

    return False


def pairwise_swath_intersect(
    satsensors_a: set,
    satsensors_b: set,
    swath_directory: Union[Path, str]
):
    """
    Function intersect geometries of sensors for a given solar day

    Returns intersection geometry
    """
    _LOG.debug(
        "Running intersection for " + str(satsensors_a) + " " + str(satsensors_b)
    )
    satsensors_a = [w.replace(" ", "_") for w in satsensors_a]
    satsensors_b = [w.replace(" ", "_") for w in satsensors_b]

    files_a = []
    files_b = []

    # Add a time interval for the intersection based on start_time end_time

    for sat in satsensors_a:
        files_a.extend(
            [
                fp
                for fp in Path(swath_directory).iterdir()
                if (sat in fp.name) and ("swath.geojson" in fp.name)
            ]
        )

    for sat in satsensors_b:
        files_b.extend(
            [
                fp
                for fp in Path(swath_directory).iterdir()
                if (sat in fp.name) and ("swath.geojson" in fp.name)
            ]
        )

    gpdlist_a = []
    for _fp in files_a:
        df = gpd.read_file(_fp.as_posix())
        gpdlist_a.append(df)
    gpdlist_b = []
    for _fp in files_b:
        df = gpd.read_file(_fp.as_posix())
        gpdlist_b.append(df)
    return pd.concat(gpdlist_a), pd.concat(gpdlist_b)


def get_nearest_hotspots(
    gdfa: gpd.GeoDataFrame,
    gdfb: gpd.GeoDataFrame,
    geosat_flag: bool,
    swath_directory: Union[Path, str]
) -> Union[None, gpd.GeoDataFrame]:
    """Method to compute nearest hotspots between two GeoDataFrame.

    :param gdfa: The GeoDataFrame
    :param gdfb: The GeoDataFrame
    :param geosat_flag: The flag to indicate if hotspot is from geostationary statellite.
    :param swath_directory: The directory where swath files for solar_date is locate.

    :returns:
        None if no intersections or fails in pairwise swath intersects.
        GeoDataFrame from cdknearest method.
    """
    if not geosat_flag:
        try:
            gpd1, gpd2 = pairwise_swath_intersect(
                set(gdfa["satellite"]), set(gdfb["satellite"]), swath_directory
            )
        except Exception as err:
            _LOG.debug(err)
            return None

        gpd1 = gpd1.unary_union
        gpd2 = gpd2.unary_union

        intersection = gpd1.intersection(gpd2)
        if intersection is None:
            _LOG.debug(f"Intersection is None for {swath_directory.name}")
            return None

        maska = gdfa.within(intersection)
        gdfa = gdfa.loc[maska]
        gdfa.reset_index(drop=True, inplace=True)

        maskb = gdfb.within(intersection)
        gdfb = gdfb.loc[maskb]
        gdfb.reset_index(drop=True, inplace=True)
    else:
        gdfa.reset_index(drop=True, inplace=True)
        gdfb.reset_index(drop=True, inplace=True)

    if gdfa.empty | gdfb.empty:
        _LOG.debug("Nothing to input to cdknearest")
        return None

    nearest_hotspots = ckdnearest(gdfa, gdfb)

    return nearest_hotspots


def swath_generation_tasks(
    start_dt: datetime,
    end_dt: datetime,
    lon_east: float,
    lon_west: float,
    swath_directory: Optional[Union[Path, str]] = None,
    config_file: Optional[Union[Path, str]] = "s3vtconfig.yaml"
) -> List[dask.delayed]:
    """Method to genrate satellite swaths from start_dt to end_date at 1 day interval.

    :param start_dt: The start datetime to generate the swath from.
    :param end_dt: The end datetime to end the swath genration.
    :param lon_east: The eastern longitude used in spatial subset.
    :param lon_west: The western longitude used in a spatial subset.
    :param swath_directory: The parent directory to store the solar_date swath files.
    :param config_file: The config file to be used in swath generation.

    :returns:
        List of delayed tasks to generate daily satellite swaths.
    """
    if swath_directory is None:
        swath_directory = Path(os.getcwd()).joinpath(f"swaths_{int(lon_east)}_{int(lon_west)}")
        swath_directory.mkdir(exist_ok=True)

    dts = [start_dt + timedelta(days=day) for day in range((end_dt - start_dt).days + 1)]

    swath_tasks = [
        delayed(get_satellite_swaths)(
            config_file,
            _dt,
            lon_east,
            lon_west,
            swath_directory
        )
        for _dt in dts
    ]
    return swath_tasks


def hotspots_compare(
    gdf_a: gpd.GeoDataFrame,
    gdf_b: gpd.GeoDataFrame,
    column_name: str,
    geosat_flag: bool,
    swath_directory: Union[Path, str]
) -> Union[None, pd.DataFrame]:
    """Function compares sensor GeoDataFrame from two satellite sensor product.

    Subsets hotspots to common imaged area and determines points nearest to productA from hotspots_gdf.
    Computes distance between points.

    :param gdf_a: The GeoDataFrame from satellite sensor product a.
    :param gdf_b: The GeoDataFrame from satellite sensor product b.
    :param column_name: The name of the column to resample the data on.
    :param geosat_flag: The flag to indicate if hotspot is from geostationary statellite.
    :param swath_directory: The parent directory to store the solar_date swath files.

    :returns:
        GeoDataFrame of nearest hotspots
    """

    nearest_hotspots_df = []
    for index_a, gdf_ra in gdf_a.resample("D", on=column_name):
        index_a_tasks = []
        for index_b, gdf_rb in gdf_b.resample("D", on=column_name):
            if index_a == index_b:
                solar_date = str(index_a.date())
                # skip if swath directory for the solar_date is missing.
                solar_date_swath_directory = Path(swath_directory).joinpath(solar_date)
                if not solar_date_swath_directory.exists():
                    continue
                index_a_tasks.append(
                    delayed(get_nearest_hotspots)(
                        gdf_ra,
                        gdf_rb,
                        geosat_flag,
                        solar_date_swath_directory
                    )
                )
        index_a_dfs = [df for df in dask.compute(*index_a_tasks) if df is not None]
        if index_a_dfs:
            index_a_df_merged = nearest_hotspots_df.append(
                pd.concat(index_a_dfs, ignore_index=True)
            )
            nearest_hotspots_df.append(index_a_df_merged)
    if nearest_hotspots_df:
        return pd.concat(
            nearest_hotspots_df,
            ignore_index=True
        )
    return None


def solar_day_start_stop_period(longitude_east, longitude_west, _solar_day):
    """
    Function solar day start time from longitude and solar day in utc

    Returns datetime start stop in utc and period between in minutes
    """
    # Solar day time relative to UTC and local longitude
    seconds_per_degree = 240
    # Offset for eastern limb
    offset_seconds_east = int(longitude_east * seconds_per_degree)
    offset_seconds_east = np.timedelta64(offset_seconds_east, "s")
    # offset for wester limb
    offset_seconds_west = int(longitude_west * seconds_per_degree)
    offset_seconds_west = np.timedelta64(offset_seconds_west, "s")
    # time between two limbs
    offset_day = np.timedelta64(1440, "m") + abs(
        offset_seconds_east - offset_seconds_west
    )
    # ten_am_crossing_adjustment = np.timedelta64(120, 'm')
    # Solar day start at eastern limb
    solar_day_start_utc = (
        np.datetime64(_solar_day) - offset_seconds_east
    ).astype(datetime)
    # Solar day finish at western limb
    solar_day_finish_utc = (
        (np.datetime64(_solar_day) + offset_day) - offset_seconds_east
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
    seconds_per_degree = 240
    offset_seconds = int(longitude * seconds_per_degree)
    offset = np.timedelta64(offset_seconds, "s")
    return (np.datetime64(utc) + offset).astype(datetime)


def solar_night(utc, longitude):
    """
    Function solar night for a given UTC time and longitude input

    Returns datetime object representing solar day
    """
    seconds_per_degree = 240
    offset_seconds = int(longitude * seconds_per_degree)
    offset = np.timedelta64(offset_seconds, "s")
    dt = (np.datetime64(utc) + offset).astype(datetime)
    return dt - timedelta(hours=12)


def ckdnearest(gdf_a, gdf_b):
    """
    Function to find points in "B" nearest to "A" geopandas dataframe

    Returns geopandas dataframe with records representing matches
    """
    na = np.array(list(zip(gdf_a.geometry.x, gdf_a.geometry.y)))
    nb = np.array(list(zip(gdf_b.geometry.x, gdf_b.geometry.y)))
    btree = cKDTree(nb)
    dist, idx = btree.query(na, k=1)
    gdf = gpd.GeoDataFrame(
        pd.concat(
            [
                gdf_a.reset_index(drop=True),
                gdf_b.loc[idx].reset_index(drop=True).add_prefix("2_"),
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
    :param dtype: The data type to supply to dask or pandas read method.
    :param skiprows: The rows to skips.
    """
    # if not lazy loading then use pandas dataframe to read.
    if not lazy_load:
        return pd.read_csv(csv_file, dtype=dtype, skiprows=skiprows)

    # if lazy load then use dask.dataframe to lazy load
    return dd.read_csv(csv_file, dtype=dtype, skiprows=skiprows)


def load_geojson(
    geojson_file: Union[Path, str],
    bbox: Optional[Tuple] = None,
    ignore_fields: Optional[List] = None,
) -> gpd.GeoDataFrame:
    """Method to read a geojson features spatially subsetted by a bbox.

    :param geojson_file: Full path to a geojson file to read.
    :param bbox: The bounding box to subset features of geojson file.
    :param ignore_fields: The fields to ignore from when reading geojson file.

    :returns:
        The GeoDataFrame loaded from geojson file.
    """
    if ignore_fields is None:
        gdf = gpd.read_file(geojson_file, bbox=bbox)
    else:
        try:
            # try to load by ignoring fields if fiona version supports.
            gdf = gpd.read_file(
                geojson_file, bbox=bbox, ignore_fields=ignore_fields
            )
        except DriverError:
            gdf = gpd.read_file(geojson_file, bbox=bbox)
    return gdf


def temporal_subset_df(
    df: gpd.GeoDataFrame,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
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


def chunk_gpd(df: gpd.GeoDataFrame, num_chunks: Optional[int] = 1) -> List:
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
    gdf["solar_night"] = gdf["solar_day"] - pd.Timedelta(hours=12)

    gdf = temporal_subset_df(gdf, start_time, end_time, start_date, end_date)
    return gdf


def slstr_temporal_subset_normalize(
    gdf: gpd.GeoDataFrame,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    provider: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Method to temporally subset and normalize features attributes for SLSTR product.

    :param gdf: The GeoDataFrame dataset.
    :param start_time: The start time to subset the data.
    :param end_time: The end time to subset the data.
    :param start_date: The start date to subset the data.
    :param end_date: The end date to subset the data
    :param provider: The hotspots algorithm associated provider.
    :returns:
        The subsetted GeoDataFrame.
    """
    gdf["datetime"] = pd.to_datetime(gdf["date"])
    gdf["solar_day"] = pd.to_datetime(gdf["solar_day"])
    gdf["solar_night"] = gdf["solar_day"] - pd.Timedelta(hours=12)

    if provider == "eumetsat":
        # TODO fix this while write geojson files.
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
    num_chunks: Optional[int] = 1,
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
        task = delayed(modis_viirs_temporal_subset_normalize)(
            df, start_time, end_time, start_date, end_date
        )
        gdf_tasks.append(task)
    return gdf_tasks


def process_dea_hotspots(
    geojson_file: Union[Path, str],
    bbox: Optional[Tuple] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    num_chunks: Optional[int] = 1,
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
        List of dask delayed tasks that would return subsetted GeoDataFrame and
        normalize features."""
    gdf = load_geojson(geojson_file, bbox=bbox)
    gdf_chunks = chunk_gpd(gdf, num_chunks)

    gdf_tasks = []
    for df in gdf_chunks:
        task = delayed(modis_viirs_temporal_subset_normalize)(
            df, start_time, end_time, start_date, end_date
        )
        gdf_tasks.append(task)
    return gdf_tasks


def process_landgate_hotspots(
    geojson_file: Union[Path, str],
    bbox: Optional[Tuple] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    num_chunks: Optional[int] = 1,
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
        List of dask delayed tasks that would return subsetted GeoDataFrame and
        normalize features.
    """
    gdf = load_geojson(geojson_file, bbox=bbox)
    gdf_chunks = chunk_gpd(gdf, num_chunks)

    gdf_tasks = []
    for df in gdf_chunks:
        task = delayed(modis_viirs_temporal_subset_normalize)(
            df, start_time, end_time, start_date, end_date
        )
        gdf_tasks.append(task)
    return gdf_tasks


def process_eumetsat_hotspots(
    geojson_file: Union[Path, str],
    bbox: Optional[Tuple] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    num_chunks: Optional[int] = 1,
) -> List[dask.delayed]:
    """Method to load, subset and normalize features attributes for EUMETSAT product.

    :param geojson_file: Fupp path to a geojson file to read.
    :param bbox: The bounding box to subset features of geojson file.
    :param start_time: The start time to subset the data.
    :param end_time: The end time to subset the data.
    :param start_date: The start date to subset the data.
    :param end_date: The end date to subset the data
    :param num_chunks: The number of blocks to be sub-divided into.

    :returns:
        List of dask delayed tasks that would return subsetted GeoDataFrame and
        normalize features.
    """
    gdf = load_geojson(
        geojson_file, bbox=bbox, ignore_fields=__slstr_ignore_attrs__
    )
    gdf_chunks = chunk_gpd(gdf, num_chunks)

    gdf_tasks = []
    for df in gdf_chunks:
        task = delayed(slstr_temporal_subset_normalize)(
            df, start_time, end_time, start_date, end_date, "eumetsat"
        )
        gdf_tasks.append(task)
    return gdf_tasks


def process_esa_hotspots(
    geojson_file: Union[Path, str],
    bbox: Optional[Tuple] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    num_chunks: Optional[int] = 1,
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
    gdf = load_geojson(
        geojson_file, bbox=bbox, ignore_fields=__slstr_ignore_attrs__
    )
    gdf_chunks = chunk_gpd(gdf, num_chunks)

    gdf_tasks = []
    for df in gdf_chunks:
        task = delayed(slstr_temporal_subset_normalize)(
            df, start_time, end_time, start_date, end_date, "esa"
        )
        gdf_tasks.append(task)
    return gdf_tasks


def get_all_hotspots_tasks(
    input_files_dict: Dict,
    bbox: Optional[Tuple] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    num_chunks: Optional[int] = 1,
) -> List[dask.delayed]:
    """Method to prepare hotspots geo dataframes from input geojson files.

    The data from geojson files are spatially subset to bbox extent and
    temporally subset to hours between start and end time, and dates between
    start and end date.

    :param input_files_dict: The dictionary with key as data provider and value as file path.
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
        _LOG.info(f"reading and sub-setting GeoDataFrame for {name}: {fid}")
        kwargs = {
            "bbox": bbox,
            "start_time": start_time,
            "end_time": end_time,
            "start_date": start_date,
            "end_date": end_date,
            "num_chunks": num_chunks,
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
            tasks = process_dea_hotspots(fid, **kwargs)
        else:
            _LOG.info(f"{name} not Implemented. Skipped processing.")
            tasks = []
        hotspots_tasks += tasks
    return hotspots_tasks


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


def csv_to_dataframe(
    nearest_csv_files: List,
    index_column: Optional[str] = "solar_day"
) -> dd.DataFrame:
    """Helper method to convert neareast hotspots csv files to dataframe.
    
    In this method, datetime fields are converted to pandas datetime type
    and additional fields dist_m, timedelta and count are added.

    :param nearest_csv_files: The List of nearest_hotspots csv files.
    :param index_column: The column name to set the dataframe index.
    
    :returns:
        Dask DataFrame with all the data from csv file concatenated and additional fields
        required for analysis added. 
    """

    df_list_tasks = [
        delayed(load_csv)
        (fp)
        for fp in nearest_csv_files
        if Path(fp).exists()
    ]
    dfs = [df for df in dask.compute(*df_list_tasks)]
    df = dd.concat(dfs)
    df["dist_m"] = df.map_partitions(_distance)
    df["datetime"] = dd.to_datetime(df["datetime"])
    df["solar_day"] = dd.to_datetime(df["solar_day"])
    df["solar_night"] = dd.to_datetime(df["solar_night"])
    df["2_datetime"] = dd.to_datetime(df["2_datetime"])
    df["timedelta"] = abs(df["datetime"] - df["2_datetime"])
    df["count"] = 1
    df = df.astype({"dist_m": float})
    df = df.set_index(index_column)
    return df


def to_timedelta(df: pd.DataFrame) -> pd.Series:
    """Helper method to calculate time difference.

    :param df: The DataFrame to compute distance

    :returns:
        The pandas Series object with computed time difference.
    """

    return abs(df["datetime"] - df["2_datetime"])


def pandas_pivot_table(
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


def dask_pivot_table(
    df: dd.DataFrame,
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


def fetch_hotspots_files(
    hotspots_files_dict: dict,
    s3_client: boto3.Session.client,
    outdir: Union[Path, str]
) -> Dict:
    """Utility to download hotspots files from s3 location.
    
    :param hotspots_files_dict: The hotspots files dictionary with key as a hotspots
                           file provider name and value as a file location. The
                           files are only downloaded if the file path matches
                           s3 pattern. Eg. s3://<bucket>/<key>
    :param s3_client: The boto3 s3 client that has previleges to download files
                      from s3 files location.
    :param outdir: The output directory to store the files downloaded from s3.
    
    :returns:
        The dictionary with same key in hotspots_files_dict and values replaced
        with the downloaded file-location in the local directory.
    """
    hotspots_files = dict()
    for provider, fp in hotspots_files_dict.items():
        if fp is None:
            _LOG.info(f"{provider} Hotspots FRP  is None. excluding from analysis.")
            continue
        if re.match(__s3_pattern__, str(fp)):
            _, bucket, _key, _ = re.split(__s3_pattern__, str(fp))
            outfile = Path(outdir).joinpath(Path(fp).name)
            hotspots_files[provider] = outfile
            if outfile.exists():
                _LOG.info(f"{fp} exists: skipped download")
                continue
            _LOG.info(f"downloading {fp} to {outfile.as_posix()}")
            s3_client.download_file(bucket, _key, outfile.as_posix())
    return hotspots_files
