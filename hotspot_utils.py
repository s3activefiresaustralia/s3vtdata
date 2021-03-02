#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Tuple, Union, Optional, List, Dict

import geopandas as gpd
import pandas as pd

import subprocess
from datetime import datetime

import os
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path

import dask
import dask.dataframe as dd
from dask import delayed

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
        _LOG.debug(str(solar_day) + " exists - skipping swath generation")
        success = True
    else:
        dirpath.mkdir(parents=True, exist_ok=True)

        try:
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
            _LOG.debug("Swath generation failed")

    return success


def pairwise_swath_intersect(satsensorsA, satsensorsB, solar_day):
    """
    Function intersect geometries of sensors for a given solar day

    Returns intersection geometry
    """
    _LOG.info(
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
    return pd.concat(gpdlistA), pd.concat(gpdlistB)


def get_nearest_hotspots(
    gdfa: gpd.GeoDataFrame,
    gdfb: gpd.GeoDataFrame,
    solar_date: str,
    geosat_flag: bool
) -> Union[None, gpd.GeoDataFrame]:
    """Method to compute nearest hotspots between two GeoDataFrame.

    :param gdfa: The GeoDataFrame
    :param gdfb: The GeoDataFrame
    :param solar_date: The solar date
    :param geosat_flag: The flag to indicate if hotspot is from geostationary statellite.

    :returns:
        None if no intersections or fails in pairwise swath intersects.
        GeoDataFrame from cdknearest method.
    """
    if not geosat_flag:
        try:
            gpd1, gpd2 = pairwise_swath_intersect(
                set(gdfa["satellite"]),
                set(gdfb["satellite"]),
                solar_date
            )
        except Exception as err:
            _LOG.debug(err)
            return None

        gpd1 = gpd1.unary_union
        gpd2 = gpd2.unary_union

        intersection = gpd1.intersection(gpd2)
        if intersection is None:
            _LOG.debug(f"Intersection is None: {solar_date}")
            return None

        maska = gdfa.within(intersection)
        gdfa = gdfa.loc[maska]
        maskb = gdfb.within(intersection)
        gdfb = gdfb.loc[maskb]

    gdfa.reset_index(drop=True, inplace=True)
    gdfb.reset_index(drop=True, inplace=True)

    if gdfa.empty | gdfb.empty:
        _LOG.debug("Nothing to input to cdknearest")
        return None

    nearest_hotspots = ckdnearest(gdfa, gdfb)

    return nearest_hotspots


def hotspots_compare(
    gdf_a: gpd.GeoDataFrame,
    gdf_b: gpd.GeoDataFrame,
    lon_east: float,
    lon_west: float,
    column_name: str,
    config_file: Union[Path, str],
    geosat_flag: bool,
) -> Union[None, pd.DataFrame]:
    """Function compares sensor GeoDataFrame from two satellite sensor product.

    Subsets hotspots to common imaged area and determines points nearest to productA from hotspots_gdf.
    Computes distance between points.

    :param gdf_a: The GeoDataFrame from satellite sensor product a.
    :param gdf_b: The GeoDataFrame from satellite sensor product b.
    :param lon_east: The eastern longitude used in spatial subset.
    :param lon_west: The western longitude used in a spatial subset.
    :param column_name: The name of the column to resample the data on.
    :param config_file: The config file to be used in swath generation.
    :param geosat_flag: The flag to indicate if hotspot is from geostationary statellite.

    :returns:
        None if no nearest hotspots detected.
        nearest hotspots DataFrame if there are hotspots.
    """
    nearest_hotspots_df = []
    for index_a, gdf_ra in gdf_a.resample('D', on=column_name):
        for index_b, gdf_rb in gdf_b.resample('D', on=column_name):
            if (index_a == index_b) and (not gdf_ra.empty | gdf_rb.empty):
                solar_date = str(index_a.date())
                min_dt, max_dt, delta_dt = solar_day_start_stop_period(lon_east, lon_west, index_a)
                get_satellite_swaths(
                    config_file,
                    min_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    str(int(delta_dt.total_seconds()/60)),
                    solar_date
                )
                nearest_hotspots = get_nearest_hotspots(gdf_ra, gdf_rb, solar_date, geosat_flag)
                if nearest_hotspots is not None:
                    nearest_hotspots_df.append(nearest_hotspots)
    if nearest_hotspots_df:
        return pd.concat(nearest_hotspots_df)
    return None


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
    :param provider: The hotspots algorithm associated provider.
    :returns:
        The subsetted GeoDataFrame.
    """
    gdf["datetime"] = pd.to_datetime(gdf["date"])
    gdf["solar_day"] = pd.to_datetime(gdf["solar_day"])

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
        _LOG.info(f"reading and subsetting GeoDataFrame for {name}: {fid}")
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
            tasks = process_dea_hotspots(fid, **kwargs)
        hotspots_tasks += tasks
    return hotspots_tasks


