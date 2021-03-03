#!/usr/bin/env python3

import logging
import logging.config

import os
from typing import Optional, Tuple, Union, Dict, List
from pathlib import Path

from datetime import datetime

import pandas as pd
import geopandas as gpd
import click

import dask
import dask.dataframe as dd
from dask.distributed import Client
from shapely import wkt
from geopy.distance import distance

import hotspot_utils as util


def _distance(
    df: pd.DataFrame
) -> List:
    """Helper method to compute distance.
    
    :param df: The DataFrame to compute distance.
    
    :returns:
        List of distance computed from DataFrame.
    """
    dist = [ 
        distance((x["latitude"], x["longitude"]), (x["2_latitude"], x["2_longitude"])).meters
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
    df = df.astype({'dist_m': float})
    df = df.set_index("solar_day")
    return df


def _to_timedelta(
    df: pd.DataFrame
) -> pd.Series:
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
    :param index_columns: The names of the columns to make new DataFrame's index.
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
        df,
        index=index,
        columns=column, 
        values=values,
        aggfunc=aggfunc
    )
    return _df




def process_cooccurence(
    nearest_csv_files_dict: Dict,
    dist_threshold: Optional[float] = 5000.,
    name_prefix: Optional[str] = ''
):
    """Process to compute the co-occurrence matrix between different product types.
    
    :param nearest_csv_files_dict: The dictionary with key as satellite sensor product type
                                   and value as file path to .csv file.
    :param dist_threshold: The distance threhold to subset the data.
    :param name_prefix: The name prefix that will be added to output file.
    """
    df_list = [util.load_csv(fp) for _, fp in nearest_hotspots_csv_files.items() if Path(fp).exists()]
    df = dd.concat(df_list)
    df["dist_m"] = df.map_partitions(_distance)
    df = _prepare_pivot_dataframe(df)
    
    # compute numerator
    numerator = _dask_pivot_table(
        df[df["dist_m"] < dist_threshold],
        index='2_satellite_sensor_product', 
        column='satellite_sensor_product',
        values='count',
        aggfunc='count'
    ).compute()
    numerator.to_csv(f"{name_prefix}_{dist_threshold}m_numerator_pivot_table.csv")
    
    # compute denominator
    denominator = _dask_pivot_table(
        df,
        index='2_satellite_sensor_product', 
        column='satellite_sensor_product',
        values='count',
        aggfunc='count'
    ).compute()
    denominator.to_csv(f"{name_prefix}_{dist_threshold}m_denominator_pivot_table.csv")
    
    # compute difference
    difference = denominator - numerator
    difference.to_csv("difference_pivot_table.csv")
    
    # compute percentage
    percentage = (numerator / denominator).style.format("{:.0%}") 
    percentage.to_csv(f"{name_prefix}_{dist_threshold}m_percentage.csv")
    
    # compute mean time
    timemean = _dask_pivot_table(
        df[df["dist_m"] < dist_threshold],
        index='2_satellite_sensor_product', 
        column='satellite_sensor_product',
        values='timedelta',
        aggfunc='mean'
    ).compute()
    timemean = timemean.style.format("{:}")
    timemean.to_csv(f"{name_prefix}_{dist_threshold}m_timemean.csv")
    
    # compute average_distance
    averagedist = dask_pivot_table(
        df[df["dist_m"] < dist_threshold],
        index='2_satellite_sensor_product', 
        column='satellite_sensor_product',
        values='dist_m',
        aggfunc='mean'
    ).compute()
    numerator.to_csv(f"{name_prefix}_{dist_threshold}m_averagedist.csv")
    
    
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
    outdir: Optional[Union[Path, str]] = Path(os.getcwd())
) -> None:
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
    :param num_chunks: The number of blocks to be sub-divided into.
    :param outdir: The output directory to save the output files.
    
    :returns:
        None.
        All the .csv output files containing the nearest hotspots details are written into outdir.
    """
    
    _LOG = logging.getLogger(__name__)
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
    
    _LOG.info(f"The merged hotspots DataFrame sample:\n {hotspots_gdf.head()}")

    _LOG.info("The spatial and temporal extents of merged hotspots.")
    _LOG.info(f"minimum datetime: {hotspots_gdf['solar_day'].min()}, maximum datetime: {hotspots_gdf['solar_day'].max()}")
    _LOG.info(f"longitude range: {hotspots_gdf['longitude'].min()}  to {hotspots_gdf['longitude'].max()}")
    _LOG.info(f"latitude range: {hotspots_gdf['latitude'].min()}  to {hotspots_gdf['latitude'].max()}")

    unique_products = [p for p in hotspots_gdf["satellite_sensor_product"].unique()]
    nearest_hotspots_product_files = dict() 
    for productA in unique_products:
        hotspots_compare_tasks = []
        for productB in unique_products:
            geosat_flag = False

            if ('AHI' in [productA, productB]) | ('INS1' in [productA, productB]):
                geosat_flag = True

            gdfA = hotspots_gdf[hotspots_gdf["satellite_sensor_product"] == productA]
            gdfB = hotspots_gdf[hotspots_gdf["satellite_sensor_product"] == productB]
            hotspots_compare_tasks.append(
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
        outfile = outdir.joinpath(f"nearest_points.{productA}.csv")
        productA_nearest_gdfs = dask.compute(*hotspots_compare_tasks)
        productA_nearest_gdfs_merged = pd.concat([df for df in productA_nearest_gdfs if df is not None])
        productA_nearest_gdfs_merged.reset_index(inplace=True, drop=True)
        productA_nearest_gdfs_merged.to_csv(outfile.as_posix())
        nearest_hotspots_product_files[productA] = outfile
    return nearest_hotspots_products_files



if __name__ == "__main__":

    # Configure log here for now, move it to __init__ at top level once
    # code is configured to run as module

    LOG_CONFIG = Path(__file__).parent.joinpath("logging.cfg")
    logging.config.fileConfig(LOG_CONFIG.as_posix())
    
    # sys.exit()
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
    '''
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
        chunks=chunks
    )
    '''
    
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
        "TERRA_MODIS_NASA6.03": "nearest_points.TERRA_MODIS_NASA6.03.csv"
        
    }
    
    process_cooccurence(nearest_hotspots_csv_files, dist_threshold=5000., name_prefix=f"{start_date.replace('-', '')}_{end_date.replace('_', '')}")

