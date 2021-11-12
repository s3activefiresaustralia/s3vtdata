#!/usr/bin/env python3

import datetime
from pathlib import Path
from typing import Optional, Iterable, List, Union
import tempfile
import re
import logging
import os
import sys
from multiprocessing import Pool as ProcessPool
from datetime import timedelta

import click
import geopandas as gpd
import pandas as pd
import numpy as np
import zipfile
import boto3
from botocore.exceptions import ClientError

S3_CLIENT = None

_LOG = logging.getLogger(__name__)


class DummyPool:
    """Helper class to expand arguments for multi-processing"""

    def __enter__(self):  # pragma: no cover
        return self

    def starmap(self, func, args):  # pragma: no cover
        return [func(*arg) for arg in args]


def Pool(processes):
    """Helper method for multi-processing."""
    if not processes:  # pragma: no cover
        return DummyPool()

    return ProcessPool(processes=processes)



def solar_day(utc, longitude):
    """
    Function solar day for a given UTC time and longitude input
    
    Returns datetime object representing solar day
    """
    SECONDS_PER_DEGREE = 240
    offset_seconds = int(longitude * SECONDS_PER_DEGREE)
    offset = np.timedelta64(offset_seconds, 's')
    return (np.datetime64(utc) + offset).astype(datetime.datetime)


def datetime_range(
    start_dt: datetime.datetime,
    end_dt: datetime.datetime,
    delta_days: Optional[int] = 1,
) -> Iterable[datetime.datetime]:
    """Generator to return dates between the start and end date.

    :param start_dt: The start datetime.
    :param end_dt: The end datetime.
    :param delta_days: The timedelta to step through start and end datetime.

    :returns:
        Generator for datetime between start and end datetime.
    """

    span = end_dt - start_dt
    for i in range(span.days + delta_days):
        yield start_dt + timedelta(days=i)

        
def get_fhs_gdf(
    bucket: str,
    s3_filepath: str,
    outfile: str,
) -> Union[str, None]:
    """Downloads file with given prefix from s3 bucket.

    :param bucket: The name of the s3 bucket to download file from.
    :param s3_filepath: The filepath of a file to download relative to s3 bucket.
    :param outfile: The name of a outfile to save file.

    :return:
        The outfile if downloaded else None if download failed.
    """
    try:
        S3_CLIENT.download_file(bucket, s3_filepath, outfile)
    except Exception as err:
        _LOG.info(f"Failed to download s3://{bucket}/{s3_filepath}: {err}")
        return None
    
    with zipfile.ZipFile(outfile, 'r') as zip_ref:
        zip_ref.extractall(path=Path(outfile).parent.as_posix())
        try:
            gdf = gpd.GeoDataFrame.from_file(Path(outfile).with_suffix(".shp"))
            if "MOD" in s3_filepath:
                sensor="MODIS"
            elif "AVH" in s3_filepath:
                sensor="AVHRR"
            elif "VII" in s3_filepath:
                sensor="VIIRS"
            else:
                 return gpd.GeoDataFrame()
            gdf['sensor'] = sensor
        except Exception:
            print(f"{outfile} not complete")
            return gpd.GeoDataFrame()
        return gdf

        
def s3_bucket_list(
    bucket: str,
    prefix: Optional[str] = "",
    suffix: Optional[str] = None,
    exclude: Optional[str] = None
):
    """List all the contents of a bucket relative to prefix.

    :param bucket:
        The name of the s3 bucket to download file from.
    :param prefix:
        The prefix to s3 file structure.
    :param suffix:
        The suffix to return only matched suffix items.
    :param exclude:
        The string match to exclude.
    """
    paginator = S3_CLIENT.get_paginator("list_objects_v2")
    Keys = []
    for _page in paginator.paginate(
        Bucket=bucket,
        Prefix=prefix
    ):
        for content in _page["Contents"]:
            _key = content['Key']
            if suffix is not None:
                if not _key.endswith(suffix):
                    continue
            if exclude is not None:
                if exclude in Path(_key).name:
                    continue
            Keys.append(content["Key"])
    return Keys


def get_s3_listings(
    start_dt: datetime.datetime,
    end_dt: datetime.datetime,
    s3_bucket: str,
    fhs_prefix: str,
    fhs_suffix: str,
    exclude_str: str,
    nprocs: int
):
    """Returns the list of s3 files after applying filters.
    
    :param start_dt:
        The start datetime to subset the s3 listing.
    :param end_dt:
        The end datetime to subset the s3 listing.
    :param s3_bucket:
        The name of s3 bucket where FHS .zip file are located.
    :param fhs_prefix:
        The folder path relative to s3_bucket.
    :param fhs_suffix:
        The extension of the filename.
    :param exclude_str:
        The string to match in s3 listing which will be excluded.
    :param nprocs:
        The number of processer to use in multi-processing.
    
    :return:
        The list of s3 file paths.
    
    """
    fhs_all_zip_list = []
    with Pool(processes=nprocs) as pool:
        results = pool.starmap(
            s3_bucket_list,
            [
                (
                    s3_bucket,
                    f"{fhs_prefix}/{_dt.year}/{_dt.month:02}/{_dt.day:02}/",
                    fhs_suffix,
                    exclude_str
                )
                for _dt in datetime_range(start_dt, end_dt, delta_days=1)
            ],
        )
        fhs_all_zip_list.append(sum(results, []))
    fhs_all_zip_list = sum(fhs_all_zip_list, [])
    return fhs_all_zip_list


def remove_duplicates(df, time_diff_threshold: Optional[int] = 1800):
    df_new = []
    for _, row_m in df.iterrows():
        sub = df[(df['longitude'] == row_m.longitude) & (df['latitude'] == row_m.latitude)]
        sub = sub.sort_values(by=['datetime'])
        if len(sub) > 1:
            num_rows = 0
            for idx, row in sub.iterrows():
                if num_rows == 0:
                    df_new.append(sub[sub.index == idx])
                    num_rows += 1
                    continue
                if num_rows < len(sub) - 1:
                    time_diff = abs((sub.iloc[num_rows-1].datetime - sub.iloc[num_rows].datetime).total_seconds())
                    if time_diff > time_diff_threshold:
                        df_new.append(sub[sub.index == idx])
                num_rows += 1
        else:
            df_new.append(sub)
    if len(df_new) >= 1:
        df_new = pd.concat(df_new, ignore_index=True)
        df_new.drop_duplicates(inplace=True)
        return df_new
    return None


def clean_gdf(gdf, nprocs: Optional[int]  = 1, time_diff_threshold: Optional[int] = 1800):
    """Method to clean up GeoDataFrame and make it confirm with other FHS provider.
    
    This method will rename of GeoDataframe columns to conform with other FHS providers
    column names. In duplicates removal method, any acquisition of the same satellite
    within the bounds of +/- time_diff_threshold will be removed. Only first copy of the
    FHS records from duplicates will be retained.
    
    :param gdf:
        The GeoDataFrame to clean and remove duplicates.
    :param nprocs:
        The number processor to use in clean up process.
    :param time_diff_threshold:
        The time difference (in seconds) threshold used in removal of duplicates.
        Defaults to 30 minutes (1800 seconds)
    
    :returns:
        The cleaned GeoDataFrame that has duplicates removed and column renamed.
    """
    gdf.rename(
        columns = {
            'Latitude':'latitude',
            'Longitude':'longitude', 
            'Satellite':'satellite', 
            'Confidence':'confidence',
            'Intensity':'power'
        },
        inplace = True
    )
    gdf['datetime'] = gdf.Date + 'T' + gdf.Time
    gdf['datetime'] = pd.to_datetime(gdf['datetime'])
    gdf['solar_day'] = gdf.apply(lambda row: solar_day(row.datetime, row.longitude), axis = 1)
    gdf['satellite'] = (gdf['satellite'].replace(['JPSS-1'],'NOAA 20'))
    gdf['satellite'] = (gdf['satellite'].replace(['Terra'],'TERRA'))
    gdf['satellite'] = (gdf['satellite'].replace(['Aqua'],'AQUA'))
    gdf['satellite'] = (gdf['satellite'].replace(['SNPP'],'SUOMI NPP'))
    gdf.drop(['ID', 'Descriptio', 'Orbit', 'Time', 'Date', 'SatZenith', 'Location'], axis=1, inplace=True)

    with Pool(processes=nprocs) as pool:
        gdfs = pool.starmap(
            remove_duplicates,
            [
                (
                    gdf[gdf['satellite'] == sat],
                    time_diff_threshold,
                )
                for sat in gdf.satellite.unique()
            ]
        )
    
    return pd.concat([_df for _df in gdfs if _df is not None], ignore_index=True)
    

def get_landgate_geojson(
    start_dt: datetime.datetime,
    end_dt: datetime.datetime,
    fhs_prefix: Optional[List] = ["data/MOD/FHS", "data/AVH/FHS", "data/VII/FHS"],
    fhs_exclude: Optional[List] = ["MOD14", None, "NFHS"],
    fhs_suffix: Optional[str] = ".zip",
    s3_bucket: Optional[str] = "srss-data",
    nprocs: Optional[str] = 8
):
    aws_session = boto3.Session(
        region_name="ap-southeast-2",
        aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    global S3_CLIENT 
    S3_CLIENT = aws_session.client('s3')
 
    # get list of all the zip files containing Landgate FHS
    fhs_all_list = []
    for _prefix, _exclude in zip(fhs_prefix, fhs_exclude):
        fhs_list = get_s3_listings(
            start_dt, end_dt, s3_bucket, _prefix, fhs_suffix, _exclude, nprocs
        )
        for fp in fhs_list:
            fhs_all_list.append(fp)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with Pool(processes=nprocs) as pool:
            gdfs = pool.starmap(
                get_fhs_gdf,
                [
                    (
                        s3_bucket,
                        s3_path,
                        Path(tmpdir).joinpath(Path(s3_path).name).as_posix()
                    )
                    for s3_path in fhs_all_list
                ]
            )
    
    hotspots_gdf = pd.concat([gdf for gdf in gdfs if len(gdf) > 1], ignore_index=True)
    # print(hotspots_gdf)
    return clean_gdf(hotspots_gdf, nprocs=nprocs)
    

@click.command(
    "--process-landgate-fhs",
    help="Processes Landgates FHS into combined GeoDataFrame/Geojson file"
)
@click.option(
    "--start-date",
    help="The start date to subset the FHS record",
    type=click.DateTime(formats=['%Y-%m-%d']),
    required=True
)
@click.option(
    "--end-date",
    help="The end date to subset the FHS record",
    type=click.DateTime(formats=['%Y-%m-%d']),
    required=True
)
@click.option(
    "--fhs-prefix",
    help="The folder relative to S3 Bucket where FHS data",
    type=click.Tuple([str, str, str]),
    default=["data/MOD/FHS", "data/AVH/FHS", "data/VII/FHS"],
    show_default=True
)
@click.option(
    "--fhs-exclude",
    help="The matched string will be excluded",
    type=click.Tuple([str, str, str]),
    default=["MOD14", None, "NFHS"],
    show_default=True
)
@click.option(
    "--fhs-suffix",
    help="only files with matched suffix will be processed",
    type=click.STRING,
    default=".zip",
    show_default=True
)
@click.option(
    "--s3-bucket",
    help="Name of S3 bucket where FHS files are located",
    type=click.STRING,
    default="srss-data",
    show_default=True
)
@click.option(
    "--nprocs",
    help="Number of processor used in multi-processing",
    type=click.INT,
    default=1,
    show_default=True
)
def main(
    start_date,
    end_date,
    fhs_prefix,
    fhs_exclude,
    fhs_suffix,
    s3_bucket,
    nprocs
):
    print(start_date)
    print(end_date)
    gdf_hotspots = get_landgate_geojson(
        start_date,
        end_date,
        fhs_prefix=fhs_prefix,
        fhs_exclude=fhs_exclude,
        fhs_suffix=fhs_suffix,
        s3_bucket=s3_bucket,
        nprocs=nprocs

    )
    gdf_hotspots = gdf_hotspots[gdf_hotspots['satellite'] != 'NOAA-23']
    gdf_hotspots['satellite_sensor_product'] = gdf_hotspots['satellite']+'_'+gdf_hotspots['sensor']+'_LANDGATE'
    gdf_hotspots.to_file('landgate_hotspots_gdf_v2.geojson', driver='GeoJSON')
    

if __name__ == "__main__":
    main()