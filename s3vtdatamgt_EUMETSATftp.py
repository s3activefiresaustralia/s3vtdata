#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import importlib
import sys
from pathlib import Path
from typing import Union, List, Iterable, Optional, Tuple
from datetime import datetime
import subprocess
import tempfile
import logging as logger
import ftplib
import re 
import shutil 

from netCDF4 import Dataset
import pandas as pd
import geopandas as gpd
import numpy as np
import boto3
from botocore.exceptions import ClientError
import yaml
import numpy as np


logger.basicConfig(format='%(levelname)s:%(message)s', level=logger.INFO)


def solar_day(utc, longitude):
    """
    Copied from 3vtdatamgt_ESAftp file
    Function solar day for a given UTC time and longitude input
    
    Returns datetime object representing solar day
    """
    SECONDS_PER_DEGREE = 240
    offset_seconds = int(longitude * SECONDS_PER_DEGREE)
    offset = np.timedelta64(offset_seconds, 's')
    return (np.datetime64(utc) + offset).astype(datetime)


def IPF_FRP_read(filename):
    # Copied from s3vtdatamgt_ESAftp file
    try:
        dataset = Dataset(filename)
    except (RuntimeError):
        logger.info("Unable to open ",filename)
        pass
        
    IPF_FRP =  gpd.GeoDataFrame()
    for var in dataset.variables:
        temp = dataset[var]
        if len(temp.shape) < 2:
            IPF_FRP[var] = dataset[var][:]
    IPF_FRP.geometry = gpd.points_from_xy(IPF_FRP.longitude, IPF_FRP.latitude)
    return IPF_FRP 


def eumetsat_ftp_file_list(
    username: str,
    password: str,
    url: Union[str, Path],
    directory: Union[str, Path],
) -> List[Iterable]:
    """Gets the listing of ftp sites.
     Copied from s3vtdatamgt_ESAftp file
    
    :param username: The username to log for the ftp site (url).
    :param password: The password for the ftp site.
    :param url: The eumetsat's ftp site's url.
    :param directory: The directory in ftp site to list its contents.
    
    :return:
        The list of contents from the directory.
    """
    ftp = ftplib.FTP(url)
    ftp.login(username, password)
    ftp.pwd()
    ftp.cwd(directory)
    return(ftp.nlst())


def get_eumetsat_dir(
    username: str,
    password: str,
    url: Union[str, Path],
    out_directory: Union[str, Path],
) -> int:
    """Gets the listing of ftp sites.
    
    :param username: The username to log for the ftp site (url).
    :param password: The password for the ftp site.
    :param url: The eumetsat's ftp site's url with full path to a directory to be downloaded.
    :param directory: The directory in ftp site to list its contents.
    """
    out_directory = f"{str(out_directory)}/{Path(url).name}"
    try:
        cmd = [
                'wget', 
                '-q',
                '--user='+username,
                '--password='+password,
                '--recursive',
                url,
                '-nH',
                '--cut-dirs=9',
                '--directory-prefix='+out_directory
            ]
        ret_code = subprocess.check_call(cmd)
        logger.info(["".join(item) for item in cmd])
    except:
        ret_code = 1
        # logger.info("Remote file retrieval failed "+str(['wget', '-q', '--user='+username, '--password='+password, url]))
    return ret_code
    

def recursive_mlsd(ftp_object, path="", maxdepth=8, match_suffix=None):
    """Run the FTP's MLSD command recursively
    modified from: 
        https://codereview.stackexchange.com/questions/232647/recursively-listing-the-content-of-an-ftp-server
    
    The MLSD is returned as a list of tuples with (name, properties) for each
    object found on the FTP server. This function adds the non-standard
    property "children" which is then again an MLSD listing, possibly with more
    "children".

    Parameters
    ----------
    ftp_object: ftplib.FTP or ftplib.FTP_TLS
        the (authenticated) FTP client object used to make the calls to the
        server
    path: str
        path to start the recursive listing from
    maxdepth: {None, int}, optional
        maximum recursion depth, has to be >= 0 or None (i.e. no limit).
    match_suffix: {None, str}, optional
        the folder extension to match
    Returns
    -------
    list
        the directory paths list with matched suffix if not None, else all
        directory paths.

    See also
    --------
    ftplib.FTP.mlsd : the non-recursive version of this function
    """
    if maxdepth is not None:
        maxdepth = int(maxdepth)
        if maxdepth < 0:
            raise ValueError("maxdepth is supposed to be >= 0")
    frp_paths = []
    
    def _inner(path_, depth_):
        if maxdepth is not None and depth_ > maxdepth:
            return
        inner_mlsd = list(ftp_object.mlsd(path=path_))
        for name, properties in inner_mlsd:
            if properties["type"] == "dir":
                rec_path = path_+"/"+name if path_ else name
                res = _inner(rec_path, depth_+1)
                if res is not None:
                    properties["children"] = res
                if match_suffix is not None:
                    if name.endswith(match_suffix):
                        frp_paths.append(rec_path)
                else:
                    frp_paths.append(rec_path) 
                    
        return inner_mlsd
    _inner(path, 0)
    return frp_paths


def s3_download_file(
    bucket: str,
    s3_client: boto3.Session.client,
    filename: str,
    out_dir: Path,
    prefix: Optional[str] = "",
) -> None:
    """Downloads file with given prefix from s3 bucket.
    :param bucket: The name of the s3 bucket to download file from.
    :pram s3_client: The s3 client to download the files.
    :param filename: The name of a file in s3 bucket.
    :param prefix: The prefix to s3 file structure.
    :param out_dir: The output directory to store the the file.
    :return:
        None. The file is downloaded to out_dir.
    :raises:
        FileNotFoundError: If `filename` is not in s3 bucket.
    """
    paginator = s3_client.get_paginator("list_objects_v2",)
    for _page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for content in _page["Contents"]:
            _key = content["Key"]
            if Path(_key).name == filename:
                s3_client.download_file(
                    bucket, _key, out_dir.joinpath(Path(_key).name).as_posix()
                )
                return
    raise FileNotFoundError(f"{filename} not found in s3 {bucket}")
    

def s3_upload_file(
    up_file: Union[str, Path],
    s3_client: boto3.Session.client,
    bucket: str,
    prefix: Optional[str] = "",
) -> None:
    """Uploads file into s3_bucket in a prefix folder.
    :param up_file: The full path to a file to be uploaded.
    :param s3_client: s3 client to upload the files. 
    :param bucket: The name of s3 bucket to upload the files into. 
    :param prefix: The output directory to put file in s3 bucket.  
    
    :raises: 
        ValueError if file fails to upload to s3 bucket.
    """

    filename = Path(up_file).name
    s3_path = f"{prefix}/{filename}"
    try:
        s3_client.head_object(Bucket=s3_bucket, Key=s3_path)
        logger.info(f"{filename} exists in {bucket}/{prefix}, skipped upload")
    except:
        try:
            s3_client.upload_file(up_file.as_posix(), bucket, Key=s3_path)
        except ClientError as e:
            raise ValueError(f"failed to upload {filename} at {bucket}/{prefix}")
            
            
def s3_frp_list(
    aws_access_key_id: str,
    aws_secret_access_key: str,
    s3_bucket_name: str,
    exclude_s3_key: str,
    match_suffix: str,
    outfile: Union[str, Path]
) -> None:
    """Writes a list of all the s3 objects with same match_suffix from s3_bucket.
    
    :param aws_access_key_id: The aws_access_key_id with permission to access s3 bucket.
    :param aws_secret_access_key: The aws_secret_access_key associated with access key id.
    :param s3_bucket_name: The name of s3 bucket.
    :param exclude_s3_key: The key to identify data folder to skip listing its
                            contents when queried.
    :param match_suffix: The suffix match the folder name.
    :param outfile: The file to write esa's s3_object list.
    """
    # Assess inventory against AWS bucket listing
    s3 = boto3.resource(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    s3bucket = s3.Bucket(s3_bucket_name,)
    s3_file_list = []
    for bucket_object in s3bucket.objects.all():
        key_name = bucket_object.key
        # TODO better implementation to skip eumetsat's s3 data folder
        if exclude_s3_key in key_name:
            continue 
        if match_suffix == '.SEN3':
            if Path(key_name).parent.suffix == match_suffix:
                s3_file_list.append(Path(key_name).parent.name)
        if match_suffix == '.geojson':
            if Path(key_name).suffix == match_suffix:
                s3_file_list.append(Path(key_name).name)
                
    with open(outfile.as_posix(), "w") as fid:
        for item in set(s3_file_list):
            fid.write(f"{item}\n")

            
def get_eumetsat_frp_dir_list(
    username: str,
    password: str,
    url: Union[str, Path],
    directory: Optional[Union[str, Path]] = None,
    match_suffix: Optional[str] = ".SEN3",
    outfile: Optional[Union[str, Path]] = (
        Path(os.getcwd()).joinpath("eumetsat_frp_list.csv")
    )
) -> None:
    """Returns matched eumetsat's frp with esa's list.
    
    :param username: The username to log for the ftp site (url).
    :param password: The password for the ftp site.
    :param url: The eumetsat's ftp site's url.
    :param directory: The name of ftp directory if known.
    :param match_suffix: The suffix match the folder name.
    :param outfile: The outfile to store the eumetsat_frp_list.
    """
    with ftplib.FTP(host=url) as ftp:
        ftp.login(username, password)
        if directory is None:
            directory = ""
        dir_listing = recursive_mlsd(ftp, directory, match_suffix=match_suffix)
        with open(outfile.as_posix(), "w") as fid:
            for item in dir_listing:
                fid.write(f"{item}\n")
    
    
def _get_frp_attributes_from_name(
    search_string: str,
) -> Tuple:
    date_pattern = r"[0-9]{8}T[0-9]{6}" 
    sensor_pattern = r"^S3[A|B]"
    relorb_pattern = r"_[0-9]{3}_"
    # first match is start-time, second is stop-time and third processing time
    date_matches = re.findall(date_pattern, search_string)
    sensor_matches = re.findall(sensor_pattern, search_string)
    relorb_matches = re.search(relorb_pattern, search_string[68:73])
    return date_matches[0], sensor_matches[0], relorb_matches[0][1:-1]
    
    
def subset_eumetsat_frp_list(
    esa_frp_file: Union[Path, str],
    eumetsat_frp_file: Union[Path, str],
    read_pickle: Optional[bool] = False
) -> pd.DataFrame:
    """Returns the list of EUMETSAT's FRP that has same attributes as ESA's FRP.
    
    :param esa_frp_file: The path to esa's file with FRP products.
    :param eumetsat_frp_file: The path to eumetsat's file with FRP product.
    :param read_pickle: Bool flag to read pickled object with FRP attributes. 
        The file with .pkl extension will be searched if True with same name 
        at same location. if available, then will reuse to save generating
        attribute dataframes.
        
    :return:
        DataFrame with EUMETSAT's FRP attributes that matches the
        ESA's FRP in esa_frp_file.
    """
    
    if not read_pickle:
        esa_df = pd.read_csv(esa_frp_file, names=['title'], header=None)
        eu_df = pd.read_csv(eumetsat_frp_file, names=['title'], header=None)

        # create new dataframes with attributes needed to match between esa and eumetsat
        column_names = ["title", "start_date", "sensor", 'relative_orbit']
        esa_attrs_df = pd.DataFrame(columns=column_names)
        eu_attrs_df = pd.DataFrame(columns=column_names)

        # populate esa's df attribute
        for idx, row in esa_df.iterrows():
            start_dt, sensor, relorb = _get_frp_attributes_from_name(row["title"])
            esa_attrs_df = esa_attrs_df.append(
                {
                    'title': row["title"],
                    'start_date': start_dt,
                    'sensor': sensor,
                    'relative_orbit': relorb
                },
                ignore_index=True
            )

        # populate eumetsat's df attribute
        for idx, row in eu_df.iterrows():
            start_dt, sensor, relorb = _get_frp_attributes_from_name(Path(row["title"]).name)
            eu_attrs_df = eu_attrs_df.append(
                {
                    'title': row["title"],
                    'start_date': start_dt,
                    'sensor': sensor,
                    'relative_orbit': relorb
                },
                ignore_index=True
            )

        # save attributes dataframes as pickle object for reuse
        esa_attrs_df.to_pickle(Path(esa_frp_file).with_suffix(".pkl"))
        eu_attrs_df.to_pickle(Path(eumetsat_frp_file).with_suffix(".pkl"))
    else:
        esa_attrs_df = pd.read_pickle(Path(esa_frp_file).with_suffix(".pkl"))
        eu_attrs_df = pd.read_pickle(Path(eumetsat_frp_file).with_suffix(".pkl"))
    
    esa_attrs_df["start_date"] = pd.to_datetime(esa_attrs_df["start_date"], format="%Y%m%dT%H%M%S")
    eu_attrs_df["start_date"] = pd.to_datetime(eu_attrs_df["start_date"], format="%Y%m%dT%H%M%S")
    
    # esa_attrs_df_s3a = esa_attrs_df[esa_attrs_df["sensor"] == "S3A"]
    # esa_attrs_df_s3b = esa_attrs_df[esa_attrs_df["sensor"] == "S3B"]
    
    # perform merge of eumetsat's df and esa's df for common start_date, sensor
    # and relative orbit attributes.
    common_attrs_df = eu_attrs_df.merge(
        esa_attrs_df,
        how="inner",
        on=["start_date", "sensor", "relative_orbit"]
    )
    
    return common_attrs_df
    
    
def _get_eumetsat_ftp_listing(
    ftp_username: str,
    ftp_password: str,
    ftp_url: str,
    ftp_directory: str,
    eumetsat_ftp_frp_list_file: Union[Path, str],
):
    """
    Check's if eumetsat_ftp_frp_list.csv exists if file path
    is provided. Generate list with FRP products if eumetsat_ftp_frp_list_file 
    is None.
    """
    if eumetsat_ftp_frp_list_file is not None:
        if not Path(eumetsat_ftp_frp_list_file).exists():
            logger.info(
                f"{eumetsat_ftp_frp_list_file} does not exist, generating list from {ftp_url}"
            )
            # generate the frp directory listings from eumetsat's ftp site
            get_eumetsat_frp_dir_list(
                ftp_username,
                ftp_password,
                ftp_url,
                ftp_directory,
                outfile=eumetsat_ftp_frp_list_file
            )
        else:
            logger.info(
                f"eumetsat frp list read from {eumetsat_ftp_frp_list_file}"
            )
    else:
        eumetsat_ftp_frp_list_file = Path(os.getcwd()).joinpath("eumetsat_ftp_frp_list.csv")
        if eumetsat_ftp_frp_list_file.exists():
            logger.info(
                f"{eumetsat_ftp_frp_list_file} exists, reading from the available file."
            )
            return eumetsat_ftp_frp_list_file
        logger.info(f"generating list from {ftp_url} at {eumetsat_ftp_frp_list_file}")
        get_eumetsat_frp_dir_list(
            ftp_username,
            ftp_password,
            ftp_url,
            ftp_directory,
            outfile=eumetsat_ftp_frp_list_file
        )
    return eumetsat_ftp_frp_list_file


def _get_eumetsat_s3_listing(
    aws_access_key_id: str,
    aws_secret_access_key: str,
    s3_bucket_name: str,
    eumetsat_s3_frp_list_file: Union[str, Path],
    exclude_s3_key: Optional[str] = 'data/',
    match_suffix: Optional[str] = '.SEN3'
):
    """
    # check eumetsat_s3_frp_list_file exists, if file path is provided
    # generate the s3_listing of eumetsat's file if not available,
    # if eumetsat_s3_frp_list_file is None, then generate the dir listing of eumetsat's frp.
    """
    if eumetsat_s3_frp_list_file is not None:
        if not Path(eumetsat_s3_frp_list_file).exists():
            logger.info(
                f"{eumetsat_s3_frp_list_file} does not exist: generating list from s3://{s3_bucket_name}"
            )
            s3_frp_list(
                aws_access_key_id,
                aws_secret_access_key,
                s3_bucket_name,
                exclude_s3_key=exclude_s3_key,  # this key will exclude where esa's file are stored in s3
                match_suffix=match_suffix,
                outfile=eumetsat_s3_frp_list_file
            )
        else:
            logger.info(
                f"eumetsat s3 {match_suffix} frp list read from {eumetsat_s3_frp_list_file}"
            )
    else:
        eumetsat_s3_frp_list_file = Path(os.getcwd()).joinpath(f"eumetsat_s3_frp_{match_suffix[1:].lower()}_list.csv")
        if eumetsat_s3_frp_list_file.exists():
            logger.info(
                f"{eumetsat_s3_frp_list_file} exist, reading from the available file."
            )
            return eumetsat_s3_frp_list_file
        
        logger.info(
            f"generating eumetsat {match_suffix} frp list from s3://{s3_bucket_name} at {eumetsat_s3_frp_list_file}"
        )
        s3_frp_list(
            aws_access_key_id,
            aws_secret_access_key,
            s3_bucket_name,
            exclude_s3_key=exclude_s3_key,  # this key will exclude where esa's file are stored in s3
            match_suffix=match_suffix,
            outfile=eumetsat_s3_frp_list_file
        )
    return eumetsat_s3_frp_list_file

    
def _get_esa_s3_listing(
    aws_access_key_id: str,
    aws_secret_access_key: str,
    s3_bucket_name: str,
    esa_s3_frp_list_file: Union[str, Path],
    exclude_s3_key: Optional[str] = 'eumetsat_data/',
    match_suffix: Optional[str] = '.SEN3'
):
    """
    Check if esa_frp_list_file exists, if file path is provided,
    Generate the s3_listing if esa's frp file is not available or None.
    """
    if esa_s3_frp_list_file is not None:
        if not Path(esa_s3_frp_list_file).exists():
            logger.info(
                f"{esa_s3_frp_list_file} does not exist, generating list from s3://{s3_bucket_name}"
            )

            s3_frp_list(
                aws_access_key_id,
                aws_secret_access_key,
                s3_bucket_name,
                exclude_s3_key=exclude_s3_key,  # this key will exlude where eumetsat's frp are stored in s3
                match_suffix=match_suffix,
                outfile=esa_s3_frp_list_file
            )
        else:
            logger.info(
                f"esa s3 {match_suffix} frp list read from {esa_s3_frp_list_file}"
            )
    else:
        esa_s3_frp_list_file = Path(os.getcwd()).joinpath(f"esa_s3_frp_{match_suffix[1:].lower()}_list.csv")
        if esa_s3_frp_list_file.exists():
            logger.info(
                f"{esa_s3_frp_list_file} exist, reading from the available file."
            )
            return esa_s3_frp_list_file

        logger.info(f"generating esa {match_suffix} frp list from s3://{s3_bucket_name} at {esa_s3_frp_list_file}")
        s3_frp_list(
            aws_access_key_id,
            aws_secret_access_key,
            s3_bucket_name,
            exclude_s3_key=exclude_s3_key, # this key will exclude where eumetsat's frp are stored in s3.
            match_suffix=match_suffix,
            outfile=esa_s3_frp_list_file
        )
    return esa_s3_frp_list_file