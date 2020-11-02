#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import importlib
import sys
from pathlib import Path
from typing import Union, List, Iterable, Optional, Tuple
from datetime import datetime
from multiprocessing import Pool as ProcessPool
import subprocess
import tempfile
import time
import logging as logger
import ftplib
import re
import lxml
from lxml import etree
import shutil 
import requests
import zipfile

from shapely.geometry import Polygon

from netCDF4 import Dataset
import pandas as pd
import geopandas as gpd
import numpy as np
import boto3
from botocore.exceptions import ClientError
from bs4 import BeautifulSoup
import yaml
import numpy as np


logger.basicConfig(format='%(levelname)s:%(message)s', level=logger.INFO)
count_number = 0
total_counts = 0

class DummyPool:
    def __enter__(self):
        return self

    def starmap(self, func, args):
        return [func(*arg) for arg in args]


def Pool(processes):
    if not processes:
        return DummyPool()

    return ProcessPool(processes=processes)


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


def download_cophub(
    url: Union[str, Path],
    out_directory: Union[Path, str]
) -> Union[Path, str]:
    """Download file from Cophub"""
    try:
        cmd = [
            'wget', 
            '-q',
            '--recursive',
            str(url),
            '-nH',
            '--cut-dirs=10',
            '--directory-prefix='+out_directory
            ]
        ret_code = subprocess.check_call(cmd)
    except:
        ret_code = 1
    zipf_name = Path(out_directory).joinpath(Path(url).name)
    xfd_file = Path(out_directory).joinpath(f"{Path(url).stem}.SEN3/xfdumanifest.xml")                          
    if ret_code == 0:
        with zipfile.ZipFile(zipf_name.as_posix(), 'r') as zip_ref:
            zip_ref.extractall(str(out_directory))
        os.remove(zipf_name)
    if xfd_file.exists():
        return xfd_file
    return ret_code


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
    print(out_directory)
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
        print(ret_code)
        logger.info(["".join(item) for item in cmd])
    except:
        ret_code = 1
        # logger.info("Remote file retrieval failed "+str(['wget', '-q', '--user='+username, '--password='+password, url]))
    return ret_code
    

def recursive_mlsd(ftp_object, path="", maxdepth=7, match_suffix=None):
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
        if key_name.startswith(exclude_s3_key):
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
    
    
def get_polygon(xml_manifest: Union[Path, str]):
    """Get Polygon from xmldummanifest.xml file"""
    try:
        tree = etree.parse(xml_manifest.as_posix())
    except:
        tree = etree.parse(xml_manifest)
    root = tree.getroot()
    extracted = root.findall('./metadataSection/metadataObject/metadataWrap/xmlData/sentinel-safe:frameSet/sentinel-safe:footPrint/gml:posList', root.nsmap)
    coords = extracted[0].text
    split = re.split(" ", coords)
    coordsx = []
    coordsy = []
    final_list = []
    for c in range(0, len(split)):
        if c % 2 == 0:
            coordsx.append(float(split[c]))
        else:
            coordsy.append(float(split[c]))
    for c in range(0, len(coordsx)):
        final_list.append((coordsy[c], coordsx[c]))
    return Polygon(final_list)
    
    
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


def _get_cophub_frp_listing(
    domain: Optional[str] = 'http://dapds00.nci.org.au',
    ext: Optional[str] = '.zip',
    params: Optional[dict] = None,
    start_year: Optional[int] = 2020,
    end_year: Optional[int] = 2020,
    outfile: Optional[Union[str, Path]] = Path(os.getcwd()).joinpath("cophub_frp_list.csv")
) -> Union[Path, str]:
    """Method to get the listing Sentinel-3 FRP data from cophub.
    
    Any file that has processing center other than `MAR` will be excluded. The MAR
    processing center is assumed to have used EUMETSAT FRP algorithm.
    
    :param domain: The domain name of cophub data host site.
    :param ext: The extension of the FRP product file.
    :param params: The parameter {username: USERNAME, password: PASSWORD} if required.
    :param start_year: The start year to execute search for.
    :param end_year: The end year to execute search for.
    :param outfile: The output file to save the FRP product listing from the site.
    
    :return:
        The full path to a file with the listing of FRP products from the site.
    """
    
    if params is None:
        params = {}
        
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            dir_listing = []
            year_month_file = Path(outfile).parent.joinpath(f"monthly_cophub_frp_list_{year}{month:02}.csv")
            if year_month_file.exists():
                continue
            logger.info(f"getting frp listing for {year}-{month:02} from {domain}")
            month_url = f"{domain}/thredds/catalog/fj7/Copernicus/Sentinel-3/SLSTR/SL_2_FRP___/{year}/{year}-{month:02}/catalog.html"
            try:
                response = requests.get(month_url, params=params)
                response.raise_for_status()
            except requests.exceptions.HTTPError as err:
                logger.info(err)
                continue
            
            for day in range(1, 32):
                day_url = f"{domain}/thredds/catalog/fj7/Copernicus/Sentinel-3/SLSTR/SL_2_FRP___/{year}/{year}-{month:02}/{year}-{month:02}-{day:02}/catalog.html"
                try:
                    response = requests.get(day_url, params=params)
                except requests.exceptions.HTTPError as err:
                    logger.info(f'Excepted HTTP Error: {err}')
                    continue
                    
                response_text = response.text
                soup = BeautifulSoup(response_text, 'html.parser')
                parent = [node.get('href').split('=')[1] for node in soup.find_all('a') if node.get('href').endswith(ext)]
                
                # if Timeout Error occurs, retry once more after sleeping for 10s
                ret_code = int(response.status_code)
                while ret_code == 504: 
                    logger.info(f"Received status code of 504 for {year}-{month:02}-{day:02}: Retrying again")
                    time.sleep(10)
                    response = requests.get(day_url, params=params)
                    ret_code = int(response.status_code)
                    response_text = response.text
                    soup = BeautifulSoup(response_text, 'html.parser')
                    parent = [node.get('href').split('=')[1] for node in soup.find_all('a') if node.get('href').endswith(ext)]
 
                print(f"{year}-{month:02}-{day:02} : {len(parent)}")
                for p in parent:
                    if "_MAR_" in p:  # only copy the files that is from EUMETSAT
                        # print(p)
                        dir_listing.append(f"http://dapds00.nci.org.au/thredds/fileServer/fj7/Copernicus/Sentinel-3/{p[14:]}")
            with open(year_month_file.as_posix(), "w") as mfid:
                for item in dir_listing:
                     mfid.write(f"{item}\n")
    
    with open(outfile.as_posix(), "w") as outfid:
        for mfile in outfile.parent.iterdir():
            if mfile.stem.startswith('monthly'):
                with open(mfile, 'r') as infid:
                    for line in infid.readlines():
                        outfid.write(f"{line}")
    return outfile  


def get_gpd_attrs(
    aws_access_key_id,
    aws_secret_access_key,
    s3_bucket_name,
    frp_file,
    s3_folder,
):
    global count_number
    global total_counts
    count_number += 16
    print(f"processing {count_number} of {total_counts}")
    title = Path(frp_file).name
    _dt, sensor, relorb = _get_frp_attributes_from_name(title)
    aws_session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name='ap-southeast-2'
    )
    s3_client = aws_session.client('s3')
    with tempfile.TemporaryDirectory() as tmpdir:
        xfd_file = Path(tmpdir).joinpath("xfdumanifest.xml")
        if s3_folder is not None:
            try:
                s3_download_file(
                    s3_bucket_name,
                    s3_client,
                    filename="xfdumanifest.xml",
                    out_dir=Path(tmpdir),
                    prefix=f"{s3_folder}/{_dt[0:4]}-{_dt[4:6]}-{_dt[6:8]}/{title}"
                )
            except FileNotFoundError as err:
                return None
        else:
            xfd_file = download_cophub(frp_file, tmpdir)
        poly = get_polygon(xfd_file)
        return {
            'title': frp_file,
            'start_date': _dt,
            'sensor': sensor,
            'relative_orbit': relorb,
            'geometry': poly
        }

    
def create_cophub_frp_df(
    esa_s3_frp_file: Union[Path, str],
    eumetsat_s3_frp_file: Union[Path, str],
    cophub_frp_file: Union[Path, str],
    aws_access_key_id: str,
    aws_secret_access_key: str,
    s3_bucket_name: Optional[str] = "s3vtaustralia",
    nprocs: Optional[int] = 1
) -> pd.DataFrame:
    """Creates GeoDataFrame to hold attributes required to subset cophub datasets.
    
    :param esa_s3_frp_file: The path to esa's file with FRP products.
    :param eumetsat_s3_frp_file: The path to eumetsat's file with FRP product.
    :param cophub_frp_file: The path to Cophub FRP file.
    :param aws_access_key_id: The AWS_ACCESS_KEY_ID.
    :param aws_secret_access_key: The AWS_SECERET_ACCESS_KEY
    :param s3_bucket_name: The name of the s3 bucket.
    :param nprocs: The number of processor used in parallel processing.
        
    :return:
        Tuple of Cophub, ESA and EUMETSET GeoDataFrame with FRP attributes.
    """
    global total_counts
    
    esa_df = pd.read_csv(esa_s3_frp_file, names=['title'], header=None)
    eu_df = pd.read_csv(eumetsat_s3_frp_file, names=['title'], header=None)
    cophub_df = pd.read_csv(cophub_frp_file, names=['title'], header=None)
    
    # create new dataframes with attributes needed to match between cophub, esa and eumetsat
    column_names = ["title", "start_date", "sensor", 'relative_orbit']
    
    # populate esa's df attribute
    if not Path(esa_s3_frp_file).with_suffix(".pkl").exists():
        logger.info("processing ESA FRP GeoDataFrame...")
        esa_attrs_df = gpd.GeoDataFrame(columns=column_names)
        esa_frp_files = [row['title'] for _, row in esa_df.iterrows()]
        with Pool(processes=nprocs) as pool:
            results = pool.starmap(
                get_gpd_attrs,
                [
                    (
                        aws_access_key_id,
                        aws_secret_access_key,
                        s3_bucket_name,
                        frp_file,
                        "data"
                    )
                    for frp_file in esa_frp_files
                ],
            )
        for gpd_attrs in results:
            esa_attrs_df = esa_attrs_df.append(gpd_attrs, ignore_index=True)
        esa_attrs_df.to_pickle(Path(esa_s3_frp_file).with_suffix(".pkl"))
    else:
        esa_attrs_df = pd.read_pickle(Path(esa_s3_frp_file).with_suffix(".pkl"))
    # populate eumetsat's df attribute
    if not Path(eumetsat_s3_frp_file).with_suffix(".pkl").exists():
        logger.info("processing EUMETSAT FRP GeoDataFrame...")
        eu_attrs_df = gpd.GeoDataFrame(columns=column_names)
        eu_frp_files = [row['title'] for _, row in eu_df.iterrows()]
        with Pool(processes=nprocs) as pool:
            results = pool.starmap(
                get_gpd_attrs,
                [
                    (
                        aws_access_key_id,
                        aws_secret_access_key,
                        s3_bucket_name,
                        frp_file,
                        'eumetsat_data'
                    )
                    for frp_file in eu_frp_files
                ]
            )
        for gpd_attrs in results:
            eu_attrs_df = eu_attrs_df.append(gpd_attrs, ignore_index=True)
        eu_attrs_df.to_pickle(Path(eumetsat_s3_frp_file).with_suffix(".pkl"))
    else:
        eu_attrs_df = pd.read_pickle(Path(eumetsat_s3_frp_file).with_suffix(".pkl"))
        
    # populate cophub's df attribute
    if not Path(cophub_frp_file).with_suffix(".pkl").exists():
        logger.info("processing Cophub FRP GeoDataFrame...")
        cophub_attrs_df = gpd.GeoDataFrame(columns=column_names)
        cop_frp_files = [row['title'] for _, row in cophub_df.iterrows()]
        total_counts = len(cop_frp_files)
        with Pool(processes=nprocs) as pool:
            results = pool.starmap(
                get_gpd_attrs,
                [
                    (
                        aws_access_key_id,
                        aws_secret_access_key,
                        s3_bucket_name,
                        frp_file,
                        None
                    )
                    for idx, frp_file in enumerate(cop_frp_files)
                ]
            )
        for gpd_attrs in results:
            if gpd_attrs is not None:
                cophub_attrs_df = cophub_attrs_df.append(gpd_attrs, ignore_index=True)
        cophub_attrs_df.to_pickle(Path(cophub_frp_file).with_suffix(".pkl"))
    else:
        cophub_attrs_df = pd.read_pickle(Path(cophub_frp_file).with_suffix(".pkl"))
    
    return cophub_attrs_df, esa_attrs_df, eu_attrs_df


def subset_cophub_from_esa(
    esa_df: gpd.GeoDataFrame,
    cop_df: gpd.GeoDataFrame,
    outdir: Optional[Union[Path, str]] = Path(os.getcwd()),
    save_file: Optional[bool] = True
) -> gpd.GeoDataFrame:
    """Subsets Cophub list based on the ESA list if their Geometry(Footprint) intersects
       for a given day.
    
    :param esa_df: The GeoDataFrame with attributes for ESA S3 listing.
    :param cop_df: The GeoDataFrame with attributes for Cophub(NCI) file listing.
    :param save_file: Flag to save the subset download list of not.
    
    :return:
        GeoDataFrame with Cophub FRP product footprint that intersects with ESA FRP
        footprint for a given day.
    """
    # convert datetime string to datetime stamp
    esa_df['start_date'] = pd.to_datetime(esa_df["start_date"], format="%Y%m%dT%H%M%S")
    cop_df['start_date'] = pd.to_datetime(cop_df["start_date"], format="%Y%m%dT%H%M%S")
    
    # assign crs to GeoDataFrame
    esa_df.crs = 'EPSG:4326'
    cop_df.crs = 'EPSG:4326'
    
    esa_df['date'] = pd.to_datetime(esa_df['start_date']).dt.date
    cop_df['date'] = pd.to_datetime(cop_df['start_date']).dt.date
    
    
    column_names = ["title", "start_date", "sensor", 'relative_orbit', 'geometry']
    cop_download_df = gpd.GeoDataFrame(columns=column_names)
    for idx_esa, esa_row in esa_df.iterrows():
        cophub_df_subset = cop_df[cop_df['date'][:] == esa_row['date']].copy()
        cophub_df_subset['intersects'] = False
        esa_geom = esa_row['geometry'].buffer(0)
        for idx_cop, cop_row in cophub_df_subset.iterrows():
            cop_geom = cop_row['geometry'].buffer(0)
            if esa_geom.intersects(cop_geom):
                cop_download_df = cop_download_df.append(
                    {
                        'title': cop_row["title"],
                        'start_date': cop_row['start_date'],
                        'sensor': cop_row['sensor'],
                        'relative_orbit': cop_row['relative_orbit'],
                        'geometry': cop_row['geometry'],
                        'esa_geometry': esa_row['geometry']
                    },
                    ignore_index=True
                )
    subset_cop_download_df = cop_download_df.drop_duplicates(['title'])
    if save_file:
        subset_cop_download_df.to_csv(outdir.joinpath('cophub_download_list.csv'))
    return subset_cop_download_df


def generate_hotspot_geojson(
    download_url: Union[Path, str],
    aws_access_key_id: str,
    aws_secret_access_key: str,
    s3_bucket_name: str,
    s3_upload: Optional[bool] = False,
    outdir: Optional[Union[Path, str]] = Path(os.getcwd()).joinpath("COPHUB_GEOJSON")
) -> Union[Path, str]:
    """Generates FRP geojson and uploads to s3-bucker if s3_upload is set to True.
    
    :param download_url: The url to a FRP file, nci cophub site.
    :param aws_access_key_id: The AWS_ACCESS_KEY_ID with privilage to upload to s3-bucket if s3_upload is True.
    :param aws_secret_access_key: THE AWS_SECRET_ACCESS_KEY with privileges to upload.
    :param s3_bucket_name: The name of the s3-bucket to upload data.
    :param s3_upload: The flag to set if upload to s3 bucket.
    :param outdir: The directory to save .geojson file in local file system.
    
    :return:
        The url of the downloaded FRP file.
    """
    aws_session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name='ap-southeast-2'
    )
    s3_client = aws_session.client('s3')
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_frp_url = download_cophub(download_url, tmpdir)  # this returns the full path to .xml file. 
            _frp_dir = Path(xml_frp_url).parent  # the parent parth to xml_frp_file is .SEN3 folder
            _frp_name = Path(xml_frp_url).parent.name
    
            acq_date = re.findall(r"[0-9]{8}T[0-9]{6}" , _frp_name)[0]
            _frp_dir_name = f"eumetsat_data/{acq_date[0:4]}-{acq_date[4:6]}-{acq_date[6:8]}" # directory to store the eumetsat's .geojson files for processing
            for item in Path(_frp_dir).iterdir():
                if s3_upload:
                    s3_upload_file(
                        item,
                        s3_client,
                        s3_bucket_name,
                        prefix=f"{_frp_dir_name}/{_frp_name}"
                    )
                if item.name == "FRP_in.nc":
                    gpd_hotspotfile = _frp_dir.with_suffix(".FRP.geojson")
                    s3hotspotsgpd = IPF_FRP_read(item)
                    if len(s3hotspotsgpd) > 0:
                        s3hotspotsgpd.to_file(gpd_hotspotfile, driver='GeoJSON')
                        if s3_upload:
                            print(gpd_hotspotfile)
                            s3_upload_file(gpd_hotspotfile, s3_client, s3_bucket_name, prefix=_frp_dir_name)
                        outdir.joinpath(_frp_dir_name).mkdir(parents=True, exist_ok=True)
                        shutil.move(gpd_hotspotfile.as_posix(), f"{outdir.as_posix()}/{_frp_dir_name}/{gpd_hotspotfile.name}")
        return download_url
    except Exception as err:
        logger.info(f"failed to process {download_url}: {err}")
        return None


def process_cophub_subset(
    aws_access_key_id: str,
    aws_secret_access_key: str,
    esa_s3_frp_file: Optional[Union[Path, str]] = None,
    eumetsat_s3_frp_file: Optional[Union[Path, str]] = None,
    cophub_frp_file: Optional[Union[Path, str]] = None,
    output_dir: Union[Path, str] = Path(os.getcwd()).joinpath('data'),
    s3vt_s3_bucket_name: Optional[str] = 's3vtaustralia',
    start_year: Optional[int] = 2020,
    end_year: Optional[int] = 2020,
    nprocs: Optional[int] = 1
):
    """Main processing pipeline to subset Cophub datasets to ESA equivalent in S3
    
    :param aws_access_key_id: The AWS_ACCESS_KEY_ID.
    :param aws_secret_access_key: The AWS_SECERET_ACCESS_KEY
    :param esa_s3_frp_file: The .csv file containing the list of ESA FRP product.
    :param eumetsat_s3_frp_file: The .csv file containing the list of EUMETSAT FRP product.
    :param cophub_frp_file: The .csv file containing the list of Cophub FRP product.
    :param output_dir: The output directory to store the files generated from this process.
    :param s3vt_s3_bucket_name: The s3 bucket name where FRP product resides.
    :param start_year: The start year to limit the FRP list from Cophub.
    :param end_year: The end year to limit the FRP list from Cophub.
    :param nprocs: The number of processor used in parallel processing.

    """
    if not output_dir.exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    try:
        if (aws_access_key_id is not None) & (aws_secret_access_key is not None):
            aws_session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name='ap-southeast-2'
            )
        else:
            aws_session = boto3.Session(region_name='ap-southeast-2')
    except Exception:
        logger.info("failed to initialize aws session")
        sys.exit(0)
    
    # generate esa s3 listing if the file does not exist
    if esa_s3_frp_file is not None:
        if Path(esa_s3_frp_file).exists():
            logger.info(f"Using ESA FRP listing from {str(esa_s3_frp_file)}")
        else:
            logger.info(f"Generating ESA FRP listing from {s3vt_s3_bucket_name} at {str(esa_s3_frp_file)}")
            esa_s3_sen3_frp_list_file = _get_esa_s3_listing(
                aws_access_key_id,
                aws_secret_access_key,
                s3vt_s3_bucket_name,
                esa_s3_frp_file,
                exclude_s3_key='eumetsat_data/',  # this key is to exclude the s3 folder where eumetsat's FRP product resides.
                match_suffix='.SEN3'
            )
    else:
        esa_s3_frp_file = Path(output_dir).joinpath("esa_s3_frp_sen3_list.csv")
        logger.info(f"Generating ESA FRP listing from {s3vt_s3_bucket_name} at {str(esa_s3_frp_file)}")
        esa_s3_sen3_frp_list_file = _get_esa_s3_listing(
            aws_access_key_id,
            aws_secret_access_key,
            s3vt_s3_bucket_name,
            esa_s3_frp_file,
            exclude_s3_key='eumetsat_data/',  # this key is to exclude the s3 folder where eumetsat's FRP product resides.
            match_suffix='.SEN3'
        )
    
    # generate eumetsat FRP s3 listing if the file does not exist
    if eumetsat_s3_frp_file is not None:
        if Path(eumetsat_s3_frp_file).exists():
            logger.info(f"Using EUMETSAT FRP listing from {str(eumetsat_s3_frp_file)}")
        else:
            logger.info(f"Generating EUMETSAT FRP listing from {s3vt_s3_bucket_name} at {str(eumetsat_s3_frp_file)}")
            eumetsat_s3_frp_file = _get_eumetsat_s3_listing(
                aws_access_key_id,
                aws_secret_access_key,
                s3vt_s3_bucket_name,
                eumetsat_s3_frp_file,
                exclude_s3_key='data/', # key to exlude s3 bucket where esa's FRP are stored.
                match_suffix='.SEN3'
            )
    else:
        eumetsat_s3_frp_file = Path(output_dir).joinpath("eumetsat_s3_frp_sen3_list.csv")
        logger.info(f"Generating EUMETSAT FRP listing from {s3vt_s3_bucket_name} at {str(eumetsat_s3_frp_file)}")
        eumetsat_s3_frp_file = _get_eumetsat_s3_listing(
            aws_access_key_id,
            aws_secret_access_key,
            s3vt_s3_bucket_name,
            eumetsat_s3_frp_file,
            exclude_s3_key='data/', # key to exlude s3 bucket where esa's FRP are stored.
            match_suffix='.SEN3'
        )
    
    # generate Cophub FRP listing if the file does not exist
    if cophub_frp_file is not None:
        if Path(cophub_frp_file).exists():
            logger.info(f"Using Cophub FRP listing from {str(cophub_frp_file)}")
        else:
            logger.info(f"Generating Cophub FRP listing from http://dapds00.nci.org.au/ at {str(cophub_frp_file)}")
            _get_cophub_frp_listing(
                'http://dapds00.nci.org.au',
                '.zip',
                None,
                start_year,
                end_year,
                cophub_frp_file
            )
    else:
        cophub_frp_file = Path(output_dir).joinpath("cophub_frp_list.csv")
        logger.info(f"Generating Cophub FRP listing from http://dapds00.nci.org.au/ at {str(cophub_frp_file)}")
        _get_cophub_frp_listing(
            'http://dapds00.nci.org.au',
            '.zip',
            None,
            start_year,
            end_year,
            cophub_frp_file
        )
    
    # create GeoDataFrame with required attributes to enable subset.
    # This step needs to be run in parallel because it takes a while to 
    # create a dafaframe because files from its xml file with metadata 
    # from the source needs to be downloaded to extract geospatial footprint.
    logger.info("Generating a GeoDataFrame for ESA, EUMETSAT and Cophub FRP...")
    # s3_client = aws_session.client("s3")
    cophub_attrs_df, esa_attrs_df, eu_attrs_df = create_cophub_frp_df(
        esa_s3_frp_file,
        eumetsat_s3_frp_file,
        cophub_frp_file,
        aws_access_key_id,
        aws_secret_access_key,
        s3_bucket_name=s3vt_s3_bucket_name,
        nprocs=nprocs
    )
    
    # subsetting the Cophub data to that of ESA
    cophub_subset_file = output_dir.joinpath('cophub_download_list.csv')
    if not cophub_subset_file.exists():
        logger.info("Subsetting Cophub FRP...")
        subset_cop_download_df = subset_cophub_from_esa(
            esa_attrs_df,
            cophub_attrs_df,
            output_dir
        )
    else:
        logger.info(f"Reading cophub subset from {cophub_subset_file}")
        subset_cop_download_df = pd.read_csv(cophub_subset_file)
    # process the .geojson file and upload to S3 if s3_upload is True.
    s3_upload = False # set to True if files are to be uploaded to s3
    cop_download_list = [row['title'] for _, row in subset_cop_download_df.iterrows()]
    with Pool(processes=nprocs) as pool:
        results = pool.starmap(
            generate_hotspot_geojson,
            [
                (
                    frp_url,
                    aws_access_key_id,
                    aws_secret_access_key,
                    s3vt_s3_bucket_name,
                    s3_upload
                )
                for frp_url in cop_download_list
            ]
        )
        with open(Path(output_dir).joinpath("cophub_hotspot_completed.csv"), "w") as fid:
            for res in results:
                if res is not None:
                    fid.writeline(str(res))
                    
        
if __name__ =='__main__':
    process_cophub_subset(
        aws_access_key_id,
        aws_secret_access_key,
        nprocs=16
    )