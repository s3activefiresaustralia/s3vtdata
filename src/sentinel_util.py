#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import re
import subprocess
import tempfile
from datetime import datetime
from datetime import date
from datetime import timedelta
from pathlib import Path
from typing import Tuple, Union, Optional, List, Dict
import shutil
import xml.etree.ElementTree as ET
import shapely
import boto3
import geopandas as gpd
import numpy as np
import pandas as pd
from dask import delayed

from xml_util import getNamespaces


def _parse_datetime(dt_str):
    return datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S.%fZ')
    
def swath_from_xml(xml_file):
    namespaces = getNamespaces(xml_file)
    tree = ET.parse(xml_file)
    gml_strs = tree.find(".//gml:posList", namespaces).text.split(" ")
    geom = shapely.geometry.Polygon(
        [(float(gml_strs[i+1]), float(gml_strs[i])) for i in range(0, len(gml_strs), 2)]
    )
    acq_start_dt = _parse_datetime(tree.find(".//sentinel-safe:startTime", namespaces).text)
    acq_stop_dt = _parse_datetime(tree.find(".//sentinel-safe:stopTime", namespaces).text)
    diff_dt = acq_stop_dt - acq_start_dt
    swath_attrs = {
        'geometry': geom,
        'AcquisitionOfSignalLocal': acq_start_dt + timedelta(hours=9.5),
        'AcquisitionOfSignalUTC': acq_start_dt,
        'LossOfSignalUTC': acq_stop_dt,
        'TransitTime': acq_start_dt + timedelta(seconds=diff_dt.seconds/2),
        'Satellite': f"Sentinel_3{tree.find('.//sentinel-safe:number', namespaces).text}",
        'Sensor': 'SLSTR',
        'OrbitNumber': tree.find('.//sentinel-safe:orbitNumber', namespaces).text,
        'OrbitHeight': 814.5,
        'Node': tree.find('.//sentinel-safe:orbitNumber', namespaces).attrib['groundTrackDirection']
    }
    # gdf = gpd.GeoDataFrame(swath_attrs.items(), crs="EPSG:4326")
    return swath_attrs

def get_xml_from_s3(sentinel3_swath_pkl, prefix):
    client = boto3.client("s3")
    bucket = "s3vtaustralia"
    paginator = client.get_paginator('list_objects')
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    swath_attrs = {
        'geometry': [],
        'AcquisitionOfSignalLocal': [],
        'AcquisitionOfSignalUTC':[],
        'LossOfSignalUTC': [],
        'TransitTime': [],
        'Satellite': [],
        'Sensor': [],
        'OrbitNumber': [],
        'OrbitHeight': [],
        'Node': []

    }
    with tempfile.TemporaryDirectory() as outdir:
        for page in page_iterator:
            for obj in page['Contents']:
                _key = obj['Key']
                _name = Path(_key).name
                if _name == 'xfdumanifest.xml':
                    xml_file = Path(outdir).joinpath(_name).as_posix()
                    client.download_file(
                        bucket, _key, xml_file
                    )
                    _attrs = swath_from_xml(xml_file)
                    for k in swath_attrs.keys():
                        swath_attrs[k].append(_attrs[k])
    swath_gdfs = gpd.GeoDataFrame(swath_attrs)
    swath_gdfs.to_pickle(sentinel3_swath_pkl)

    
def main(outdir: Path):
    swath_dfs = []
    for _prefix in ["data", "eumetsat_data"]:
        if _prefix == "data":
            swath_pkl = outdir.joinpath('sentinel3_swath_gdfs_esa.pkl')
        else:
            swath_pkl = outdir.joinpath('sentinel3_swath_gdfs_eumetsat.pkl')
        if not swath_pkl.exists():
            get_xml_from_s3(swath_pkl, _prefix)
        
        swath_df = pd.read_pickle(swath_pkl)
        swath_dfs.append(swath_df)
    
    swath_gdf = pd.concat(swath_dfs, ignore_index=True)
    swath_gdf.drop_duplicates(subset=['AcquisitionOfSignalUTC'])
    swath_gdf.to_pickle(outdir.joinpath("sentinel3_swath_gdfs.pkl"))
    print(swath_gdf.count())
    
    
if __name__ == "__main__":
    outdir = Path('/home/jovyan/s3vt_dask/s3vtdata/')
    main(outdir)
    