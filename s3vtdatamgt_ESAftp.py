# -*- coding: utf-8 -*-
##################
# Sentinel 3 Validation Team project
# Sentinel 3 Active Fires from Fire Radiative Power
#
# Get S3 data to EC2
# Extract hotspots and extract hotspots
# Extract hotspots and push multiple formats to accompany data files 
# Copy data to AWS S3
# Push summary of download inventory and status to AWS S3
##################

##################
# Import modules #
##################
import ftplib
import boto3
from netCDF4 import Dataset
import datetime as dt
import geopandas as gpd
import pandas as pd
import yaml
import argparse
import os
import subprocess
import logging as logger
import sys

logger.basicConfig(format='%(levelname)s:%(message)s', level=logger.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('--configuration', dest='configuration', default='config.yaml',help='configuration yaml file')
args = parser.parse_args()

def get_ftp_file_list(username, password, url, directory):
    ftp = ftplib.FTP(url)
    ftp.login(username, password)
    ftp.pwd()
    ftp.cwd(directory)
    return(ftp.nlst())

def get_ftp_dir(username, password, url, directory, subdirectory):
    directory = os.path.join(directory, subdirectory)
    url = os.path.join('ftp://', url)
    url.split()
    target_path = str(os.path.join(url, directory))
    # TODO - use depth of folder structure to determine --cut-dirs number
    try:
        subprocess.call(['wget', '--user='+username, '--password='+password, '--recursive', target_path, '-nH', '--cut-dirs=2', '--directory-prefix='+subdirectory])
        logger.info(str(['wget', '--user='+username, '--password='+password, '--resursive', target_path, '-nH', '--cut-dirs=2', '--directory-prefix='+subdirectory]))
        success = True
    except:
        logger.info("Remote file retrieval failed "+str(['wget', '--user='+username, '--password='+password, target_path]))
        success = False
    return(success)

def get_polygon_from_gml(gml_dict):
    listoftuples = []
    for i in list(gml_dict.split(" ")):
        x = float(i.split(',')[1]) 
        if x <= 30.0:
            x = x + 360.0
            
        pair = (x, float(i.split(',')[0]))
        listoftuples.append(pair)
    return(listoftuples)


def IPF_FRP_read(filename):
    print ("Processing ",filename)
    try:
        dataset = Dataset(filename)
    except (RuntimeError):
        print ("Unable to open ",filename)
        sys.exit(1)
        
    IPF_FRP =  gpd.GeoDataFrame()
    for var in dataset.variables:
        # print (var)
        temp = dataset[var]
        if len(temp.shape) < 2:
            IPF_FRP[var] = dataset[var][:]
    IPF_FRP.geometry = gpd.points_from_xy(IPF_FRP.longitude, IPF_FRP.latitude)
    return IPF_FRP 

if __name__ == '__main__':
    # Get configurations
    with open(args.configuration, 'r') as config:
            cfg = yaml.load(config, Loader=yaml.Loader)    
    
    for configuration in cfg['configurations']:
        
        logger.info(configuration['ftpusername']+' '+configuration['ftppassword']+' '+configuration['ftpurl'])

        # Get File List from server - remove inventory file to refresh
        
        if not os.path.exists('s3mpc_inventory.txt'):
        
            # Determine number of records to retrieve
            sen3 = get_ftp_file_list(configuration['ftpusername'], configuration['ftppassword'], configuration['ftpurl'], configuration['ftpdirectory']) 
            with open('s3mpc_inventory.txt', 'w') as outfile:
                for item in sen3:
                    outfile.write("%s\n" % item)
        
        # Assess inventory against AWS bucket listing
        s3 = boto3.resource('s3', aws_access_key_id=configuration['awskeyid'],
                            aws_secret_access_key=configuration['awskeypass'])
        
        s3folderlist = []
        s3geojsonlist = []
        s3bucket = s3.Bucket(configuration['awss3bucket'])
        
        for bucket_object in s3bucket.objects.all():
            s3bucketobject = str(bucket_object.key).split("/")[2]
            if '.SEN3' in s3bucketobject:
                s3folderlist.append(s3bucketobject)
            if '.FRP.geojson' in s3bucketobject:
                s3geojsonlist.append(s3bucketobject)
            logger.info(str(bucket_object.key).split("/")[2])                        

        s3vtgpd = pd.read_csv('s3mpc_inventory.txt')
        s3vtgpd.columns = ['title']
        # Add fields to enable monitoring
        s3vtgpd['hotspot'] = 0
        s3vtgpd['download'] = 0
        s3vtgpd['s3bucket'] = 0
        dataframelength = len(s3vtgpd)
            
        # Check if folder already downloaded and flag in gpd
        for i in range(dataframelength):
            if s3vtgpd.loc[i]['title']+'.SEN3' in set(s3folderlist):
                s3vtgpd.at[i, 'download'] = 1
            if s3vtgpd.loc[i]['title']+'.FRP.geojson' in set(s3folderlist):
                s3vtgpd.at[i, 'hotspot'] = 1    
        
        # TODO - do something useful with the dataframe - gather polygon info and metadata    
        for i in range(dataframelength):
        
            if s3vtgpd.loc[i]['download'] == 0:
        
                filename = s3vtgpd.loc[i]['title']+'/FRP_in.nc'
                s3hotspots = s3vtgpd.loc[i]['title'][:-5]+'.FRP.geojson'
                if get_ftp_dir(configuration['ftpusername'], configuration['ftppassword'], configuration['ftpurl'], configuration['ftpdirectory'], s3vtgpd.loc[i]['title']) == False:
                    s3vtgpd.at[i, 'download'] = 0
                else:
                    s3vtgpd.at[i, 'download'] = 1
                    s3hotspotsgpd = IPF_FRP_read(filename)
                    if len(s3hotspotsgpd) != 0:
                        s3hotspotsgpd.to_file(s3hotspots, driver='GeoJSON')
                        #s3vthostpotsgpdlist.append(s3hotspotsgpd)
                        s3vtgpd.at[i, 'hotspot'] = 1
                    else:
                        s3vtgpd.at[i, 'hotspot'] = 0
        
                # Assumes AWScli configured
                folderdate = (dt.datetime.strptime(str(s3vtgpd.loc[i]['title']).split('_')[7], "%Y%m%dT%H%M%S")).strftime("%Y-%m-%d")
                
                try:
                    subprocess.call(['aws', 's3', 'cp', s3vtgpd.loc[i]['title'], 's3://s3vtaustralia/data/'+folderdate+'/'+s3vtgpd.loc[i]['title'], '--recursive'])
                    subprocess.call(['aws', 's3', 'cp', s3vtgpd.loc[i]['title'][:-5]+'.FRP.geojson', 's3://s3vtaustralia/data/'+folderdate+'/'])
                except:
                    logger.info("Upload failed "+s3vtgpd.loc[i]['title'])
                else:
                    s3vtgpd.at[i, 's3bucket'] = 1
                    subprocess.call(['rm', '-rf', s3vtgpd.loc[i]['title']])
                    #subprocess.call(['rm', '-rf', s3vtgpd.loc[i]['title']+'.zip'])
                    logger.info("Deleted "+s3vtgpd.loc[i]['title'])             
            else:
                logger.info(s3vtgpd.loc[i]['title']+' already exists')
            