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

import boto3
from botocore.exceptions import ClientError
from netCDF4 import Dataset
import datetime as dt
import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import xmltodict
import yaml
import argparse
import os
import subprocess
from datetime import date
import logging as logger
import os
import sys

logger.basicConfig(format='%(levelname)s:%(message)s', level=logger.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('--configuration', dest='configuration', default='config.yaml',help='configuration yaml file')
args = parser.parse_args()

from jinja2 import Environment, FileSystemLoader


def get_file_list(username, password, aoi, startrecord):
    # rows returned is limited to 100, add pagination but looking at number of records and incrementing by 100 each iteration
    try:
        subprocess.call(['wget','--no-check-certificate', '--user='+username, '--password='+password, '--output-document=filelist.txt', 'https://131.176.236.38/dhus/search?q=footprint:"Intersects('+aoi+')" AND platformname:Sentinel-3 AND producttype:SL_2_FRP___&rows=100&start='+startrecord+'&format=json'])
        logger.info(str(['wget','--no-check-certificate', '--user='+username, '--password='+password, '--output-document=filelist.txt', 'https://131.176.236.38/dhus/search?q=footprint:"Intersects('+aoi+')" AND platformname:Sentinel-3 AND producttype:SL_2_FRP___&rows=100&start='+startrecord+'&format=json']))
        success = True
    except:
        logger.info("Remote file listing process failed "+str(['wget','--no-check-certificate', '--user='+username, '--password='+password, '--output-document=filelist.txt', 'https://131.176.236.38/dhus/search?q=footprint:"Intersects('+aoi+')" AND platformname:Sentinel-3 AND producttype:SL_2_FRP___&rows=100&start='+startrecord+'&format=json']))
        success = False
        
    return(success)

def get_file(username, password, uuid, zipname):
    # rows returned is limited to 100, add pagination but looking at number of records and incrementing by 100 each iteration
    try:
        subprocess.call(['wget', '--no-check-certificate', '--content-disposition',  '--continue', '--user='+username, '--password='+password, "https://131.176.236.38/dhus/odata/v1/Products('"+uuid+"')/$value"])
        logger.info(str(['wget', '--no-check-certificate', '--content-disposition',  '--continue', '--user='+username, '--password='+password, "https://131.176.236.38/dhus/odata/v1/Products('"+uuid+"')/$value"]))
        success = True
    except:
        logger.info("Remote file retrieval failed "+str(['wget', '--no-check-certificate', '--content-disposition',  '--continue', '--user='+username, '--password='+password, "https://131.176.236.38/dhus/odata/v1/Products('"+uuid+"')/$value"]))
        success = False
        
    try:
        subprocess.call(['unzip', zipname])
        logger.info
        success = True
    except:
        success = False
    return(success)


def get_polygon_from_gml(gml_dict):
    listoftuples = []
    for i in list(gml_dict.split(" ")):
        pair = (float(i.split(',')[1]), float(i.split(',')[0]))
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


def filter(sensors):

    for sensordict in sensors:
        
        filter_string = ''
        count = 0
        
        for sensor in sensordict.keys():
            filter_string = filter_string+'(sensor=%27'+sensor+'%27%20AND%20(product=%27'
            product_count = 0
            for product in sensordict[sensor]:
                filter_string = filter_string+product+'%27'
                if product_count < (len(sensordict[sensor])-1):
                    filter_string = filter_string+'%20OR%20product=%27'
                else:
                    filter_string = filter_string+'))' 
                product_count = product_count + 1
            if count < (len(sensordict.keys())-1):        
                filter_string = filter_string+'%20OR%20'
            count = count+1

    return(filter_string)    
    
    
def load_hotspots(filter_string, time_period, bbox, max_features, min_confidence, to_date):
    y_max = bbox[0]
    x_min = bbox[1]
    y_min = bbox[2]
    x_max = bbox[3]
    if to_date is None:
        
        to_date = dt.datetime.now()
    
    logger.info(str(to_date)+' '+str(type(to_date)))
    from_date = (to_date - dt.timedelta(days=time_period)).strftime('%Y-%m-%d')
    
    # trim datetime to enable WFS 
    to_date = to_date.strftime('%Y-%m-%d')
      
    # TODO - sort out paging - looks like there is a limit to WFS requests number returned per query
    logger.info(f"https://hotspots.dea.ga.gov.au/geoserver/public/wfs?service=WFS&version=1.1.0&request=GetFeature&typeName=public:hotspots&outputFormat=application/json&CQL_FILTER=({filter_string})%20AND%20datetime%20%3E%20%27{from_date}%27%20AND%20datetime%20%3C%20%27{to_date}%27%20AND%20INTERSECTS(location,%20POLYGON(({y_max}%20{x_min},%20{y_max}%20{x_max},%20{y_min}%20{x_max},%20{y_min}%20{x_min},%20{y_max}%20{x_min})))&maxFeatures={max_features}&startIndex=0&sortBy=sensor%20A")
    url = f"https://hotspots.dea.ga.gov.au/geoserver/public/wfs?service=WFS&version=1.1.0&request=GetFeature&typeName=public:hotspots&outputFormat=application/json&CQL_FILTER=({filter_string})%20AND%20datetime%20%3E%20%27{from_date}%27%20AND%20datetime%20%3C%20%27{to_date}%27%20AND%20INTERSECTS(location,%20POLYGON(({y_max}%20{x_min},%20{y_max}%20{x_max},%20{y_min}%20{x_max},%20{y_min}%20{x_min},%20{y_max}%20{x_min})))&maxFeatures={max_features}&startIndex=0&sortBy=sensor%20A"
    
    hotspots_gdf = gpd.read_file(url)
    logger.info(str(hotspots_gdf['stop_dt']))
    
    # TODO - improved None value handling  -currently just look at first and apply that to all
    if hotspots_gdf['confidence'][0] == None:
        logger.info('Skipping confidence filter as confidence not populated')
    else:

        # Filter by confidence
        hotspots_gdf = hotspots_gdf.loc[hotspots_gdf.confidence >= min_confidence]

    # Fix datetime
    if hotspots_gdf['start_dt'][0] == None:
        logger.info('Start date field is not populated')
        hotspots_gdf['datetime'] = pd.to_datetime(hotspots_gdf['datetime'])
    else:
        hotspots_gdf['datetime'] = pd.to_datetime(hotspots_gdf['start_dt'])

    # Extract required columns
    hotspots_gdf = hotspots_gdf.loc[:, [
            'datetime', 'latitude', 'longitude', 'confidence', 'geometry'
            ]]
    hotspots_gdf.sort_values('datetime', ascending=True, inplace=True)
    logger.info('Hotspots loaded successfully '+str(hotspots_gdf.geometry.total_bounds))

    return(hotspots_gdf)


if __name__ == '__main__':
    file_loader = FileSystemLoader("templates")
    env = Environment(loader=file_loader)

    # Get configurations
    satellites = []
    with open(args.configuration, 'r') as config:
        cfg = yaml.load(config, Loader=yaml.Loader)

    for configuration in cfg['configurations']:
        
        logger.info(configuration['username']+' '+configuration['password']+' '+configuration['url'])

        # Get File List from server
        # Run this if no local inventory exists
        # Remote s3vt_inventory to refresh listing
        
        if not os.path.exists('s3vt_inventory.json'):
            startrecord = 0
        
            responselist = [] 
        
            # Determine number of records to retrieve
            get_file_list(configuration['username'], configuration['password'], configuration['aoi'], str(startrecord))
            with open('filelist.txt') as results:
                for i in results: 
                    response = json.loads(i)
                    responselist.append(response)
        
            upperlimit = int(response['feed']['opensearch:totalResults'])
            
            ##### Below for testing only #####
            #upperlimit = 200
            ##################################
            
            # Get the full list of records
        
            while startrecord <= upperlimit:
                startrecord = startrecord+100
        
                get_file_list(configuration['username'], configuration['password'], configuration['aoi'], str(startrecord)) 
                with open('filelist.txt') as results:
                    for i in results: responselist.append(json.loads(i))
        
                    # Dump the results to an inventory file
                    with open('s3vt_inventory.json', 'w') as f:
                        json.dump(responselist, f)

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

        # Read inventory to geopandas - write to geojson       
                
        with open('s3vt_inventory.json') as inventory:
            frames = []
            for p in inventory:
                pages = json.loads(p)
                        
                for page in pages:
                    for entry in page['feed']['entry']:
                                
                        df = pd.DataFrame.from_dict(entry, orient='index')
                                
                        polygon = get_polygon_from_gml(xmltodict.parse(entry['str'][2]['content'])['gml:Polygon']['gml:outerBoundaryIs']['gml:LinearRing']['gml:coordinates'])
                        
                        df = df.transpose()
                        df['Coordinates'] = Polygon(polygon)
                        for d in entry['str']:
                            if d['name'] ==  'orbitdirection':
                                df['orbitdirection'] = d['content']
                            if d['name'] ==  'platformidentifier':
                                df['platformidentifier'] = d['content'] 
                            if d['name'] ==  'filename':
                                df['filename'] = d['content']
                            if d['name'] ==  'instrumentshortname':
                                df['instrumentshortname'] = d['content']
                            if d['name'] ==  'passnumber':
                                df['passnumber'] = d['content']        
                        s3vtdf = gpd.GeoDataFrame(df, geometry='Coordinates')
                        
                        frames.append(s3vtdf) 
                            
        s3vtgpd = pd.concat(frames)
        
        # Not sure why we need to index but do it anyway
        s3vtgpd = s3vtgpd.reset_index(drop=True)
        s3vtgpd['date'] = pd.to_datetime(s3vtgpd.summary.str.split(",", expand= True)[0].str.split(' ', expand=True)[1])
        # Some fields are lists and geojson translation doesn't like it
        
        s3vtgpd = s3vtgpd.drop(['link', 'int', 'str', 'summary'], axis=1)
        s3vtgpd.to_file('s3vt_geometry.geojson', driver='GeoJSON')
        
        dataframelength = len(s3vtgpd)
        # Add fields to enable monitoring
        s3vtgpd['hotspot'] = 0
        s3vtgpd['download'] = 0
        s3vtgpd['s3bucket'] = 0
        
        #s3vthostpotsgpdlist = []
        
        
        # Check if folder already downloaded and flag in gpd
        for i in range(dataframelength):
            if s3vtgpd.loc[i]['title']+'.SEN3' in set(s3folderlist):
                s3vtgpd.at[i, 'download'] = 1
            if s3vtgpd.loc[i]['title']+'.FRP.geojson' in set(s3folderlist):
                s3vtgpd.at[i, 'hotspot'] = 1
                #s3vthostpotsgpdlist.append(s3hotspotsgpd)
                
        # Uncomment this when running all #        
        for i in range(dataframelength):
        
            #for i in range(1):
            if s3vtgpd.loc[i]['download'] == 0:
                zipname = s3vtgpd.loc[i]['title']+'.zip'
                uuid = s3vtgpd.loc[i]['id']
                filename = s3vtgpd.loc[i]['title']+'.SEN3/FRP_in.nc'
                s3hotspots = s3vtgpd.loc[i]['title']+'.FRP.geojson'
                
        
                if get_file(configuration['username'], configuration['password'], uuid, zipname) == False:
                    s3vtgpd.at[i, 'download'] = 0
                    break
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
                folderdate = s3vtgpd.loc[i]['date'].strftime("%Y-%m-%d")
                
                try:
                    subprocess.call(['aws', 's3', 'cp', s3vtgpd.loc[i]['title']+'.SEN3/', 's3://s3vtaustralia/data/'+folderdate+'/'+s3vtgpd.loc[i]['title']+'.SEN3/', '--recursive'])
                    subprocess.call(['aws', 's3', 'cp', s3vtgpd.loc[i]['title']+'.FRP.geojson', 's3://s3vtaustralia/data/'+folderdate+'/'])
                except:
                    logger.info("Upload failed "+s3vtgpd.loc[i]['title'])
                else:
                    s3vtgpd.at[i, 's3bucket'] = 1
                    subprocess.call(['rm', '-rf', s3vtgpd.loc[i]['title']+'.SEN3'])
                    subprocess.call(['rm', '-rf', s3vtgpd.loc[i]['title']+'.zip'])
                    logger.info("Deleted "+s3vtgpd.loc[i]['title'])             
            else:
                logger.info(s3vtgpd.loc[i]['title']+'.zip already exists')
        

                
        '''
        hotspots_gdf = load_hotspots(filter(configuration['sensors']),
                                     configuration['time_period'],
                                     configuration['bbox'],
                                     configuration['max_features'], 
                                     configuration['min_confidence'],
                                     configuration['to_date'])
        '''