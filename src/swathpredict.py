#!/usr/bin/env python
# coding: utf-8
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
import git
parser = argparse.ArgumentParser()
import time
from datetime import timedelta, datetime, date
from dateutil import tz
from jinja2 import Environment, FileSystemLoader
import yaml
import json
import pandas
import math
import requests
from space_track_api import SpaceTrackApi
import ephem
from pyorbital.orbital import Orbital
from pyorbital import tlefile
from geographiclib.geodesic import Geodesic
import osgeo.ogr
import osgeo.osr
from osgeo import ogr
import simplekml
import folium
from folium import plugins
from colour import Color

parser.add_argument('--configuration', dest='configuration', default='config.yaml',help='ground station configuration')
parser.add_argument('--start', dest='start', help='start time YYYY-MM-DDTHH:MM:SSZ format')
parser.add_argument('--period', dest='period', help='ground station configuration')
parser.add_argument('--output_path', dest='output_path', help='ground station configuration')
#parser.add_argument('--space_track_user', dest='space_track_user', help='SpaceTrack.org username')
#parser.add_argument('--space_track_pass', dest='space_track_pass', help='SpaceTrack.org password')

args = parser.parse_args()

def tleiscurrent(tlefile, satname, targetdate, acceptabletimedelta=None):
    """
    Return TLE JSON from SpaceTrack.org, delta from start time to TLE epoch, and True if TLE within 30 days
    """
    # scenarios
    # time precedes launch date
    # time delta exceeds threshold
    
    # tlefile is json format retrieved by updatetlefile() written using pandas to_json method
    # satname is NORAD compatible satellite name eg. TERRA
    # targetdate is pd.timestamp requested swathpredict date - could be a date range??
    # acceptabletimedelta is timedelta64
    
    if acceptabletimedelta is None:
        acceptabletimedelta = timedelta(days = 30)
        
    df = pandas.read_json('tle.json')
    df['EPOCH'] = pandas.to_datetime(df['EPOCH'])
    
    #dt = pd.Timestamp(str(targetdate))
    
    df = df[df['TLE_LINE0'].str.contains(satellite)]
    tle = df.iloc[(abs(df.EPOCH-targetdate)).argsort()[:1]]
    delta = tle['EPOCH'] - targetdate
    result = delta < acceptabletimedelta
    
    return(tle, delta, result)
    
def updatetlefile(norad_sat_ids,user, password, mindatetime=None, maxdatetime=None ):
    """
    Return updated TLE as JSON
    """
    # norad_sat_ids as tuple of norad identifiers (12134,12121,23243,)
    # mindatetime as pandas timestamp
    # maxdatetime as pandas timestamp
    
    if mindatetime is None:
        d = date(1970, 1, 1)
        mindatetime = pandas.Timestamp(str(d))
    if maxdatetime is None:
        maxdatetime = pandas.Timestamp(datetime.now())       
    
    with SpaceTrackApi(login=user, password=password) as api:

        #convert mindatetime and maxdatetime to YYYY-MM-DD format
        mindatetime = mindatetime.strftime("%Y-%m-%d")
        maxdatetime = maxdatetime.strftime("%Y-%m-%d")
        tle_list = api.tle(EPOCH=mindatetime+"--"+maxdatetime,
                           NORAD_CAT_ID=norad_sat_ids,
                           order_by=('EPOCH desc', 'NORAD_CAT_ID',),
                           predicate=('EPOCH', 'NORAD_CAT_ID', 'TLE_LINE0', 'TLE_LINE1', 'TLE_LINE2',))
        df = pandas.DataFrame(tle_list)
        df.to_json('tle.json')
    
    return(df)
    
def getredtoblack(number):
    """
    Return a list of colours for a give number of samples
    """
    if number < 256:
        numbercolours = 256
    else:
        numbercolours = number
        
    rangevalue = int((numbercolours)/4)
    
    if number == 1:
        return(['red'])
    if number == 2:
        return(['red', 'black'])
    
    red = Color("red")
    orange = Color("orange")
    yellow = Color("yellow")
    white = Color("white")
    black = Color("black")
    
    redorange = tuple(red.range_to(orange, rangevalue))
    orangeyellow = tuple(orange.range_to(yellow, rangevalue+1))
    yellowwhite = tuple(yellow.range_to(white,rangevalue+1))
    whiteblack = tuple(white.range_to(black,rangevalue+1))
    redtoblack = redorange + orangeyellow[1:] + yellowwhite[1:] + whiteblack[1:]
    
    redtoblacklist = list(redtoblack)
    colours = []
    position = 0
    increment = int(len(redtoblacklist)/(number-1))
    
    while (position < len(redtoblacklist)):

        colours.append(redtoblacklist[position])
        position = position + increment
    colours[len(colours)-1] = black   
    return(colours)


def local_time(utc, local):
    """Return a local time representation of UTC"""
    to_zone = tz.gettz(local)
    from_zone = tz.tzutc()

    # Set UTC datetime object to UTC
    utc = utc.replace(tzinfo=from_zone)
    # Convert time zone
    return utc.astimezone(to_zone)


def download_file(url):
    """Return local filename from input URL"""
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    return local_filename


def get_tles():
    """ Return a list of tuples of kepler parameters for each satellite"""

    tle_file_basenames = []

    for url in tle_files:
        tle_file_basename = os.path.basename(url)
        tle_file_basenames.append(tle_file_basename)
        try:
            os.remove(tle_file_basename)
        except OSError:
            pass
        try:
            download_file(url)
            logging.info("Downloading URL from configuration sources:" + tle_file_basename)
        except OSError:
            logging.info("Process could not download file:" + tle_file_basename)
            return ()

    with open('tles.txt', 'w') as outfile:
        for fname in tle_file_basenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)


    return clean_combine_tles()


def clean_combine_tles():
    tles = open('tles.txt', 'r').readlines()

    logging.info("Combining TLE from configuration sources")
    # strip off the header tokens and newlines
    tles = [item.strip() for item in tles]

    # clean up the lines
    tles = [(tles[i], tles[i + 1], tles[i + 2]) for i in range(0, len(tles) - 2, 3)]
    return(tles)


def check_tle_is_current():
    """Returns whether the TLE file was retrieved less than 24hrs ago and initiates download"""
    # Check if TLE exists and get it if not
    if os.path.exists('tles.txt'):
        tle_retrieve_time = datetime.fromtimestamp(os.path.getmtime('tles.txt'))
    else:
        get_tles()
        logging.info('tle.txt does not exist')
        tle_retrieve_time = datetime.utcnow()

    # check for stale tle and update if required
    tle_timesinceretrieve = datetime.now() - tle_retrieve_time

    # Compare TLE age against daily update requirement
    if tle_timesinceretrieve > timedelta(hours=24):
        tle_retrieve_time = datetime.utcnow()
        get_tles()
        logging.info("Stale TLE detected and replaced at" + str(tle_retrieve_time))
    else:
        logging.info("TLE file current and does not require update")
    logging.info("HH:MM:SS since TLE retrieval:"+ str(tle_timesinceretrieve))
    return clean_combine_tles()

def update_points_crossing_antimeridian(listofpoints):
    """Return list of points which account for crossing of antimeridian"""
    if len(listofpoints) == 0:
        return(listofpoints)
    antimeridianlistofpoints = []
    diff = 325.0
    referencepointlon = listofpoints[0]['lon2']
    
    crossingeasttowest = False
    for point in listofpoints:
        # Confirm no crossing of antimeridian between points
        if (abs(referencepointlon - point['lon2']) <= diff):
            antimeridianlistofpoints.append(point)
            referencepointlon = listofpoints[0]['lon2']
        else:
            # if crossing antimeridian west to east add 360 i.e. diff will be negative
            if ((referencepointlon - point['lon2']) >= diff):
                point['lon2'] = point['lon2']+360
                antimeridianlistofpoints.append(point)
                referencepointlon = point['lon2'] 
            # if crossing antimeridian east to west minus 360 i.e. diff will be negative
            if ((referencepointlon - point['lon2']) <= (diff*-1)):        
                point['lon2'] = point['lon2']-360
                antimeridianlistofpoints.append(point)
                referencepointlon = point['lon2'] 
                # Crossing east to west
                crossingeasttowest = True
    if crossingeasttowest == True:
        for point in antimeridianlistofpoints:
            point['lon2'] = point['lon2']+360
        
    return(antimeridianlistofpoints)


def get_vector_file(attributes, input_points, poly_or_line, ogr_output, ogr_format):
    """ Returns spatial layer built on inputs - attributes, points, polygon or line, output in specified format"""

    input_points = update_points_crossing_antimeridian(input_points)
    spatialReference = osgeo.osr.SpatialReference()
    spatialReference.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')

    # if no points passed for ogr build return
    if len(input_points) == 0:
        return ()
    try:
        os.remove(ogr_output)
    except OSError:
        pass
    ogr.UseExceptions()

    driver = ogr.GetDriverByName(ogr_format)
    
    if os.path.exists(ogr_output):
        driver.DeleteDataSource(ogr_output)
    ds = driver.CreateDataSource(ogr_output)


    if poly_or_line == 'polygon':
        geomtype = ogr.wkbPolygon
    if poly_or_line == 'line':
        geomtype = ogr.wkbLineString
    if poly_or_line == 'point':
        geomtype = ogr.wkbPoint

    if ds is None:
        logging.info("Process could not create file")
        sys.exit(1)
    layer = ds.CreateLayer(attributes['Satellite name'], geom_type=geomtype)

    field_definition= ogr.FieldDefn('Satellite               :', ogr.OFTString)
    field_definition.SetWidth(30)
    layer.CreateField(field_definition)
    field_definition = ogr.FieldDefn('Sensor               :', ogr.OFTString)
    field_definition.SetWidth(30)
    layer.CreateField(field_definition)
    field_definition = ogr.FieldDefn('Orbit height                 :', ogr.OFTString)
    field_definition.SetWidth(30)
    layer.CreateField(field_definition)
    layer.CreateField(ogr.FieldDefn('Orbit number                 :', ogr.OFTInteger))
    '''
    field_definition = ogr.FieldDefn('Current UTC time             :', ogr.OFTString)
    field_definition.SetWidth(30)
    layer.CreateField(field_definition)
    field_definition = ogr.FieldDefn('Minutes to horizon           :', ogr.OFTString)
    field_definition.SetWidth(30)
    layer.CreateField(field_definition)
    '''
    field_definition = ogr.FieldDefn('Acquisition of Signal Local    :', ogr.OFTString)
    field_definition.SetWidth(30)
    layer.CreateField(field_definition)
    field_definition = ogr.FieldDefn('Acquisition of Signal UTC    :', ogr.OFTString)
    field_definition.SetWidth(30)
    layer.CreateField(field_definition)
    field_definition = ogr.FieldDefn('Loss of Signal UTC           :', ogr.OFTString)
    field_definition.SetWidth(30)
    layer.CreateField(field_definition)
    field_definition = ogr.FieldDefn('Transit time                 :', ogr.OFTString)
    field_definition.SetWidth(30)
    layer.CreateField(field_definition)
    field_definition = ogr.FieldDefn('Node                         :', ogr.OFTString)
    field_definition.SetWidth(30)
    layer.CreateField(field_definition)
    feature_definition = layer.GetLayerDefn()
    feature = ogr.Feature(feature_definition)
    feature.SetField('Satellite               :', attributes['Satellite name'])
    feature.SetField('Sensor               :', attributes['Sensor code'])
    feature.SetField('Orbit height                 :', attributes['Orbit height'])
    feature.SetField('Orbit number                 :', attributes['Orbit'])
    '''
    feature.SetField('Current UTC time             :', str(attributes['Current time']))
    feature.SetField('Minutes to horizon           :', attributes['Minutes to horizon'])
    '''
    feature.SetField('Acquisition of Signal Local    :', attributes['Local time'])
    feature.SetField('Acquisition of Signal UTC    :', str(attributes['AOS time']))
    feature.SetField('Loss of Signal UTC           :', str(attributes['LOS time']))
    feature.SetField('Transit time                 :', str(attributes['Transit time']))
    feature.SetField('Node                         :', attributes['Node'])


    
    if poly_or_line == 'point':
        point = ogr.Geometry(ogr.wkbPoint)
        for x in input_points:
            point.AddPoint(x['lon2'], x['lat2'], x['alt2'])

        feature.SetGeometry(point)
        layer.CreateFeature(feature)

        point.Destroy()
    
    
    
    if poly_or_line == 'line':
        line = ogr.Geometry(type=ogr.wkbLineString)
        for x in input_points:
      
            
           
            line.AddPoint(x['lon2'], x['lat2'], x['alt2'])

        feature.SetGeometry(line)
        layer.CreateFeature(feature)

        line.Destroy()

    if poly_or_line == 'polygon':
        
        ring = ogr.Geometry(ogr.wkbLinearRing)
        
        #input_points = update_points_crossing_antimeridian(input_points, ogr_format, 'antimeridian.geojson')
        for x in input_points:
   
            ring.AddPoint(x['lon2'], x['lat2'])
                        
        poly = ogr.Geometry(ogr.wkbPolygon)
        ring.color = "red"
        
        poly.AddGeometry(ring)
        
        
        feature.SetGeometry(poly)

        layer.CreateFeature(feature)

        ring.Destroy()
        poly.Destroy()

    feature.Destroy()
   
    ds.Destroy()
    # for KML - Add altitude to GeoJSON if ogr_format=="GeoJSON" and change colour of track to yellow
    if ogr_format == "GeoJSON":
        if poly_or_line == 'line':
            replace_string_in_file(ogr_output, '<LineString>', '<LineString><altitudeMode>absolute</altitudeMode>')
            replace_string_in_file(ogr_output, 'ff0000ff', 'ffffffff')
        if poly_or_line == 'point':
            replace_string_in_file(ogr_output, '<Point>', '<Point><altitudeMode>absolute</altitudeMode>')
        if poly_or_line == 'polygon':
            replace_string_in_file(ogr_output, '<PolyStyle><fill>0</fill>',
                                   '<PolyStyle><color>7f0000ff</color><fill>1</fill>')

    return ()


def replace_string_in_file(infile, text_to_find, text_to_insert):
    in_file = open(infile, 'r')
    temporary = open(os.path.join(output_path, 'tmp.txt'), 'w')
    for line in in_file:
        temporary.write(line.replace(text_to_find, text_to_insert))
    in_file.close()
    temporary.close()
    os.remove(infile)
    shutil.move(os.path.join(output_path, 'tmp.txt'), infile)
    return ()


def get_effective_heading(satellite, oi_deg, latitude, longitude, tle_orbit_radius, daily_revolutions):
    """Returns the effective heading of the satellite"""
    lat_rad = math.radians(latitude)  # Latitude in radians
    oi_rad = math.radians(oi_deg)  # Orbital Inclination (OI) [radians]
    orbit_radius = tle_orbit_radius * 1000.0  # Orbit Radius (R) [m]
    # np = 5925.816                   # Nodal Period [sec] = 5925.816
    np = (24 * 60 * 60) / daily_revolutions
    av = 2 * math.pi / np  # Angular Velocity (V0) [rad/sec] =	 0.001060307189285 =2*PI()/E8
    #sr = 0  # Sensor Roll (r) [degrees] =	0

    # TODO put earth parameters into a dict and add support for other spheroids GRS1980 etc.
    # Earth Stuff (WGS84)
    one_on_f = 298.257223563  # Inverse flattening 1/f = 298.257223563
    #f = 1 / one_on_f  # flattening
    r = 6378137  # Radius (a) [m] =	 6378137
    e = 1 - math.pow((1 - 1 / one_on_f), 2)  # Eccentricity (e^2) = 0.00669438 =1-(1-1/I5)^2
    wO = 0.000072722052  # rotation (w0) [rad/sec] = 7.2722052E-05

    xfac = math.sqrt(1 - e * (2 - e) * (math.pow(math.sin(math.radians(latitude)), 2)))
    phi_rad = math.asin((1 - e) * math.sin(math.radians(latitude)) / xfac)  # Phi0' (Geocentric latitude)
    # phi_deg = math.degrees(phi_rad)  # Phi0' (Degrees)
    n = r / math.sqrt(1 - e * (math.pow(math.sin(math.radians(latitude)), 2)))  # N
    altphi_rad = latitude - 180 * math.asin(
        n * e * math.sin(lat_rad) * math.cos(lat_rad) / orbit_radius) / math.pi  # Alt Phi0'(Radians)
    rho_rad = math.acos(math.sin(altphi_rad * math.pi / 180) / math.sin(oi_rad))  # Rho (Radians)
    beta = -1 * (math.atan(1 / (math.tan(oi_rad) * math.sin(rho_rad))) * 180 / math.pi)  # Heading Beta (degrees)
    #xn = n * xfac  # Xn
    #altitude = (orbit_radius - xn) / 1000  # altitude
    #altitude_ = (orbit_radius * math.cos(altphi_rad / 180 * math.pi) / math.cos(lat_rad) - n) / 1000
    rotation = math.atan((wO * math.cos(phi_rad) * math.cos(beta * math.pi / 180)) / (
            av + wO * math.cos(phi_rad) * math.sin(beta * math.pi / 180))) * 180 / math.pi
    eh = beta + rotation
    alpha12 = eh
    #s = 0.5 * 185000  # s = distance in metres
    effective_heading = alpha12
    return effective_heading

def folium_timespan_geojson_html(schedule, satname):
    timespan_map = folium.Map(location=[-26, 132], tiles='OpenStreetMap',zoom_start=3)
    #lines=[]
    polygons = []




    lines=[]
    polygons = []
    #polygontuples = ()
    
    colorindex = 0
    
    colorlut = getredtoblack(len(schedule)+1)

    for i in schedule:
        
        pointlist = []
        polygonlist = []
        #timeslist = []
        datelist = []
        for x in i['Orbit line']: 
            pointlist.append([x['lon2'], x['lat2']])
            datelist.append(str(x['time']).replace(" ","T"))
            
            # folium expects a time for each point - could use iterate for len of points and time period to get times per point - or add to dict from original function which uses time
        lines.append({'coordinates': pointlist, 'dates': datelist, 'color': str(colorlut[colorindex]),'weight': 2})
        
        datelist = []  
            
        for x in i['Swath polygon']:
            #polygonlist.append([x['lat2'],x['lon2']])
            pointtuple = (x['lon2'],x['lat2'])
            polygonlist.append(pointtuple)
            
            datelist.append(str(x['time']).replace(" ","T"))
        
        polygons.append({'coordinates': [(tuple(polygonlist),)], 'dates': datelist, 'color': str(colorlut[colorindex]),'weight': 2})
        colorindex = colorindex +1
    
    
    features = [
        {
            'type': 'Feature',
            'geometry': {
                'type': 'MultiPolygon',
                'coordinates': polygon['coordinates'],            
            },
            'properties': {
                'times': polygon['dates'],
                'style': {
                    'color': polygon['color'],
                    'opacity': 0.1,
                    'weight': polygon['weight'] if 'weight' in polygon else 5
                    
                }
            }
        }
        
        #for line in lines
        for polygon in polygons 
    ]
    
    
    featureslines = [
        {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': line['coordinates'],
            },
            'properties': {
                'times': line['dates'],
                'style': {
                    'color': line['color'],
                    'dash-array': '[4]',
                    'weight': line['weight'] if 'weight' in line else 5
                
                }
            }
        }
        
        for line in lines
        #for polygon in polygons 
    ]
    
    
    
    for featureline in featureslines:
        features.append(featureline)
    
    
    plugins.TimestampedGeoJson({
        'type': 'FeatureCollection',
        'features': features,
    }, period='PT1M', duration=None,add_last_point=False, auto_play=True, transition_time=1, time_slider_drag_update=True).add_to(timespan_map)
    
    plugins.Fullscreen(
        position='topright',
        title='Expand me',
        title_cancel='Exit me',
        force_separate_button=True
    ).add_to(timespan_map)
    
    timespan_map




    '''

    colorindex = 0
    for i in schedule:
        
        pointlist = []
        polygonlist = []
        #timeslist = []
        datelist = []
        for x in i['Orbit line']: 
            pointlist.append([x['lon2'], x['lat2']])
            datelist.append(str(x['time']).replace(" ","T"))
            
            # folium expects a time for each point - could use iterate for len of points and time period to get times per point - or add to dict from original function which uses time
        polygons.append({'coordinates': pointlist, 'dates': datelist, 'color': colorlut[colorindex],'weight': 2})
        colorindex = colorindex +1
        datelist = []
    
        for x in i['Swath polygon']:
            #polygonlist.append([x['lat2'],x['lon2']])
            pointtuple = (x['lon2'],x['lat2'])
            polygonlist.append(pointtuple)
            
            datelist.append(str(x['time']).replace(" ","T"))
        
        polygons.append({'coordinates': [(tuple(polygonlist),)], 'dates': datelist, 'color': colorlut[colorindex],'weight': 2})
        #colorindex = colorindex +1
    #print(polygonlist)  
    
    features = [
        {
            'type': 'Feature',
            'geometry': {
                #'type': 'LineString',
                #'coordinates': line['coordinates'],
                'type': 'MultiPolygon',
                'coordinates': polygon['coordinates'],            
            },
            'properties': {
                #'times': line['dates'],
                #'style': {
                #    'color': line['color'],
                #    'weight': line['weight'] if 'weight' in line else 5
                'times': polygon['dates'],
                'style': {
                     'color': polygon['color'],
                     'weight': polygon['weight'] if 'weight' in polygon else 5           
                
                
                
                }
            }
        }
        #for line in lines
        for polygon in polygons
    ]
    
    #print(features)
    
    #plugins.TimestampedGeoJson({
    #    'type': 'FeatureCollection',
    #    'features': features,
    #}, period='PT1M', duration=None,add_last_point=False, auto_play=True, transition_time=1).add_to(timespan_map)
 
    style_function = lambda x: {'fillColor': '#00ffff'}
    gj = folium.GeoJson({
        'type': 'FeatureCollection',
        'features': features,
    }, style_function=style_function)

    #gj.add_child(folium.GeoJsonTooltip(fields=["Satellite               :", "Sensor               :", \
    #                                           "Orbit height                 :", "Orbit number                 :", \
    #                                           "Acquisition of Signal Local    :", "Acquisition of Signal UTC    :", \
    #                                           "Loss of Signal UTC           :", "Transit time                 :", \
    #                                           "Node                         :"]))
    
    gj.add_to(timespan_map)    
    #timespan_map.add_child(folium.GeoJsonTooltip(fields=["Satellite               :", "Sensor               :", \
    #                                           "Orbit height                 :", "Orbit number                 :", \
    #                                           "Acquisition of Signal Local    :", "Acquisition of Signal UTC    :", \
    #                                           "Loss of Signal UTC           :", "Transit time                 :", \
    #                                           "Node                         :"]))
    
    plugins.Fullscreen(
        position='topright',
        title='Expand me',
        title_cancel='Exit me',
        force_separate_button=True
    ).add_to(timespan_map)
    '''
    foliumtimespanhtml = os.path.join(output_path, satname + "." + ground_station_name + ".timespan.map.html")
    timespan_map.save(foliumtimespanhtml)


def add_layer_to_map(swathfile, layername, color):
    """Adds the input layer to the folium map"""
    if not os.path.isfile(swathfile):
        return ()

    geojsonlayer = json.loads(open(swathfile).read())
    df = pandas.DataFrame(geojsonlayer)
    ds = pandas.Series(df.features[0]['properties'])
    df = ds.to_frame()
    gj = folium.GeoJson(geojsonlayer, name=layername)

    gj.add_child(folium.GeoJsonTooltip(fields=["Satellite               :", "Sensor               :", \
                                               "Orbit height                 :", "Orbit number                 :", \
                                               "Acquisition of Signal Local    :", "Acquisition of Signal UTC    :", \
                                               "Loss of Signal UTC           :", "Transit time                 :", \
                                               "Node                         :"]))
    gj.add_to(satellite_map)


def to_datetime(ephemtime):
    """Return datetime object from input ephem time"""
    try:
        stringtime = (datetime.strptime(str(ephemtime), "%Y/%m/%d %H:%M:%S"))
    
    except ValueError:
        stringtime = (datetime.strptime(str(ephemtime), "%Y-%m-%d %H:%M:%S"))
    
    return (stringtime)


def get_satellite_name(tle):
    """Return input string with spaces and dashes replaced with underscore"""
    satname = str(tle[0]).replace(" ", "_")
    satname = satname.replace("-", "_")
    return (satname)


def get_orbit_node(aos_lat, los_lat, oi):
    """Return the orbit node - ascending or descending, based on time of AOS and LOS latitude """
    if (aos_lat > los_lat):
        print("PASS                 = descending")
        node = "descending"
    else:
        print("PASS                 = ascending")
        node = "ascending"
        oi = 360 - oi
    return (oi, node)


def get_subsat_oneincrement(sat, deltatime, timestep):
    """Return to sub satellite latitude and longitude of the next satellite position"""    
    deltatime = deltatime + timestep
    sat.compute(deltatime)
    subsatlat1 = sat.sublat.real * (180 / math.pi)
    subsatlon1 = sat.sublong.real * (180 / math.pi)
    return(subsatlat1, subsatlon1)


def get_upcoming_passes(satellite_name, passes_begin_time, passes_period):
    """Returns potential satellite pass information for input satellite and Two Line Elements over temporal period"""
    kml = simplekml.Kml()
    observer = ephem.Observer()
    observer.lat = ground_station[0]
    observer.long = ground_station[1]
    observer.horizon = observer_horizon
    swathincrement = 1
    # make a list to hold dicts of attributes for the upcoming pass information for the selected satellites
    schedule = []
    observer.date = passes_begin_time

    print("---------------------------------------")

    
    '''
    for tle in tles:
        
        if tle[0] == satellite_name:

            # Update satellite names for use in filename
            satname = get_satellite_name(tle)
            sat = ephem.readtle(tle[0], tle[1], tle[2])

            # Report currency of TLE
            twole = tlefile.read(tle[0], 'tles.txt')
            now = datetime.utcnow()
            timesinceepoch = now - twole.epoch.astype(datetime)

            print("TLE EPOCH:", twole.epoch.astype(datetime))
            print("TLE age:", timesinceepoch)
            print("---------------------------------------")

            # While lostime of next pass is less than or equal to pass_begin_time + period do something
    
    Indented below 2 tabs
    
    '''
    lostime = start  # passes_begin_time + passes_period
    
    
    while lostime <= (passes_begin_time + timedelta(minutes=passes_period)):
        
        # get TLE for satellite
        
        df = tledf[tledf['TLE_LINE0'].str.contains(satellite_name)]

        lostle = df.iloc[(abs(df.EPOCH-pandas.Timestamp(str(lostime)))).argsort()[:1]]
        tle = {}
        tle[0] = lostle['TLE_LINE0'].values[0][2:]
        tle[1] = lostle['TLE_LINE1'].values[0]
        tle[2] = lostle['TLE_LINE2'].values[0]
        
       
        with open('tles.txt', 'w') as f:
            for i in tle:
                f.write(tle[i])
                f.write('\n')
        
        
        satname = get_satellite_name(tle)
        sat = ephem.readtle(tle[0], tle[1], tle[2])

        # Report currency of TLE
        twole = tlefile.read(tle[0], 'tles.txt')
        now = datetime.utcnow()
        timesinceepoch = now - twole.epoch.astype(datetime)

        print("TLE EPOCH:", twole.epoch.astype(datetime))
        print("TLE age:", timesinceepoch)
        print("---------------------------------------")







        oi = float(tle[2][9:16])      
        
        orb = Orbital(tle[0], "tles.txt", tle[1], tle[2])

        rt, ra, tt, ta, st, sa = observer.next_pass(sat)

        # Confirm that observer details have been computed i.e. are not 'Null'

        if rt is None:
            print(rt + "is none")
            logging.info("Rise time of satellite not calculated - pass currently under way")
            observer.date = (lostime + timedelta(minutes=90))

            return ()
        sat.compute(rt)
        aos_lat = sat.sublat.real * (180 / math.pi)

        sat.compute(st)
        los_lat = sat.sublat.real * (180 / math.pi)

        # Determine if pass descending or ascending
        oi, node = get_orbit_node(aos_lat, los_lat, oi)

        aostime = datetime.strptime(str(rt), "%Y/%m/%d %H:%M:%S")


        loctime = local_time(aostime, 'Australia/Sydney')
        minutesaway = ((aostime - start).seconds / 60.0) + ((aostime - start).days * 1440.0)
        orad = orb.get_lonlatalt(to_datetime(rt))[2]
        orbitnumber = orb.get_orbit_number(to_datetime(rt))+orbitoffset

        # Get code version for provenance - TODO embed this in HTML
        try:
            repo = git.Repo(os.getcwd())
            tag = repo.tags[len(repo.tags)-1].tag.tag # TODO work out if git repo, if not skip TRY
        except:
            tag = '0.0.0'
        print("code tag             = ", tag)
        print("------------------------------------------")
        print("Satellite            = ", satname)
        print("Orbit                = ", orbitnumber)
        print("Minutes to horizon   = ", minutesaway)
        print("AOStime local        = ", loctime)
        print("AOStime UTC          = ", to_datetime(rt))
        print("LOStime UTC          = ", to_datetime(st))
        print("Transit time UTC     = ", to_datetime(tt))

        SWATH_FILENAME = os.path.join(output_path, satname + "." + str(
            orbitnumber) + "." + ground_station_name + ".orbit_swath.geojson")
        ORBIT_FILENAME = os.path.join(output_path, satname + "." + str(
            orbitnumber) + "." + ground_station_name + ".orbit_track.geojson")



        # Step from AOS to LOS by configuration timestep interval
        deltatime = to_datetime(rt)

        geoeastpoint = []
        geowestpoint = []
        geotrack = []
        # TODO - 1/10 steps out to east and west limbs of pass from track
        startpoint = True

        # TODO - confirm nextpass method works when satellite within horizon
        subsatlat1 = None
        while deltatime < to_datetime(st):
            #if subsatlat1 is None:
            #    subsatlat1 = sat.sublat.real * (180 / math.pi)
            #    subsatlon1 = sat.sublong.real * (180 / math.pi)
            #    print("got to here")
            
            sat.compute(deltatime)
            
            subsatlat = sat.sublat.real * (180 / math.pi)
            subsatlon = sat.sublong.real * (180 / math.pi)
            
            orbaltitude = orb.get_lonlatalt(to_datetime(rt))[2] * 1000

            geotrack.append(
                {'lat2': subsatlat, 'lon2': subsatlon,
                 'alt2': orbaltitude, 'time': str(deltatime)})
            
            # Original heading calculation
            #effectiveheading = get_effective_heading(sat, oi, subsatlat,
            #                                       subsatlon, orad, sat._n)

            # TODO Alternate simple heading
            subsatlat1, subsatlon1 = get_subsat_oneincrement(sat, deltatime, timestep)
            
            effectiveheading = Geodesic.WGS84.Inverse(subsatlat1, subsatlon1, subsatlat, subsatlon)['azi1']

            eastaz = effectiveheading + 90
            westaz = effectiveheading + 270

            # 1/10 swath steps out to east and west limbs of pass from track start and end
            #          <-- 1| 2 --> reverse
            #   |   .................   |
            #   V   .       .       .   V  reverse
            #       .       .       .
            #       .................
            #          <-- 3| 4 -->
            #  (reverse 3 append to 1) append (append 4 to reversed 2)



            incrementtoswath = swathincrement
            if startpoint is True:
                # Step from sub satellite point to limb of swath with increasing step size
                while math.pow(incrementtoswath, 5) < swath:
                    
                    geoeastpointdict = Geodesic.WGS84.Direct(subsatlat, subsatlon, eastaz, math.pow(incrementtoswath, 5))
                    geoeastpointdict['time']=str(deltatime)
                    geoeastpoint.append(geoeastpointdict)
                    
                    geowestpointdict = Geodesic.WGS84.Direct(subsatlat, subsatlon, westaz, math.pow(incrementtoswath, 5))
                    geowestpointdict['time']=str(deltatime)
                    geowestpoint.append(geowestpointdict)
                    incrementtoswath = incrementtoswath + 1
                    startpoint = False
            # Trace the eastern limb of the swath
            geoeastpointdict = Geodesic.WGS84.Direct(subsatlat, subsatlon, eastaz, swath)
            geoeastpointdict['time']=str(deltatime)
            geoeastpoint.append(geoeastpointdict)
            
            # Trace the western limb of the swath
            
            geowestpointdict = Geodesic.WGS84.Direct(subsatlat, subsatlon, westaz, swath)
            geowestpointdict['time']=str(deltatime)
            geowestpoint.append(geowestpointdict)

            deltatime = deltatime + timestep

            time.sleep(0.01)
        # When the end of the track is reached
        # Step from sub satellite point to limb of swath with increasing step size

        geoloseastpoint = []
        geoloswestpoint = []
        incrementtoswath = swathincrement

        while math.pow(incrementtoswath, 5) < swath:
            
            geodesiceastazdict = Geodesic.WGS84.Direct(subsatlat, subsatlon, eastaz, math.pow(incrementtoswath, 5))
            geodesiceastazdict['time']=str(deltatime)
            geoloseastpoint.append(geodesiceastazdict)
            
            geodesicwestazdict = Geodesic.WGS84.Direct(subsatlat, subsatlon, westaz, math.pow(incrementtoswath, 5))
            geodesicwestazdict['time']=str(deltatime)
            geoloswestpoint.append(geodesicwestazdict)

            incrementtoswath = incrementtoswath + 1
            
        # Append reversed geoloswestpoint to geoloseastpoint
        reversedwest = []
        for x in reversed(geoloswestpoint):
            reversedwest.append(x)

        for x in geoloseastpoint:
            reversedwest.append(x)
        for x in reversedwest:
            geowestpoint.append(x)

        polypoints = []

        for x in geowestpoint:
            polypoints.append({'lat2': x['lat2'], 'lon2': x['lon2'], 'time': x['time']})

        for x in reversed(geoeastpoint):
            polypoints.append({'lat2': x['lat2'], 'lon2': x['lon2'], 'time': x['time']})

        if len(polypoints) > 0:
            polypoints.append({'lat2': geowestpoint[0]['lat2'], 'lon2': geowestpoint[0]['lon2'], 'time': geowestpoint[0]['time']})


        # TODO add local solar time for AOSTime Lat Lon
        attributes = {'Satellite name': satname,
                      'Sensor code': sensor,
                      'Orbit height': orad,
                      'Orbit': orbitnumber,
                      #'Current time': str(now),
                      'Minutes to horizon': minutesaway,
                      'Local time': str(loctime),
                      'AOS time': str(rt),
                      'LOS time': str(st),
                      'Transit time': str(tt),
                      'Node': node, 'SWATH_FILENAME': (
                    satname + "." + str(orbitnumber) + "." + ground_station_name + ".orbit_swath.geojson"),
                      'Orbit filename': ORBIT_FILENAME,
                      'Orbit line': geotrack,
                      'Swath filename': SWATH_FILENAME,
                      'Swath polygon': polypoints
                      }

        # Append the attributes to the list of acquisitions for the acquisition period
        #print('ATTRIBUTES:', attributes)
        #print('SCHEDULE:', schedule)
        if not any(d['SWATH_FILENAME'] == attributes['SWATH_FILENAME'] for d in schedule):
            #if not attributes in schedule:
            # if not any((x['Satellite name'] == satname and x['Orbit'] == orbitnumber) for x in schedule):
            if (imagingnode == 'both' or imagingnode == attributes['Node']) and (to_datetime(rt) < (passes_begin_time + timedelta(minutes=passes_period))):
                schedule.append(attributes)

                # Create swath footprint ogr output

                get_vector_file(attributes, polypoints, 'polygon', SWATH_FILENAME, 'GeoJSON')
                get_vector_file(attributes, geotrack, 'line', ORBIT_FILENAME, 'GeoJSON')

                add_layer_to_map(SWATH_FILENAME,
                         satname + "." + str(orbitnumber) + ".swath", 'blue')
                add_layer_to_map(ORBIT_FILENAME,
                         satname + "." + str(orbitnumber) + ".orbit", 'red')

                # Create a temporal KML
                pol = kml.newpolygon(name=satname + '_' + str(orbitnumber), description=SWATH_FILENAME)
                kml_polypoints = []

                for i in polypoints:
                    kml_polypoints.append((i['lon2'], i['lat2']))
                rt_kml = datetime.strptime(str(rt), "%Y/%m/%d %H:%M:%S")
                st_kml = datetime.strptime(str(st), "%Y/%m/%d %H:%M:%S")
                pol.outerboundaryis = kml_polypoints
                pol.style.linestyle.color = simplekml.Color.green
                pol.style.linestyle.width = 5
                pol.style.polystyle.color = simplekml.Color.changealphaint(100, simplekml.Color.green)
                pol.timespan.begin = rt_kml.strftime("%Y-%m-%dT%H:%M:%SZ")
                pol.timespan.end = st_kml.strftime("%Y-%m-%dT%H:%M:%SZ")

        observer.date = (lostime + timedelta(minutes=90))
        lostime = (datetime.strptime(str(observer.date), "%Y/%m/%d %H:%M:%S"))

    kml.save(os.path.join(output_path, satname + "." + ground_station_name + ".kml"))

    ## plot folium map
    folium.LayerControl().add_to(satellite_map)
    foliumhtml = os.path.join(output_path, satname + "." + ground_station_name + ".map.html")
    satellite_map.save(foliumhtml)
    folium_timespan_geojson_html(schedule, satname)

    # render the html schedule and write to file
    attributeshtml = []
    
    for acquisition in schedule:
        attributeshtml.append('<td>' + acquisition['Satellite name'] + '</td>'\
        '<td><a href="' + str(os.path.relpath(acquisition['Swath filename'],os.path.dirname(foliumhtml)))+ '">' + str(acquisition['Sensor code']) + '</a></td>'\
        '<td><a href="' + str(os.path.relpath(acquisition['Orbit filename'],os.path.dirname(foliumhtml))) + '">' + str(acquisition['Orbit']) + '</a></td>'\
        '<td>' + str(acquisition['Node']) + '</td>'\
        '<td>' + str(acquisition['AOS time']) + '</td>'\
        '<td>' + str(acquisition['LOS time']) + '</td>')

    satelliteslist = []
    for x in satellites:
        schedule_url = get_satellite_name([x]) + "." + ground_station_name + ".schedule.html"
        satelliteslist.append([x, schedule_url])
        satelliteslist.append(',')

    renderedoutput = template.render(content=attributeshtml, foliumhtml=os.path.relpath(foliumhtml, output_path), satelliteslist=satelliteslist )

    with open(os.path.join(output_path, satname + "." + ground_station_name + ".schedule.html"), "w") as fh:
        fh.write(renderedoutput)

    return ()


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    # Configure Jinja for HTML templating
    path_to_templates = Path(__file__).resolve()
    path_to_templates = path_to_templates.parent.parent.joinpath("templates")
    file_loader = FileSystemLoader(path_to_templates.as_posix())
    env = Environment(loader=file_loader)
    template = env.get_template("template.html")
    
    

    # Get configurations
    satellites = []
    with open(args.configuration, 'r') as config:
        cfg = yaml.load(config, Loader=yaml.Loader)
    for satellite in cfg['missions']: satellites.append(satellite['satellite']['tle'])

    if not args.start:
        if cfg['scenario']['start'] == 'now':
            start = datetime.utcnow()
            #tles = check_tle_is_current()
        else:
            start = cfg['scenario']['start']
            #tles = check_tle_is_current()
            # TODO build logic to handle requests for time ranges > + 1 mont
    else:
        start = datetime.strptime(args.start, "%Y-%m-%dT%H:%M:%SZ")
        
    # Gather all norad ids and put into tuple
    norad_satellite_ids = []
    stale_age = timedelta(days = int(cfg['scenario']['stale_age']))
    for satellite in cfg['missions']: norad_satellite_ids.append(satellite['satellite']['norad'])
    norad_satellite_ids = tuple(norad_satellite_ids)
    
    stale_test = []
    # Check if 'tle.json exists else fetch
    if not os.path.exists('tle.json'):
        logging.info('No TLE present fetching...')
        tledf = updatetlefile(norad_satellite_ids, cfg['scenario']['space_track_user'], cfg['scenario']['space_track_pass'])      
    else:
        for satellite in satellites:
            tle, delta, result = tleiscurrent('tle.json', satellite, pandas.to_datetime(start), stale_age)
            
            stale_test.append(result.values[0])
        
        if False in stale_test:
            logging.info('TLE is stale, fetching ...')
            tledf = pandas.read_json('tle.json')
            tledf = updatetlefile(norad_satellite_ids, args.space_track_user, args.space_track_pass)
        else:   
            logging.info('TLE confirmed as within age tolerance for start time')
    try:
        tldedf
    except:
        tledf = pandas.read_json('tle.json')
    tledf['EPOCH'] = pandas.to_datetime(tledf['EPOCH'])        
    # At this point we have a pandas df containing all available TLEs for all satellites that aren't stale.        
            
    
    if not args.period:
        period = cfg['scenario']['period']
    else:
        period = int(args.period)
        
    ground_station = cfg['scenario']['ground_station']  # ('-23 42', '133 54')  # Alice Spring Data Acquisition Facility
    ground_station_latitude, ground_station_longitude = ground_station
    map_centre_lat = ground_station_latitude.split()[0]+"."+str(int((float(ground_station_latitude.split()[1])/60)*100))
    map_centre_lon = ground_station_longitude.split()[0]+"."+str(int((float(ground_station_longitude.split()[1])/60)*100))

    ground_station_name = cfg['scenario']['ground_station_name']
    observer_horizon = cfg['scenario']['observer-horizon']
    timestep = timedelta(seconds=cfg['scenario']['timestep'])
    
    
    if not args.output_path:
        output_path = os.path.relpath(cfg['outputs']['output_path'], os.getcwd())
    else:
        output_path = args.output_path
    
    schedule_url_basename = cfg['outputs']['schedule_url_basename']
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("OUTPUT PATH DOESN'T EXIST", output_path)

    # Loop through satellite list and execute until end of period
    
    # while 1: do this without stopping - comment out to do once - new scenario could be run once each 24hr period, then only update satellite positions each minute
    for i in satellites:
        
        # Determine time range
        # Get representative month
        print(start.month, (start+timedelta(minutes=1440)).month)
        
        for satellite in cfg['missions']:
            # Set up folium map
            #style_function = lambda x: {'fillColor': '#000000' if x['type'] == 'Polygon' else '#00ff00'}
            satellite_map = folium.Map(location=[float(map_centre_lat), float(map_centre_lon)], tiles='OpenStreetMap',zoom_start=3)

            if satellite['satellite']['tle'] == i:
                swath = satellite['satellite']['sensor']['swath'] / 2
                imagingnode = satellite['satellite']['sensor']['node']
                sensor = satellite['satellite']['sensor']['name']
                orbitoffset = satellite['satellite']['addtoorbit']
        logging.info("Get upcoming passes for " + i)
        
        logging.info(str([i, start, period]))
        get_upcoming_passes(i, start, period)
