# s3vtdata
Data retrieval and hotspots abstraction routines for S3VT Active Fires project

## Contact details
mailto:Simon.Oliver@ga.gov.au

## config.yaml
Configurations for FTP and AWS S3 bucket access. Credentials not included in s3vt github repository on security grounds.

## DEA Sandbox
### Signup
Signup and send username to mailto:Simon.Oliver@ga.gov.au requesting addition to the S3VT group for DEA Sandbox

### Logging in 
https://app.sandbox.dea.ga.gov.au/

Select Sentinel 3 Validation Team profile following login

### s3vt_active_fire_load_hotspots.ipynb 
#### Initial run
- You will need to run this to load the available geojson hotspots to your Sandbox instance.

- xmldict does not install - work around for now is to do the following
Options to resolve:
Option 1 - launch terminal
'> cd s3vt folder
'> source sandbox_libraries.sh

Options to resolve:
launch console

* use shift/enter to execute the below
'> import sys
'> !{sys.executable} -m pip install xmltodict
'> !{sys.executable} -m pip install pyephem
'> !{sys.executable} -m pip install pyorbital
'> !{sys.executable} -m pip install simplekml
'> !{sys.executable} -m pip install colour

- You should see something like:
Collecting xmltodict
  Using cached xmltodict-0.12.0-py2.py3-none-any.whl (9.2 kB)
Installing collected packages: xmltodict
Successfully installed xmltodict-0.12.0

### swathpredict.py
Runs as part of s3vt_active_fire_load_hotspots.ipynb
Generates satellite swaths based on the configuration s3vtconfig.yaml
usage: swathpredict.py [-h] [--configuration CONFIGURATION] [--start START]
                       [--period PERIOD] [--output_path OUTPUT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --configuration CONFIGURATION
                        ground station configuration
  --start START         start time YYYY-MM-DDTHH:MM:SS format
  --period PERIOD       ground station configuration
  --output_path OUTPUT_PATH
                        ground station configuration