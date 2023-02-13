# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 15:28:51 2022

@author: z5158936
"""

import requests
import sys
import os
from os.path import isfile
# Set the URL string to point to a specific data URL. Some generic examples are:
#   https://servername/data/path/file
#   https://servername/opendap/path/file[.format[?subset]]
#   https://servername/daac-bin/OTF/HTTP_services.cgi?KEYWORD=value[&KEYWORD=value]

fldr_data = 'MERRA-2_Raw'

#slv
list_files = open('subset_M2T1NXSLV_5.12.4_20221229_045311_.txt', 'r')
list_files = open('subset_M2T1NXSLV_5.12.4_20221230_112224_.txt', 'r')
keyword1 = 'LABEL'
keyword2 = 'DATASET_VERSION'

#rad
list_files = open('subset_M2T1NXRAD_5.12.4_20221230_112824_.txt', 'r')
keyword1 = 'LABEL'
keyword2 = 'FORMAT'

Lines = list_files.readlines()
for line in Lines:
    URL = line
    print(URL)
print()

for line in Lines:
    URL = line
    FILENAME = os.path.join(fldr_data,URL.split(keyword1,1)[1].split(keyword2,1)[0][1:-1])
    if not(isfile(FILENAME)):
        result = requests.get(URL)
        try:
           result.raise_for_status()
           f = open(FILENAME,'wb')
           f.write(result.content)
           f.close()
           print('contents of URL written to '+FILENAME)
        except:
           print('requests.get() returned an error code '+str(result.status_code))
           
           with open(os.path.join(fldr_data,'Failed_files.txt'), 'a+') as f:
               f.write(URL)
    # else:
    #     print('File existing')