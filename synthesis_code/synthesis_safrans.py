import os
import sys
import errno
import re
import glob
import csv
import os.path as op
import json
from datetime import datetime, timedelta, date
from IPython.display import display, clear_output, HTML
import numpy as np
from pathlib import Path
import pandas as pd
from osgeo import osr, ogr, gdal
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.gridspec as pltg
import seaborn as sn
from matplotlib.colors import LogNorm
import calendar
import pyproj
from pyproj import Proj, transform
import warnings
import rasterio
import pickle
from rasterio.warp import calculate_default_transform , reproject, Resampling
from rasterio.mask import mask
from shapely.geometry import mapping
import geopandas as gpd
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def mkdir_p(dos):
    try:
        os.makedirs(dos)
    except OSError as exc:  
        if exc.errno == errno.EEXIST and os.path.isdir(dos):
            pass
        else:
            raise
            

def getCoords(G):
    GT = G.GetGeoTransform()
    minx = GT[0]
    maxy = GT[3]
    maxx = minx + GT[1] * G.RasterXSize
    miny = maxy + GT[5] * G.RasterYSize
    return minx, maxy, maxx, miny


def reproject(inEPSG,outEPSG,x1,y1):
    inProj = Proj(init='EPSG:' + inEPSG)
    outProj = Proj(init='EPSG:'+ outEPSG)
    x2,y2 = transform(inProj,outProj,x1,y1)
    return x2, y2


SAFRAN_tiles={
    'ALP':
    {'31TGM':['Chablais'],
     '32TLR':['Mont-Blanc','Haute-Tarentaise','Haute-Laurienne'],
     '31TGL':['Chartreuse','Aravis','Beaufortain','Vanoise','Maurienne','Bauges','Grandes-Rousses','Belledonne'],
     '31TGK':['Vercors','Oisans','Devoluy','Champsaur','Pelvoux','Embrunnais Parpaillon','Ubaye'],
     '32TLQ':['Queyras','Thabor'],
     '32TLP':['Haut-Var Haut-Verdon','Mercantour']
    },
    'PYR':
    {'30TXN':['Pays-Basque','Aspe Ossau','Navarra','Jacetiana'],
     '30TYN':['Haute-Bigorre','Aure Louron','Luchonnais','Gallego','Sobrarbe','Esera'],
     '31TCH':['Couserans','Haute-Ariege','Andorre','Aran','Ribagorcana','Pallaresa','Perafita'],
     '31TDH':['Orlu St-Barthelemy','Capcir Puymorens','Cerdagne Canigou','Ter-Freser','Cadi Moixero']
    }
}










