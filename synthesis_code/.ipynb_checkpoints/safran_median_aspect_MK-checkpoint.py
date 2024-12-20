
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
from rasterio.merge import merge
from shapely.geometry import mapping
import pymannkendall as mk
import statsmodels.api as sm
import statsmodels
import scipy.stats as st
from sklearn.metrics import r2_score, mean_squared_error
import geopandas as gpd
import argparse
from argparse import Action
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

def getDateFromStr(N):
    sepList = ["","-","_","/"]
    datev = ''
    for s in sepList :
        found = re.search('\d{4}'+ s +'\d{2}'+ s +'\d{2}', N)
        if found != None :
            datev = datetime.strptime(found.group(0), '%Y'+ s +'%m'+ s +'%d').date()
            break
    return datev

def getTimeFromStr(N,):
    sepList = ["","-","_"]
    HHMMSS = ''
    for s in sepList :
        found = re.search('-'+'\d{2}'+ s +'\d{2}'+ s +'\d{2}'+'-', N)
        if found != None :
            HHMMSS = datetime.strptime(found.group(0), '-'+'%H'+ s +'%M'+ s +'%S'+'-').time()
            break
    return HHMMSS

def reproject(inEPSG,outEPSG,x1,y1):
    
    inProj = Proj(init='EPSG:' + inEPSG)
    outProj = Proj(init='EPSG:'+ outEPSG)
    x2,y2 = transform(inProj,outProj,x1,y1)
    
    return x2, y2

def getCoords(G):
    
    
    GT = G.GetGeoTransform()
    minx = GT[0]
    maxy = GT[3]
    maxx = minx + GT[1] * G.RasterXSize
    miny = maxy + GT[5] * G.RasterYSize
    
    return minx, maxy, maxx, miny

def mkdir_p(dos):
    try:
        os.makedirs(dos)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(dos):
            pass
        else:
            raise

def getTileFromStr(N):

    tile = ''
    found = re.search('\d{2}' +'[A-Z]{3}', N)
    if found != None : tile = found.group(0)
    return tile

S2_tiles={
    "PYR":
    {
        "30TXN":{'EPSG':'32630','MINX':600000,'MINY':4690200,'MAXX':709800,'MAXY':4800000},
        '30TYN':{'EPSG':'32630','MINX':699960,'MINY':4690200,'MAXX':809760,'MAXY':4800000},
        '31TCH':{'EPSG':'32631','MINX':300000,'MINY':4690200,'MAXX':409800,'MAXY':4800000},
        '31TDH':{'EPSG':'32631','MINX':399960,'MINY':4690200,'MAXX':509760,'MAXY':4800000}
    },
    "ALP":
    {
        "31TGJ":{'EPSG':'32631','MINX':699960,'MINY':4790220,'MAXX':809760,'MAXY':4900020},
        '31TGK':{'EPSG':'32631','MINX':699960,'MINY':4890240,'MAXX':809760,'MAXY':5000040},
        '31TGL':{'EPSG':'32631','MINX':699960,'MINY':4990200,'MAXX':809760,'MAXY':5100000},
        '31TGM':{'EPSG':'32631','MINX':699960,'MINY':5090220,'MAXX':809760,'MAXY':5200020},
        "32TLP":{'EPSG':'32632','MINX':300000,'MINY':4790220,'MAXX':409800,'MAXY':4900020},
        '32TLQ':{'EPSG':'32632','MINX':300000,'MINY':4890240,'MAXX':409800,'MAXY':5000040},
        '32TLR':{'EPSG':'32632','MINX':300000,'MINY':4990200,'MAXX':409800,'MAXY':5100000},
        '32TLS':{'EPSG':'32632','MINX':300000,'MINY':5090220,'MAXX':409800,'MAXY':5200020}
    }
}


epsg_list={
    "30TXN":{'EPSG':'32630'},
    '30TYN':{'EPSG':'32630'},
    '31TCH':{'EPSG':'32631'},
    '31TDH':{'EPSG':'32631'},
    "31TGJ":{'EPSG':'32631'},
    '31TGK':{'EPSG':'32631'},
    '31TGL':{'EPSG':'32631'},
    '31TGM':{'EPSG':'32631'},
    "32TLP":{'EPSG':'32632'},
    '32TLQ':{'EPSG':'32632'},
    '32TLR':{'EPSG':'32632'},
    '32TLS':{'EPSG':'32632'}
}


S2_4326_tiles={
    "PYR":
    {
        "30TXN":{'EPSG':'32630','MINX':reproject('32630','4326',600000,4690200)[0],'MINY':reproject('32630','4326',600000,4690200)[1],'MAXX':reproject('32630','4326',709800,4800000)[0],'MAXY':reproject('32630','4326',709800,4800000)[1]
                },
        '30TYN':{'EPSG':'32630','MINX':reproject('32630','4326',699960,4690200)[0],'MINY':reproject('32630','4326',699960,4690200)[1],'MAXX':reproject('32630','4326',809760,4800000)[0],'MAXY':reproject('32630','4326',809760,4800000)[1]
                },
        '31TCH':{'EPSG':'32631','MINX':reproject('32631','4326',300000,4690200)[0],'MINY':reproject('32631','4326',300000,4690200)[1],'MAXX':reproject('32631','4326',409800,4800000)[0],'MAXY':reproject('32631','4326',409800,4800000)[1]
                },
        '31TDH':{'EPSG':'32631','MINX':reproject('32631','4326',399960,4690200)[0],'MINY':reproject('32631','4326',399960,4690200)[1],'MAXX':reproject('32631','4326',509760,4800000)[0],'MAXY':reproject('32631','4326',509760,4800000)[1]
                }
    },
    "ALP":
    {
        "31TGJ":{'EPSG':'32631','MINX':reproject('32631','4326',699960,4790220)[0],'MINY':reproject('32631','4326',699960,4790220)[1],'MAXX':reproject('32631','4326',809760,4900020)[0],'MAXY':reproject('32631','4326',809760,4900020)[1]
                },
        '31TGK':{'EPSG':'32631','MINX':reproject('32631','4326',699960,4890240)[0],'MINY':reproject('32631','4326',699960,4890240)[1],'MAXX':reproject('32631','4326',809760,5000040)[0],'MAXY':reproject('32631','4326',809760,5000040)[1]
                },
        '31TGL':{'EPSG':'32631','MINX':reproject('32631','4326',699960,4990200)[0],'MINY':reproject('32631','4326',699960,4990200)[1],'MAXX':reproject('32631','4326',809760,5100000)[0],'MAXY':reproject('32631','4326',809760,5100000)[1]
                },
        '31TGM':{'EPSG':'32631','MINX':reproject('32631','4326',699960,5090220)[0],'MINY':reproject('32631','4326',699960,5090220)[1],'MAXX':reproject('32631','4326',809760,5200020)[0],'MAXY':reproject('32631','4326',809760,5200020)[1]
                },
        "32TLP":{'EPSG':'32632','MINX':reproject('32632','4326',300000,4790220)[0],'MINY':reproject('32632','4326',300000,4790220)[1],'MAXX':reproject('32632','4326',409800,4900020)[0],'MAXY':reproject('32632','4326',409800,4900020)[1]
                },
        '32TLQ':{'EPSG':'32632','MINX':reproject('32632','4326',300000,4890240)[0],'MINY':reproject('32632','4326',300000,4890240)[1],'MAXX':reproject('32632','4326',409800,5000040)[0],'MAXY':reproject('32632','4326',409800,5000040)[1]
                },
        '32TLR':{'EPSG':'32632','MINX':reproject('32632','4326',300000,4990200)[0],'MINY':reproject('32632','4326',300000,4990200)[1],'MAXX':reproject('32632','4326',409800,5100000)[0],'MAXY':reproject('32632','4326',409800,5100000)[1]
                },
        '32TLS':{'EPSG':'32632','MINX':reproject('32632','4326',300000,5090220)[0],'MINY':reproject('32632','4326',300000,5090220)[1],'MAXX':reproject('32632','4326',409800,5200020)[0],'MAXY':reproject('32632','4326',409800,5200020)[1]
                }
    }
}



LANDSAT_wrs={
    "ALP":
    {
        "195029":["31TGJ",'31TGK','31TGL',"32TLP",'32TLQ','32TLR'],
        "195028":['31TGL','31TGM','32TLQ','32TLR','32TLS'],
        "196029":["31TGJ",'31TGK','31TGL','32TLQ','32TLR'],
        "196028":['31TGK','31TGL','31TGM','32TLQ','32TLR','32TLS'],
        "194029":["32TLP",'32TLQ','32TLR']
    },
    "PYR":
    {
        "200030":["30TXN",'30TYN'],
        "199030":["30TXN",'30TYN','31TCH'],
        "198031":['31TCH','31TDH'],
        "198030":['31TCH','31TDH'],
        "197031":['31TDH']
    }
}

LANDSAT_tiles={
    "ALP":
    {
        "31TGJ":["195029","196029"],
        '31TGK':["195029","196029","196028"],
        '31TGL':["195029","195028","196029","196028"],
        '31TGM':["195028","196028"],
        "32TLP":["194029","195029"],
        '32TLQ':["194029","195029","195028","196029","196028"],
        '32TLR':["194029","195029","195028","196029","196028"],
        '32TLS':["195028","196028"]
    },
    "PYR":
    {
        "30TXN":["200030","199030"],
        '30TYN':["200030","199030"],
        '31TCH':["199030","198031","198030"],
        '31TDH':["198031","198030","197031"]
    }
}

SPOT_tile={
    "ALP":
    {"KMIN":46,
     "KMAX":55,
     "JMIN":254,
     "JMAX":263
    },
    "PYR":
    {"KMIN":35,
     "KMAX":48,
     "JMIN":262,
     "JMAX":265
    }
}

SAFRAN_tiles={
    "ALP":
    {"31TGM":["Chablais"],
     "32TLR":["Mont-Blanc","Haute-Tarentaise","Haute-Maurienne"],
     "31TGL":["Chartreuse","Aravis","Beaufortain","Vanoise","Maurienne","Bauges","Grandes-Rousses","Belledonne"],
     "31TGK":["Vercors","Oisans","Devoluy","Champsaur","Pelvoux","Embrunnais Parpaillon","Ubaye"],
     "32TLQ":["Queyras","Thabor"],
     "32TLP":["Haut-Var Haut-Verdon","Mercantour"]
     
    },
    "PYR":
    {"30TXN":["Pays-Basque","Aspe Ossau","Navarra","Jacetiana"],
     "30TYN":["Haute-Bigorre","Aure Louron","Luchonnais","Gallego","Sobrarbe","Esera"],
     "31TCH":["Couserans","Haute-Ariege","Andorre","Aran","Ribagorcana","Pallaresa","Perafita"],
     "31TDH":["Orlu St-Barthelemy","Capcir Puymorens","Cerdagne Canigou","Ter-Freser","Cadi Moixero"]
    }
}

SAFRAN_tiles_2={
    "ALP":
    {"Chablais":["31TGM"],
     "Mont-Blanc":["32TLR"],
     "Haute-Tarentaise":["32TLR"],
     "Haute-Maurienne":["32TLR"],
     "Chartreuse":["31TGL"],
     "Aravis":["31TGL","31TGM"],
     "Beaufortain":["31TGL"],
     "Vanoise":["31TGL"],
     "Maurienne":["31TGL"],
     "Bauges":["31TGL"],
     "Grandes-Rousses":["31TGL"],
     "Belledonne":["31TGL"],
     "Vercors":["31TGK","31TGL"],
     "Oisans":["31TGK"],
     "Devoluy":["31TGK"],
     "Champsaur":["31TGK"],
     "Pelvoux":["31TGK"],
     "Embrunnais Parpaillon":["31TGK"],
     "Ubaye":["31TGK"],
     "Queyras":["32TLQ"],
     "Thabor":["31TGK","31TGL"],
     "Haut-Var Haut-Verdon":["32TLP","32TLQ"],
     "Mercantour":["32TLP","32TLQ"]
     
    },
    "PYR":
    {"Pays-Basque":["30TXN"],
     "Aspe Ossau":["30TXN","30TYN"],
     "Navarra":["30TXN"],
     "Jacetiana":["30TXN","30TYN"],
     "Haute-Bigorre":["30TYN"],
     "Aure Louron":["30TYN"],
     "Luchonnais":["30TYN"],
     "Gallego":["30TYN"],
     "Sobrarbe":["30TYN"],
     "Esera":["30TYN"],
     "Couserans":["31TCH"],
     "Haute-Ariege":["31TCH"],
     "Andorre":["31TCH"],
     "Aran":["31TCH"],
     "Ribagorcana":["31TCH"],
     "Pallaresa":["31TCH"],
     "Perafita":["31TCH"],
     "Orlu St-Barthelemy":["31TCH","31TDH"],
     "Capcir Puymorens":["31TDH"],
     "Cerdagne Canigou":["31TDH"],
     "Ter-Freser":["31TDH"]

    }
}



SAFRAN_tiles_test={
    "ALP":
    {"31TGM":["Chablais"],
     "32TLR":["Mont-Blanc","Haute-Tarentaise","Haute-Maurienne"]
     
    },
    "PYR":
    {"30TXN":["Pays-Basque","Aspe Ossau","Navarra","Jacetiana"]
    }
}


def mannkendall(g):
    if g.VALUE.isna().sum() <=2 :
        mk_test = mk.original_test(g.VALUE, alpha=0.05)
        out = pd.Series( dict(  TAU = mk_test.Tau, TREND = mk_test.trend, P = mk_test.p, SLOPE = mk_test.slope, INTERCEPT = mk_test.intercept) )
    else:
        out = pd.Series( dict(  TAU = np.nan, TREND = np.nan, P = np.nan, SLOPE = np.nan, INTERCEPT = np.nan) )
    return out






arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

arg_parser.add_argument('-mtn', action='store', default="", nargs='?', const='', dest='mtn')
arg_parser.add_argument('-massif', action='store', default="", nargs='?', const='', dest='massif')
arg_parser.add_argument('-N', action='store', default="", nargs='?', const='', dest='N')
mtn = arg_parser.parse_args().mtn
massif = arg_parser.parse_args().massif
N = arg_parser.parse_args().N

model="TCD-BLUE_AVG-1200"
synthese_model = f"/home/ad/barrouz/zacharie/TIMESERIES_PROJECT/SYNTHESIS/LANDSAT_SWH/{model}"
THEIA_path = "/home/ad/barrouz/zacharie/TIMESERIES_PROJECT/SYNTHESIS/THEIA_MARGIN_15D_S2L8L7"
plot_path = "/home/ad/barrouz/zacharie/TIMESERIES_PROJECT/SYNTHESIS/ANALYSIS/SAFRAN/PLOTS"
#SAFRAN_PATH = "/home/ad/barrouz/zacharie/TIMESERIES_PROJECT/SAFRAN/GR_alps_safran_31.shp"
GLACIER_PATH= "/home/ad/barrouz/zacharie/TIMESERIES_PROJECT/GLACIERS"
DEM_PATH= "/home/ad/barrouz/Neige/DEM"
TCD_PATH = "/home/ad/barrouz/datalake/static_aux/TreeCoverDensity"
WATER_PATH = "/home/ad/barrouz/datalake/static_aux/MASQUES/eu_hydro/raster/20m"
ASPECT_PATH = "/home/ad/barrouz/zacharie/TIMESERIES_PROJECT/ASPECTS"
DAH_PATH = "/home/ad/barrouz/zacharie/TIMESERIES_PROJECT/DAH"
TMP_PATH = f"/home/ad/barrouz/zacharie/TIMESERIES_PROJECT/SYNTHESIS/ANALYSIS/SAFRAN/TMP/{mtn}_{N}.tif"
drv = ogr.GetDriverByName( 'ESRI Shapefile' )
#safran_shp = drv.Open(SAFRAN_PATH)
#shapefile = gpd.read_file(SAFRAN_PATH)
#geoms = (shapefile.query("title == 'Grandes-Rousses'")).geometry.values # list of shapely geometries
#geoms = [mapping(geoms[0])]





dict_massif_trends = {'ELEVATION':[],'SAFRAN':[],
                   'MTN':[], 'SMOD_MEDIAN':[],'SOD_MEDIAN':[], 'SCD_MEDIAN':[],  'PIXEL_COUNT':[], 'MIN_NOBS':[], 'TOPO':[], 'TOPO_VALUE':[], 'HYDRO_YEAR':[],'DATASET':[]}



period = range(1986,2015)
theia_period = range(2015,2023)
TCD_max = 50
DTM_step = 300
DTM_min = 1500
DTM_max= 4500
MIN_SCD = 0

#ASPECT_range = {0:[337.5,22.5],45:[22.5,67.5],90:[67.5,112.5],135:[112.5,157.5],180:[157.5,202.5],225:[202.5,247.5],270:[247.5,292.5],315:[292.5,337.5]}

ASPECT_range = {0:[315,45],90:[45,135],180:[135,225],270:[225,315]}

#DAH_range = {-0.9:[-1,-0.8],-0.7:[-0.8,-0.6],-0.5:[-0.6,-0.4],-0.3:[-0.4,-0.2],-0.1:[-0.2,0],0.1:[0,0.2],0.3:[0.2,0.4],0.5:[0.4,0.6],0.7:[0.6,0.8],0.9:[0.8,1]}
DAH_range = {-0.8:[-1,-0.6],-0.4:[-0.6,-0.2],0:[-0.2,0.2],0.4:[0.2,0.6],0.8:[0.6,1]}

tile_list = SAFRAN_tiles_2[mtn][massif]
epsg_nb = (tile_list[0])[:2]
SAFRAN_PATH = f"/home/ad/barrouz/zacharie/TIMESERIES_PROJECT/SAFRAN/safran_{epsg_nb}.shp"
shapefile = gpd.read_file(SAFRAN_PATH)
geoms = (shapefile.query(f"title == '{massif}'")).geometry.values
geoms = [mapping(geoms[0])]
#print(SAFRAN_PATH)
#print("\n")
#print(massif)


#TCD
os.system(f"rm {TMP_PATH}")
raster_list = []
for tile in tile_list:
    tcd_path = os.path.join(TCD_PATH,tile,f'TCD_{tile}.tif')
    raster_list.append(tcd_path)
merge(raster_list,dst_path = TMP_PATH,nodata=255)
with rasterio.open(TMP_PATH) as src:
    meta = src.meta.copy()
    tcd, out_transform = mask(src, geoms, crop=True,nodata=255)
    meta["count"] = tcd.shape[0]
    meta["height"] = tcd.shape[1]
    meta["width"] = tcd.shape[2]
    meta["nodata"] = 255
    meta["transform"] = out_transform
    tcd = tcd[0].flatten()
    #print(np.unique(tcd, return_counts=True))
os.system(f"rm {TMP_PATH}")

# GLACIER
raster_list = []
for tile in tile_list:
    gla_path = os.path.join(GLACIER_PATH,f'glacier_{tile}.tif')
    raster_list.append(gla_path)
merge(raster_list,dst_path = TMP_PATH,nodata=0)
with rasterio.open(TMP_PATH) as src:
    gla, out_transform = mask(src, geoms, crop=True,nodata=0)
    gla = gla[0].flatten()
    #print(np.unique(gla, return_counts=True))
os.system(f"rm {TMP_PATH}")

# WATER
raster_list = []
for tile in tile_list:
    wtr_path = os.path.join(WATER_PATH,tile,f'eu_hydro_20m_{tile}.tif')
    raster_list.append(wtr_path)
merge(raster_list,dst_path = TMP_PATH,nodata=0)
with rasterio.open(TMP_PATH) as src:
    wtr, out_transform = mask(src, geoms, crop=True,nodata=0)
    wtr = wtr[0].flatten()
    #print(np.unique(gla, return_counts=True))
os.system(f"rm {TMP_PATH}")

#ASPECTS
raster_list = []
for tile in tile_list:
    asp_path = os.path.join(ASPECT_PATH,f'ASP_{mtn}_{tile}.tif')
    raster_list.append(asp_path)
merge(raster_list,dst_path = TMP_PATH,nodata=-9999)
with rasterio.open(TMP_PATH) as src:
    asp, out_transform = mask(src, geoms, crop=True,nodata=-9999)
    asp = asp[0].flatten()
os.system(f"rm {TMP_PATH}")

#DAH
raster_list = []
for tile in tile_list:
    dah_path = os.path.join(DAH_PATH,f'DAH_{mtn}_{tile}.tif')
    raster_list.append(dah_path)
merge(raster_list,dst_path = TMP_PATH,nodata=-99999)
with rasterio.open(TMP_PATH) as src:
    dah, out_transform = mask(src, geoms, crop=True,nodata=-99999)
    dah = dah[0].flatten()
os.system(f"rm {TMP_PATH}")

# ELEVATION

raster_list = []
for tile in tile_list:
    dem_path = os.path.join(DEM_PATH,f"S2__TEST_AUX_REFDE2_T{tile}_0001.DBL.DIR",f'S2__TEST_AUX_REFDE2_T{tile}_0001_ALT_R2.TIF')
    raster_list.append(dem_path)
merge(raster_list,dst_path = TMP_PATH,nodata=0)
with rasterio.open(TMP_PATH) as src:
    dem, out_transform = mask(src, geoms, crop=True,nodata=0)
    dem = dem[0].flatten()
    #print(np.unique(dem, return_counts=True))
os.system(f"rm {TMP_PATH}")
mask_fix= np.where((gla != 1) & (wtr != 1) & (tcd <= 50.0) & (dah >= -0.1) & (asp >= 0),1,0)
tmpmask = f"/home/ad/barrouz/zacharie/TIMESERIES_PROJECT/SYNTHESIS/ANALYSIS/SAFRAN/TMP/MASK_{mtn}_{N}.tif"
with rasterio.open(tmpmask, "w", **meta) as ds:
        ds.write(mask_fix.reshape((meta["height"], meta["width"])),1)


for year in period:
    smod_list=[]
    nobs_list=[]
    scd_list=[]
    sod_list=[]
    for tile in tile_list:
        smod_path = glob.glob(op.join(synthese_model,mtn,tile,"*",f"LIS_S2L8-SNOW-SMOD_{tile}_{str(year)}*.tif"),recursive=True)
        nobs_path = glob.glob(op.join(synthese_model,mtn,tile,"*",f"LIS_S2L8-SNOW-NOBS_{tile}_{str(year)}*.tif"),recursive=True)
        scd_path = glob.glob(op.join(synthese_model,mtn,tile,"*",f"LIS_S2L8-SNOW-SCD_{tile}_{str(year)}*.tif"),recursive=True)
        sod_path = glob.glob(op.join(synthese_model,mtn,tile,"*",f"LIS_S2L8-SNOW-SOD_{tile}_{str(year)}*.tif"),recursive=True)
        if len(smod_path) == 1:
            smod_list.extend(smod_path)
            nobs_list.extend(nobs_path)
            scd_list.extend(scd_path)
            sod_list.extend(sod_path)
    if len(smod_list) == 0:
        print("ERROR MISSING SMOD SYNTHESIS FOR",mtn,tile_list,year)
    merge(smod_list,dst_path = TMP_PATH,nodata=0)
    with rasterio.open(TMP_PATH) as src:
        smod, out_transform = mask(src, geoms, crop=True,nodata=0)
        smod = smod[0].flatten()
    os.system(f"rm {TMP_PATH}")
    merge(nobs_list,dst_path = TMP_PATH,nodata=0)
    with rasterio.open(TMP_PATH) as src:
        nobs, out_transform = mask(src, geoms, crop=True,nodata=0)
        nobs = nobs[0].flatten()
    os.system(f"rm {TMP_PATH}")
    merge(scd_list,dst_path = TMP_PATH,nodata=0)
    with rasterio.open(TMP_PATH) as src:
        scd, out_transform = mask(src, geoms, crop=True,nodata=0)
        scd = scd[0].flatten()
    os.system(f"rm {TMP_PATH}")
    merge(sod_list,dst_path = TMP_PATH,nodata=0)
    with rasterio.open(TMP_PATH) as src:
        sod, out_transform = mask(src, geoms, crop=True,nodata=0)
        sod = sod[0].flatten()
    os.system(f"rm {TMP_PATH}")

    mask_valid = np.where((smod > 0) & (scd >= MIN_SCD),1,0)
    for ele in range(DTM_min,DTM_max+1,DTM_step):
        mask_ele = np.where((dem >= ele - DTM_step/2) & (dem < ele + DTM_step/2) ,1,0)
        for nobs_min in range(2,27,2):
            mask_min = np.where((nobs >=nobs_min ),1,0)
            #elevation uniquement
            mask_all = np.where((mask_fix == 1) & (mask_ele == 1)  &  (mask_valid == 1) &  (mask_min == 1),1,0)
            smod_ele = smod[(mask_all == 1) ]
            sod_ele = sod[(mask_all == 1) ]
            scd_ele = scd[(mask_all == 1) ]
            count = len(smod_ele)  
            if count >= 2:
                smod_median = np.median(smod_ele)
                sod_median = np.median(sod_ele)
                scd_median = np.median(scd_ele)
            else: 
                smod_median=np.nan
                sod_median = np.nan
                scd_median = np.nan
            dict_massif_trends['ELEVATION'].append(ele)
            dict_massif_trends['SAFRAN'].append(massif)
            dict_massif_trends['MTN'].append(mtn)
            dict_massif_trends['SMOD_MEDIAN'].append(smod_median)
            dict_massif_trends['SOD_MEDIAN'].append(sod_median)
            dict_massif_trends['SCD_MEDIAN'].append(scd_median)
            dict_massif_trends['HYDRO_YEAR'].append(year)
            dict_massif_trends['PIXEL_COUNT'].append(count)
            dict_massif_trends['TOPO'].append("NONE")
            dict_massif_trends['TOPO_VALUE'].append(np.nan)
            dict_massif_trends['MIN_NOBS'].append(nobs_min)
            dict_massif_trends['DATASET'].append("TB")
            
            #add aspect
            for aspect in ASPECT_range:
                if aspect == 0:
                    mask_topo = np.where(((asp >= ASPECT_range[aspect][0]) | (asp < ASPECT_range[aspect][1])) ,1,0)
                else:
                    mask_topo = np.where(((asp >= ASPECT_range[aspect][0]) & (asp < ASPECT_range[aspect][1])) ,1,0)
                smod_ele = smod[(mask_topo == 1) &  (mask_all == 1)  ]
                sod_ele = sod[(mask_topo == 1) &  (mask_all == 1)   ]
                scd_ele = scd[(mask_topo == 1) &  (mask_all == 1)   ]
                count = len(smod_ele)  
                if count >= 2:
                    smod_median = np.median(smod_ele)
                    sod_median = np.median(sod_ele)
                    scd_median = np.median(scd_ele)
                else: 
                    smod_median=np.nan
                    sod_median = np.nan
                    scd_median = np.nan
                dict_massif_trends['ELEVATION'].append(ele)
                dict_massif_trends['SAFRAN'].append(massif)
                dict_massif_trends['MTN'].append(mtn)
                dict_massif_trends['SMOD_MEDIAN'].append(smod_median)
                dict_massif_trends['SOD_MEDIAN'].append(sod_median)
                dict_massif_trends['SCD_MEDIAN'].append(scd_median)
                dict_massif_trends['HYDRO_YEAR'].append(year)
                dict_massif_trends['PIXEL_COUNT'].append(count)
                dict_massif_trends['TOPO'].append("ASPECT")
                dict_massif_trends['TOPO_VALUE'].append(aspect)
                dict_massif_trends['MIN_NOBS'].append(nobs_min)
                dict_massif_trends['DATASET'].append("TB")
                    


            #add dah
            for dah_value in DAH_range:
                if dah_value == 0.8:
                    mask_topo = np.where((dah >= DAH_range[dah_value][0]) & (dah <= DAH_range[dah_value][1]) ,1,0)
                else:
                    mask_topo = np.where((dah >= DAH_range[dah_value][0]) & (dah < DAH_range[dah_value][1]) ,1,0)
                    
                smod_ele = smod[(mask_topo == 1) &  (mask_all == 1)   ]
                sod_ele = sod[(mask_topo == 1) &  (mask_all == 1)  ]
                scd_ele = scd[(mask_topo == 1) &  (mask_all == 1)   ]
                count = len(smod_ele)  
                if count >= 2:
                    smod_median = np.median(smod_ele)
                    sod_median = np.median(sod_ele)
                    scd_median = np.median(scd_ele)
                else: 
                    smod_median=np.nan
                    sod_median = np.nan
                    scd_median = np.nan
                dict_massif_trends['ELEVATION'].append(ele)
                dict_massif_trends['SAFRAN'].append(massif)
                dict_massif_trends['MTN'].append(mtn)
                dict_massif_trends['SMOD_MEDIAN'].append(smod_median)
                dict_massif_trends['SOD_MEDIAN'].append(sod_median)
                dict_massif_trends['SCD_MEDIAN'].append(scd_median)
                dict_massif_trends['HYDRO_YEAR'].append(year)
                dict_massif_trends['PIXEL_COUNT'].append(count)
                dict_massif_trends['TOPO'].append("DAH")
                dict_massif_trends['TOPO_VALUE'].append(dah_value)
                dict_massif_trends['MIN_NOBS'].append(nobs_min)
                dict_massif_trends['DATASET'].append("TB")


for year in theia_period:
    smod_list=[]
    nobs_list=[]
    scd_list=[]
    sod_list=[]
    for tile in tile_list:
        smod_path = glob.glob(op.join(THEIA_path,mtn,tile,f"MULTISAT_{str(year)}*",f"*SM*.tif"),recursive=True)
        nobs_path = glob.glob(op.join(THEIA_path,mtn,tile,f"MULTISAT_{str(year)}*",f"*NOB*.tif"),recursive=True)
        scd_path = glob.glob(op.join(THEIA_path,mtn,tile,f"MULTISAT_{str(year)}*",f"*SCD*.tif"),recursive=True)
        sod_path = glob.glob(op.join(THEIA_path,mtn,tile,f"MULTISAT_{str(year)}*",f"*SOD*.tif"),recursive=True)
        if len(smod_path) == 1:
            smod_list.extend(smod_path)
            nobs_list.extend(nobs_path)
            scd_list.extend(scd_path)
            sod_list.extend(sod_path)
    if len(smod_list) == 0:
        print("ERROR MISSING SYNTHESIS FOR",mtn,tile_list,year)
    merge(smod_list,dst_path = TMP_PATH,nodata=0)
    with rasterio.open(TMP_PATH) as src:
        smod, out_transform = mask(src, geoms, crop=True,nodata=0)
        smod = smod[0].flatten()
    os.system(f"rm {TMP_PATH}")
    merge(nobs_list,dst_path = TMP_PATH,nodata=0)
    with rasterio.open(TMP_PATH) as src:
        nobs, out_transform = mask(src, geoms, crop=True,nodata=0)
        nobs = nobs[0].flatten()
    os.system(f"rm {TMP_PATH}")
    merge(scd_list,dst_path = TMP_PATH,nodata=0)
    with rasterio.open(TMP_PATH) as src:
        scd, out_transform = mask(src, geoms, crop=True,nodata=0)
        scd = scd[0].flatten()
    os.system(f"rm {TMP_PATH}")
    merge(sod_list,dst_path = TMP_PATH,nodata=0)
    with rasterio.open(TMP_PATH) as src:
        sod, out_transform = mask(src, geoms, crop=True,nodata=0)
        sod = sod[0].flatten()
    os.system(f"rm {TMP_PATH}")

    mask_valid = np.where((smod > 0) & (scd >= MIN_SCD),1,0)
    for ele in range(DTM_min,DTM_max+1,DTM_step):
        mask_ele = np.where((dem >= ele - DTM_step/2) & (dem < ele + DTM_step/2) ,1,0)
        for nobs_min in range(2,27,2):
            mask_min = np.where((nobs >=nobs_min ),1,0)
            #elevation uniquement
            mask_all = np.where((mask_fix == 1) & (mask_ele == 1)  &  (mask_valid == 1) &  (mask_min == 1),1,0)
            smod_ele = smod[(mask_all == 1) ]
            sod_ele = sod[(mask_all == 1) ]
            scd_ele = scd[(mask_all == 1) ]
            count = len(smod_ele)  
            if count >= 2:
                smod_median = np.median(smod_ele)
                sod_median = np.median(sod_ele)
                scd_median = np.median(scd_ele)
            else: 
                smod_median=np.nan
                sod_median = np.nan
                scd_median = np.nan
            dict_massif_trends['ELEVATION'].append(ele)
            dict_massif_trends['SAFRAN'].append(massif)
            dict_massif_trends['MTN'].append(mtn)
            dict_massif_trends['SMOD_MEDIAN'].append(smod_median)
            dict_massif_trends['SOD_MEDIAN'].append(sod_median)
            dict_massif_trends['SCD_MEDIAN'].append(scd_median)
            dict_massif_trends['HYDRO_YEAR'].append(year)
            dict_massif_trends['PIXEL_COUNT'].append(count)
            dict_massif_trends['TOPO'].append("NONE")
            dict_massif_trends['TOPO_VALUE'].append(np.nan)
            dict_massif_trends['MIN_NOBS'].append(nobs_min)
            dict_massif_trends['DATASET'].append("THEIA")
            
            #add aspect
            for aspect in ASPECT_range:
                if aspect == 0:
                    mask_topo = np.where(((asp >= ASPECT_range[aspect][0]) | (asp < ASPECT_range[aspect][1])) ,1,0)
                else:
                    mask_topo = np.where(((asp >= ASPECT_range[aspect][0]) & (asp < ASPECT_range[aspect][1])) ,1,0)
                smod_ele = smod[(mask_topo == 1) &  (mask_all == 1)   ]
                sod_ele = sod[(mask_topo == 1) &  (mask_all == 1)  ]
                scd_ele = scd[(mask_topo == 1) &  (mask_all == 1)  ]
                count = len(smod_ele)  
                if count >= 2:
                    smod_median = np.median(smod_ele)
                    sod_median = np.median(sod_ele)
                    scd_median = np.median(scd_ele)
                else: 
                    smod_median=np.nan
                    sod_median = np.nan
                    scd_median = np.nan
                dict_massif_trends['ELEVATION'].append(ele)
                dict_massif_trends['SAFRAN'].append(massif)
                dict_massif_trends['MTN'].append(mtn)
                dict_massif_trends['SMOD_MEDIAN'].append(smod_median)
                dict_massif_trends['SOD_MEDIAN'].append(sod_median)
                dict_massif_trends['SCD_MEDIAN'].append(scd_median)
                dict_massif_trends['HYDRO_YEAR'].append(year)
                dict_massif_trends['PIXEL_COUNT'].append(count)
                dict_massif_trends['TOPO'].append("ASPECT")
                dict_massif_trends['TOPO_VALUE'].append(aspect)
                dict_massif_trends['MIN_NOBS'].append(nobs_min)
                dict_massif_trends['DATASET'].append("THEIA")
                    


            #add dah
            for dah_value in DAH_range:
                if dah_value == 0.8:
                    mask_topo = np.where((dah >= DAH_range[dah_value][0]) & (dah <= DAH_range[dah_value][1]) ,1,0)
                else:
                    mask_topo = np.where((dah >= DAH_range[dah_value][0]) & (dah < DAH_range[dah_value][1]) ,1,0)
                smod_ele = smod[(mask_topo == 1) &  (mask_all == 1)   ]
                sod_ele = sod[(mask_topo == 1) &  (mask_all == 1)  ]
                scd_ele = scd[(mask_topo == 1) &  (mask_all == 1)  ]
                count = len(smod_ele)  
                if count >= 2:
                    smod_median = np.median(smod_ele)
                    sod_median = np.median(sod_ele)
                    scd_median = np.median(scd_ele)
                else: 
                    smod_median=np.nan
                    sod_median = np.nan
                    scd_median = np.nan
                dict_massif_trends['ELEVATION'].append(ele)
                dict_massif_trends['SAFRAN'].append(massif)
                dict_massif_trends['MTN'].append(mtn)
                dict_massif_trends['SMOD_MEDIAN'].append(smod_median)
                dict_massif_trends['SOD_MEDIAN'].append(sod_median)
                dict_massif_trends['SCD_MEDIAN'].append(scd_median)
                dict_massif_trends['HYDRO_YEAR'].append(year)
                dict_massif_trends['PIXEL_COUNT'].append(count)
                dict_massif_trends['TOPO'].append("DAH")
                dict_massif_trends['TOPO_VALUE'].append(dah_value)
                dict_massif_trends['MIN_NOBS'].append(nobs_min)
                dict_massif_trends['DATASET'].append("THEIA")





df_path = f"/home/ad/barrouz/zacharie/TIMESERIES_PROJECT/SYNTHESIS/ANALYSIS/SAFRAN/DATAFRAMES/swhlx_theia_safran_median_aspect_hydro_rgi_{mtn}_{N}.pkl"
df = pd.DataFrame(data=dict_massif_trends)

df.to_pickle(df_path)          

            

            

            

            



