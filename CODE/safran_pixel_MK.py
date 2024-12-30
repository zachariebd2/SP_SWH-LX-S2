
import os
import glob
import numpy as np
import pandas as pd
from osgeo import ogr
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import warnings
import rasterio
import pickle
from rasterio.mask import mask
from rasterio.merge import merge
from shapely.geometry import mapping
import pymannkendall as mk
import geopandas as gpd
import argparse
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 





            

SAFRAN_tiles={
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




notna_var = pd.notna
def mannkendall(g,):
    if g.query("@notna_var(VALUE)").HYDRO_YEAR.min() == 1986 and \
       g.query("@notna_var(VALUE)").HYDRO_YEAR.max() == 2022 and \
       g["VALUE"].count() >= 35:
        mk_test = mk.original_test(g.VALUE, alpha=0.05)
        out = pd.Series( dict( TREND = mk_test.trend, SLOPE = mk_test.slope, ASPECT = g.ASPECT.iloc[0], DAH = g.DAH.iloc[0]) )
    else:
        out = pd.Series( dict(   TREND = "nan",  SLOPE = np.nan, ASPECT = g.ASPECT.iloc[0], DAH = g.DAH.iloc[0]) )
    return out




arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

arg_parser.add_argument('-mtn', action='store', default="", nargs='?', const='', dest='mtn')
arg_parser.add_argument('-massif', action='store', default="", nargs='?', const='', dest='massif')
arg_parser.add_argument('-N', action='store', default="", nargs='?', const='', dest='N')
arg_parser.add_argument('-ele', action='store', default="", nargs='?', const='', dest='ELE')
arg_parser.add_argument('-project_dir', action='store', default="", nargs='?', const='', dest='project_dir')
arg_parser.add_argument('-model', action='store', default="", nargs='?', const='', dest='model')
mtn = arg_parser.parse_args().mtn
massif = arg_parser.parse_args().massif
N = arg_parser.parse_args().N
ele = int(arg_parser.parse_args().ELE)
project_dir = arg_parser.parse_args().project_dir
model = arg_parser.parse_args().model


synthese_model = os.path.join(project_dir,"DATA","SYNTHESIS","LANDSAT_SWH",model)
plot_path = os.path.join(project_dir,"DATA","SYNTHESIS","ANALYSIS","SAFRAN","PLOTS")
THEIA_path = os.path.join(project_dir,"DATA","SYNTHESIS","THEIA_MARGIN_15D_S2L8L7")
WATER_PATH = "/work/datalake/static_aux/MASQUES/eu_hydro/raster/20m"
GLACIER_PATH= os.path.join(project_dir,"DATA","GLACIERS")
DEM_PATH= "/work/CAMPUS/etudes/Neige/DEM"
TCD_PATH = "/work/datalake/static_aux/TreeCoverDensity"
ASPECT_PATH = os.path.join(project_dir,"DATA","ASPECTS")
DAH_PATH = os.path.join(project_dir,"DATA","DAH")
TMP_PATH = os.path.join(project_dir,"DATA","TMP",f"pixel_{ele}_{mtn}_{N}.tif")
drv = ogr.GetDriverByName( 'ESRI Shapefile' )
tile_list = SAFRAN_tiles[mtn][massif]
epsg_nb = (tile_list[0])[:2]
SAFRAN_PATH = os.path.join(project_dir,"DATA","SAFRAN",f"safran_{epsg_nb}.shp")


df = pd.DataFrame(data={'ELEVATION':[],'SAFRAN':[],
                   'MTN':[], 'TREND':[],'SLOPE':[], 'METRIC':[],'DAH':[],'ASPECT':[],'MIN_NOBS':[]})

period = range(1986,2015)
theia_period = range(2015,2023)
metric_list= ["SMOD"]
TCD_max = 50
MIN_NOBS = [10,18,26]
MIN_SCD = 0
DTM_step = 300


        

shapefile = gpd.read_file(SAFRAN_PATH)
geoms = (shapefile.query(f"title == '{massif}'")).geometry.values
geoms = [mapping(geoms[0])]
print("\n")
print(massif)


#TCD
raster_list = []
for tile in tile_list:
    tcd_path = os.path.join(TCD_PATH,tile,f'TCD_{tile}.tif')
    raster_list.append(tcd_path)
merge(raster_list,dst_path = TMP_PATH,nodata=255)
with rasterio.open(TMP_PATH) as src:
    tcd, out_transform = mask(src, geoms, crop=True,nodata=255)
    tcd = tcd[0].flatten()
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

#GLACIER
raster_list = []
for tile in tile_list:
    gla_path = os.path.join(GLACIER_PATH,f'glacier_{tile}.tif')
    raster_list.append(gla_path)
merge(raster_list,dst_path = TMP_PATH,nodata=0)
with rasterio.open(TMP_PATH) as src:
    gla, out_transform = mask(src, geoms, crop=True,nodata=0)
    gla = gla[0].flatten()
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


#ELEVATION
raster_list = []
for tile in tile_list:
    dem_path = os.path.join(DEM_PATH,f"S2__TEST_AUX_REFDE2_T{tile}_0001.DBL.DIR",f'S2__TEST_AUX_REFDE2_T{tile}_0001_ALT_R2.TIF')
    raster_list.append(dem_path)
merge(raster_list,dst_path = TMP_PATH,nodata=0)
with rasterio.open(TMP_PATH) as src:
    dem, out_transform = mask(src, geoms, crop=True,nodata=0)
    dem = dem[0].flatten()
os.system(f"rm {TMP_PATH}")
mask_fix= np.where((gla != 1) & (tcd <= 50.0) & (wtr != 1) & (dah >= -0.1) & (asp >= 0),1,0)

for minnobs in MIN_NOBS:
    for i,metric in enumerate(metric_list):
        mask_ele = np.where((dem >= ele - DTM_step/2) & (dem < ele + DTM_step/2),1,0)
        dah_ele = (dah[ (mask_fix == 1) & (mask_ele == 1)]).astype(float)
        asp_ele = (asp[ (mask_fix == 1) & (mask_ele == 1)]).astype(float)
        dict_MK = {'HYDRO_YEAR':[],'VALUE':[],'PIXEL':[],'DAH':[],'ASPECT':[]}
        year_list = []
        pixel_list = []
        is_ele_valid = True
        for year in period:
            raster_list=[]
            nobs_list=[]
            scd_list=[]
            for tile in tile_list:
                swh_landsat_path = glob.glob(os.path.join(synthese_model,mtn,tile,"*",f"LIS_S2L8-SNOW-SMOD_{tile}_{str(year)}*.tif"),recursive=True)
                nobs_path = glob.glob(os.path.join(synthese_model,mtn,tile,"*",f"LIS_S2L8-SNOW-NOBS_{tile}_{str(year)}*.tif"),recursive=True)
                scd_path = glob.glob(os.path.join(synthese_model,mtn,tile,"*",f"LIS_S2L8-SNOW-SCD_{tile}_{str(year)}*.tif"),recursive=True)
                if len(swh_landsat_path) == 1:
                    raster_list.extend(swh_landsat_path)
                    nobs_list.extend(nobs_path)
                    scd_list.extend(scd_path)

            if len(raster_list) == 0:
                print("ERROR MISSING SYNTHESIS FOR",mtn,tile_list,year)
            else:
                merge(raster_list,dst_path = TMP_PATH)
                with rasterio.open(TMP_PATH) as src:
                    swh_landsat, out_transform = mask(src, geoms, crop=True,nodata=0)
                    swh_landsat = swh_landsat[0].flatten().astype(float)
                os.system(f"rm {TMP_PATH}")
                merge(nobs_list,dst_path = TMP_PATH,nodata=0)
                with rasterio.open(TMP_PATH) as src:
                    nobs, out_transform = mask(src, geoms, crop=True,nodata=0)
                    nobs = nobs[0].flatten().astype(float)
                os.system(f"rm {TMP_PATH}")
                merge(scd_list,dst_path = TMP_PATH,nodata=0)
                with rasterio.open(TMP_PATH) as src:
                    scd, out_transform = mask(src, geoms, crop=True,nodata=0)
                    scd = scd[0].flatten().astype(float)
                os.system(f"rm {TMP_PATH}")
                del src
                #swh_landsat = swh_landsat[cond]  
                swh_landsat_year = (swh_landsat[ (mask_fix == 1) & (mask_ele == 1)]).astype(float)
                nobs_year = (nobs[ (mask_fix == 1) & (mask_ele == 1)]).astype(float)
                scd_year = (scd[ (mask_fix == 1) & (mask_ele == 1)]).astype(float)
                swh_landsat_year[(swh_landsat_year == 0) | (nobs_year < minnobs) | (scd_year <= MIN_SCD) ] = np.nan
                count = len(swh_landsat_year)
                if count >0:
                    dict_MK['VALUE'].extend(swh_landsat_year)
                    dict_MK['ASPECT'].extend(asp_ele)
                    dict_MK['DAH'].extend(dah_ele)
                    dict_MK['PIXEL'].extend(list(range(0, count)))
                    dict_MK['HYDRO_YEAR'].extend([year]*count)
                    del swh_landsat_year , out_transform
                    print(f"{mtn} {massif} {metric} {ele} {year} total length {len(df)} ")
                else: 
                    is_ele_valid = False
                    break

        for year in theia_period:
            raster_list=[]
            nobs_list=[]
            scd_list=[]
            for tile in tile_list:
                swh_landsat_path = glob.glob(os.path.join(THEIA_path,mtn,tile,f"*_{str(year)}0*",f"*SM*.tif"),recursive=True)
                nobs_path = glob.glob(os.path.join(THEIA_path,mtn,tile,f"*_{str(year)}0*",f"*NOB*.tif"),recursive=True)
                scd_path = glob.glob(os.path.join(THEIA_path,mtn,tile,f"*_{str(year)}0*",f"*SCD*.tif"),recursive=True)
                if len(swh_landsat_path) == 1:
                    raster_list.extend(swh_landsat_path)
                    nobs_list.extend(nobs_path)
                    scd_list.extend(scd_path)

            if len(raster_list) == 0:
                print("ERROR MISSING SYNTHESIS FOR",mtn,tile_list,year)
            else: 
                merge(raster_list,dst_path = TMP_PATH)
                with rasterio.open(TMP_PATH) as src:
                    swh_landsat, out_transform = mask(src, geoms, crop=True,nodata=0)
                    swh_landsat = swh_landsat[0].flatten().astype(float)
                os.system(f"rm {TMP_PATH}")
                merge(nobs_list,dst_path = TMP_PATH,nodata=0)
                with rasterio.open(TMP_PATH) as src:
                    nobs, out_transform = mask(src, geoms, crop=True,nodata=0)
                    nobs = nobs[0].flatten().astype(float)
                os.system(f"rm {TMP_PATH}")
                merge(scd_list,dst_path = TMP_PATH,nodata=0)
                with rasterio.open(TMP_PATH) as src:
                    scd, out_transform = mask(src, geoms, crop=True,nodata=0)
                    scd = scd[0].flatten().astype(float)
                os.system(f"rm {TMP_PATH}")
                del src
                #swh_landsat = swh_landsat[cond]  
                swh_landsat_year = (swh_landsat[ (mask_fix == 1) & (mask_ele == 1)]).astype(float)
                nobs_year = (nobs[ (mask_fix == 1) & (mask_ele == 1)]).astype(float)
                scd_year = (scd[ (mask_fix == 1) & (mask_ele == 1)]).astype(float)
                swh_landsat_year[(swh_landsat_year == 0) | (nobs_year < minnobs) | (scd_year <= MIN_SCD) ] = np.nan
                count = len(swh_landsat_year)
                if count >0:
                    dict_MK['VALUE'].extend(swh_landsat_year)
                    dict_MK['ASPECT'].extend(asp_ele)
                    dict_MK['DAH'].extend(dah_ele)
                    dict_MK['PIXEL'].extend(list(range(0, count)))
                    dict_MK['HYDRO_YEAR'].extend([year]*count)
                    del swh_landsat_year , out_transform
                    print(f"{mtn} {massif} {metric} {ele} {year} total length {len(df)} ")
                else: 
                    is_ele_valid = False
                    break

        #trend
        if is_ele_valid:
            df_MK =  pd.DataFrame(data=dict_MK)
            print(f"{mtn} {massif} {metric} {ele} MK total length {len(df)}")
            df_MK_trends = df_MK.sort_values(by=['HYDRO_YEAR']).groupby(["PIXEL"]).apply(mannkendall).reset_index()
            del df_MK
            df_MK_trends['ELEVATION'] = ele
            df_MK_trends['SAFRAN'] = massif
            df_MK_trends['MTN'] = mtn
            df_MK_trends['METRIC'] = metric
            df_MK_trends['MIN_NOBS'] = minnobs


            df = pd.concat([df,df_MK_trends], ignore_index=True)
            del df_MK_trends




            
            
            
            
            
df_path = os.path.join(project_dir,"DATA","SYNTHESIS","ANALYSIS","SAFRAN","DATAFRAMES",f"safran_pixel_{mtn}_{N}_{ele}.pkl")


df.to_pickle(df_path)         
            
            
            
            
            
            
            
            
            



