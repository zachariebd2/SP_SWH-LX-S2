
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




arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

arg_parser.add_argument('-mtn', action='store', default="", nargs='?', const='', dest='mtn')
arg_parser.add_argument('-massif', action='store', default="", nargs='?', const='', dest='massif')
arg_parser.add_argument('-N', action='store', default="", nargs='?', const='', dest='N')
arg_parser.add_argument('-project_dir', action='store', default="", nargs='?', const='', dest='project_dir')
arg_parser.add_argument('-model', action='store', default="", nargs='?', const='', dest='model')
mtn = arg_parser.parse_args().mtn
massif = arg_parser.parse_args().massif
N = arg_parser.parse_args().N
project_dir = arg_parser.parse_args().project_dir
model = arg_parser.parse_args().model

synthese_model = os.path.join(project_dir,"DATA","SYNTHESIS","LANDSAT_SWH",model)
THEIA_path = os.path.join(project_dir,"DATA","SYNTHESIS","THEIA_MARGIN_15D_S2L8L7")
plot_path = os.path.join(project_dir,"DATA","SYNTHESIS","ANALYSIS","SAFRAN","PLOTS")
GLACIER_PATH= os.path.join(project_dir,"DATA","GLACIERS")
DEM_PATH= "/work/CAMPUS/etudes/Neige/DEM"
TCD_PATH = "/work/datalake/static_aux/TreeCoverDensity"
WATER_PATH = "/work/datalake/static_aux/MASQUES/eu_hydro/raster/20m"
ASPECT_PATH = os.path.join(project_dir,"DATA","ASPECTS")
DAH_PATH = os.path.join(project_dir,"DATA","DAH")
TMP_PATH = os.path.join(project_dir,"DATA","TMP",f"median_smod_{mtn}_{N}.tif")
drv = ogr.GetDriverByName( 'ESRI Shapefile' )
tile_list = SAFRAN_tiles[mtn][massif]
epsg_nb = (tile_list[0])[:2]
SAFRAN_PATH = os.path.join(project_dir,"DATA","SAFRAN",f"safran_{epsg_nb}.shp")


dict_massif_trends = {'ELEVATION':[],'SAFRAN':[],
                   'MTN':[], 'SMOD_MEDIAN':[],'SOD_MEDIAN':[], 'SCD_MEDIAN':[],  'PIXEL_COUNT':[], 'MIN_NOBS':[], 'TOPO':[], 'TOPO_VALUE':[], 'HYDRO_YEAR':[],'DATASET':[]}


period = range(1986,2015)
theia_period = range(2015,2023)
TCD_max = 50
DTM_step = 300
DTM_min = 1500
DTM_max= 4500
MIN_SCD = 0

ASPECT_range = {0:[315,45],90:[45,135],180:[135,225],270:[225,315]}

DAH_range = {-0.8:[-1,-0.6],-0.4:[-0.6,-0.2],0:[-0.2,0.2],0.4:[0.2,0.6],0.8:[0.6,1]}


shapefile = gpd.read_file(SAFRAN_PATH)
geoms = (shapefile.query(f"title == '{massif}'")).geometry.values
geoms = [mapping(geoms[0])]


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
os.system(f"rm {TMP_PATH}")
mask_fix= np.where((gla != 1) & (wtr != 1) & (tcd <= 50.0) & (dah >= -0.1) & (asp >= 0),1,0)


for year in period:
    smod_list=[]
    nobs_list=[]
    scd_list=[]
    sod_list=[]
    for tile in tile_list:
        smod_path = glob.glob(os.path.join(synthese_model,mtn,tile,"*",f"LIS_S2L8-SNOW-SMOD_{tile}_{str(year)}*.tif"),recursive=True)
        nobs_path = glob.glob(os.path.join(synthese_model,mtn,tile,"*",f"LIS_S2L8-SNOW-NOBS_{tile}_{str(year)}*.tif"),recursive=True)
        scd_path = glob.glob(os.path.join(synthese_model,mtn,tile,"*",f"LIS_S2L8-SNOW-SCD_{tile}_{str(year)}*.tif"),recursive=True)
        sod_path = glob.glob(os.path.join(synthese_model,mtn,tile,"*",f"LIS_S2L8-SNOW-SOD_{tile}_{str(year)}*.tif"),recursive=True)
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
        smod_path = glob.glob(os.path.join(THEIA_path,mtn,tile,f"MULTISAT_{str(year)}*",f"*SM*.tif"),recursive=True)
        nobs_path = glob.glob(os.path.join(THEIA_path,mtn,tile,f"MULTISAT_{str(year)}*",f"*NOB*.tif"),recursive=True)
        scd_path = glob.glob(os.path.join(THEIA_path,mtn,tile,f"MULTISAT_{str(year)}*",f"*SCD*.tif"),recursive=True)
        sod_path = glob.glob(os.path.join(THEIA_path,mtn,tile,f"MULTISAT_{str(year)}*",f"*SOD*.tif"),recursive=True)
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





df_path = os.path.join(project_dir,"DATA","SYNTHESIS","ANALYSIS","SAFRAN","DATAFRAMES",f"swhlx_theia_safran_median_aspect_hydro_rgi_{mtn}_{N}.pkl")
df = pd.DataFrame(data=dict_massif_trends)

df.to_pickle(df_path)          

            

            

            

            




