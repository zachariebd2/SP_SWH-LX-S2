import os
import numpy as np
import glob
import torch
from torch.utils.data import DataLoader, random_split, RandomSampler
from torch import optim
import torch.nn as nn
from osgeo import osr, ogr,gdal
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
from pyproj import Transformer
import argparse
import errno
import pickle
import xml.etree.ElementTree as ET
import zipfile

from datetime import datetime









# S2 tile dictionary
# for each mountain range, S2 tiles with epsg projection and coordinates (lower left and upper right)
#print(epsg,max_y,min_y,max_x,min_x,model_type,sat_name, model_dir)
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



# mkdir directories with mkdir -p
def mkdir_p(dos):
    try:
        os.makedirs(dos)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(dos):
            pass
        else:
            raise

# reproject the coord of a point from inEPSG to outEPSG
def reproject(inEPSG,outEPSG,x1,y1):  
    transformer = Transformer.from_crs(int(inEPSG),int(outEPSG))
    x2,y2 = transformer.transform(y1, x1)
    #print("REPROJECT")
    #print(inEPSG,x1,y1)
    #print(outEPSG,x2,y2)
    return x2, y2

# find the S2 tiles overlapping the SPOT image and 
# output the overlap coordinates in the epsg projection
# only use the PYRENEANS tiles for now
# an image cannot overlap PYR and ALP at the same time
def getInferenceParams(epsg,max_y,min_y,max_x,min_x,model_type,sat_name, model_dir): 

    out_tiles={}
    for mtn in S2_tiles:
        for tile in S2_tiles[mtn]:
            epsgS2 = S2_tiles[mtn][tile]['EPSG']
            min_x_S2 , min_y_S2 = reproject(epsg,epsgS2,min_x,min_y)
            max_x_S2 , max_y_S2 = reproject(epsg,epsgS2,max_x,max_y)
            min_x_o = max(min_x_S2,S2_tiles[mtn][tile]['MINX'])
            max_y_o = min(max_y_S2,S2_tiles[mtn][tile]['MAXY'])
            max_x_o = min(max_x_S2,S2_tiles[mtn][tile]['MAXX'])
            min_y_o = max(min_y_S2,S2_tiles[mtn][tile]['MINY'])   
            if (min_x_o > max_x_o or min_y_o > max_y_o) : continue # if no intersection with tile
            area = tile if model_type == "tile" else mtn #get tile name or mountain range name
            # get model and normalisation files
            model = glob.glob(os.path.join(model_dir,area,sat_name,"STEP2",'*model*'))
            norm = glob.glob(os.path.join(model_dir,area,sat_name,"STEP2",'*norm*'))
            #print(area, model, norm)
            if model == [] or norm == []: 
                #print(f"WARNING: model or normalisation parameters not available for satellite {sat_name} and model {tile}")
                continue
            else:
                out_tiles[tile]={}
                out_tiles[tile]['MTN']=mtn
                out_tiles[tile]['EPSG']=epsgS2
                out_tiles[tile]['MINX']=min_x_o
                out_tiles[tile]['MINY']=min_y_o
                out_tiles[tile]['MAXX']=max_x_o
                out_tiles[tile]['MAXY']=max_y_o
                out_tiles[tile]['MODEL']=model[0]
                out_tiles[tile]['NORM']=norm[0]
    return out_tiles


#add a buffer of bufferDist units to a polygon shapefile inputfn 
# the result outputBufferfn is a new shapefile
#used to mask the inference pixels affected by neighbouring (~2km) nodata pixels
def createBuffer(inputfn, outputBufferfn, bufferDist):
    inputds = ogr.Open(inputfn)
    inputlyr = inputds.GetLayer()
    shpdriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputBufferfn):
        shpdriver.DeleteDataSource(outputBufferfn)
    outputBufferds = shpdriver.CreateDataSource(outputBufferfn)
    srs = inputlyr.GetSpatialRef()
    bufferlyr = outputBufferds.CreateLayer("post_buffer",srs, 
    geom_type=ogr.wkbPolygon)
    featureDefn = bufferlyr.GetLayerDefn()
    for feature in inputlyr:
        ingeom = feature.GetGeometryRef()
        geomBuffer = ingeom.Buffer(bufferDist)
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(geomBuffer)
        bufferlyr.CreateFeature(outFeature)
        outFeature = None
    outputBufferds = None



# ############## START ########################################################################################################################################################

def detect_snow(in_zip_path, out_path, job_path,model_type, model_dir, buffer, dem_path, tcd_path,keep,mask_bool):
    # use GPU 
    device = torch.device('cuda:'+torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
    print("DEVICE",device)
    torch.cuda.empty_cache()
    
    # Open zip

    zip_scene = zipfile.ZipFile(in_zip_path)
    MTD_file = ""
    XS1_file = ""
    XS2_file = ""
    XS3_file = ""
    SWIR_file = ""
    for ziped_file in zip_scene.namelist():
        if "MTD_ALL.xml" in os.path.basename(ziped_file) and ".tif.aux.xml" not in ziped_file :
            MTD_file = ziped_file
        elif "XS1" in os.path.basename(ziped_file) and ".tif.aux.xml" not in ziped_file : 
            XS1_file = ziped_file
        elif "XS2" in os.path.basename(ziped_file) and ".tif.aux.xml" not in ziped_file : 
            XS2_file = ziped_file
        elif "XS3" in os.path.basename(ziped_file) and ".tif.aux.xml" not in ziped_file : 
            XS3_file = ziped_file
        elif "SWIR" in os.path.basename(ziped_file) and ".tif.aux.xml" not in ziped_file : 
            SWIR_file = ziped_file
        else:
            continue
    
    # Parse MTD

    swh_str = zip_scene.read(MTD_file)
    tree= ET.fromstring(swh_str)
    t = tree.findall(".//Geometric_Informations/Mean_Value_List/Sun_Angles/ZENITH_ANGLE")[0]
    sun_zen = float(t.text)
    t = tree.findall(".//Geometric_Informations/Mean_Value_List/Sun_Angles/AZIMUTH_ANGLE")[0]
    sun_azi = float(t.text)
    t = tree.findall(".//Product_Characteristics/ACQUISITION_DATE")[0]
    acqui_date_time = t.text.split("T")
    acqui_date=acqui_date_time[0].replace('-', '')
    acqui_time=acqui_date_time[1].replace(':', '').split('.')
    acqui_HMS = acqui_time[0]
    acqui_SSS = acqui_time[1]
    dateobject=datetime.strptime(acqui_date,'%Y%m%d')
    year = str(dateobject.year)
    month = str(dateobject.month)
    day = str(dateobject.day)
    t = tree.findall(".//Product_Characteristics/PRODUCT_ID")[0]
    product_id = t.text
    t = tree.findall(".//Product_Characteristics/PLATFORM")[0]
    sat_name = t.text
    t = tree.findall(".//Product_Characteristics/INSTRUMENT")[0]
    ins_name = t.text
    t = tree.findall(".//Geoposition_Informations/Coordinate_Reference_System/Horizontal_Coordinate_System/HORIZONTAL_CS_CODE")[0]
    epsg = t.text
    t = tree.findall(".//Geoposition_Informations/Geopositioning/Global_Geopositioning/Point[@name='upperRight']/LAT")[0]
    ur_lat = float(t.text)
    t = tree.findall(".//Geoposition_Informations/Geopositioning/Global_Geopositioning/Point[@name='upperRight']/LON")[0]
    ur_lon = float(t.text)
    t = tree.findall(".//Geoposition_Informations/Geopositioning/Global_Geopositioning/Point[@name='lowerLeft']/LAT")[0]
    ll_lat = float(t.text)
    t = tree.findall(".//Geoposition_Informations/Geopositioning/Global_Geopositioning/Point[@name='lowerLeft']/LON")[0]
    ll_lon = float(t.text)

    # Parse zip files

    xs1_path = "/vsizip/"+in_zip_path+"/"+XS1_file
    xs2_path = "/vsizip/"+in_zip_path+"/"+XS2_file
    xs3_path = "/vsizip/"+in_zip_path+"/"+XS3_file
    
    
    
    if SWIR_file != "": 
        swir_path = "/vsizip/"+in_zip_path+"/"+SWIR_file
    else: 
        swir_path = ""
        
    #print("swir path",SWIR_file,swir_path)
    
    # Get inference params

    inferenceParams = getInferenceParams(epsg,ur_lat,ll_lat,ur_lon,ll_lon,model_type,sat_name, model_dir)
    if len(inferenceParams) == 0:
        print("INPUT SWH SCENE DOES NOT OVERLAP ANY S2 TILE WITH AVAILABLE MODEL")
        exit()
    #print(inferenceParams)
    
    # For each S2 tile do inference

    for tile in inferenceParams:  # for each S2 tile
        # get unet model and normalisation
        model_path = inferenceParams[tile]['MODEL']
        norm_path = inferenceParams[tile]['NORM']


        # set paths
        common_name = f"FSC_{acqui_date}T{acqui_HMS}_{sat_name}_{tile}" 
        tile_path = os.path.join(job_path,sat_name,year,month,day,product_id,tile,common_name) 
        mkdir_p(tile_path)
        
        xs1_tiled_path = os.path.join(tile_path,"xs1.tif")
        xs2_tiled_path = os.path.join(tile_path,"xs2.tif")
        xs3_tiled_path = os.path.join(tile_path,"xs3.tif")
        band_paths = xs1_tiled_path + " " + xs2_tiled_path + " " + xs3_tiled_path
        merged_path = os.path.join(tile_path,"merged.tif")
        dem_in_path = glob.glob(os.path.join(dem_path,'**','*'+tile+'*ALT_R2.TIF'),recursive=True)
        dem_out_path = os.path.join(tile_path,"dem.tif")
        tcd_in_path = os.path.join(tcd_path,tile,f'TCD_{tile}.tif')
        tcd_out_path = os.path.join(tile_path,"tcd.tif")
        hill_path =  os.path.join(tile_path,"hill.tif")
        pred_path = os.path.join(tile_path,f"{common_name}_FSCTOC.tif") #f"_SCA.tif""
        prob_0_path = os.path.join(tile_path,f"{common_name}_PROB_0_GROUND.tif")
        prob_1_path = os.path.join(tile_path,f"{common_name}_PROB_1_SNOW.tif")
        prob_2_path = os.path.join(tile_path,f"{common_name}_PROB_2_CLOUD.tif")
        pre_buffer_path = os.path.join(tile_path,"pre_buffer.shp")
        post_buffer_path = os.path.join(tile_path,"post_buffer.shp")
        nodata_path = os.path.join(tile_path,"nodata.tif")
        QA_path = os.path.join(tile_path,f"{common_name}_QA.tif")
        SWH_mtd = os.path.join(tile_path,os.path.basename(MTD_file))
        pred_mtd = os.path.join(tile_path,f"{common_name}_MTD.xml")
        
        
        
        #check if inference already exist
        mtn = inferenceParams[tile]['MTN']
        output_tile_path = os.path.join(out_path,mtn,sat_name,year,month,day,product_id,tile,common_name)
        if os.path.exists(output_tile_path):
            if len(glob.glob(os.path.join(output_tile_path,'*.tif'))) == 4 :
                print('skipped existing inference for',output_tile_path)
                continue


        # get overlapping coords and proj
        epsg = inferenceParams[tile]["EPSG"]
        minx = inferenceParams[tile]["MINX"]
        miny = inferenceParams[tile]["MINY"]
        maxx = inferenceParams[tile]["MAXX"]
        maxy = inferenceParams[tile]["MAXY"]

        # reproject and cut spot bands inside overlap bounds
        gdal.Warp(xs1_tiled_path,xs1_path,format= 'GTiff',dstSRS="EPSG:" + epsg,resampleAlg="cubic",outputBounds=[minx,miny,maxx,maxy],xRes= 20,yRes= 20)
        gdal.Warp(xs2_tiled_path,xs2_path,format= 'GTiff',dstSRS="EPSG:" + epsg,resampleAlg="cubic",outputBounds=[minx,miny,maxx,maxy],xRes= 20,yRes= 20)
        gdal.Warp(xs3_tiled_path,xs3_path,format= 'GTiff',dstSRS="EPSG:" + epsg,resampleAlg="cubic",outputBounds=[minx,miny,maxx,maxy],xRes= 20,yRes= 20)
        
        # add SWIR band if exist
        if swir_path != "":
            swir_tiled_path = os.path.join(tile_path,"swir.tif")
            band_paths = band_paths + " " + swir_tiled_path
            gdal.Warp(swir_tiled_path,swir_path,format= 'GTiff',dstSRS="EPSG:" + epsg,resampleAlg="cubic",outputBounds=[minx,miny,maxx,maxy],xRes= 20,yRes= 20)
        
        #print("band_paths",band_paths)
        # cut dem inside overlapping bounds
        gdal.Warp(dem_out_path,dem_in_path,format= 'GTiff',resampleAlg="cubic",outputBounds=[minx,miny,maxx,maxy],xRes= 20,yRes= 20)
        
        # generate hillshade from dem inside overlap bounds
        os.system(f"gdaldem hillshade {dem_out_path} {hill_path} -alg Horn -az {sun_azi} -alt {90 - sun_zen} -compute_edges")
        
        # merge spot bands, dem and hillshade
        os.system(f"gdal_merge.py -separate -o {merged_path} {band_paths} {hill_path} {dem_out_path}")
        # remove temp files
        os.system(f"rm {band_paths}  {hill_path} {dem_out_path}")

        # create tensor
        merged = gdal.Open(merged_path, gdal.GA_Update)
        merged_array = merged.ReadAsArray()
        norm = torch.load(norm_path,map_location=device)
        tensor =  torch.as_tensor(merged_array.copy().astype(float),device=device,dtype=torch.float32)
        del merged, merged_array
        if int(keep) != 1:
            os.system(f"rm {merged_path} ")

        # remove rows and cols until the image has appropriate dimensions for the UNET
        # size/16-7.75 must give an integer
        odd_margin_W = 0  #width
        W_size = tensor.size()[2]
        while W_size/16-7.75 != int(W_size/16-7.75):
            odd_margin_W = odd_margin_W + 1
            W_size = W_size - 1

        odd_margin_H = 0  #height
        H_size = tensor.size()[1]
        while H_size/16-7.75 != int(H_size/16-7.75):
            odd_margin_H = odd_margin_H + 1
            H_size = H_size - 1

        if odd_margin_H > 0: #remove bottom pixel rows
            tensor = tensor[:,:-1*odd_margin_H,:]
        if odd_margin_W > 0: #remove right pixel cols
            tensor = tensor[:,:,:-1*odd_margin_W]

        #make nodata mask for inference (0) and for buffer (the nodata lost in the inference are used for the buffer)
        mask = torch.where(tensor[0,...] > 0, 1.0, 0.0) #check for 0 in first band xs1      
        # normalize
        tensor = norm(tensor) # tensor [Classes,H,W] with C = 5 or 6
        # add mask
        tensor = tensor*mask
        # add batch dim
        tensor = torch.unsqueeze(tensor, dim=0) # tensor [Batches,channels,H,W] with B = 1, C = 5, and  H,W = image dim

        # load unet in eval mode
        padding="same"
        nb_channels = 6 if swir_path != "" else 5
        net = torch.load(model_path,map_location=device)
        net.eval()

        # do inference
        m = nn.Softmax(dim=1)
        with torch.no_grad():
            print("TOTO")
            prob_classes = net(tensor)
            pred_classes =  torch.squeeze(prob_classes.argmax(dim=1),0).cpu().detach().numpy()  # (H,W)    
            print(pred_classes)
            prob_classes = torch.squeeze(m(prob_classes),0) * 100
            prob_0 = prob_classes[0,...].cpu().detach().numpy() 
            prob_1 = prob_classes[1,...].cpu().detach().numpy() 
            prob_2 = prob_classes[2,...].cpu().detach().numpy() 
            
        pixel_loss_from_conv = 92 #constant value for the standard Unet architecture
        del tensor
        # create post-inference nodata mask (255)
        # because the pre-inference nodata were detected as clouds
        mask2 = mask[pixel_loss_from_conv:-1*pixel_loss_from_conv,pixel_loss_from_conv:-1*pixel_loss_from_conv].numpy() 
        pred_classes[mask2 == 0] = 255
        print("geo data")
        # output geo data
        ulx = minx+pixel_loss_from_conv*20
        uly = maxy-pixel_loss_from_conv*20
        xsize = W_size-pixel_loss_from_conv*2
        ysize = H_size-pixel_loss_from_conv*2
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(int(epsg))

        # create quality binary band (two bits)
        # bit 0: inference pixel affected by neighbouring nodata (1) or not (0)
        # bit 1: inference pixel covered by TCD > 50 (1) or not (0)
        #exemple: binary value of 2 => 01 => no TCD but presence of nodata in inference

        #get tcd array
        tcd_raster = gdal.Warp('',tcd_in_path,format= 'MEM',resampleAlg="near",outputBounds=[ulx,uly-ysize*20,ulx+xsize*20,uly],xRes= 20,yRes= 20)
        tcd_array = tcd_raster.ReadAsArray()
        tcd_array = np.where(tcd_array <= 50,0,1)

        #get inference-affected-by-nodata array (with buffer)
        driver = gdal.GetDriverByName('GTiff')
        outDs = driver.Create(nodata_path,  xsize, ysize , 1, gdal.GDT_Byte)
        #set pixel coordinates from the top-left pixel (no need for the odd_margin values)
        outDs.SetGeoTransform([ulx, 20, 0, uly, 0, -20]) 
        outDs.SetProjection(srs.ExportToWkt())
        outDs.FlushCache()
        del outDs
        #add buffer
        drv = ogr.GetDriverByName("ESRI Shapefile")
        mask_shp =  gdal.GetDriverByName('MEM').Create('', W_size, H_size, 1, gdal.GDT_Byte) #use the dim of the first mask so that the nodata lost in the inference are used for the buffer
        mask_shp.SetProjection(srs.ExportToWkt())
        mask_shp.SetGeoTransform([minx, 20, 0, maxy, 0, -20])
        mask_shp_band = mask_shp.GetRasterBand(1)
        mask = mask.numpy() 
        mask = np.where(mask == 0,1,0)
        mask_shp_band.WriteArray(mask)
        mask_shp_layername = "mask_shp"
        mask_shp_ds = drv.CreateDataSource(pre_buffer_path)
        mask_shp_layer = mask_shp_ds.CreateLayer(mask_shp_layername, srs, ogr.wkbPolygon)
        gdal.Polygonize( mask_shp_band, mask_shp_band, mask_shp_layer, -1, [], callback=None )
        mask_shp_ds = None
        createBuffer(pre_buffer_path, post_buffer_path, buffer)

        os.system(f"gdal_rasterize -burn 1  -l post_buffer {post_buffer_path} {nodata_path}")
        nodata_raster = gdal.Warp('',nodata_path,format= 'MEM',resampleAlg="near",outputBounds=[ulx,uly-ysize*20,ulx+xsize*20,uly],xRes= 20,yRes= 20)
        nodata_array = nodata_raster.ReadAsArray()

        
        

        minx = S2_tiles[mtn][tile]['MINX']
        miny = S2_tiles[mtn][tile]['MINY']
        maxx = S2_tiles[mtn][tile]['MAXX']
        maxy = S2_tiles[mtn][tile]['MAXY']

        
        
        #either separate inference and QA or combine both
        if int(mask_bool) == 0:
        
            #make binary QA raster
            QA_array = (nodata_array +tcd_array*2)*mask2
            driver = gdal.GetDriverByName('GTiff')
            outDs = driver.Create(QA_path,  xsize, ysize , 1, gdal.GDT_Byte)
            outBand = outDs.GetRasterBand(1)
            outBand.SetNoDataValue(255)
            outBand.WriteArray(QA_array)
            #set pixel coordinates from the top-left pixel (no need for the odd_margin values)
            outDs.SetGeoTransform([ulx, 20, 0, uly, 0, -20]) 
            outDs.SetProjection(srs.ExportToWkt())
            outDs.FlushCache()
            del outDs 
            gdal.Warp(QA_path,QA_path,format= 'GTiff',resampleAlg="near",xRes=20,yRes=20,outputBounds=[minx,miny,maxx,maxy],creationOptions = ['COMPRESS=DEFLATE'])
            del tcd_raster, tcd_array, nodata_raster, nodata_array
            os.system(f"rm {tile_path}/post_buffer.* {tile_path}/pre_buffer.* {tile_path}/nodata.tif ")

            #give pred LIS labels
            pred_classes[pred_classes == 1] = 100
            pred_classes[pred_classes == 2] = 205
            pred_classes[pred_classes == 3] = 255
            #create prediction raster
            driver = gdal.GetDriverByName('GTiff')
            outDs = driver.Create(pred_path,  xsize, ysize , 1, gdal.GDT_Byte)
            outBand = outDs.GetRasterBand(1)
            outBand.SetNoDataValue(255)
            outBand.WriteArray(pred_classes)
            #set pixel coordinates from the top-left pixel (no need for the odd_margin values)
            outDs.SetGeoTransform([ulx, 20, 0, uly, 0, -20]) 
            outDs.SetProjection(srs.ExportToWkt())
            outDs.FlushCache()
            del outDs
            gdal.Warp(pred_path,pred_path,format= 'GTiff',resampleAlg="near",xRes=20,yRes=20,outputBounds=[minx,miny,maxx,maxy],creationOptions = ['COMPRESS=DEFLATE'])
            
        
        else:
            QA_array = (nodata_array +tcd_array*2)*mask2
            #give pred LIS labels
            pred_classes[pred_classes == 1] = 100
            pred_classes[pred_classes == 2] = 205
            pred_classes[pred_classes == 3] = 255
            #mask pred
            pred_classes[QA_array > 0] = 255
            #create prediction raster
            driver = gdal.GetDriverByName('GTiff')
            outDs = driver.Create(pred_path,  xsize, ysize , 1, gdal.GDT_Byte)
            outBand = outDs.GetRasterBand(1)
            outBand.SetNoDataValue(255)
            outBand.WriteArray(pred_classes)
            #set pixel coordinates from the top-left pixel (no need for the odd_margin values)
            outDs.SetGeoTransform([ulx, 20, 0, uly, 0, -20]) 
            outDs.SetProjection(srs.ExportToWkt())
            outDs.FlushCache()
            del outDs
            gdal.Warp(pred_path,pred_path,format= 'GTiff',resampleAlg="near",xRes=20,yRes=20,outputBounds=[minx,miny,maxx,maxy],creationOptions = ['COMPRESS=DEFLATE'])

            os.system(f"rm {tile_path}/post_buffer.* {tile_path}/pre_buffer.* {tile_path}/nodata.tif ")
            del tcd_raster, tcd_array, nodata_raster, nodata_array
            
            
            
        #create probability rasters
        prob_0[pred_classes == 255] = 255
        driver = gdal.GetDriverByName('GTiff')
        outDs = driver.Create(prob_0_path,  xsize, ysize , 1, gdal.GDT_Byte)
        outBand = outDs.GetRasterBand(1)
        outBand.SetNoDataValue(255)
        outBand.WriteArray(prob_0)
        #set pixel coordinates from the top-left pixel (no need for the odd_margin values)
        outDs.SetGeoTransform([ulx, 20, 0, uly, 0, -20]) 
        outDs.SetProjection(srs.ExportToWkt())
        outDs.FlushCache()
        del outDs
        gdal.Warp(prob_0_path,prob_0_path,format= 'GTiff',resampleAlg="cubic",xRes=20,yRes=20,outputBounds=[minx,miny,maxx,maxy],creationOptions = ['COMPRESS=DEFLATE',"PREDICTOR=2"])

        del  prob_0
        
        prob_1[pred_classes == 255] = 255
        driver = gdal.GetDriverByName('GTiff')
        outDs = driver.Create(prob_1_path,  xsize, ysize , 1, gdal.GDT_Byte)
        outBand = outDs.GetRasterBand(1)
        outBand.SetNoDataValue(255)
        outBand.WriteArray(prob_1)
        #set pixel coordinates from the top-left pixel (no need for the odd_margin values)
        outDs.SetGeoTransform([ulx, 20, 0, uly, 0, -20]) 
        outDs.SetProjection(srs.ExportToWkt())
        outDs.FlushCache()
        del outDs, prob_1
        gdal.Warp(prob_1_path,prob_1_path,format= 'GTiff',resampleAlg="cubic",xRes=20,yRes=20,outputBounds=[minx,miny,maxx,maxy],creationOptions = ['COMPRESS=DEFLATE',"PREDICTOR=2"])

        
        
        prob_2[pred_classes == 255] = 255
        driver = gdal.GetDriverByName('GTiff')
        outDs = driver.Create(prob_2_path,  xsize, ysize , 1, gdal.GDT_Byte)
        outBand = outDs.GetRasterBand(1)
        outBand.SetNoDataValue(255)
        outBand.WriteArray(prob_2)
        #set pixel coordinates from the top-left pixel (no need for the odd_margin values)
        outDs.SetGeoTransform([ulx, 20, 0, uly, 0, -20]) 
        outDs.SetProjection(srs.ExportToWkt())
        outDs.FlushCache()
        del outDs, prob_2
        gdal.Warp(prob_2_path,prob_2_path,format= 'GTiff',resampleAlg="cubic",xRes=20,yRes=20,outputBounds=[minx,miny,maxx,maxy],creationOptions = ['COMPRESS=DEFLATE',"PREDICTOR=2"])

        
        


        #copy SWH metadata
        with open(SWH_mtd, "wb") as f:
            f.write(swh_str)
            
        
        mkdir_p(output_tile_path)
        
        print("copy to output")
        os.system(f"cp -f {tile_path}/* {output_tile_path}/")
