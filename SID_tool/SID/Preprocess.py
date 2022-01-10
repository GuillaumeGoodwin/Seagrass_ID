"""
Base_mask.py

This script is to make a basemap raster

"""



import os
import sys
import numpy as np
import platform
from datetime import datetime, timedelta
import pandas as pd
import geopandas as gpd
from osgeo import gdal, ogr, osr
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import csv
import tarfile as tar
import matplotlib.pyplot as plt
import subprocess as sp

from osgeo import gdal, ogr, osr
from shapely.geometry import Polygon
import rasterio
import fiona

from pathlib import Path

from rasterstats.io import Raster

###################################################################################
###################################################################################
def reproject_vectorfile(shp, EPSG): # OK works
    shapes = gpd.read_file(shp)
    shapes_goodcrs = shapes.to_crs({'init': 'epsg:'+ str(EPSG)}) 
    nameonly = shp.split('.')[0]

    shapes_goodcrs.to_file(driver = 'ESRI Shapefile', filename = nameonly+"_"+str(EPSG)+".shp")

    
    return nameonly+"_"+str(EPSG)+".shp"



###################################################################################
###################################################################################
def make_maskraster (basearr, ref_raster, basearr_dir, zoneraster, aoi, EPSG, maskfilename = 'Mask.bil'):

    maskfilename = basearr_dir+maskfilename

    if not os.path.isfile(maskfilename[:-4]+'_clip.bil'):

        #Retrieve shp files from the maskfile folder
        maskfiles = [basearr_dir+f for f in os.listdir(basearr_dir) if f.endswith('.shp')]

        # create an array to fill
        mask_arr = 255*np.ones(basearr.shape)

        zones = rasterio.open(zoneraster)
        zonearr = zones.read(1)
        mask_arr[zonearr > 0] = 0
        zones.close()

        #make the number codes for the mask
        maskdict ={}; count = 1
        for f in maskfiles:
            maskdict[f]=count; count +=1

        radagaster = rasterio.open(ref_raster)

        for shp in maskdict:
            # reproject shapefiles
            #shp3003 = reproject_shp(shp, 3003)
            #print (shp3003)

            # open the extent raster as a reference
            with Raster(ref_raster) as raster_obj: 
                # open the survey shapefile
                with fiona.open(shp) as f: 
                    # for each polygon
                    for m in range(len(f)):
                        geom = f[m]['geometry'] # the polygon geometry

                        # create a True/False mask of the dimensions of the extent raster
                        # True means the presence of a feature in the concerned layer
                        polygon_mask = rasterio.features.geometry_mask(geometries=[geom], out_shape=(raster_obj.shape[0], raster_obj.shape[1]), transform=raster_obj.affine, all_touched=True, invert=True)

                        # these are the pixels touched by the polygon
                        where = np.where(polygon_mask == True)
                        
                        if len(where[0]) > 0:
                            print (m , '/', len(f))
                            # attribute the code value
                            mask_arr[where] = maskdict[shp]
        
        with rasterio.open(maskfilename,'w', driver='ENVI', width=radagaster.width, height=radagaster.height, count=1, crs=radagaster.crs,transform=radagaster.transform, dtype=radagaster.dtypes[0]) as raster:
                raster.write(mask_arr,1) 
        raster.close()
        radagaster.close()

        #Clip the mask
        #define the GDAL options
        options = gdal.WarpOptions(cutlineDSName=aoi, cropToCutline=True, dstSRS='EPSG:' + str(EPSG), dstNodata = 255)
        outBand = gdal.Warp(srcDSOrSrcDSTab=maskfilename,destNameOrDestDS=maskfilename[:-4]+'_clip.bil', options=options); outBand= None


    return maskfilename[:-4]+'_clip.bil'


###################################################################################
###################################################################################
def make_zoneraster (basearr, ref_raster, basearr_poly, aoi, EPSG, maskfilename = 'Zones.bil'):

    maskfilename = basearr_poly[:-4]+'.bil'

    if not os.path.isfile(maskfilename[:-4]+'_clip.bil'):

        # create an array to fill
        mask_arr = np.zeros(basearr.shape)
       
        radagaster = rasterio.open(ref_raster)

        # open the extent raster as a reference
        with Raster(ref_raster) as raster_obj: 
            # open the survey shapefile
            with fiona.open(basearr_poly) as f: 
                # for each polygon
                for m in range(len(f)):
                    print (m , '/', len(f))
                    geom = f[m]['geometry'] # the polygon geometry

                    # create a True/False mask of the dimensions of the extent raster
                    # True means the presence of a feature in the concerned layer
                    polygon_mask = rasterio.features.geometry_mask(geometries=[geom], out_shape=(raster_obj.shape[0], raster_obj.shape[1]), transform=raster_obj.affine, all_touched=True, invert=True)

                    # these are the pixels touched by the polygon
                    where = np.where(polygon_mask == True)

                    # attribute the code value
                    mask_arr[where] = int(f[m]['id'])+1

        with rasterio.open(maskfilename,'w', driver='ENVI', width=radagaster.width, height=radagaster.height, count=1, crs=radagaster.crs,transform=radagaster.transform, dtype=radagaster.dtypes[0]) as raster:
                raster.write(mask_arr,1) 
        raster.close()
        radagaster.close()

        #Clip the mask
        #define the GDAL options
        options = gdal.WarpOptions(cutlineDSName=aoi, cropToCutline=True, dstSRS='EPSG:' + str(EPSG))
        outBand = gdal.Warp(srcDSOrSrcDSTab=maskfilename,destNameOrDestDS=maskfilename[:-4]+'_clip.bil', options=options); outBand= None


    return maskfilename[:-4]+'_clip.bil'

    
     
 

###################################################################################
###################################################################################
def make_vegraster (basearr, ref_raster, veg_shpfiles, vegdates, aoi, EPSG):
    print ('making vegetation rasters')
    vegdict = {}

    veg_arr = np.zeros(basearr.shape)

    cnt = 0
    for f in veg_shpfiles:
        print ('rasterising', f)

        if f.endswith ('.bil'):
            f_clipped = f[:-4]+'_clipped.bil'
            if not os.path.isfile(f_clipped):
                f_clipped = clip_bilfile(f, aoi, EPSG)
                # Link the new raster files to the date of the associated image
            vegdict[vegdates[cnt]] = f_clipped

        else:
            #Make sure we are all in the same EPSG
            f_goodepsg = reproject_vectorfile(f, EPSG)

            if not os.path.isfile(f_goodepsg[:-4]+'_clip.bil'):

                # create an array to fill
                veg_arr = np.zeros(basearr.shape)

                radagaster = rasterio.open(ref_raster)

                # open the extent raster as a reference
                with Raster(ref_raster) as raster_obj: 
                    # open the survey shapefile
                    with fiona.open(f_goodepsg) as f: 
                        # for each polygon
                        for m in range(len(f)):
                            print (m , '/', len(f))
                            geom = f[m]['geometry'] # the polygon geometry

                            # create a True/False mask of the dimensions of the extent raster
                            # True means the presence of a feature in the concerned layer
                            polygon_mask = rasterio.features.geometry_mask(geometries=[geom], out_shape=(raster_obj.shape[0], raster_obj.shape[1]), transform=raster_obj.affine, all_touched=False, invert=True)

                            # these are the pixels touched by the polygon
                            where = np.where(polygon_mask == True)

                            # attribute the code value
                            veg_arr[where] = 1

                newname = f_goodepsg[:-4]+'.bil'
                with rasterio.open(newname,'w', driver='ENVI', width=radagaster.width, height=radagaster.height, count=1, crs=radagaster.crs,transform=radagaster.transform, dtype=radagaster.dtypes[0]) as raster:
                        raster.write(veg_arr,1) 
                raster.close()
                radagaster.close()

                #define the GDAL options
                options = gdal.WarpOptions(cutlineDSName=aoi, cropToCutline=True, dstSRS='EPSG:' + str(EPSG))

                #Clip the mask
                outBand = gdal.Warp(srcDSOrSrcDSTab=newname,destNameOrDestDS=newname[:-4]+'_clip.bil', options=options); outBand= None


            # Link the new raster files to the date of the associated image
            vegdict[vegdates[cnt]] = f_goodepsg[:-4]+'_clip.bil'

        cnt += 1


    print (vegdict)
    return vegdict

    
 



################################################################################
################################################################################
def preprocess(baseraster, aoi, vegfiles, vegdates, maskdir, nomask_val, EPSG, zonedir = None, ooi = [''], tideparams = None):

    #convert aoi from geojson to shp if needed and put in the right epsg
    shp_trans=reproject_vectorfile(aoi, EPSG)

    #load a template raster
    rgbn = rasterio.open(baseraster)
    basearr = rgbn.read(1)

    #make the zones from polygons
    zoneraster = make_zoneraster (basearr, baseraster, zonedir, shp_trans, EPSG)

    #make a mask from polygons
    maskraster = make_maskraster (basearr, baseraster, maskdir, zoneraster, shp_trans, EPSG)


    #make the vegetation rasters
    vegdict = make_vegraster (basearr, baseraster, vegfiles, vegdates, shp_trans, EPSG)
    
    #make the ooi rasters
    width = rgbn.width
    height = rgbn.height
    transform = rgbn.transform

    ooi_rasters = []
    if ooi != ['']:
        for ooi_xyz in ooi:
            if ooi_xyz.endswith('.bil'):
                # just clip it
                ooi_raster = clip_bilfile(ooi_xyz, shp_trans, EPSG)
                ooi_rasters.append(ooi_raster)
            else:
                ooi_raster = grid_xyz(ooi_xyz, baseraster, shp_trans, EPSG, transform, width, height)
                ooi_rasters.append(ooi_raster)
    
    #make the tide data
    if tideparams != None:
        print (tideparams)
        tidelist = []
        for k in tideparams.keys():
            print (k)
            tidedf = stations2panda([k], [tideparams[k]])
            tidelist.append(tidedf)

    return maskraster, zoneraster, vegdict, ooi_rasters, tidelist





################################################################################
################################################################################
def clip_bilfile(mytif, cutline, EPSG):
    print ('correcting shift and clipping...')
    options = gdal.WarpOptions(cutlineDSName=cutline,cropToCutline=True, dstSRS='EPSG:' + str(EPSG), format = 'ENVI')
    mybil = mytif[:-4]+'_clip.bil'
    outBand = gdal.Warp(srcDSOrSrcDSTab=mytif,destNameOrDestDS=mybil, options=options)
    outBand= None

    return mybil




################################################################################
################################################################################
def stations2panda(allstations_dirlist, tide_stations):
    counter = 0
    for stationdir in allstations_dirlist: # for each station
        files = sorted(os.listdir(stationdir))

        # Combine the yearly files
        for f in range(len(files)):
            if f == 0:
                data = pd.read_csv(stationdir+files[f], sep = ';', header = 1)
            else:
                temp_data = pd.read_csv(stationdir+files[f], sep = ';', header = 1)
                data = data.append(temp_data, ignore_index = True)

        # keep only necessary data
        istide = False; iswind = False
        for t in tide_stations:
            if files[f].startswith(t) and not files[f].startswith('chioggiaporto'):
                istide = True; break
        #for w in wind_stations:
        #    if files[f].startswith(w):
        #        iswind = True; break

        if istide is True:
            data = data[['Data_ora_solare', 'Livello_(m)']]
            data = data.rename(columns={"Livello_(m)":'Livello_(m)_'+t})
        elif iswind is True:
            data = data[['Data_ora_solare', 'Direzione_vento_(GN)', 'Velocita_vento_(m/s)']]
            data = data.rename(columns={"Direzione_vento_(GN)":'Direzione_vento_(GN)_'+w, "Velocita_vento_(m/s)":'Velocita_vento_(m/s)_'+w})

        data = data.set_index('Data_ora_solare')

        if counter == 0:
            big_data = data.copy(deep = True)
        else:
            big_data = pd.concat([big_data,data], ignore_index=False, axis=1)

        counter += 1

    # Put the index in datetime format and sort
    big_data.index = pd.to_datetime(big_data.index)
    big_data = big_data.sort_index(ascending = True)

    # Delete unnecessary columns
    for c in big_data.columns.values:
        #if len([ele for ele in tide_stations+wind_stations if(ele in c)]) == 0:
        if len([ele for ele in tide_stations if(ele in c)]) == 0:
            if c in big_data.columns.values:
                big_data = big_data.drop(c, axis=1)

    # correct the wind direction
    for c in big_data.columns.values:
        if 'Direzione' in c:
            big_data[c] = 270 - big_data[c]

    # eliminate nan values
    big_data = big_data.fillna(0)

    return big_data









################################################################################
################################################################################
def grid_xyz(xyzfile, ext_raster, cutline, EPSG, transform, width, height):

    # Make a vrt
    my_vrt = make_vrt(xyzfile)

    """
    # open the reference raster
    arr, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(ext_raster)

    # get the extent coordinates. We are in 3003 here
    xs = geotransform[0]
    ys = geotransform[3]
    pxx = geotransform[1]
    pxy = geotransform[5]
    xe = xs + len(arr[0]) * pxx
    ye = ys + len(arr) * pxy
    """

    print ('gridding once for extent...')
    options = gdal.GridOptions(algorithm = 'nearest', zfield = 'Elevation', format = 'GTiff', outputSRS = 'EPSG:'+EPSG)

    my_tif = my_vrt[:-4]+'.tif'
    output = gdal.Grid(my_tif,my_vrt, options = options) 
    output = None

    # Read the first grid produced to get its extent
    raster = rasterio.open(my_tif)
    width_r = raster.width
    height_r = raster.height
    transform_r = raster.transform

    width_m = width_r * transform_r[0]
    width_new = width_m / transform[0]

    height_m = height_r * transform_r[4]
    height_new = height_m / -transform[4]


    print ('gridding twice for pixel size...')
    options = gdal.GridOptions(algorithm = 'nearest', zfield = 'Elevation', format = 'GTiff', outputSRS = 'EPSG:'+EPSG, width = width_new, height = height_new)

    my_tif2 = my_vrt[:-4]+'2.tif'
    output = gdal.Grid(my_tif2,my_vrt, options = options) 
    output = None



    #os.system('gdal_grid --config GDAL_DATA ' + gcs_path +' -ot Float32 -of GTIFF -zfield Elevation -a nearest -a_srs EPSG:3003  -txe ' + str(xs) + ' ' + str(xe) + ' -tye ' + str(ys) + ' ' + str(ye) + ' -outsize '+ str(len(arr[0]))+' '+ str(len(arr)) +' -of GTiff -ot Float64 -l '+ myname + ' ' + my_vrt + ' ' + my_tif)
    #os.system('gdal_grid -ot Float32 -of GTIFF -zfield Elevation -a nearest -a_srs EPSG:3003  -txe ' + str(xs) + ' ' + str(xe) + ' -tye ' + str(ys) + ' ' + str(ye) + ' -outsize '+ str(len(arr[0]))+' '+ str(len(arr)) +' -of GTiff -ot Float64 -l '+ myname + ' ' + my_vrt + ' ' + my_tif)


    # Now get this into a bil file
    # PB how correct the systematic gdalwarp shift? Specify the  target extent!
    print ('correcting shift and clipping...')
    options = gdal.WarpOptions(cutlineDSName=cutline,cropToCutline=True, dstSRS='EPSG:' + str(EPSG), format = 'ENVI')

    my_bil = my_tif[:-4]+'_clip.bil'
    outBand = gdal.Warp(srcDSOrSrcDSTab=my_tif2,destNameOrDestDS=my_bil, options=options)
    outBand= None


    #os.system('gdalwarp --config GDAL_DATA ' + gcs_path + ' -overwrite -of ENVI -cutline '+ cutline + ' -crop_to_cutline -te ' +  str(xs) + ' ' + str(ye) + ' ' + str(xe) + ' ' +  str(ys) + ' ' + my_tif + ' ' + my_bil )

    return my_bil


################################################################################
################################################################################
def make_vrt(xyzfile):

    my_csv = xyzfile[:-4]+'.csv'
    my_vrt = xyzfile[:-4]+'.vrt'

    path_to = os.sep.join(xyzfile.split(os.sep)[:-1])+os.sep
    name = xyzfile.split(os.sep)[-1][:-4]

    A = []
    # Open the .xyz file
    f = open(xyzfile, "r", encoding = 'utf-8')
    for line in f :
        A.append([np.float(line.split()[0]), np.float(line.split()[1]), np.float(line.split()[2])])
    f.close()
    A = np.asarray(A)

    # make a .csv file with pandas
    Apd = pd.DataFrame({'Easting':A[:,0], 'Northing':A[:,1], 'Elevation':A[:,2]})
    Apd.to_csv(my_csv)

    # Create the corresponding .vrt for gridding with gdal_grid
    # make a .xyz file
    f = open(my_vrt, "w", encoding = 'utf-8')
    f.write('<OGRVRTDataSource>' + '\n')
    f.write('    <OGRVRTLayer name="'+ name +'">' + '\n')
    f.write('        <SrcDataSource>'+my_csv+'</SrcDataSource>' + '\n')
    f.write('        <GeometryType>wkbPoint</GeometryType>' + '\n')
    f.write('        <GeometryField encoding="PointFromColumns" x="Easting" y="Northing" z="Elevation"/>' + '\n')
    f.write('    </OGRVRTLayer>' + '\n')
    f.write('</OGRVRTDataSource>' + '\n')
    f.close()

    return my_vrt





################################################################################
################################################################################
def grid_shp(mydir, myname, ext_raster, cutline):


    my_shp = mydir +myname+ ".shp"
    my_tif = mydir + myname + '_nearest.tif'
    my_bil = my_tif[:-4]+'_laguna3003.bil'

    # open the reference raster
    arr, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(ext_raster)

    # get the extent coordinates. We are in 3003 here
    xs = geotransform[0]
    ys = geotransform[3]
    pxx = geotransform[1]
    pxy = geotransform[5]
    xe = xs + len(arr[0]) * pxx
    ye = ys + len(arr) * pxy


    if not os.path.isfile(my_bil):
        print ('gridding ...')
        os.system('gdal_grid -ot Float32 -of GTIFF -zfield Elevation -a nearest -a_srs EPSG:3003  -txe ' + str(xs) + ' ' + str(xe) + ' -tye ' + str(ys) + ' ' + str(ye) + ' -outsize '+ str(len(arr[0]))+' '+ str(len(arr)) +' -of GTiff -ot Float64 -l '+ myname + ' ' + my_shp + ' ' + my_tif)

        # Now get this into a bil file
        # PB how correct the systematic gdalwarp shift? Specify the  target extent!
        print ('correcting shift ...')
        os.system('gdalwarp --config GDAL_DATA ' + gcs_path + ' -overwrite -of ENVI -cutline '+ cutline + ' -crop_to_cutline -te ' +  str(xs) + ' ' + str(ye) + ' ' + str(xe) + ' ' +  str(ys) + ' ' + my_tif + ' ' + my_bil )







##################################################################################
##################################################################################
def grid_zone_shp (zonefile, aoi, EPSG, basearr):

    # reproject zone shapefile
    shp3003 = reproject_shp(zonefile, EPSG)

    # open the extent raster as a reference
    with Raster(ref_raster) as raster_obj: 
        # open the survey shapefile
        with fiona.open(shp3003) as f: 
            # for each polygon
            for m in range(len(f)):
                print (m , '/', len(f))
                value = f[m]['properties']['id']
                geom = f[m]['geometry'] # the polygon geometry

                # create a True/False mask of the dimensions of the extent raster
                # True means the presence of a feature in the concerned layer
                polygon_mask = rasterio.features.geometry_mask(geometries=[geom], out_shape=(raster_obj.shape[0], raster_obj.shape[1]), transform=raster_obj.affine, all_touched=True, invert=True)

                # these are the pixels touched by the polygon
                where = np.where(polygon_mask == True)

                # attribute the code value
                basearr[where] = int(value)

    # save the new raster
    ENVI_raster_binary_from_2d_array((geotransform, inDs), basearr_dir+"Inlets_AoI.bil", pixelWidth, basearr)         
    return basearr, basearr_dir+"Inlets_AoI.bil"



###################################################################################
###################################################################################
def makezonearr (basearr, zone_shp, ref_raster, geotransform, inDs, pixelWidth, basearr_dir):

    # reproject shapefiles
    shp3003 = reproject_shp(zone_shp, 3003)
    print (shp3003)

    # open the extent raster as a reference
    with Raster(ref_raster) as raster_obj: 
        # open the survey shapefile
        with fiona.open(shp3003) as f: 
            # for each polygon
            for m in range(len(f)):
                print (m , '/', len(f))
                value = f[m]['properties']['id']
                geom = f[m]['geometry'] # the polygon geometry

                # create a True/False mask of the dimensions of the extent raster
                # True means the presence of a feature in the concerned layer
                polygon_mask = rasterio.features.geometry_mask(geometries=[geom], out_shape=(raster_obj.shape[0], raster_obj.shape[1]), transform=raster_obj.affine, all_touched=True, invert=True)

                # these are the pixels touched by the polygon
                where = np.where(polygon_mask == True)

                # attribute the code value
                basearr[where] = int(value)

    # save the new raster
    ENVI_raster_binary_from_2d_array((geotransform, inDs), basearr_dir+"Inlets_AoI.bil", pixelWidth, basearr)         
    return basearr, basearr_dir+"Inlets_AoI.bil"












################################################################################
################################################################################
def ENVI_raster_binary_to_3d_array(file_name):
    """
    This function transforms a raster into a numpy array.

    Args:
        file_name (ENVI raster): the raster you want to work on.
        gauge (string): a name for your file

    Returns:
        image_array (2-D numpy array): the array corresponding to the raster you loaded
        pixelWidth (geotransform, inDs) (float): the size of the pixel corresponding to an element in the output array.

    Source: http://chris35wills.github.io/python-gdal-raster-io/
    """


    driver = gdal.GetDriverByName('ENVI')

    driver.Register()

    inDs = gdal.Open(file_name, GA_ReadOnly)

    if inDs is None:
        print ("Couldn't open this file: " + file_name)
        print ("Perhaps you need an ENVI .hdr file? ")
        sys.exit("Try again!")
    else:
        print ("%s opened successfully" %file_name)

        #print '~~~~~~~~~~~~~~'
        #print 'Get image size'
        #print '~~~~~~~~~~~~~~'
        cols = inDs.RasterXSize
        rows = inDs.RasterYSize
        bands = inDs.RasterCount

        #print "columns: %i" %cols
        #print "rows: %i" %rows
        #print "bands: %i" %bands

        #print '~~~~~~~~~~~~~~'
        #print 'Get georeference information'
        #print '~~~~~~~~~~~~~~'
        geotransform = inDs.GetGeoTransform()
        originX = geotransform[0]
        originY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]

        #print "origin x: %i" %originX
        #print "origin y: %i" %originY
        #print "width: %2.2f" %pixelWidth
        #print "height: %2.2f" %pixelHeight

        # Set pixel offset.....
        #print '~~~~~~~~~~~~~~'
        #print 'Convert image to 2D array'
        #print '~~~~~~~~~~~~~~'

        for n in range(inDs.RasterCount):
            band = inDs.GetRasterBand(n+1)
            if n == 0:
                arr = band.ReadAsArray(0, 0, cols, rows)
            else:
                temp_arr = band.ReadAsArray(0, 0, cols, rows)
                arr = np.dstack((arr,temp_arr))


        
        #band = inDs.GetRasterBand(1)
        #print band
        #image_array = band.ReadAsArray(0, 0, cols, rows)
        image_array_name = file_name
        #print type(image_array)
        #print image_array.shape

        return arr, pixelWidth, (geotransform, inDs)




################################################################################
################################################################################
def ENVI_raster_binary_from_2d_array(envidata, file_out, post, image_array):
    """
    This function transforms a numpy array into a raster.

    Args:
        envidata: the geospatial data needed to create your raster
        file_out (string): the name of the output file
        post: coordinates for the goegraphical transformation
        image_array (2-D numpy array): the input raster

    Returns:
        new_geotransform
        new_projection: the projection in which the raster
        file_out (ENVI raster): the raster you wanted

    Source: http://chris35wills.github.io/python-gdal-raster-io/
    """

    driver = gdal.GetDriverByName('ENVI')

    original_geotransform, inDs = envidata

    rows, cols = image_array.shape
    bands = 1

    # Creates a new raster data source
    outDs = driver.Create(file_out, cols, rows, bands, gdal.GDT_Float32)

    # Write metadata
    originX = original_geotransform[0]
    originY = original_geotransform[3]

    outDs.SetGeoTransform([originX, post, 0.0, originY, 0.0, -post])
    outDs.SetProjection(inDs.GetProjection())

    #Write raster datasets
    outBand = outDs.GetRasterBand(1)
    outBand.WriteArray(image_array)

    new_geotransform = outDs.GetGeoTransform()
    new_projection = outDs.GetProjection()

    print ("Output binary saved: ", file_out)
    return new_geotransform,new_projection,file_out


###################################################################################
###################################################################################
def reproject_shp(shp, EPSG):
    shapes = gpd.read_file(shp)
    shapes_goodcrs = shapes.to_crs({'init': 'epsg:'+ str(EPSG)}) #
    shapes_goodcrs.to_file(driver = 'ESRI Shapefile', filename = shp[:-4]+"_"+str(EPSG)+".shp")
    return shp[:-4]+"_"+str(EPSG)+".shp"









                