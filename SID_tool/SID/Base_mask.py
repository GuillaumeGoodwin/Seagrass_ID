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
from rasterstats.io import Raster

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import scikitplot as skplt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
import seaborn as sn
from sklearn.metrics import confusion_matrix


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


###################################################################################
###################################################################################
def make_basearr (basearr, ref_raster, shp_dict, geotransform, inDs, pixelWidth, basearr_dir):
	for shp in shp_dict:
		# reproject shapefiles
		shp3003 = reproject_shp(shp, 3003)
		print (shp3003)

		# open the extent raster as a reference
		with Raster(ref_raster) as raster_obj: 
		    # open the survey shapefile
		    with fiona.open(shp3003) as f: 
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
		            basearr[where] = shp_dict[shp]

		            #if m >= 100:
		            #	break


	# save the new raster
	ENVI_raster_binary_from_2d_array((geotransform, inDs), basearr_dir+"basemap.bil", pixelWidth, basearr)         
	return basearr, basearr_dir+"basemap.bil"







###################################################################################
###################################################################################
# First set up the directory structure

basemap_dir = "/home/willgoodwin/PostDoc/Padova/Data/Basemaps/"
urban_shp = basemap_dir + "aree_urbane.shp"
emersed_shp = basemap_dir + "cb13_terre_emerse.shp"
naturalmarsh_shp = basemap_dir + "cb13_shp065barnat_10.shp"
artificialmarsh_shp = basemap_dir + "cb13_strutture_morfologiche_a_barena.shp"
canals_shp = basemap_dir + "cb13_canali_lagunari.shp"
adriatic_shp = basemap_dir + "mare_adriatico.shp"


outline_dir ="/home/willgoodwin/PostDoc/Padova/Data/Outlines/"
laguna_outline = outline_dir + "Laguna_outline3003.shp"


gcs_path = "/home/willgoodwin/Software/anaconda3/envs/coastalsat/lib/python3.7/site-packages/fiona/gdal_data/"

###################################################################################
###################################################################################

basearr_code_dict = {adriatic_shp:6, emersed_shp: 1, urban_shp: 2, naturalmarsh_shp: 3, artificialmarsh_shp: 4, canals_shp:5}

ext_dir = "/home/willgoodwin/PostDoc/Padova/Data/VL_Bathymetry/"
ext_raster = ext_dir + "bathy_2012_nearest_laguna3003.bil"
ext_arr, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(ext_raster)

###################################################################################
###################################################################################

basearr = np.copy(ext_arr)
basearr[basearr == 0.] = -9999.
basearr[basearr > -9999.] = 0

basearr, basemap_raster = make_basearr (basearr, ext_raster, basearr_code_dict, geotransform, inDs, pixelWidth, basemap_dir)


# get the extent coordinates. We are in 3003 here
xs = geotransform[0]
ys = geotransform[3]
pxx = geotransform[1]
pxy = geotransform[5]
xe = xs + len(ext_arr[0]) * pxx
ye = ys + len(ext_arr) * pxy


print (ext_arr.shape)
print (basearr.shape)


os.system('gdalwarp --config GDAL_DATA ' + gcs_path + ' -of ENVI -t_srs EPSG:3003 ' + ' -cutline '+ laguna_outline + ' -crop_to_cutline -te ' +  str(xs) + ' ' + str(ye) + ' ' + str(xe) + ' ' +  str(ys) + ' -overwrite -srcnodata -9999 -dstnodata -9999 ' + basemap_raster + ' ' + basemap_raster[:-4]+'_laguna.bil' )

#os.system('gdalwarp --config GDAL_DATA ' + gcs_path + ' -of ENVI -t_srs EPSG:3003 -tap' + ' -cutline '+ laguna_outline + ' -crop_to_cutline -tr ' +  str(pxx) + ' ' + str(pxy) + ' -overwrite -srcnodata -9999 -dstnodata -9999 -te ' +  str(xs) + ' ' + str(ye) + ' ' + str(xe) + ' ' +  str(ys) + ' ' + basemap_raster + ' ' + basemap_raster[:-4]+'_laguna.bil' )




quit()









                