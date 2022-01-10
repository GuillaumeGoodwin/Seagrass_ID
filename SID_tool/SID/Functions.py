################################################################################
################################################################################
"""Import Python packages"""
################################################################################
################################################################################

import os
import glob
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
from matplotlib.patches import Rectangle
import math
from itertools import permutations

from osgeo import gdal, ogr, osr
from shapely.geometry import Polygon
import rasterio
import fiona
#from rasterstats.io import Raster

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
#import scikitplot as skplt
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
#import seaborn as sn
from sklearn.metrics import confusion_matrix

from PIL import Image, ImageFilter, ImageOps
from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#import Lee_model as lee
#import ML_functions as ml


# Protect these values for now
Density_vals = [1,2,3,4]
Density_vals = [0,1,2,3,4]
Zone_vals = [1,2,3,4,5]
Surv_idx = list([39,67,71,133])
Model_idx = list((0,1,2,3))






"""
#Collect the right bands (rgbn + angles?) and save to image
granule10_dir = workdir+unzipped+'/GRANULE/'
granule10_dir = granule10_dir + os.listdir(granule10_dir)[0] + '/IMG_DATA/R10m/'

tile_id = zipped.split('_')[5]
date_id = zipped.split('_')[2]

blue = rasterio.open(granule10_dir+'/'+tile_id+'_'+date_id+'_B02_10m.jp2')
green = rasterio.open(granule10_dir+'/'+tile_id+'_'+date_id+'_B03_10m.jp2') 
red = rasterio.open(granule10_dir+'/'+tile_id+'_'+date_id+'_B04_10m.jp2')
nir = rasterio.open(granule10_dir+'/'+tile_id+'_'+date_id+'_B08_10m.jp2')


with rasterio.open(granule10_dir+'rgbn.bil','w',driver='ENVI', width=blue.width, height=blue.height, count=4, crs=blue.crs,transform=blue.transform, dtype=blue.dtypes[0]) as rgb:
    rgbn.write(blue.read(1),4) 
    rgbn.write(green.read(1),3) 
    rgbn.write(red.read(1),2) 
    rgbn.write(nir.read(1),1) 
rgbn.close()


#Clip the .bil files
rgbn = rasterio.open(granule10_dir+'rgbn.bil')

options = gdal.WarpOptions(cutlineDSName='Chioggia_test.geojson',cropToCutline=True)
outBand = gdal.Warp(srcDSOrSrcDSTab=granule10_dir+'rgb.bil',destNameOrDestDS=granule10_dir+'rgbn_clip.bil',options=options)
outBand= None
"""




###################################################################à
###################################################################à
def read_paramfile(this_dir, paramfile):
    param_open = open(this_dir+'/'+paramfile, 'r')
    param_lines = param_open.read().splitlines()
    
    for l in range(1, len(param_lines) ):
        
        if param_lines[l-1].startswith('#aoi'):
            aoi = param_lines[l]
        elif param_lines[l-1].startswith('#EPSG'):
            EPSG = param_lines[l]
        elif param_lines[l-1].startswith('#Mask_file'):
            maskfile = param_lines[l] 
        elif param_lines[l-1].startswith('#Transparent_mask_value'):
            nomask_val = param_lines[l] 
        elif param_lines[l-1].startswith('#Zone_file'):
            zonefile = param_lines[l] 
        elif param_lines[l-1].startswith('#Image_data_directory'):
            imgdir = param_lines[l]
        elif param_lines[l-1].startswith('#base_raster'):
            baseraster = param_lines[l] 
        elif param_lines[l-1].startswith('#Out_of_image_data_files'):
            ooi = param_lines[l].split(' ')
        elif param_lines[l-1].startswith('#vegetation_dates'):
            vegdates = param_lines[l].split(' ')
        elif param_lines[l-1].startswith('#vegetation_shp'):
            vegdf = param_lines[l].split(' ')
        elif param_lines[l-1].startswith('#tide_data'):
            tidetemp = param_lines[l].split(' ') 
            print (tidetemp)
            tidedict = {}
            for i in range(0,len(tidetemp),2):
                tidedict[tidetemp[i]] = tidetemp[i+1]

    param_open.close()

    print (ooi)
    print (vegdates)
    print (vegdf)

    print(tidedict)


    return aoi, EPSG, maskfile, nomask_val, zonefile, imgdir, baseraster, ooi, vegdates, vegdf, tidedict




######################################################################
######################################################################
def add_remove_stats(diffmap,diffmapfil, show = False):

    if show == True:
    
        fig = plt.figure(0, facecolor='White',figsize=[13, 7])
        ax1 = plt.subplot2grid((1,2), (0,0), rowspan=1, colspan=1)
        ax2 = plt.subplot2grid((1,2), (0,1), rowspan=1, colspan=1)

        #ax1.imshow(diffmap,cmap = plt.cm.seismic, vmin = -3, vmax = 3)
        #ax2.imshow(diffmapfil,cmap = plt.cm.seismic, vmin = -3, vmax = 3)

        ax1.imshow(diffmap,cmap = plt.cm.viridis, vmin = 0, vmax = 3)
        ax2.imshow(diffmapfil,cmap = plt.cm.viridis, vmin = 0, vmax = 3)
        plt.show()
            


    tot = len(np.where(diffmap != 0)[0])
    #print (tot)

    Correct_added = diffmapfil[np.logical_and(diffmap == 0, diffmapfil > 0)]
    Incorrect_added = diffmapfil[np.logical_and(diffmap == 0, diffmapfil < 0)]

    Correct_removed = diffmap[np.logical_and(diffmapfil == 0, diffmap > 0)]
    Incorrect_removed = diffmap[np.logical_and(diffmapfil == 0, diffmap < 0)]

    return len(Correct_added)/tot, len(Incorrect_added)/tot, len(Correct_removed)/tot, len(Incorrect_removed)/tot







######################################################################
######################################################################
def make_edge_contrast (classif, band, mask, zones):
    # make survey edges
    classif_edge = make_classif_edges_arr (classif)
    classif_edge = np.ma.masked_where(mask != 0, classif_edge)
    classif_edge = np.ma.masked_where(zones < 1, classif_edge)

    ##############################################################
    # Make the contrast metric
    edge_contrast = np.zeros(kso.shape, dtype = np.float32)
    

    start = datetime.now()
    for r in range(len(kso)):
    #for r in range(400,600):
        print (r)
        for c in range(len(kso[0])):
            if mask[r,c] == 0 and arr[r,c] >= 0:
                # Make a kernel
                kernel, k_indices, indices, k_centre = fn.kernel_circle (arr, 9, r, c)

                # Make contrast
                contrast[r,c] = np.nanstd(kernel)

    contrast = np.ma.masked_where(mask != 0, contrast)
    contrast = np.ma.masked_where(arr <= 0, contrast)

    timer = datetime.now()-start
    print(timer)

    Contrast.append(contrast)




######################################################################
######################################################################
def count_seasons (pred_times):

    seasons_4 = np.zeros(len(pred_times), dtype = np.uint16)
    seasons_2 = np.zeros(len(pred_times), dtype = np.uint16)

    summer_months = [7,8,9]
    autumn_months = [10,11,12]
    winter_months = [1,2,3]
    spring_months = [4,5,6]

    growth_months = [4,5,6,7,8,9]
    decay1_months = [10,11,12]
    decay2_months = [1,2,3]

    for i in range(len(pred_times)):

        if pred_times[i].month in winter_months: S = 1
        if pred_times[i].month in spring_months: S = 2
        if pred_times[i].month in summer_months: S = 3
        if pred_times[i].month in autumn_months: S = 4

        seasons_4[i] = 10*pred_times[i].year + S

        if pred_times[i].month in growth_months:
            S = 1; seasons_2[i] = 10*pred_times[i].year + S
        elif pred_times[i].month in decay1_months:
            S = 2; seasons_2[i] = 10*pred_times[i].year + S
        elif pred_times[i].month in decay2_months: 
            S = 2; seasons_2[i] = 10*(pred_times[i].year-1) + S

    return seasons_4, seasons_2




######################################################################
######################################################################
def Surface_outline (Outline_array, Surface_array, Outline_value, Nodata_value):
    """
    This function calculates the inner outline of a surface within an array (Surface_array) where the element to outline has a value of 1, and stores that outline in a second array (Outline_array) under the value Outline_value.
    Args:
        Surface_array (2D numpy array): a 2-D array containing the surface to outline with the value 1. Undesirable elements have the value 0 or Nodata_value.
        Outline_array (2D numpy array): a 2-D array destined to store the outline.
        Outline_value (float): The value to be given to outline cells
        Nodata_value (float): The value for empty cells

    Returns:
        Outline_array (2D numpy array): a 2-D array populated with the outline cells.

    Author: GCHG
    """

    Surface_array[Surface_array > 0. ] = 1
    Inside = np.where(Surface_array == 1)

    for i in range(len(Inside[1])):
        x = Inside[0][i]; y = Inside[1][i]
        K, Kx, Ky = kernel (Surface_array, 3, x, y)

        if np.count_nonzero(K) <=  K.size-1 :
            Outline_array[x, y] = Outline_value

    """ THIS PART IS CURRENTLY USELESS.
    for i in range(len(Outline[0])):
        x = Outline[0][i]; y = Outline[1][i]
        K_r, Kx_r, Ky_r = kernel (Surface_array, 3, x, y)
        K, Kx, Ky = kernel (Twin, 3, x, y)
        if np.sum(K_r) == 0:
            Outline_array[x, y] = 0"""

    return Outline_array, Outline_value


######################################################################
######################################################################
def calculate_slope (DEM, DEM_file):
    ml.ENVI_raster_binary_from_2d_array(geodata, DEM_file, pxw, DEM)
    gdal.DEMProcessing('slope.tif', DEM_file, 'slope')
    with rasterio.open('slope.tif') as dataset:
        slope=dataset.read(1)
    return slope

######################################################################
######################################################################
def buffer_patch_edges (arr, patch_idx, patch_value):
    D = 3
    for i in range(len(patch_idx[0])):
        K, k_idx, idx, k_centre = ml.kernel_circle (arr, D, patch_idx[0][i], patch_idx[1][i])
        # find the ones that are not full
        max_total = len(np.where(np.isnan(K)==False)[0]) * K[k_centre[0], k_centre[1]]
        if np.nansum(K) < max_total:
            # and fill them
            arr[idx] = patch_value #K[k_centre[0], k_centre[1]]
    return arr





#########################################################################
#########################################################################
def expand_patch (param, patch_id, ini_patch, proba, zones, zonevals):
    # Find the areas in the buffer that have no veg, high proba and low proba slope
    # Include these areas in the patch

    # Examine and modify proba inside your buffer
    # Question: do you boost proba or do you allow the patch to expand into lower proba?
    # For now we allow the model to expand into lower probas
    # remember this is a model of vegetative expansion.

    rp = patch_id.ravel()

    # Make a buffer
    ###########################
    mybuffer = np.copy(patch_id)

    border_col = 255*np.ones((len(patch_id),1))
    border_row = 255*np.ones((1,len(patch_id[0])))

    myedges_c = patch_id[:,:-1] - patch_id[:,1:]
    myedges_c2 = patch_id[:,1:] - patch_id[:,:-1]
    myedges_r = patch_id[:-1,:] - patch_id[1:,:]
    myedges_r2 = patch_id[1:,:] - patch_id[:-1,:]


    myedges_c = np.concatenate((border_col, myedges_c), axis=1)
    myedges_r = np.concatenate((border_row, myedges_r), axis=0)    
    myedges_c2 = np.concatenate((myedges_c2, border_col), axis=1)
    myedges_r2 = np.concatenate((myedges_r2, border_row), axis=0)


    patchedges = 255*np.ones(patch_id.shape)
    patchedges[myedges_c == 0] = 0
    patchedges[myedges_r == 0] = 0
    patchedges[myedges_c2 == 0] = 0
    patchedges[myedges_r2 == 0] = 0


    
    for p in np.arange(1,np.amax(patch_id)+1):

        """
        Find a way to only look at the patch edges and not at the inside!
        It doesn't matter if the edge doesn't include diagonals because the kernel does!
        """
        patch_idx = np.where(np.logical_and(patch_id == p,patchedges == 0))


        """
        fig = plt.figure(0, facecolor='White',figsize=[23, 20])
        ax = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)
        #ax.hist(proba[patch_idx])
        ax.imshow(patch_id)
        plt.show()
        quit()
        """
        
        mybuffer = buffer_patch_edges(mybuffer, patch_idx, p)
    mybuffer[proba == 255] = 0

    #################################################
    for p in np.arange(1,np.amax(mybuffer)+1):
        for z in zonevals:

            ri = ini_patch.ravel()
            rb = mybuffer.ravel()
            rp = patch_id.ravel()
            rq = proba.ravel()
            rz = zones.ravel()

            # the threshold is defined relative to the initial patch stats
            # This is to avoid runaway patch expansion, but maybe that could be controlled with slope
            if len(rq[ri == p])>0:
                threshold = param*np.percentile(rq[ri == p],1)
            else:
                threshold = 100

            # Maybe also set a minimum threshold

            # The slope threshold needs to be refined as well to avoid large drops
            # Think about how this works
            # Maybe finding a way of defining the drop from a high place non-locally would be faster
            # Because this is essentially what we are doing: finding non-local ways to describe local properties.

            w1 = np.zeros(rb.shape, dtype = int); w1[rb == p] = 1 # it's in the buffer
            w2 = np.zeros(rb.shape, dtype = int); w2[rp == 0] = 1 # it's not in the patch
            w3 = np.zeros(rb.shape, dtype = int); w3[rq >= threshold] = 1 #  high proba
            w4 = np.zeros(rb.shape, dtype = int); w3[rz == z] = 1 # has high proba
            wsum = w1+w2+w3+w4
            W = np.where(wsum == 4)[0]

            rp[W] = p # attribute value

    P = rp.reshape(patch_id.shape)

    return P, mybuffer









############################################################################################
############################################################################################
def combine_model_probas (weights, probarray, zones, mask, Sepvals):
    # This function averages the votes of the 4 models according to the input WEIGHTS, in the zones defined in Z_LIST

    # The sum of all weights must always be equal to the number of models used
    # e.g. if I use models 1 and 4, then the sum must be 2
    n_models = np.count_nonzero(np.asarray(weights))
    total_weight = np.sum(weights)
    for i in range(len(weights)):
        weights[i] = n_models*weights[i]/total_weight

    combo = 255*np.ones(probarray.shape[1:], dtype = np.uint8)
    combosep = [0,0,0,0,0]

    for z in range(1,6):
        # Mix the Sepvals
        # Avg
        combosep[z-1] = (weights[0] * Sepvals[0][z-1] + weights[1] * Sepvals[1][z-1] + weights[2] * Sepvals[2][z-1] + weights[3] * Sepvals[3][z-1]) / n_models

        # Median
        """
        combosep[z-1] = np.uint8(
                np.percentile(
                    np.asarray(
                        [weights[0] * Sepvals[0][z-1],
                        weights[1] * Sepvals[1][z-1],
                        weights[2] * Sepvals[2][z-1],
                        weights[3] * Sepvals[3][z-1]]) , 50, axis = 0))
        """

        # Mix the proba values
        w = np.where(zones == z)
        for t in range(0,len(probarray[0])):
            # Average
            #combo[t,w[0],w[1]] = np.uint8((weights[0] * probarray[0,t,w[0],w[1]] + weights[1] * probarray[1,t,w[0],w[1]] + weights[2] * probarray[2,t,w[0],w[1]] + weights[3] * probarray[3,t,w[0],w[1]]) / n_models)

            # Median
            combo[t,w[0],w[1]] = np.uint8(
                np.percentile(
                    np.asarray(
                        [weights[0] * probarray[0,t,w[0],w[1]],
                        weights[1] * probarray[1,t,w[0],w[1]],
                        weights[2] * probarray[2,t,w[0],w[1]],
                        weights[3] * probarray[3,t,w[0],w[1]]]) , 50, axis = 0))
            
            combo[t][mask != 0] = 255
            combo[t][probarray[0,t,:,:] == 255] = 255

    return combo, combosep


###############################################################################################
###############################################################################################
def load_survey (i, prediction_scenes, pred_times, Model_scenes, Inletscenes, Digiscenes,
    dwld_dir = "/home/willgoodwin/PostDoc/Padova/Data/Landsat/" + 'Scenes/All/', maskraster = "/home/willgoodwin/PostDoc/Padova/Data/Basemaps/basemap_laguna.bil", veg_dir = "/home/willgoodwin/PostDoc/Padova/Data/VL_Vegetation/"):



    mask, pixelWidth, (geotransform, inDs) = ml.ENVI_raster_binary_to_3d_array(maskraster)
    mask = mask[:-1] # get rid of the last row of mask ... FIX THIS ANOTHER WAY! HOW????


    # Define the scene and time
    scene = prediction_scenes[i]
    time = pred_times[i]
    txt_time = str(time.year)+str(time.month).rjust(2,'0')+str(time.day).rjust(2,'0')
    #print (scene, txt_time)


    exist_survey = True
    # Here comes the tricky bit: Choosing the survey to compare to and load the aoi
    if scene in Model_scenes:
        print ('Model Scene', scene)

        # This is the vegetation state at kso_idx
        kso, kso_density, pxw, geodata = ml.load_vetted_vegetation_data(veg_dir, time, mask)

    elif scene in Inletscenes:
        print ('Inlet Scene', scene)
        kso, kso_density, pxw, geodata = ml.load_vegetation_data(veg_dir, time, mask)
        # load an additional mask for the aoi
        aoi, pixelWidth, (geotransform, inDs) = ml.ENVI_raster_binary_to_3d_array('/home/willgoodwin/PostDoc/Padova/Data/Basemaps/Inlets_AoI_laguna.bil')
        aoi = aoi[:-1]

        kso[aoi < 1] = 255
        kso_density[aoi < 1] = 255

    elif scene in Digiscenes:
        print ('Digitised Scene', scene)
        kso, kso_density, pxw, geodata = ml.load_digitised_vegetation_data(dwld_dir+scene+'/Digi/', time, mask)
        # load an additional mask for the aoi
        aoi, pixelWidth, (geotransform, inDs) = ml.ENVI_raster_binary_to_3d_array('/home/willgoodwin/PostDoc/Padova/Data/Basemaps/Digi_AoI_laguna.bil')
        aoi = aoi[:-1]

        kso[aoi < 1] = 255
        kso_density[aoi < 1] = 255

    else:
        kso = None; kso_density = None; aoi = None
        exist_survey = False

    return exist_survey, kso, kso_density






############################################################################################
############################################################################################
def proba2pred (proba, sepvals, zones, mask):

    pred = 255*np.ones(proba.shape, dtype = np.uint8)


    for z in range(1, int(np.amax(zones))):
        ws = np.where(np.logical_and(zones == z,proba >= 100*sepvals[z-1]))
        wb = np.where(np.logical_and(zones == z,proba <  100*sepvals[z-1]))
        pred[ws] = 1 
        pred[wb] = 0

    wmask = np.where(mask!=0)
    pred[wmask] = 255

    return pred





###########################################################################################
###########################################################################################
def model_mixing (weights, probarray, Sepvals, zones, mask, prediction_scenes, pred_times, Model_scenes, Inletscenes, Digiscenes):
    print ('Mixing models')

    # To make sure you don't evaluate the mixed models on themselves
    # Find the scenes not to look at
    testing_modelscenes = []
    for i in range(len(Model_scenes)):
        if weights[i] == 0.:
            testing_modelscenes.append(Model_scenes[i])


    # Apply the combo
    #timer = datetime.now()
    #Pro,Sep = combine_model_probas(weights, probarray, zones, mask, Sepvals)
    #timer = datetime.now()-timer
    #print ('combined probabilities in:', timer)

    # This is slow but it doesn't really matter because it only needs to be done once. Right?
    SS = []; PP = []; Times = []
    # Make a performance figure for all the scenes
    for i in range(len(prediction_scenes)):

        # Locate and load a survey
        exist_survey, kso, kso_index = load_survey (i, prediction_scenes, pred_times, testing_modelscenes, Inletscenes, Digiscenes)
        
        # evaluate the perf
        if exist_survey == True:


            # Apply the combo
            timer = datetime.now()
            Pro, Sep = combine_model_probas(weights, probarray[:,i:i+1,:,:], zones, mask, Sepvals)
            timer = datetime.now()-timer
            print ('combined probabilities in:', timer)

            # Make sure you don't evaluate the mixed models on themselves
            scene = prediction_scenes[i]

            # Define the prediction and proba
            proba = Pro[i]
            time = pred_times[i]

            #print ('optimise threshold Sepvals')
            #quit()

            pred = proba2pred (proba, Sep, zones, mask)
            

            # Make the difference map and performance metrics
            S, P, A, QFN , QFP = make_zonewise_perf(pred, proba, kso, zones)

            SS.append(S); PP.append(P); Times.append(time)

    SS = np.asarray(SS); PP = np.asarray(PP)

    # the optimisation things are SS and PP
    print ('Zonal performance')
    print (SS)
    print (PP)
    opt_objective = (np.mean(SS[:,-1])+ np.mean(PP[:,-1])**2)/2
    print ('Overall performance', opt_objective)

    return weights, opt_objective




##############################################################################################
##############################################################################################
def make_initial_pop(n_models, probarray, Sepvals, zones, mask, prediction_scenes, pred_times, Model_scenes, Inletscenes, Digiscenes, pop_size = 6):
    """
    This function creates a population of weights and evaluates their performance
    """
    # Make the sets to return as lists
    W = []; O = []

    # try the most obvious & simple combinations. Those are good initialisers
    weights = [1 for i in range(n_models)]
    for i in range(4-n_models):
        weights.append(0)
    perm = permutations(weights)
    W = []
    for i in list(perm):
        if list(i) not in W:
            W.append(list(i))


    for n in range(pop_size - len(W)):
        # Define the initial weights and the rules of weight attribution
        weights = [0,0,0,0]
        for i in range(n_models):
            r = int(len(weights) * np.random.rand())
            if weights[r] == 0: weights[r] = np.random.rand()
            else: weights[r-1] = np.random.rand()

        # Make sure the weights equal to n_weights
        total_weight = np.sum(weights)
        for i in range(len(weights)):
            weights[i] = n_models*weights[i]/total_weight

        W.append(weights)


    for n in range(len(W)):

        #Evaluate model
        #optimised_weights, opt_objective = model_mixing (weights, probarray, Sepvals, zones, mask, prediction_scenes, pred_times, Model_scenes, Inletscenes, Digiscenes)
        w, o = model_mixing (W[n], probarray, Sepvals, zones, mask, prediction_scenes, pred_times, Model_scenes, Inletscenes, Digiscenes)

        O.append(o)

    return W,O



##############################################################################################
##############################################################################################
def make_nth_pop(new_weights, probarray, Sepvals, zones, mask, prediction_scenes, pred_times, Model_scenes, Inletscenes, Digiscenes):
    #do a model run
    W = []; O = []
    for i in range(len(new_weights)):
        weights = new_weights[i]

        # Make sure the weights equal to n_weights
        n_models = np.count_nonzero(weights)
        total_weight = np.sum(weights)
        for i in range(len(weights)):
            weights[i] = n_models*weights[i]/total_weight

        optimised_weights, opt_objective = model_mixing (weights, probarray, Sepvals, zones, mask, prediction_scenes, pred_times, Model_scenes, Inletscenes, Digiscenes)
        W.append(optimised_weights)
        O.append(opt_objective)

    O = np.asarray(O)

    return W,O


#############################################################################################
#############################################################################################
def select_fittest(n_fittest, W, O):
    print ('Selecting', n_fittest, 'fittest individuals')

    new_weights = []; new_obj = []
    for i in range(n_fittest):
        #print (i)
        #print (O)
        #print (np.asarray(O))
        X = np.where( O == np.amax(O) )[0][0]
        new_weights.append(W[X])
        new_obj.append(O[X])
        O[X] = 0

    return new_weights, new_obj


##############################################################################################
##############################################################################################
def breed_mutate (new_weights, k):
    print ('Reproduction of solutions')

    new_weights = list(new_weights)
    for i in range(len(new_weights)):
        w = new_weights[i]
    
        # make one random one
        w_mut = []
        for j in range(len(w)):
            #sgn = np.sign( (-1)**int(100*np.random.rand()) )
            w_mut.append( np.random.rand()*w[j])
        new_weights.append(w_mut)

        # make one close
        w_mut = []
        for j in range(len(w)):
            sgn = np.sign( (-1)**int(100*np.random.rand()) )
            w_mut.append((1+sgn*k*np.random.rand())*w[j])
        new_weights.append(w_mut)
        
        # make one far
        w_mut = []
        for j in range(len(w)):
            sgn = np.sign( (-1)**int(100*np.random.rand()) )
            w_mut.append((1+sgn*2*k*np.random.rand())*w[j])
        new_weights.append(w_mut)   

        # make one very close
        w_mut = []
        for j in range(len(w)):
            sgn = np.sign( (-1)**int(100*np.random.rand()) )
            w_mut.append((1+sgn*0.2*k*np.random.rand())*w[j])
        new_weights.append(w_mut)       

    new_weights = np.asarray(new_weights)

    return new_weights


#############################################################################################
#############################################################################################
def optimise_model_mixing (n_models, probarray, Sepvals, zones, mask, prediction_scenes, pred_times, Model_scenes, Inletscenes, Digiscenes):

    # Initial set
    W,O = make_initial_pop(n_models,probarray, Sepvals, zones, mask, prediction_scenes, pred_times, Model_scenes, Inletscenes, Digiscenes)

    O_mem = 0*np.copy(O)


    L = 40
    best_perf = 0
    best_perf_diff = 1
    i = 0

    while best_perf_diff>0.001 or best_perf_diff == 0 and i < L:# or i < 20:

    #for i in range(3):
        print ('iteration', i)

        # Find the 5 best performances and use them to populate a new set of weights
        W,O = select_fittest(5, W, O)
        #print (O)

        # Now populate new_weights with mutations
        W = breed_mutate (W, 0.5)

        #print (W)

        # Run the mixing based on the new pop
        W,O = make_nth_pop(W, probarray, Sepvals, zones, mask, prediction_scenes, pred_times, Model_scenes, Inletscenes, Digiscenes)

        best_perf_diff = np.amax(O) - best_perf
        print (best_perf_diff)
        best_perf = np.amax(O)
        print (best_perf)

        i += 1
        #print (O)
        #print (O_mem)

    X = np.where(O == np.amax(O))[0][0]

    opt_weights = W[X]
    opt_obj = O[X]

    print (opt_weights)
    print ()
    print (opt_obj)
    print ('iterations:', i)

    return opt_weights, opt_obj




###################################################################################
###################################################################################
def extract_station_record (datatype, stations, stations_dir = "/home/willgoodwin/PostDoc/Padova/Data/VL_Stations/"):

    allstations_dirlist =  [f.path+'/' for f in os.scandir(stations_dir) if f.is_dir()]

    counter = 0
    for stationdir in allstations_dirlist: # for each station
        files = sorted(os.listdir(stationdir))

        for S in stations:
            if files[0].startswith(S):
                station = S
                break
        try:
            A = station
        except:
            continue

        # Combine the yearly files
        for f in range(len(files)):
            if f == 0:
                data = pd.read_csv(stationdir+files[f], sep = ';', header = 1)
            else:
                temp_data = pd.read_csv(stationdir+files[f], sep = ';', header = 1)
                data = data.append(temp_data, ignore_index = True)

        if datatype in data.columns.values:
            data = data[['Data_ora_solare', datatype]]
            data = data.rename(columns={datatype:datatype+'_'+station})
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
        if len([ele for ele in stations if(ele in c)]) == 0:
            if c in big_data.columns.values:
                big_data = big_data.drop(c, axis=1)

    return big_data




###################################################################################
###################################################################################
def make_diff (pred, survey):

    mysum = survey + pred
    mydif = survey - pred

    diffmap = np.zeros(pred.shape, dtype = np.float32)
    diffmap [mysum == 0] = 1 # TN
    diffmap [mysum == 2] = 2 # TP
    diffmap [mydif == 1] = -1 # FN
    diffmap [mydif == -1] = -2 # FP

    diffmap[survey < 0] = 0
    diffmap[survey > 1] = 0
    diffmap[pred < 0] = 0
    diffmap[pred > 1] = 0

    TN = len(np.where(diffmap == 1)[0])
    TP = len(np.where(diffmap == 2)[0])
    FN = len(np.where(diffmap == -1)[0])
    FP = len(np.where(diffmap == -2)[0])

    S = (TP+0.0000000000001) / (TP+FN+0.0000000000001)
    P = (TP+0.0000000000001) / (TP+FP+0.0000000000001)
    A = (TP+TN+0.0000000000001) / (TP+TN+FP+FN+0.0000000000001)




    return diffmap, S, P, A

###################################################################################
###################################################################################
def make_zonewise_perf (pred, proba, survey, zones):

    mask = np.ma.getmask(proba)
    survey = np.ma.masked_where(mask == True, survey)
    survey = np.ma.masked_where(proba == 255, survey)

    mysum = survey + pred
    mydif = survey - pred

    #diffmap = np.zeros(pred.shape, dtype = np.int8)
    #diffmap [mysum == 0] = 1 # TN
    #diffmap [mysum == 2] = 2 # TP
    #diffmap [mydif == 1] = -1 # FN
    #diffmap [mydif == -1] = -2 # FP

    diffmap = np.zeros(pred.shape, dtype = np.float)
    diffmap [mysum == 0] = 1. # TN
    diffmap [mysum == 2] = 2. # TP
    diffmap [mydif == 1] = -1. # FN
    diffmap [mydif == -1] = -2. # FP

    diffmap[survey < 0] = 0
    diffmap[survey > 1] = 0
    diffmap[pred < 0] = 0
    diffmap[pred > 1] = 0


    S,P,A,QFN,QFP = [],[],[],[],[]

    for z in range(1,int(np.amax(zones)+1)):

        wTN = np.where(np.logical_and(diffmap == 1,zones==z))
        wTP = np.where(np.logical_and(diffmap == 2,zones==z))
        wFN = np.where(np.logical_and(diffmap == -1,zones==z))
        wFP = np.where(np.logical_and(diffmap == -2,zones==z))

        TN = len(wTN[0])
        TP = len(wTP[0])
        FN = len(wFN[0])
        FP = len(wFP[0])

        if FN > 0:
            qFN = [np.percentile(proba[wFN],25),np.percentile(proba[wFN],50),np.percentile(proba[wFN],75)]
        else:
            qFN = [0,0,0]

        if FP > 0:
            qFP = [np.percentile(proba[wFP],25),np.percentile(proba[wFP],50),np.percentile(proba[wFP],75)]
        else:
            qFP = [0,0,0]


        s = TP / (TP+FN+0.0000000000001)
        p = TP / (TP+FP+0.0000000000001)
        a = (TP+TN) / (TP+TN+FP+FN+0.0000000000001)

        S.append(s); P.append(p); A.append(a); QFN.append(qFN); QFP.append(qFP)

    wTN = np.where(diffmap == 1)
    wTP = np.where(diffmap == 2)
    wFN = np.where(diffmap == -1)
    wFP = np.where(diffmap == -2)

    TN = len(wTN[0])
    TP = len(wTP[0])
    FN = len(wFN[0])
    FP = len(wFP[0])

    s = TP / (TP+FN+0.0000000000001)
    p = TP / (TP+FP+0.0000000000001)
    a = (TP+TN) / (TP+TN+FP+FN+0.0000000000001)

    S.append(s); P.append(p); A.append(a)

    return diffmap, S, P, A, QFN , QFP




###################################################################################
###################################################################################
def make_zonewise_metrics (diffmap, zones):


    S,P,A,QFN,QFP = [],[],[],[],[]

    for z in range(1,int(np.amax(zones)+1)):

        wTN = np.where(np.logical_and(diffmap == 1,zones==z))
        wTP = np.where(np.logical_and(diffmap == 2,zones==z))
        wFN = np.where(np.logical_and(diffmap == -1,zones==z))
        wFP = np.where(np.logical_and(diffmap == -2,zones==z))

        TN = len(wTN[0])
        TP = len(wTP[0])
        FN = len(wFN[0])
        FP = len(wFP[0])


        s = TP / (TP+FN+0.0000000000001)
        p = TP / (TP+FP+0.0000000000001)
        a = (TP+TN) / (TP+TN+FP+FN+0.0000000000001)

        S.append(s); P.append(p); A.append(a)

    wTN = np.where(diffmap == 1)
    wTP = np.where(diffmap == 2)
    wFN = np.where(diffmap == -1)
    wFP = np.where(diffmap == -2)

    TN = len(wTN[0])
    TP = len(wTP[0])
    FN = len(wFN[0])
    FP = len(wFP[0])

    s = TP / (TP+FN+0.0000000000001)
    p = TP / (TP+FP+0.0000000000001)
    a = (TP+TN) / (TP+TN+FP+FN+0.0000000000001)

    S.append(s); P.append(p); A.append(a)

    return S, P, A




###############################
def make_hillshade_arr (proba):
    im = Image.fromarray(np.uint8(plt.cm.jet(proba/100)*255))
    imod = im.filter(EMBOSS)
    imarr = ImageOps.grayscale(imod)
    arr_hs= np.array(imarr)

    return arr_hs


###############################
def make_edges_arr (proba):
    im = Image.fromarray(np.uint8(plt.cm.jet(proba/100)*255))
    imod = im.filter(FIND_EDGES)
    imarr = ImageOps.grayscale(imod)
    arr_hs= np.array(imarr)

    return arr_hs

###############################
def make_classif_edges_arr (pred):
    im = Image.fromarray(np.uint8(plt.cm.jet(pred)*255))
    imod = im.filter(FIND_EDGES)
    imarr = ImageOps.grayscale(imod)
    arr_hs= np.array(imarr)

    return arr_hs


###############################
def make_contour_arr (proba):
    im = Image.fromarray(np.uint8(plt.cm.jet(proba/100)*255))
    imod = im.filter(CONTOUR)
    imarr = ImageOps.grayscale(imod)
    arr_hs= np.array(imarr)

    return arr_hs


########################################
def plot_colorbar (fig, axis, image, pad, void_ticks = False):
    axins1 = inset_axes(axis,
                        width="5%",  # width = 50% of parent_bbox width
                        height="50%",  # height : 5%
                        loc='lower right',
                        borderpad=pad)
    if void_ticks == True:
        fig.colorbar(image, cax=axins1, orientation="vertical", ticks = [])
    else:
        fig.colorbar(image, cax=axins1, orientation="vertical")

    axins1.xaxis.set_ticks_position("bottom")




################################################
def plot_zoomed_axis_copy (zoomed_axis, fullmap_axis, pred, proba, hs, xs, xe, ys, ye, colormap, FNcolor, FPcolor, diffmap = None, hs_vmin = 80, hs_vmax = 180):

    fullmap_axis.add_patch(Rectangle((ys, xs), ye-ys, xe-xs, edgecolor='black', facecolor='none', lw=2))

    zoomed_axis.imshow(hs[xs:xe,ys:ye], cmap = plt.cm.Greys_r, vmin = hs_vmin, vmax = hs_vmax)


    try:
        test = diffmap[0]
        # plot TP
        zoomed_axis.imshow(np.ma.masked_where(diffmap[xs:xe,ys:ye] !=1 ,proba[xs:xe,ys:ye]), cmap = colormap, vmin = 0, vmax = 100, alpha = 0.5)
        # plot FN
        zoomed_axis.imshow(np.ma.masked_where(diffmap[xs:xe,ys:ye] !=-1 ,proba[xs:xe,ys:ye]), cmap = FNcolor, vmin = 0, vmax = 100, alpha = 0.5)
        # plot FP
        zoomed_axis.imshow(np.ma.masked_where(diffmap[xs:xe,ys:ye] !=-2 ,proba[xs:xe,ys:ye]), cmap = FPcolor, vmin = 0, vmax = 100, alpha = 0.5)
        
    except:
        zoomed_axis.imshow(np.ma.masked_where(pred[xs:xe,ys:ye] <1, proba[xs:xe,ys:ye]), cmap = colormap, vmin = 0, vmax = 100, alpha = 0.58)



################################################
def plot_zoomed_axis (zoomed_axis, fullmap_axis, pred, proba, L, hs, mask, xs, xe, ys, ye, colormap, FNcolor, FPcolor, diffmap = None, kso_poly = None, geodata= None, hs_vmin = 80, hs_vmax = 180):

    fullmap_axis.add_patch(Rectangle((ys, xs), ye-ys, xe-xs, edgecolor='black', facecolor='none', lw=2))

    print (np.amax(proba))
    proba[mask!=0] = 255
    pred[pred!=1] = 0

  
    for i in range(len(kso_poly)):
        poly = kso_poly.iloc[i]
        x,y = poly.exterior.xy
        x = (np.asarray(x) - geodata[0][0])/geodata[0][1] - ys
        y = (np.asarray(y) - geodata[0][3])/geodata[0][5] - xs
        
        zoomed_axis.plot(x,y, c = 'r')
    

    try:
        test = diffmap[0]
    except:
        print ('no kso')

    pred[mask!=0] = 255

    im = zoomed_axis.imshow(np.ma.masked_where(pred[xs:xe,ys:ye] != 1, L[xs:xe,ys:ye]), cmap = colormap, vmin = 0, vmax = 52, alpha = 0.7)
    #plot_colorbar (figure, fullmap_axis, im, 6, void_ticks = False)
        
    im2 = zoomed_axis.imshow(np.ma.masked_where(pred[xs:xe,ys:ye] != 0, L[xs:xe,ys:ye]), cmap = FNcolor, vmin = 0, vmax = 52, alpha = 0.7)
    #plot_colorbar (figure, fullmap_axis, im2, 4, void_ticks = False)
        
    im3 = zoomed_axis.imshow(np.ma.masked_where(mask[xs:xe,ys:ye] == 0, mask[xs:xe,ys:ye]), cmap = plt.cm.Greys, vmin = 0, vmax = 10, alpha = 1)



################################################
def plot_fullmap_axis (figure, fullmap_axis, pred, proba, L, hs, mask, colormap, FNcolor, FPcolor, S=0, P=0, A=0, kso = None, kso_poly = None, geodata = None, hs_vmin = 80, hs_vmax = 180):

    print (np.amax(proba))
    proba[mask!=0] = 255
    pred[pred!=1] = 0
    pred[mask!=0] = 255

    try:
        test = kso[0]
        for i in range(len(kso_poly)):
            poly = kso_poly.iloc[i]
            x,y = poly.exterior.xy
            x = (np.asarray(x) - geodata[0][0])/geodata[0][1]
            y = (np.asarray(y) - geodata[0][3])/geodata[0][5]
            
            fullmap_axis.plot(x,y, c = 'r')

    except:
        print ('no kso')

    im = fullmap_axis.imshow(np.ma.masked_where(pred != 1, L), cmap = colormap, vmin = 0, vmax = 52, alpha = 0.7)
    #plot_colorbar (figure, fullmap_axis, im, 6, void_ticks = False)
        
    im2 = fullmap_axis.imshow(np.ma.masked_where(pred != 0, L), cmap = FNcolor, vmin = 0, vmax = 52, alpha = 0.7)
    #plot_colorbar (figure, fullmap_axis, im2, 4, void_ticks = False)
        
    im3 = fullmap_axis.imshow(np.ma.masked_where(mask == 0, mask), cmap = plt.cm.Greys, vmin = 0, vmax = 10, alpha = 1)
    




################################################
def plot_fullmap_axis_copy (figure, fullmap_axis, pred, proba, hs, colormap, FNcolor, FPcolor, S=0, P=0, A=0, diffmap = None, hs_vmin = 80, hs_vmax = 180):


    # background probability map
    fullmap_axis.imshow(hs, cmap = plt.cm.Greys_r, vmin = hs_vmin, vmax = hs_vmax)

   
    proba = np.ma.masked_where(proba == 255, proba)

    try:
        test = diffmap[0]
        # plot TP
        im = fullmap_axis.imshow(np.ma.masked_where(diffmap !=1, proba), cmap = colormap, vmin = 0, vmax = 100, alpha = 0.5)
        plot_colorbar (figure, fullmap_axis, im, 6, void_ticks = False)
        
        # plot FN
        imFN = fullmap_axis.imshow(np.ma.masked_where(diffmap !=-1, proba), cmap = FNcolor, vmin = 0, vmax = 100, alpha = 0.5)
        plot_colorbar (figure, fullmap_axis, imFN, 5, void_ticks = False)
        
        # plot FP
        imFP = fullmap_axis.imshow(np.ma.masked_where(diffmap !=-2 ,proba), cmap = FPcolor, vmin = 0, vmax = 100, alpha = 0.5)
        plot_colorbar (figure, fullmap_axis, imFP, 4, void_ticks = False)

        fullmap_axis.set_title(str(round(S,2)) + '//' + str(round(P,2)) + '//' + str(round(A,2)) )
    
    except:
        print ('no kso')
        im = fullmap_axis.imshow(np.ma.masked_where(pred <1, proba), cmap = colormap, vmin = 0, vmax = 100, alpha = 0.58)
        plot_colorbar (figure, fullmap_axis, im, 6, void_ticks = False)
 


################################################
def plot_patch_proba_distribution (axis, pred, proba, Zone_vals, zones, Density_vals, survey_density = None, sepvals = None):

    pos = 0
    # Look at each zone 
    for Z in Zone_vals:
        print ("\nZone", Z)
        # Identify patches for each density
        for D in Density_vals:
            print ('\nFinding contiguous patches of density', D)

            try:
                test = survey_density[0]
                contiguous_area, patch_id = ml.getContiguousAreas(survey_density, D, min_pixels = 2)
            except:
                contiguous_area, patch_id = ml.getContiguousAreas(pred, D, min_pixels = 2)

            # For each patch
            for p in range(1,np.amax(patch_id)+1):
                # these are the patch indices
                pidx = np.where(patch_id == p)
                here_Z = zones[pidx][0]
                if here_Z == Z:
                    #print ('patch number:', p, 'of', np.amax(patch_id)+1)

                    X = proba; XX = X[pidx]
                    Q25=np.percentile(XX,25);Q50=np.percentile(XX,50);Q75=np.percentile(XX,75)

                    try:
                        test = survey_density[0]
                        axis.plot ([pos-5,pos+5], [100*sepvals[Z-1],100*sepvals[Z-1]], '-k', lw=1, alpha = 0.8)
                    except:
                        test = 0

                    axis.plot ([pos,pos], [Q25,Q75], c = plt.cm.jet(D/4), lw = 0.7, alpha = 0.6)
                    axis.scatter (pos, Q50, facecolor = plt.cm.jet(D/4), marker = 'o', lw = 0, s = 10 + 2*np.sqrt(len(pidx[0])), alpha = 0.6)
                    # Move one step right
                    pos += 1
            # Change density value
            pos += 5
        # Change zone and plot a big line
        pos += 10
        if Z != 5:
            axis.plot ([pos-5,pos-5], [-5,105], c = 'k', lw = 1.5)

    # Put a line for each year
    pos += 10
    axis.plot([pos-5,pos-5], [-5,105], 'k', lw = 4.5)
    axis.set_ylim(-5,105)
    axis.set_xlim(-2,pos-8)
    axis.set_ylabel(' P_sg for patches ')


################################################
################################################
def make_stability_plot (figname, fignum, pred, proba, L, hs, mask, zones, S=0, P=0, A=0, diffmap = None, survey_density = None, survey_poly = None, geodata = None, Sepvals = None):

    #if diffmap != None:
    #mask = np.ma.getmask(proba)
    #diffmap = np.ma.masked_where(mask == True, diffmap)
    #diffmap = np.ma.masked_where(proba == 255, diffmap)


    # This is the figure. Hi figure!    
    fig = plt.figure(1, facecolor='White',figsize=[31, 30])

    # Define the axes
    axmap = plt.subplot2grid(  (10,10), (0,0), rowspan=3, colspan=2)
    axzoom1 = plt.subplot2grid((10,10), (0,2), rowspan=3, colspan=4)
    axzoom2 = plt.subplot2grid((10,10), (3,0), rowspan=7, colspan=6)
    axzoom3 = plt.subplot2grid((10,10), (0,6), rowspan=10, colspan=4)
    #axzoom4 = plt.subplot2grid((4,7), (2,5), rowspan=2, colspan=2)

    # Define some colour stuff
    hs_vmin = 80; hs_vmax = 180
    colormap = plt.cm.Wistia
    FNcolor = plt.cm.winter_r
    FPcolor = plt.cm.cool

    # Plot main map
    plot_fullmap_axis (fig, axmap, pred, proba, L, hs, mask, colormap, FNcolor, FPcolor, S, P, A,survey_density, survey_poly, geodata)
    #plot_fullmap_axis (fig, axmap, pred, proba, hs, colormap, FNcolor, FPcolor, S, P, A,diffmap)

    # Zoom1 
    xs=100; xe=400; ys=620;ye=1100
    plot_zoomed_axis (axzoom1, axmap, pred, proba, L, hs, mask, xs, xe, ys, ye, colormap, FNcolor, FPcolor, diffmap, survey_poly, geodata)

    # Zoom2 
    xs=410; xe=905; ys=350;ye=785
    plot_zoomed_axis (axzoom2, axmap, pred, proba, L, hs, mask, xs, xe, ys, ye, colormap, FNcolor, FPcolor, diffmap, survey_poly, geodata)

    # Zoom3 
    xs=870; xe=1500; ys=240;ye=500
    plot_zoomed_axis (axzoom3, axmap, pred, proba, L, hs, mask, xs, xe, ys, ye, colormap, FNcolor, FPcolor, diffmap, survey_poly, geodata)

    # Zoom4 
    #xs=1245; xe=1330; ys=340;ye=420
    #plot_zoomed_axis (axzoom4, axmap, pred, proba, hs, xs, xe, ys, ye, colormap, FNcolor, FPcolor, diffmap)

    #try:
    #    test = survey_density[0]

    print (S,P,A)

    axzoom1.set_title('S:'+str(np.round(S[1],2))+' // P:'+str(np.round(P[1],2))+' // A:'+str(np.round(A[1],2)), fontsize = 28) 
    axzoom2.set_title('S:'+str(np.round(S[2],2))+' // P:'+str(np.round(P[2],2))+' // A:'+str(np.round(A[2],2)), fontsize = 28)
    axzoom3.set_title('S:'+str(np.round(S[3],2))+' // P:'+str(np.round(P[3],2))+' // A:'+str(np.round(A[3],2)), fontsize = 28)
        

    #except:
    #    print ('no survey')

    plt.tight_layout()
    plt.savefig(figname)
    fig.clear(True)
    plt.clf()

################################################
################################################
def make_staple_plot (figname, fignum, proba, pred, hs, zones, S=0, P=0, A=0, diffmap = None, survey_density = None, Sepvals = None):

    #if diffmap != None:
    #mask = np.ma.getmask(proba)
    #diffmap = np.ma.masked_where(mask == True, diffmap)
    #diffmap = np.ma.masked_where(proba == 255, diffmap)

    
    # This is the figure. Hi figure!    
    fig = plt.figure(fignum, facecolor='White',figsize=[23, 20])

    # Define the axes
    axmap = plt.subplot2grid((7,7),   (0,0), rowspan=4, colspan=3)
    axzoom1 = plt.subplot2grid((7,7), (0,3), rowspan=2, colspan=2)
    axzoom2 = plt.subplot2grid((7,7), (0,5), rowspan=2, colspan=2)
    axzoom3 = plt.subplot2grid((7,7), (2,3), rowspan=2, colspan=2)
    axzoom4 = plt.subplot2grid((7,7), (2,5), rowspan=2, colspan=2)
    axpatch1 = plt.subplot2grid((7,7),(4,0), rowspan=1, colspan=7)
    axpatch2 = plt.subplot2grid((7,7),(5,0), rowspan=1, colspan=7)
    axpatch3 = plt.subplot2grid((7,7),(6,0), rowspan=1, colspan=7)

    # Define some colour stuff
    hs_vmin = 80; hs_vmax = 180
    colormap = plt.cm.Greens
    FNcolor = plt.cm.autumn_r
    FPcolor = plt.cm.cool

    # Plot main map
    plot_fullmap_axis (fig, axmap, pred, proba, hs, colormap, FNcolor, FPcolor, S, P, A,diffmap)

    # Zoom1 
    xs=145; xe=240; ys=960;ye=1060
    plot_zoomed_axis (axzoom1, axmap, pred, proba, hs, xs, xe, ys, ye, colormap, FNcolor, FPcolor, diffmap)

    # Zoom2 
    xs=410; xe=580; ys=625;ye=785
    plot_zoomed_axis (axzoom2, axmap, pred, proba, hs, xs, xe, ys, ye, colormap, FNcolor, FPcolor, diffmap)

    # Zoom3 
    xs=830; xe=990; ys=350;ye=510
    plot_zoomed_axis (axzoom3, axmap, pred, proba, hs, xs, xe, ys, ye, colormap, FNcolor, FPcolor, diffmap)

    # Zoom4 
    xs=1245; xe=1330; ys=340;ye=420
    plot_zoomed_axis (axzoom4, axmap, pred, proba, hs, xs, xe, ys, ye, colormap, FNcolor, FPcolor, diffmap)

    # Plot distro of proba within the patches
    #plot_patch_proba_distribution (axpatch1, pred, proba, Zone_vals,zones, [0,1])

    # Plot distro of proba within the surveyed patches
    sepvals = Sepvals[3]
    try: 
        A = survey_density[0]
        Exist_survey = True
    except:
        Exist_survey = False

    if Exist_survey == True:
        print ('plotting surveyed patch proba')
        #plot_patch_proba_distribution (axpatch2, pred, proba, Zone_vals,zones, Density_vals, survey_density, sepvals)


    plt.tight_layout()
    plt.savefig(figname)
    fig.clear(True)
    plt.clf()



###################################################################################
###################################################################################
def confusion_metrix (m):
    mysum = 0
    for r in range(len(m)):
        mysum += m[r,r]
        S = m[r,r] / np.sum(m[r,:])
        P = m[r,r] / np.sum(m[:,r])
    A = mysum/np.sum(m)

    return S,P,A


###################################################################################
###################################################################################
def nested_train_test_other (data_train, data_test, cols, tolerance = 0.1):

    # select the data based on the fraction cover
    bare_data = data_train[data_train['fraction_veg_cover'] < tolerance]
    veg_data = data_train[data_train['fraction_veg_cover'] > 1 - tolerance]
    short_data = bare_data.append(veg_data)

    bare_data = data_test[data_test['fraction_veg_cover'] < tolerance]
    veg_data = data_test[data_test['fraction_veg_cover'] > 1 - tolerance]
    short_data_test = bare_data.append(veg_data)

    """
    # Calculate McFeeters NDWI
    Gm = np.asarray(short_data['green_mean'])
    Nm = np.asarray(short_data['nir_mean'])
    NDWIm = (Gm-Nm) / (Gm+Nm)
    short_data['NDWI_mean'] = NDWIm

    Gm = np.asarray(short_data_test['green_mean'])
    Nm = np.asarray(short_data_test['nir_mean'])
    NDWIm = (Gm-Nm) / (Gm+Nm)
    short_data_test['NDWI_mean'] = NDWIm
    
    cols = ['NDWI_mean'] + cols
    """

    """
    # something is weird ... make maps ...
    sr = short_data.index.get_level_values(0).values
    sc = short_data.index.get_level_values(1).values

    svr = shortv_data.index.get_level_values(0).values
    svc = shortv_data.index.get_level_values(1).values

    srmax = np.amax(sr)
    scmax = np.amax(sc)

    svrmax = np.amax(svr)
    svcmax = np.amax(svc)

    arr = -1*np.ones((srmax+1,scmax+1), dtype = np.float)
    arrv = -1*np.ones((svrmax+1,svcmax+1), dtype = np.float)

    for i in range(len(sr)):
        #print (srmax, scmax)
        #print (arr.shape)
        #print (sr[i],sc[i])
        arr[sr[i],sc[i]] = short_data.loc[(sr[i],sc[i]),'fraction_veg_cover']
    for i in range(len(svr)):
        arrv[svr[i],svc[i]] = shortv_data.loc[(svr[i],svc[i]),'fraction_veg_cover']


    arr[arr>0.9] = 1
    arrv[arrv>0.9] = 1


    arr = arr[:svrmax+1, :svcmax+1]


    fig = plt.figure(1, facecolor='White',figsize=[24, 24])
    ax1 =  plt.subplot2grid((1,3),(0,0),colspan=1, rowspan=1)
    ax2 =  plt.subplot2grid((1,3),(0,1),colspan=1, rowspan=1)
    ax3 =  plt.subplot2grid((1,3),(0,2),colspan=1, rowspan=1)

    ax1.imshow(arr, cmap = plt.cm.seismic, vmin = -1, vmax = 1)
    ax2.imshow(arrv, cmap = plt.cm.seismic, vmin = -1, vmax = 1)
    ax3.imshow(arr-arrv, cmap = plt.cm.seismic, vmin = -1, vmax = 1)

    plt.savefig('Figures/2004/MAPTEST.png')

    quit()
    """
        
    # make a copy of the training set just in case
    long_data = short_data.copy()

    # subsample the training data to fit the area of the test data
    idx = short_data_test.index.values
    short_data = short_data.loc[short_data.index.intersection(idx)]

    # Make sure this is a reciprocal thing
    idx = short_data.index.values
    short_data_test = short_data_test.loc[short_data_test.index.intersection(idx)]


    # only look at NDWI >= 0.3
    #short_data = short_data[short_data['NDWI_mean'] >= 0.25]
    #short_data_test = short_data_test[short_data_test['NDWI_mean'] >= 0.25]

    # only look at stuff below 0 bathy
    short_data = short_data[short_data['bathy'] <= 0]
    short_data_test = short_data_test[short_data_test['bathy'] <= 0]

    # build a reference set of pixels for transformation
    # We are transforming things according to buildings!
    ref = short_data[short_data['mask'] == 5]
    ref_test = short_data_test[short_data_test['mask'] == 5]

    # Now look at te bits likely to have seagrass
    short_data = short_data[short_data['mask'] == 0]
    #long_data = long_data[long_data['mask'] == 0]
    short_data_test = short_data_test[short_data_test['mask'] == 0]

    # Select areas that are likely to have seagrass.
    # Tidal flats or tidal flats and canals? Try both
    canal_data = short_data[short_data['mask'] == 5.]
    tf_data = short_data[short_data['mask'] == 0.]
    short_data = tf_data.append(canal_data)

    canal_data = short_data_test[short_data_test['mask'] == 5.]
    tf_data = short_data_test[short_data_test['mask'] == 0.]
    short_data_test = tf_data.append(canal_data)


    # Transform test features with the reference transformation
    for band in ref.columns.values:
        if band.startswith('blue') or band.startswith('green') or band.startswith('red') or band.startswith('nir'):

            # the reference rasters for fitting
            A = np.asarray(ref[band])
            A_test = np.asarray(ref_test[band])

            # the raster we want to transform with the fitting model
            B = np.asarray(short_data_test[band])

            # the fitting model
            model = np.polyfit(A_test,A,2)
            fitline = np.poly1d(model)

            # the transformed array
            B_trans = fitline(B)

            # now replace the values in the array
            short_data_test[band] = list(B_trans)
                
    # get data into array
    X = np.asarray(short_data.loc[:,cols])
    X = np.asarray(long_data.loc[:,cols])
    X_test = np.asarray(short_data_test.loc[:,cols])

    # get rid of nan-containing rows in array
    X = X[~np.isnan(X).any(axis=1), :]
    X_test = X_test[~np.isnan(X_test).any(axis=1), :]

    # separate the features (X) from classification columns (Y)
    Y = X[:,-1]; X = X[:,:-1]
    Y_test = X_test[:,-1]; X_test = X_test[:,:-1]

    # normalise features
    for x in range(len(X[0])):
        X[:,x] = (X[:,x] - min(X[:,x])) / ( max(X[:,x]) - min(X[:,x]) )
    for x in range(len(X_test[0])):
        X_test[:,x] = (X_test[:,x] - min(X_test[:,x])) / ( max(X_test[:,x]) - min(X_test[:,x]) )

    # Make binary classif. Keep a copy of Y for further non-binary classif
    Ynb = np.copy(Y)
    Y[Y > 0] = 1
    Ynb_test = np.copy(Y_test)
    Y_test[Y_test > 0] = 1

    # attribute the sets
    X_train = X; X_val = X_test
    Y_train = Y; Y_val = Y_test
    Ynb_train = Ynb; Ynb_val = Ynb_test

    wherev_train = np.where(Ynb_train >=1)
    wherev_val = np.where(Ynb_val >=1)

    # make non-binary classif among positives 
    Xnb_train, Xnb_val = X_train[wherev_train], X_val[wherev_val]
    Ynb_train, Ynb_val = Ynb_train[wherev_train], Ynb_val[wherev_val]


    return X_train, X_val, Y_train, Y_val, Xnb_train, Xnb_val, Ynb_train, Ynb_val, wherev_train, wherev_val


###################################################################################
###################################################################################
def nested_train_test_self (data, cols, folds = 5, tolerance = 0.1):
    # select the data based on the fraction cover
    bare_data = data[data['fraction_veg_cover'] < tolerance]
    veg_data = data[data['fraction_veg_cover'] > 1 - tolerance]
    short_data = bare_data.append(veg_data)

    """
    # Only look at stuff with McFeeters NDWI above 0.3
    Gm = np.asarray(short_data['green_mean'])
    Nm = np.asarray(short_data['nir_mean'])
    NDWIm = (Gm-Nm) / (Gm+Nm)
    short_data['NDWI_mean'] = NDWIm
    cols = ['NDWI_mean'] + cols
    """

    # only look at NDWI >= 0.3
    #short_data = short_data[short_data['NDWI_mean'] >= 0.25]

    # only look at stuff below 0 bathy. Not reliable because of the gridding process
    short_data = short_data[short_data['bathy'] <= 0]

    """
    # something is weird ... make maps ...
    sr = short_data.index.get_level_values(0).values
    sc = short_data.index.get_level_values(1).values
    srmax = np.amax(sr)
    scmax = np.amax(sc)

    arr = -1*np.ones((srmax+1,scmax+1), dtype = np.float)

    for i in range(len(sr)):
        arr[sr[i],sc[i]] = short_data.loc[(sr[i],sc[i]),'density_class']

    fig = plt.figure(1, facecolor='White',figsize=[24, 24])
    ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

    arr = arr[400:, 200:800]

    arr = np.ma.masked_where(arr <= -1, arr)

    ax1.imshow(arr, cmap = plt.cm.summer_r, vmin = 0, vmax = 4)

    plt.tight_layout()
    plt.savefig('/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Validation/MAPTEST_density_NDWI.png')
    quit()
    """

    # Select areas that are likely to have seagrass.
    # Tidal flats or tidal flats and canals? Try both
    canal_data = short_data[short_data['mask'] == 5.]
    tf_data = short_data[short_data['mask'] == 0.]
    short_data = tf_data.append(canal_data)

    # get data into array
    X = np.asarray(short_data.loc[:,cols]) # get rid of ndwi

    # get rid of nan-containing rows in array
    X = X[~np.isnan(X).any(axis=1), :]

    # separate the features (X) from classification columns (Y)
    Y = X[:,-1]; X = X[:,:-1]

    # normalise features
    for x in range(len(X[0])):
        X[:,x] = (X[:,x] - min(X[:,x])) / ( max(X[:,x]) - min(X[:,x]) )

    # Make binary classif. Keep a copy of Y for further non-binary classif
    Ynb = np.copy(Y)
    Y[Y > 0] = 1

    # split the data
    kf = KFold(n_splits=folds, shuffle = True)
    kf.get_n_splits(X)

    for train_idx, test_idx in kf.split(X):
        # create training and validation datasets for model 1
        X_train, X_val = X[train_idx], X[test_idx]
        Y_train, Y_val = Y[train_idx], Y[test_idx]

        Ynb_train, Ynb_val = Ynb[train_idx], Ynb[test_idx]
        wherev_train = np.where(Ynb_train >=1)
        wherev_val = np.where(Ynb_val >=1)


        # make non-binary classif among positives 
        Xnb_train, Xnb_val = X_train[wherev_train], X_val[wherev_val]
        Ynb_train, Ynb_val = Ynb_train[wherev_train], Ynb_val[wherev_val]

        break

    return X_train, X_val, Y_train, Y_val, Xnb_train, Xnb_val, Ynb_train, Ynb_val, wherev_train, wherev_val


###################################################################################
###################################################################################
def RF_fit_predict_score(X_train,Y_train,X_val,Y_val, ne = 100, md = 30):
    timer_start = datetime.now() 

    model = RandomForestClassifier(n_estimators = ne, max_depth = md)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_val)

    timer  = datetime.now() - timer_start
    print ('RF fitted and predicted in:', timer)

    m = confusion_matrix(Y_val, Y_pred) # rows = pred value ; cols = actual value

    return model, Y_pred, m


###################################################################################
###################################################################################
def make_metrics (data, useful_bands, bands_to_drop, summer_idx = [0,0]):

    for b in useful_bands:
        X = np.asarray(data.loc[:,b])
        X = np.asarray(list(map(lambda m_ls: np.asarray(m_ls), X)), dtype = np.float)

        X[X == -9999.] = np.nan
        if 'bot' in b:
            X[X > 0.5] = np.nan

        # Only use a summer trio
        if summer_idx[0] == 0 and summer_idx[1] == 0:
            X = X[:,:]
        else:
            X = X[:,summer_idx[0]:summer_idx[1]]

        # Make the relevant metrics
        data[b[:-2]+'_max'] = list(np.nanmax(X, axis = 1)) 
        data[b[:-2]+'_min'] = list(np.nanmin(X, axis = 1)) 
        data[b[:-2]+'_mean'] = list(np.nanmean(X, axis = 1)) 
        data[b[:-2]+'_std'] = list(np.nanstd(X, axis = 1)) 

        # discard the legnthy column
        data = data.drop(b, axis = 1) 

    # discard useless bands
    for b in bands_to_drop:
        data = data.drop(b, axis = 1) 

    return data

###################################################################################
###################################################################################
def add_multiple_srf_rows(mypanda, veg, density, cover, Rsurf, Angles, date, where):

    A = where[0]; A = A.T
    AA = where[1]; AA = AA.T

    B = Rsurf[0]; B = B[where]
    C = Rsurf[1]; C = C[where]
    D = Rsurf[2]; D = D[where]
    E = Rsurf[3]; E = E[where]

    P = Angles[0]; P = P[where]
    Q = Angles[1]; Q = Q[where]

    temp = pd.DataFrame({'vegetation':['Nonveg']*len(where[0]),
                          'density_class':np.zeros(len(where[0])),
                          'fraction_veg_cover':np.zeros(len(where[0])), 
                          'row':A,
                          'col':AA,
                          'blue':B,             
                          'green':C,
                          'red':D,
                          'nir':E,
                          'sensor_zenith':P,
                          'solar_zenith':Q,
                          'survey':date})

    mypanda = mypanda.append(temp, ignore_index = True)

    return mypanda

###################################################################################
###################################################################################
def add_srf_row(mypanda, veg, density, cover, Rsurf, Angles, date, row,col):

    # Add a row to the df
    mypanda = mypanda.append({'vegetation':veg,
                              'density_class':density,
                              'fraction_veg_cover':cover, 
                              'row':row,
                              'col':col,
                              'blue':Rsurf[0,row,col],
                              'green':Rsurf[1,row,col],
                              'red':Rsurf[2,row,col],
                              'nir':Rsurf[3,row,col],
                              'sensor_zenith':Angles[0,row,col],
                              'solar_zenith':Angles[1,row,col],
                              'survey':date},
                              ignore_index=True)

    return mypanda


###################################################################################
###################################################################################
def fill_srf_pixel_df(pixels_pd, pixels_surveyed, veg_state, Rsurf, Angles, shp, ref_raster, geotransform, date):

    # open the depth raster as a reference
    with Raster(ref_raster) as raster_obj: 
        # open the survey shapefile
        with fiona.open(shp) as f: 
            # for each polygon
            for m in range(len(f)):
                print (m , '/', len(f))
                geom = f[m]['geometry'] # the polygon geometry
                if veg_state == True:
                    veg = f[m]['properties']['COD'] # vegetation assemblage
                    density = f[m]['properties']['CON'] # density class
                else:
                    veg = 'Nonveg'
                    density = 0

                # create a True/False mask of the dimensions of the depth raster
                # True means the presence of  vegetation
                polygon_mask = rasterio.features.geometry_mask(geometries=[geom], out_shape=(raster_obj.shape[0], raster_obj.shape[1]), transform=raster_obj.affine, all_touched=True, invert=True)

                # these are the pixels touched by the polygon
                where = np.where(polygon_mask == True)
                print (len(where[0]))

                # for each of these pixels, we must determine the area covered by the polygon
                if len(where[0]) > 0:
                    # there is some veg
                    if veg_state == True:
                        for p in range(len(where[0])):
                            row = where[0][p]; col = where[1][p]
                            pixels_surveyed.append([row,col])
                            x_l = geotransform[0] + col * geotransform[1]
                            y_t = geotransform[3] + row * geotransform[5]
                            x_r = x_l + geotransform[1]
                            y_b = y_t + geotransform[5]

                            pixelpoly = Polygon([(x_l,y_t), (x_l,y_b), (x_r,y_b), (x_r,y_t)])

                            try: 
                                A = Polygon(geom['coordinates'][0])

                                inter = pixelpoly.intersection(A)
                                inter_area = inter.area
                                relative_coverage = inter_area / pixelpoly.area

                                pixels_pd = add_srf_row(pixels_pd, veg, density, relative_coverage, Rsurf, Angles, date, row,col)

                            except:
                                print ('geometry error')
                                pass
                            


                    #if m >= 10:
                    #    break
                    # if there is no veg
                    else:

                        relative_coverage = 0
                        pixels_pd = add_multiple_srf_rows(pixels_pd, veg, density, relative_coverage, Rsurf, Angles, date, where)

    return pixels_pd, pixels_surveyed
                




###################################################################################
###################################################################################
def make_srf_pixel_df(Rsurf, Angles, seagrass_shp, AoI_shp, bathyraster, processed_dir, geotransform, date):

    # Create an empty dataframe to hold polygon information enriched with spectral and depth data
    pixels_pd=pd.DataFrame(columns=['vegetation', 'density_class', 'fraction_veg_cover', 'row', 'col', 'blue', 'green', 'red', 'nir', 'bathy', 'mask', 'solar_zenith', 'sensor_zenith', 'survey'])

    cols = pixels_pd.columns.values

    pixels_surveyed = []

    pixels_pd, pixels_surveyed = fill_srf_pixel_df(pixels_pd, pixels_surveyed, True, Rsurf, Angles, seagrass_shp, bathyraster, geotransform, date)

    pixels_pd, pixels_surveyed = fill_srf_pixel_df(pixels_pd, pixels_surveyed, False, Rsurf, Angles, AoI_shp, bathyraster, geotransform, date)

    pixels_pd.to_csv(processed_dir + 'prep_data.csv')

    return pixels_pd



###################################################################################
###################################################################################
def add_multiple_rows(mypanda, veg, density, cover, Rbot,  Rsynth, Rsurf, Hydro, Angles, date, where):

    A = where[0]; A = A.T
    AA = where[1]; AA = AA.T

    B = Rsurf[0]; B = B[where]
    C = Rsurf[1]; C = C[where]
    D = Rsurf[2]; D = D[where]
    E = Rsurf[3]; E = E[where]
    F = Rsurf[4]; F = F[where]

    G = Rbot[0]; G = G[where]
    H = Rbot[1]; H = H[where]
    I = Rbot[2]; I = I[where]
    J = Rbot[3]; J = J[where]
    K = Rbot[4]; K = K[where]

    Gs = Rsynth[0]; Gs = Gs[where]
    Hs = Rsynth[1]; Hs = Hs[where]
    Is = Rsynth[2]; Is = Is[where]
    Js = Rsynth[3]; Js = Js[where]
    Ks = Rsynth[4]; Ks = Ks[where]

    L = Hydro[0]; L = L[where]
    M = Hydro[1]; M = M[where]
    N = Hydro[2]; N = N[where]
    O = Hydro[3]; O = O[where]

    P = Angles[0]; P = P[where]
    Q = Angles[1]; Q = Q[where]

    temp = pd.DataFrame({'vegetation':['Nonveg']*len(where[0]),
                          'density_class':np.zeros(len(where[0])),
                          'fraction_veg_cover':np.zeros(len(where[0])), 
                          'row':A,
                          'col':AA,
                          'blue':B,             
                          'green':C,
                          'red':D,
                          'nir':E,
                          'swir':F,
                          'blue_bot':G,
                          'green_bot':H,
                          'red_bot':I,
                          'nir_bot':J,
                          'swir_bot':K,
                          'blue_synth':Gs,
                          'green_synth':Hs,
                          'red_synth':Is,
                          'nir_synth':Js,
                          'swir_synth':Ks,
                          'bathy':L,
                          'depth':M,
                          'waveheight':N,
                          'spm':O,
                          'sensor_zenith':P,
                          'solar_zenith':Q,
                          'survey':date})

    mypanda = mypanda.append(temp, ignore_index = True)

    return mypanda

###################################################################################
###################################################################################
def add_a_row(mypanda, veg, density, cover, Rbot,  Rsynth, Rsurf, Hydro, Angles, date, row,col):

    # Add a row to the df
    mypanda = mypanda.append({'vegetation':veg,
                              'density_class':density,
                              'fraction_veg_cover':cover, 
                              'row':row,
                              'col':col,
                              'blue':Rsurf[0,row,col],
                              'green':Rsurf[1,row,col],
                              'red':Rsurf[2,row,col],
                              'nir':Rsurf[3,row,col],
                              'swir':Rsurf[4,row,col],
                              'blue_bot':Rbot[0,row,col],
                              'green_bot':Rbot[1,row,col],
                              'red_bot':Rbot[2,row,col],
                              'nir_bot':Rbot[3,row,col],
                              'swir_bot':Rbot[4,row,col],
                              'blue_synth':Rsynth[0,row,col],
                              'green_synth':Rsynth[1,row,col],
                              'red_synth':Rsynth[2,row,col],
                              'nir_synth':Rsynth[3,row,col],
                              'swir_synth':Rsynth[4,row,col],
                              'bathy':Hydro[0,row,col],
                              'depth':Hydro[1,row,col],
                              'waveheight':Hydro[2,row,col],
                              'spm':Hydro[3,row,col],
                              'sensor_zenith':Angles[0,row,col],
                              'solar_zenith':Angles[1,row,col],
                              'survey':date},
                              ignore_index=True)

    return mypanda


###################################################################################
###################################################################################
def fill_pixel_df(pixels_pd, pixels_surveyed, veg_state, Rbot, Rsynth, Rsurf, Hydro, Angles, shp, ref_raster, geotransform, date):

    # open the depth raster as a reference
    with Raster(ref_raster) as raster_obj: 
        # open the survey shapefile
        with fiona.open(shp) as f: 
            # for each polygon
            for m in range(7437,len(f)):
                print (m , '/', len(f))
                geom = f[m]['geometry'] # the polygon geometry
                if veg_state == True:
                    veg = f[m]['properties']['COD'] # vegetation assemblage
                    density = f[m]['properties']['CON'] # density class
                    print(density)
                else:
                    veg = 'Nonveg'
                    density = 0
                quit()

                # create a True/False mask of the dimensions of the depth raster
                # True means the presence of  vegetation
                polygon_mask = rasterio.features.geometry_mask(geometries=[geom], out_shape=(raster_obj.shape[0], raster_obj.shape[1]), transform=raster_obj.affine, all_touched=True, invert=True)

                # these are the pixels touched by the polygon
                where = np.where(polygon_mask == True)
                print (len(where[0]))

                # for each of these pixels, we must determine the area covered by the polygon
                if len(where[0]) > 0:
                    # there is some veg
                    if veg_state == True:
                        for p in range(len(where[0])):
                            row = where[0][p]; col = where[1][p]
                            pixels_surveyed.append([row,col])
                            x_l = geotransform[0] + col * geotransform[1]
                            y_t = geotransform[3] + row * geotransform[5]
                            x_r = x_l + geotransform[1]
                            y_b = y_t + geotransform[5]

                            pixelpoly = Polygon([(x_l,y_t), (x_l,y_b), (x_r,y_b), (x_r,y_t)])

                            try: 
                                A = Polygon(geom['coordinates'][0])

                                inter = pixelpoly.intersection(A)
                                inter_area = inter.area
                                relative_coverage = inter_area / pixelpoly.area

                                pixels_pd = add_a_row(pixels_pd, veg, density, relative_coverage, Rbot,  Rsynth, Rsurf, Hydro, Angles, date, row,col)

                            except:
                                print ('geometry error')
                                pass

                            
                    # if there is no veg
                    else:

                        # if the pixel isn't already part of the partly vegetated ones
                        # Actually not doing this here as it's easier to process it in the subsequent pandas
 
                        relative_coverage = 0
                        pixels_pd = add_multiple_rows(pixels_pd, veg, density, relative_coverage, Rbot,  Rsynth, Rsurf, Hydro, Angles, date, where)

    return pixels_pd, pixels_surveyed
                











###################################################################################
###################################################################################
def fill_pixels(shp, cover_arr, density_arr, raster, veg_state = True):

    mask_arr, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(raster)

    # open the depth raster as a reference
    with Raster(raster) as raster_obj: 
        # open the survey shapefile
        with fiona.open(shp) as f: 
            # for each polygon
            for m in range(len(f)):
                print (m , '/', len(f))
                geom = f[m]['geometry'] # the polygon geometry
                if veg_state == True:
                    try:
                        density = f[m]['properties']['CON'] # density class
                    except:
                        density = 3 # an arbitrary value id density does not exist in the record
                    if density == None:
                        density = 4
                else:
                    density = 0

                # create a True/False mask of the dimensions of the depth raster
                # True means the presence of  vegetation
                try:
                    polygon_mask = rasterio.features.geometry_mask(geometries=[geom], out_shape=(raster_obj.shape[0], raster_obj.shape[1]), transform=raster_obj.affine, all_touched=True, invert=True)

                    # these are the pixels touched by the polygon
                    where = np.where(polygon_mask == True)
                    print (len(where[0]))

                    # for each of these pixels, we must determine the area covered by the polygon
                    if len(where[0]) > 0:
                        # there is some veg
                        if veg_state == True:
                            for p in range(len(where[0])):
                                row = where[0][p]; col = where[1][p]
                                x_l = geotransform[0] + col * geotransform[1]
                                y_t = geotransform[3] + row * geotransform[5]
                                x_r = x_l + geotransform[1]
                                y_b = y_t + geotransform[5]

                                pixelpoly = Polygon([(x_l,y_t), (x_l,y_b), (x_r,y_b), (x_r,y_t)])


                                A = Polygon(geom['coordinates'][0])

                                inter = pixelpoly.intersection(A)
                                inter_area = inter.area
                                relative_coverage = inter_area / pixelpoly.area

                                if mask_arr[row,col] == 0 or mask_arr[row,col] == 5:
                                    coverval = min(1, relative_coverage+max(cover_arr[row,col],0))
                                    cover_arr[row,col] = coverval
                                    density_arr[row,col] = max(density_arr[row,col], density)

                        # there is no veg
                        elif veg_state == False:
                            # Create a temporary raster
                            temp = -9999. * np.ones(cover_arr.shape, dtype = np.float)

                            # Fill the AoI
                            temp[where] = 0

                            # Put back to nodata in the emersed areas and the sea
                            for invalid in [1,2,3,4,6]:
                                where_inv = np.where(mask_arr == invalid)
                                temp[where_inv] = -9999.

                            # Put back to nodata where there is already seagrass
                            where_sg = np.where(density_arr > 0)
                            temp[where_sg] = -9999.

                            # fill the cover and density in the determined bare areas
                            where_bare = np.where(temp == 0)
                            cover_arr[where_bare] = 0
                            density_arr[where_bare] = 0
                except:
                    print('invalid or empty geometry')

    return cover_arr, density_arr

###################################################################################
###################################################################################
def make_cover_density_rasters(seagrass_shp, AoI_shp, maskraster, txt_yr):
    mask_arr, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(maskraster)

    cover_arr = -9999. * np.ones(mask_arr.shape, dtype  = np.float)
    density_arr = -9999. * np.ones(mask_arr.shape, dtype  = np.float)

    # Fill the density an veg cover
    cover_arr, density_arr = fill_pixels(seagrass_shp, cover_arr, density_arr, maskraster, veg_state = True)

    # Fill the empty in-AoI pixels
    cover_arr, density_arr = fill_pixels(AoI_shp, cover_arr, density_arr, maskraster, veg_state = False)

    #save the rasters
    cover_shp = '/'.join(seagrass_shp.split('/')[:-1]) + '/' + txt_yr + '_cover_vetted.bil'
    ENVI_raster_binary_from_2d_array((geotransform, inDs), cover_shp, pixelWidth, cover_arr)

    density_shp = '/'.join(seagrass_shp.split('/')[:-1]) + '/' + txt_yr + '_density_vetted.bil'
    ENVI_raster_binary_from_2d_array((geotransform, inDs), density_shp, pixelWidth, density_arr)


    return cover_arr, density_arr































###################################################################################
###################################################################################
def make_pixel_df(Rbot, Rsynth, Rsurf, Hydro, Angles, seagrass_shp, AoI_shp, bathyraster, processed_dir, geotransform, date):

    # Create an empty dataframe to hold polygon information enriched with spectral and depth data
    pixels_pd=pd.DataFrame(columns=['vegetation', 'density_class', 'fraction_veg_cover', 'row', 'col', 'blue', 'green', 'red', 'nir', 'swir', 'blue_bot', 'green_bot', 'red_bot', 'nir_bot', 'swir_bot', 'blue_synth', 'green_synth', 'red_synth', 'nir_synth', 'swir_synth', 'bathy', 'depth', 'waveheight', 'spm', 'solar_zenith', 'sensor_zenith', 'survey'])

    cols = pixels_pd.columns.values

    pixels_surveyed = []

    pixels_pd, pixels_surveyed = fill_pixel_df(pixels_pd, pixels_surveyed, True, Rbot,  Rsynth, Rsurf, Hydro, Angles, seagrass_shp, bathyraster, geotransform, date)


    pixels_pd, pixels_surveyed = fill_pixel_df(pixels_pd, pixels_surveyed, False, Rbot,  Rsynth, Rsurf, Hydro, Angles, AoI_shp, bathyraster, geotransform, date)

    pixels_pd.to_csv(processed_dir + 'calibration_data.csv')

    return pixels_pd



###################################################################################
###################################################################################
def multispectral_Rsynth(bands_dict, wavelength_dict, aw_dict, used_bands, band_arrs, bathy, H, mud, sand, theta_w, theta_v, eta = 0.46949532, gamma = 0.20439084, chla = 3 ):

    # Inputs
    depth = H - bathy
    spm = mud + sand

    # Synthetic bottom reflectance measured by Sonia
    baresoil_ref = {'blue':np.mean([0.0500735,0.0646339,0.0160046,0.0179186]),
                    'green':np.mean([0.0705193,0.0861473,0.0227438,0.0248304]),
                    'red':np.mean([0.0782468,0.0938513,0.0242921,0.0262757]),
                    'nir':np.mean([0.100798,0.121318,0.0267086,0.0353639]),
                    'swir1': np.mean([0,0,0,0])}


    Bigarr = np.zeros([len(band_arrs), len(band_arrs[0]), len(band_arrs[0,0])], dtype = np.float)

    count = 0
    for b in used_bands:
        print ('processing band', bands_dict[b]+1)

        # this is the wavelength of the band centre
        l = wavelength_dict[b]
        aw = aw_dict[b]
        # accomodate for the satellite number
        if bands_dict['blue'] == 1:
            ID = bands_dict[b]-1
        else:
            ID = bands_dict[b]

        band_arr = band_arrs[ID, :, :]
 
        for i in range(len(depth)):
            for j in range(len(depth[0])):
                if H[i,j] > 0.01:
                    
                    # this is reflectance
                    Rsynth = lee.calculate_surface_reflectance_synthetic_Rbot (spm[i,j], baresoil_ref[b], l, depth[i,j], chla, theta_v[i,j], theta_w[i,j], aw)

                    Bigarr[count, i, j] = Rsynth

        count += 1

    return Bigarr


###################################################################################
###################################################################################
def multispectral_Rbot(bands_dict, wavelength_dict, aw_dict, used_bands, band_arrs, bathy, H, mud, sand, theta_w, theta_v, eta = 0.46949532, gamma = 0.20439084, chla = 3 ):

    # Inputs
    depth = H - bathy
    spm = mud + sand

    Bigarr = np.zeros([len(band_arrs), len(band_arrs[0]), len(band_arrs[0,0])], dtype = np.float)

    count = 0
    for b in used_bands:
        print ('processing band', bands_dict[b]+1)

        # this is the wavelength of the band centre
        l = wavelength_dict[b]
        aw = aw_dict[b]
        # accomodate for the satellite number
        if bands_dict['blue'] == 1:
            ID = bands_dict[b]-1
        else:
            ID = bands_dict[b]

        band_arr = band_arrs[ID, :, :]
 
        for i in range(len(depth)):
            for j in range(len(depth[0])):
                if H[i,j] > 0.01:
                    
                    # this is reflectance
                    Rbot = lee.calculate_bottom_reflectance_known_SPM (spm[i,j], band_arr[i,j], l, depth[i,j], chla, theta_v[i,j], theta_w[i,j], aw)
                    Bigarr[count, i, j] = Rbot

        count += 1

    return Bigarr



###################################################################################
###################################################################################
def reproject_shp(shp, EPSG):
    shapes = gpd.read_file(shp)
    shapes_goodcrs = shapes.to_crs({'init': 'epsg:'+ str(EPSG)}) #
    shapes_goodcrs.to_file(driver = 'ESRI Shapefile', filename = shp[:-4]+"_"+str(EPSG)+".shp")
    return shp[:-4]+"_"+str(EPSG)+".shp"


###################################################################################
###################################################################################
def load_rasters(scene_dir, processed_dir, scene, bands_dict, bathyraster, maskraster):

    blueraster = processed_dir + scene + '_sr_band' + str(bands_dict['blue']+1) + '_laguna3003.bil'
    blue, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(blueraster)

    greenraster = processed_dir + scene + '_sr_band'+str(bands_dict['green']+1)+'_laguna3003.bil'
    green, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(greenraster)

    redraster = processed_dir + scene + '_sr_band'+str(bands_dict['red']+1)+'_laguna3003.bil'
    red, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(redraster)

    nirraster = processed_dir + scene + '_sr_band'+str(bands_dict['nir']+1)+'_laguna3003.bil'
    nir, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(nirraster)

    bathy, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(bathyraster)
    mask, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(maskraster)

    ###
    mask = mask[:-1] # get rid of the last row of mask ... FIX THIS ANOTHER WAY! HOW????
    ###

    Hraster = processed_dir + 'IDRO_nearest_laguna3003.bil'
    H, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(Hraster)

    mudraster = processed_dir + 'SEDIm_nearest_laguna3003.bil'
    mud, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(mudraster)

    sandraster = processed_dir + 'SEDIs_nearest_laguna3003.bil'
    sand, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(sandraster)

    # angles are tricky
    scenefiles = os.listdir(scene_dir)
    solarbil = [i for i in scenefiles if i.endswith('solar_B01_laguna3003.bil')][0]
    sensorbil = [i for i in scenefiles if i.endswith('sensor_B01_laguna3003.bil')][0]

    theta_w_arr, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(scene_dir+solarbil)
    theta_v_arr, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(scene_dir+sensorbil)

    theta_w = theta_w_arr[:,:, 1] * np.pi/(180*100) # original value is in deg*100. This is now in rad
    theta_v = theta_v_arr[:,:, 1] * np.pi/(180*100) # original value is in deg*100. This is now in rad

    # Convert angles to subsurface angles
    n_air = 1
    n_water = 1.33

    theta_v = np.arcsin(np.sin(theta_v) * n_water / n_air) # Snell's Law
    theta_w = np.arcsin(np.sin(theta_w) * n_water / n_air) # Snell's Law

    # Get spm
    spm = mud + sand

    return blue, green, red, nir, bathy, mask, H, spm, theta_w, theta_v, pixelWidth, (geotransform, inDs)

###################################################################################
###################################################################################
def load_srfrasters(scene_dir, processed_dir, scene, bands_dict):

    blueraster = processed_dir + scene + '_sr_band' + str(bands_dict['blue']+1) + '_laguna3003.bil'
    blue, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(blueraster)

    greenraster = processed_dir + scene + '_sr_band'+str(bands_dict['green']+1)+'_laguna3003.bil'
    green, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(greenraster)

    redraster = processed_dir + scene + '_sr_band'+str(bands_dict['red']+1)+'_laguna3003.bil'
    red, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(redraster)

    nirraster = processed_dir + scene + '_sr_band'+str(bands_dict['nir']+1)+'_laguna3003.bil'
    nir, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(nirraster)

    # Scale to undo the ESPA integral storage format
    blue = blue/10000; green = green/10000; red = red/10000; nir = nir/10000

    # angles are tricky
    scenefiles = os.listdir(scene_dir)
    solarbil = [i for i in scenefiles if i.endswith('solar_B01_laguna3003.bil')][0]
    sensorbil = [i for i in scenefiles if i.endswith('sensor_B01_laguna3003.bil')][0]

    theta_w_arr, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(scene_dir+solarbil)
    theta_v_arr, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(scene_dir+sensorbil)

    theta_w = theta_w_arr[:,:, 1] * np.pi/(180*100) # original value is in deg*100. This is now in rad
    theta_v = theta_v_arr[:,:, 1] * np.pi/(180*100) # original value is in deg*100. This is now in rad

    # Convert angles to subsurface angles
    n_air = 1
    n_water = 1.33

    theta_v = np.arcsin(np.sin(theta_v) * n_water / n_air) # Snell's Law
    theta_w = np.arcsin(np.sin(theta_w) * n_water / n_air) # Snell's Law

    return blue, green, red, nir, theta_w, theta_v, pixelWidth, (geotransform, inDs)

###################################################################################
###################################################################################
def load_allrasters(scene_dir, processed_dir, scene, bands_dict, bathyraster):

    blueraster = processed_dir + scene + '_sr_band' + str(bands_dict['blue']+1) + '_laguna3003.bil'
    blue, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(blueraster)

    greenraster = processed_dir + scene + '_sr_band'+str(bands_dict['green']+1)+'_laguna3003.bil'
    green, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(greenraster)

    redraster = processed_dir + scene + '_sr_band'+str(bands_dict['red']+1)+'_laguna3003.bil'
    red, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(redraster)

    nirraster = processed_dir + scene + '_sr_band'+str(bands_dict['nir']+1)+'_laguna3003.bil'
    nir, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(nirraster)

    swirraster = processed_dir + scene + '_sr_band'+str(bands_dict['swir1']+1)+'_laguna3003.bil'
    swir, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(swirraster)

    bathy, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(bathyraster)

    Hraster = processed_dir + 'IDRO_nearest_laguna3003.bil'
    H, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(Hraster)

    waveraster = processed_dir + 'MO_nearest_laguna3003.bil'
    waveheight, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(waveraster)

    mudraster = processed_dir + 'SEDIm_nearest_laguna3003.bil'
    mud, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(mudraster)

    sandraster = processed_dir + 'SEDIs_nearest_laguna3003.bil'
    sand, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(sandraster)

    # angles are tricky
    scenefiles = os.listdir(scene_dir)
    solarbil = [i for i in scenefiles if i.endswith('solar_B01_laguna3003.bil')][0]
    sensorbil = [i for i in scenefiles if i.endswith('sensor_B01_laguna3003.bil')][0]

    theta_w_arr, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(scene_dir+solarbil)
    theta_v_arr, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(scene_dir+sensorbil)

    theta_w = theta_w_arr[:,:, 1] * np.pi/(180*100) # original value is in deg*100. This is now in rad
    theta_v = theta_v_arr[:,:, 1] * np.pi/(180*100) # original value is in deg*100. This is now in rad

    # Convert angles to subsurface angles
    n_air = 1
    n_water = 1.33

    theta_v = np.arcsin(np.sin(theta_v) * n_water / n_air) # Snell's Law
    theta_w = np.arcsin(np.sin(theta_w) * n_water / n_air) # Snell's Law

    return blue, green, red, nir, swir, bathy, H, waveheight, mud, sand, theta_w, theta_v, pixelWidth, (geotransform, inDs)

###################################################################################
###################################################################################
def satnumber_dicts(satnumber):
    if satnumber == '7':
        bands_dict = {'blue':0, 'green':1, 'red':2, 'nir':3, 'swir1':4, 'swir2':5}
        wavelength_dict = {'blue':485, 'green':560, 'red':660, 'nir':835, 'swir1':1650, 'swir2':2700}
        aw_dict = {'blue':0.0136, 'green':0.0619, 'red':0.410, 'nir':3.08, 'swir1':6.07, 'swir2':193}

    elif satnumber == '8':
        # https://landsat.gsfc.nasa.gov/landsat-8/landsat-8-bands/
        bands_dict = {'aerosol':0, 'blue':1, 'green':2, 'red':3, 'nir':4, 'swir1':5, 'swir2':6}
        wavelength_dict = {'aerosol':443, 'blue':482.5, 'green':562.5, 'red':655, 'nir':865, 'swir1':1610, 'swir2':2200}

        # THIS IS NOT CORRECT
        aw_dict = {'aerosol':0.00696, 'blue':0.0136, 'green':0.0619, 'red':0.410, 'nir':4.28, 'swir1':6.80, 'swir2':1.93}

    # References for pure water absorption:
    # Pope and Fry 1993 (440-727.5 nm)
    # Kou Labrie, Chylek 1993 (730-2000 nm) at 22°C - NOPE
    # Palmer and Williams 1974 (730-2700 nm)and relation refr_index_k = wavelength_l * absorption_a / 4*pi
    # note: swir2 is not very reliable

    return bands_dict, wavelength_dict, aw_dict

###################################################################################
###################################################################################
def read_outfile (outfile, Nnodes, Nelems, Nvals, coords):
    with open(outfile,'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    f.close()

    # get rid of header
    data = data[4:]

    # Read nodes and element values
    H = [] # water level
    Qx = []; Qy = [] # discharge, in x and y

    # Go through the data in batches of Nval
    for n in range(0,int(len(data)/Nvals)):
        # get levels
        H.append(data[:Nnodes])
        # delete the elements in data
        data = data[Nnodes:]

        # same for qx
        Qx.append(data[:Nelems])
        data = data[Nelems:]

        # same for qy
        Qy.append(data[:Nelems])
        data = data[Nelems:]

    H = np.asarray(H)
    Qx = np.asarray(Qx)
    Qy = np.asarray(Qy)

    # prepare the levels for x, y, z extraction
    H_last = H[-1].reshape(Nnodes,1)
    A = np.concatenate( (coords, H_last), axis = 1)

    return A

###################################################################################
###################################################################################
def read_geofile(geofile):
    # Read the number of nodes and elements in the .geo
    # Read the position of the nodes
    geo = open(geofile, 'r')

    # initiate coordinates
    coords = []

    counter = 0
    for line in geo:
        # Count the elements
        if counter == 2:
            Nnodes = int(line.split()[0])
            Nelems = int(line.split()[2])

        # Retrieve node positions
        if counter >= 7 and counter < 7 + Nnodes:
            coords.append( [np.float(line.split()[0]), np.float(line.split()[1]) ])     
        
        counter+=1

    coords = np.asarray(coords)

    # Number of values in one timestep
    Nvals = Nnodes + 2*Nelems

    return Nnodes, Nelems, Nvals, coords



##################################################################################
##################################################################################
def array_to_xyz_csv(arr, mydir, myname):
    xyz = mydir + myname + ".xyz"
    csv = mydir + myname + ".csv"

    f = open(xyz, "w", encoding = 'utf-8')
    for i in range(len(arr)):
        f.write(str(arr[i][0]) + ' ' + str(arr[i][1]) + ' ' + str(arr[i][2]) + '\n')
    f.close()

    # make a .csv file with pandas
    arr_pd = pd.DataFrame({'Easting':arr[:,0], 'Northing':arr[:,1], 'Elevation':arr[:,2]})
    arr_pd.to_csv(csv)


##################################################################################
##################################################################################
def make_anglerasters(Landsat_number, scene_dir, angle_file):

    #for Landsat 7
    if Landsat_number == "7":
        angle_software_dir = "/home/willgoodwin/PostDoc/Padova/Scripts/RS_Software/landsat_angles/"
        angle_img_start = 'angle_'
        sp.run(['./landsat_angles', angle_file], cwd = angle_software_dir)

    # for Landsat 8
    if Landsat_number == "8":
        angle_software_dir = "/home/willgoodwin/PostDoc/Padova/Scripts/RS_Software/l8_angles/"
        angle_img_start = scene_dir.split('/')[-2]
        sp.run(['./l8_angles', angle_file, 'BOTH', '1', '-b', '1'], cwd = angle_software_dir)


    # this produces WAY TOO MANY files. Remove all except the band 1 which you can rename.
    files = os.listdir(angle_software_dir)
  
    angle_img_list = [i for i in files if i.startswith(angle_img_start)]

    for m in angle_img_list:
        if m.endswith('B01.img') or m.endswith('B01.img.hdr'):
            print ('got here')
            sp.run(['mv', angle_software_dir+m, scene_dir+m])
        else:
            print ('got here')
            sp.run(['rm', angle_software_dir+m])


################################################################################
################################################################################
def find_espa_scene(LandSat_dir, processed_dir, scene_no_underscore):

    if not os.path.isdir(processed_dir):
        sp.call(['mkdir', processed_dir])

    # Locate scene
    files = os.listdir(LandSat_dir)

    try:
        compressed_scene = [i for i in files if scene_no_underscore[8:22] in i][0]
        print (compressed_scene)
        found = True
    except:
        print ('file does not exist, moving on')
        found = False
    
    return found

################################################################################
################################################################################
def extract_espa_products(LandSat_dir, processed_dir, scene_no_underscore):

    if not os.path.isdir(processed_dir):
        sp.call(['mkdir', processed_dir])

    # Locate scene
    files = os.listdir(LandSat_dir)

    try:
        compressed_scene = [i for i in files if scene_no_underscore[8:22] in i][0]
        sp.call(['cp', LandSat_dir + compressed_scene, processed_dir + compressed_scene])
    except:
        print ('file already moved, moving on')
        l2files = os.listdir(processed_dir)
        compressed_scene = [i for i in l2files if scene_no_underscore[8:22] in i][0]
    
    # Decompress files
    tarfile = tar.open(processed_dir+compressed_scene)
    tarfile.extractall(path = processed_dir)
    tarfile.close()

    # move the tarball back to its original place
    if os.path.isfile(processed_dir + compressed_scene):
         sp.call(['rm', processed_dir + compressed_scene])


################################################################################
################################################################################
def stations2panda(allstations_dirlist, wind_stations, tide_stations):
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
        for w in wind_stations:
            if files[f].startswith(w):
                iswind = True; break

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
        if len([ele for ele in tide_stations+wind_stations if(ele in c)]) == 0:
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
def extract_overpasstime (preselected, scenes_csv, metadata_file, diff_fm_GMT, datetime_list, index):

    mtl = open(metadata_file,'r').read().split('\n')
    for m in mtl:
        if 'DATE_ACQUIRED' in m:
            acq_date_str = m.split(' = ')[1]
        if 'SCENE_CENTER_TIME' in m:
            acq_time_str = m.split(' = ')[1][1:9]
        
    acq_datetime = datetime.strptime(acq_date_str+acq_time_str, "%Y-%m-%d%H:%M:%S")+diff_fm_GMT
    datetime_list.append(acq_datetime)
    preselected.loc[index, 'datetime'] = acq_datetime
    preselected.to_csv(scenes_csv[:-4]+'_overpasstimes.csv')

    return preselected, datetime_list

################################################################################
################################################################################
def landsat_scenes_to_panda (scenes_csv):
    my_selection = pd.read_csv(scenes_csv)

    # Add date information
    datelist = []
    datetime_list = []
    for n in range(len(my_selection)):
        mydate = my_selection['scene'].iloc[n].split('_')[3]
        datelist.append( datetime.strptime(mydate, '%Y%m%d' ) )
    my_selection['datetime'] = datelist

    return my_selection, datelist

################################################################################
################################################################################
def grid_xyz(mydir, myname, ext_raster, cutline, gcs_path):

    my_xyz = mydir +myname+ ".xyz"
    my_csv = mydir +myname+ ".csv"
    my_vrt = mydir +myname+ ".vrt"
    my_tif = mydir + myname + '_nearest.tif'
    my_bil = my_tif[:-4]+'_laguna3003.bil'

    make_vrt(mydir, myname)


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
        os.system('gdal_grid --config GDAL_DATA ' + gcs_path +' -ot Float32 -of GTIFF -zfield Elevation -a nearest -a_srs EPSG:3003  -txe ' + str(xs) + ' ' + str(xe) + ' -tye ' + str(ys) + ' ' + str(ye) + ' -outsize '+ str(len(arr[0]))+' '+ str(len(arr)) +' -of GTiff -ot Float64 -l '+ myname + ' ' + my_vrt + ' ' + my_tif)
        #os.system('gdal_grid -ot Float32 -of GTIFF -zfield Elevation -a nearest -a_srs EPSG:3003  -txe ' + str(xs) + ' ' + str(xe) + ' -tye ' + str(ys) + ' ' + str(ye) + ' -outsize '+ str(len(arr[0]))+' '+ str(len(arr)) +' -of GTiff -ot Float64 -l '+ myname + ' ' + my_vrt + ' ' + my_tif)

        # Now get this into a bil file
        # PB how correct the systematic gdalwarp shift? Specify the  target extent!
        print ('correcting shift ...')
        os.system('gdalwarp --config GDAL_DATA ' + gcs_path + ' -overwrite -of ENVI -cutline '+ cutline + ' -crop_to_cutline -te ' +  str(xs) + ' ' + str(ye) + ' ' + str(xe) + ' ' +  str(ys) + ' ' + my_tif + ' ' + my_bil )

################################################################################
################################################################################
def grid_shp(mydir, myname, ext_raster, cutline, gcs_path):


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


################################################################################
################################################################################
def make_vrt(mydir, myname):

    my_xyz = mydir +myname+ ".xyz"
    my_csv = mydir +myname+ ".csv"
    my_vrt = mydir +myname+ ".vrt"

    A = []
    # Open the .xyz file
    f = open(my_xyz, "r", encoding = 'utf-8')
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
    f.write('    <OGRVRTLayer name="'+myname+'">' + '\n')
    f.write('        <SrcDataSource>'+my_csv+'</SrcDataSource>' + '\n')
    f.write('        <GeometryType>wkbPoint</GeometryType>' + '\n')
    f.write('        <GeometryField encoding="PointFromColumns" x="Easting" y="Northing" z="Elevation"/>' + '\n')
    f.write('    </OGRVRTLayer>' + '\n')
    f.write('</OGRVRTDataSource>' + '\n')
    f.close()

    #gdal_cmd = 'gdal_grid -a invdist:power=2.0:smoothing=1.0 -zfield "Concentration" -of GTiff -ot Float64 -l %s %s %s' % (lyr_name, vrt_fn, out_tif)


################################################################################
################################################################################

################################################################################
################################################################################

################################################################################
################################################################################

################################################################################
################################################################################

################################################################################
################################################################################


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

    #print 'WOOO'
    #print envidata
    #print original_geotransform
    #print inDs
    #print inDs.GetProjection()

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


#####################################################################################
#####################################################################################
def cool_violin_1D(position, data, step, axis, quantiles = [10,25,50,75,90], kerntype = 'gaussian', colour = 'k', scale = 1):

    # stats
    q_values = []
    for q in quantiles:
        q_values.append(np.percentile(data,q))

    datarange = np.arange(0.8*min(data), 1.2*max(data)+step, step)
    fmt_datarange = datarange[:, np.newaxis]

    data = data.reshape(-1, 1)
    kde = KernelDensity(kernel=kerntype, bandwidth=5*step).fit(data)
    density = np.exp(kde.score_samples(fmt_datarange))

    upupper = 0; upper = 0; lower = 0; lolower = 0

    for q in range(len(q_values)):
        #print ('quartile', q)

        lesser = np.where(datarange <= q_values[q])[0]
        greater = np.where(datarange >= q_values[q])[0]

        procedeless = 0; procedegreat = 0
        
        if len(lesser) >= 1:
            lesser = lesser[-1]
            procedeless = 1
        else:
            print ('error', lesser, q_values[q], min(datarange))

        if len(greater) >=1:
            greater = greater[0]
            procedegreat = 1
        else:
            print ('error', greater, q_values[q], max(datarange))

        if procedeless == 1:
            if q == 0:
                lolower = lesser
        if procedegreat == 1:
            if q == 4:
                upupper = greater

        if q == 2: # draw wide median line
            axis.plot([position-density[greater], position+density[greater]], [q_values[q], q_values[q]], c = colour, alpha = 0.9, lw = 1.5)
            axis.plot([position-density[lesser], position+density[lesser]], [q_values[q], q_values[q]], c = colour, alpha = 0.9, lw = 1.5)

        if q == 1: #first quartile
            lower = lesser
        if q == 3: #third quartile
            upper = greater
        
    # plot between the 10th and 90th percentile
    axis.plot(position+scale*density[lolower:upupper], datarange[lolower:upupper], lw = 1, c = colour)
    axis.plot(position-scale*density[lolower:upupper], datarange[lolower:upupper], lw = 1, c = colour)
    
    # plot the rest of the density
    axis.plot(position+scale*density, datarange, lw = 0.5, c = colour, alpha = 0.5)
    axis.plot(position-density, datarange, lw = 0.5, c = colour, alpha = 0.5)

    # fill between the quartiles
    axis.fill_betweenx(datarange[lower:upper], position-scale*density[lower:upper], position+scale*density[lower:upper], lw = 1, facecolor = colour, alpha = 0.5)




#####################################################################################
#####################################################################################
def half_violin_1D(position, data, step, axis, side, kerntype = 'gaussian', colour = 'k', linetype = '-', scale = 1):

    # stats

    datarange = np.arange(0.8*min(data), 1.2*max(data)+step, step)
    fmt_datarange = datarange[:, np.newaxis]

    data = data.reshape(-1, 1)
    kde = KernelDensity(kernel=kerntype, bandwidth=5*step).fit(data)
    density = np.exp(kde.score_samples(fmt_datarange))

    if side == 'right':
        axis.plot(position+scale*density, datarange, linetype, lw = 1, c = colour)
    elif side == 'left':
        axis.plot(position-scale*density, datarange, linetype, lw = 1, c = colour)
    










###################################################################################
###################################################################################
def getContiguousAreas(data, value, min_pixels = 1, contiguity = 'queen'):
    from scipy import ndimage
    from scipy.ndimage.measurements import label

    '''
    Get pixels that come from the same contigous area.

    Args:
        data: A numpy array
        value: Pixel value to include in contiguous_area (e.g. True for forest)
        min_area: What minimum area should be included (number of pixels)
        contuguity: Set to rook (4-way) or queen (8-way) connectivity constraint. Defaults to 'queen'.

    Returns:
        A binary array of pixels that meet the conditions
    '''

    assert contiguity in ['rook', 'queen'], "Contiguity must be either 'rook' or 'queen'. Input recieved was <%s>."%str(contiguity)

    # Extract area that meets condition
    binary_array = (data == value) * 1

    # If masked, we use this flag to save the mask for later.
    masked = np.ma.isMaskedArray(binary_array)

    # Set masked areas to non-contiguous value
    if masked:
        mask = np.ma.getmaskarray(binary_array)
        binary_array = binary_array.filled(0)

    # Label contigous areas with a number
    if contiguity == 'rook':
        structure = ndimage.generate_binary_structure(2,1) # 4-way connectivity
    elif contiguity == 'queen':
        structure = ndimage.generate_binary_structure(2,2) # 8-way connectivity

    location_id, n_areas = label(binary_array, structure = structure)

    # Get count of each value in array
    label_area = np.bincount(location_id.flatten())[1:]

    # Find those IDs that meet minimum area requirements
    include_id = np.arange(1, n_areas + 1)[label_area >= min_pixels]

    # Get a binary array of location_id pixels that meet the minimum area requirement
    contiguous_area = np.in1d(location_id, include_id).reshape(data.shape).astype(np.bool)

    # Return an array giving values to each area
    location_id[contiguous_area == False] = 0

    # Re-number location_ids 1 to n, given that some unique value shave now been removed
    location_id_unique, location_id_indices = np.unique(location_id, return_inverse = True)
    location_id = np.arange(0, location_id_unique.shape[0], 1)[location_id_indices].reshape(data.shape)

    # Put mask back in if input was a masked array
    if masked:
        contiguous_area = np.ma.array(contiguous_area, mask = mask)
        location_id = np.ma.array(location_id, mask = mask)

    return contiguous_area, location_id



###################################################################################
###################################################################################
def load_vegetation_data(veg_dir, this_time, mask, thresh_cover = 0.5):  
    # Load the vegetation cover and density
    cover, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(veg_dir+str(this_time.year)+'/'+str(this_time.year)+'_cover.bil')
    density, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(veg_dir+str(this_time.year)+'/'+str(this_time.year)+'_density.bil')

    shapes = gpd.read_file(glob.glob (os.path.join (veg_dir+str(this_time.year)+'/', '*fixed_sgpart_3003.shp'))[0])
    shapes = shapes['geometry']


    ###
    cover = cover[:-1] # get rid of the last row of mask ... FIX THIS ANOTHER WAY! HOW????
    density = density[:-1] # get rid of the last row of mask ... FIX THIS ANOTHER WAY! HOW????
    ###

    cover[cover < thresh_cover] = 0
    cover[cover >= thresh_cover] = 1

    cover = np.ma.masked_where(mask != 0, cover)
    density = np.ma.masked_where(mask != 0, density)



    return cover, density, shapes, pixelWidth, (geotransform, inDs)


###################################################################################
###################################################################################
def load_digitised_vegetation_data(thisdir, this_time, mask, thresh_cover = 0.5): 
    # Load the vegetation cover and density
    cover, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(thisdir+str(this_time.year)+'_cover_vetted.bil')
    density, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(thisdir+str(this_time.year)+'_density_vetted.bil')

    shapes = gpd.read_file(thisdir+'Veg_3003.shp')
    shapes = shapes['geometry']

    ###
    cover = cover[:-1] # get rid of the last row of mask ... FIX THIS ANOTHER WAY! HOW????
    density = density[:-1] # get rid of the last row of mask ... FIX THIS ANOTHER WAY! HOW????
    ###

    cover[cover < thresh_cover] = 0
    cover[cover >= thresh_cover] = 1

    cover = np.ma.masked_where(mask != 0, cover)
    density = np.ma.masked_where(mask != 0, density)



    return cover, density, shapes, pixelWidth, (geotransform, inDs)



###################################################################################
###################################################################################
def load_vetted_vegetation_data(veg_dir, this_time, mask, thresh_cover = 0.5): 
    print (veg_dir+str(this_time.year)+'/'+str(this_time.year)+'_cover_vetted.bil') 
    # Load the vegetation cover and density
    cover, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(veg_dir+str(this_time.year)+'/'+str(this_time.year)+'_cover_vetted.bil')
    density, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(veg_dir+str(this_time.year)+'/'+str(this_time.year)+'_density_vetted.bil')

    #print ( glob.glob (os.path.join (veg_dir+str(this_time.year)+'/', '*vetted*_fixedsgpart_3003.shp') ) [0] )

    if this_time.year == 2002:
        file = veg_dir + '/' +'2002'+ '/' + 'Fanerogame2002_sgpart.shp'
        shapes = gpd.read_file(file); shapes = shapes['geometry']

    if this_time.year == 2004:
        file = veg_dir + '/' +'2004'+ '/' + 'vetted_sg_2004_fixed_sgpart_3003.shp'
        shapes = gpd.read_file(file); shapes = shapes['geometry']

    if this_time.year == 2009:
        file = veg_dir + '/' +'2009'+ '/' + 'vetted_sg_2009_fixed_sgpart_3003.shp'
        shapes = gpd.read_file(file); shapes = shapes['geometry']

    if this_time.year == 2010:
        file = veg_dir + '/' +'2010'+ '/' + 'rilievi_fanerogame_2010_vetted_fixedsgpart_3003.shp'
        shapes = gpd.read_file(file); shapes = shapes['geometry']

    if this_time.year == 2017:
        filesud = veg_dir + '/' +'2017'+ '/' + 'Fanerogame_sud_2017_vetted_fixedsgpart.shp'
        filecen = veg_dir + '/' +'2017'+ '/' + 'Fanerogame_centro_2017_vetted_fixedsgpart.shp'
        filenor = veg_dir + '/' +'2017'+ '/' + 'Fanerogame_nord_2017_vetted_fixedsgpart.shp'
        
        shapesud = gpd.read_file(filesud); shapecen = gpd.read_file(filesud); shapenor = gpd.read_file(filenor)
        shapes=shapesud.append(shapecen, ignore_index = True)
        shapes=shapes.append(shapenor, ignore_index = True)

        shapes = shapes['geometry']




    ###
    cover = cover[:-1] # get rid of the last row of mask ... FIX THIS ANOTHER WAY! HOW????
    density = density[:-1] # get rid of the last row of mask ... FIX THIS ANOTHER WAY! HOW????
    ###

    cover[cover < thresh_cover] = 0
    cover[cover >= thresh_cover] = 1

    cover = np.ma.masked_where(mask != 0, cover)
    density = np.ma.masked_where(mask != 0, density)



    return cover, density, shapes, pixelWidth, (geotransform, inDs)


###################################################################################
###################################################################################
def gettime(overpass_times, scene):
    this_time = overpass_times[overpass_times['scene'] == scene]['datetime'].values[0]
    this_time = datetime.strptime(this_time, '%Y-%m-%d %H:%M:%S')
    return this_time


###################################################################################
###################################################################################
def load_rasters(scene_dir, processed_dir, scene, bands_dict, bathyraster, maskraster):

    blueraster = processed_dir + scene + '_sr_band' + str(bands_dict['blue']+1) + '_laguna3003.bil'
    blue, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(blueraster)
    blue = blue / 10000

    greenraster = processed_dir + scene + '_sr_band'+str(bands_dict['green']+1)+'_laguna3003.bil'
    green, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(greenraster)
    green = green / 10000

    redraster = processed_dir + scene + '_sr_band'+str(bands_dict['red']+1)+'_laguna3003.bil'
    red, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(redraster)
    red = red / 10000

    nirraster = processed_dir + scene + '_sr_band'+str(bands_dict['nir']+1)+'_laguna3003.bil'
    nir, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(nirraster)
    nir = nir / 10000

    bathy, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(bathyraster)
    mask, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(maskraster)

    ###
    mask = mask[:-1] # get rid of the last row of mask ... FIX THIS ANOTHER WAY! HOW????
    ###
    #
    mask [mask == 5] = 0 # get the channels to be tif as well
    #

    Hraster = processed_dir + 'IDRO_nearest_laguna3003.bil'
    H, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(Hraster)

    mudraster = processed_dir + 'SEDIm_nearest_laguna3003.bil'
    mud, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(mudraster)

    sandraster = processed_dir + 'SEDIs_nearest_laguna3003.bil'
    sand, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(sandraster)

    # angles are tricky
    scenefiles = os.listdir(scene_dir)
    solarbil = [i for i in scenefiles if i.endswith('solar_B01_laguna3003.bil')][0]
    sensorbil = [i for i in scenefiles if i.endswith('sensor_B01_laguna3003.bil')][0]

    theta_w_arr, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(scene_dir+solarbil)
    theta_v_arr, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(scene_dir+sensorbil)

    theta_w = theta_w_arr[:,:, 1] * np.pi/(180*100) # original value is in deg*100. This is now in rad
    theta_v = theta_v_arr[:,:, 1] * np.pi/(180*100) # original value is in deg*100. This is now in rad

    # Convert angles to subsurface angles
    n_air = 1
    n_water = 1.33

    theta_v = np.arcsin(np.sin(theta_v) * n_water / n_air) # Snell's Law
    theta_w = np.arcsin(np.sin(theta_w) * n_water / n_air) # Snell's Law

    # Get spm
    spm = mud + sand

    # mask the spectral arrays
    blue = np.ma.masked_where(blue < 0, blue)
    blue = np.ma.masked_where(mask != 0, blue)

    green = np.ma.masked_where(green < 0, green)
    green = np.ma.masked_where(mask != 0, green)

    red = np.ma.masked_where(red < 0, red)
    red = np.ma.masked_where(mask != 0, red)

    nir = np.ma.masked_where(nir < 0, nir)
    nir = np.ma.masked_where(mask != 0, nir)

    # Mask the other arrays
    bathy = np.ma.masked_where(mask != 0, bathy)
    H = np.ma.masked_where(mask != 0, H)
    spm = np.ma.masked_where(mask != 0, spm)



    return blue, green, red, nir, bathy, mask, H, spm, theta_w, theta_v, pixelWidth, (geotransform, inDs)



###################################################################################
###################################################################################
def makerowcol(blue):
    loc_rows = np.zeros(blue.shape, dtype = np.float); loc_cols = np.zeros(blue.shape, dtype = np.float)
    for k in range(len(loc_rows)): loc_rows[k,:] = k
    for k in range(len(loc_cols[0])): loc_cols[:,k] = k

    return loc_rows, loc_cols

###################################################################################
###################################################################################
def load_rasters_no_WWTM(dwld_dir, scene, bathyraster, maskraster, zoneraster):

    scene_dir = dwld_dir + scene + '/'
    processed_dir = dwld_dir + scene + '/Level2/'

    satnumber = scene.split('_')[0][-1]
    # Identify relevant bands and their associated properties
    bands_dict, wavelength_dict, aw_dict = fn.satnumber_dicts (satnumber)

    blueraster = processed_dir + scene + '_sr_band' + str(bands_dict['blue']+1) + '_laguna3003.bil'
    blue, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(blueraster)
    blue = blue / 10000

    greenraster = processed_dir + scene + '_sr_band'+str(bands_dict['green']+1)+'_laguna3003.bil'
    green, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(greenraster)
    green = green / 10000

    redraster = processed_dir + scene + '_sr_band'+str(bands_dict['red']+1)+'_laguna3003.bil'
    red, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(redraster)
    red = red / 10000

    nirraster = processed_dir + scene + '_sr_band'+str(bands_dict['nir']+1)+'_laguna3003.bil'
    nir, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(nirraster)
    nir = nir / 10000

    bathy, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(bathyraster)
    mask, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(maskraster)
    zones, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(zoneraster)

    ###
    mask = mask[:-1] # get rid of the last row of mask ... FIX THIS ANOTHER WAY! HOW????
    zones = zones[:-1] # get rid of the last row of mask ... FIX THIS ANOTHER WAY! HOW????
    ###
    #
    #mask [mask == 5] = 0 # get the channels to be tif as well
    #

    # angles are tricky
    scenefiles = os.listdir(scene_dir)
    solarbil = [i for i in scenefiles if i.endswith('solar_B01_laguna3003.bil')][0]
    sensorbil = [i for i in scenefiles if i.endswith('sensor_B01_laguna3003.bil')][0]

    theta_w_arr, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(scene_dir+solarbil)
    theta_v_arr, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(scene_dir+sensorbil)

    theta_w = theta_w_arr[:,:, 1] * np.pi/(180*100) # original value is in deg*100. This is now in rad
    theta_v = theta_v_arr[:,:, 1] * np.pi/(180*100) # original value is in deg*100. This is now in rad

    # Convert angles to subsurface angles
    n_air = 1
    n_water = 1.33

    theta_v = np.arcsin(np.sin(theta_v) * n_water / n_air) # Snell's Law
    theta_w = np.arcsin(np.sin(theta_w) * n_water / n_air) # Snell's Law


    # mask the spectral arrays
    blue = np.ma.masked_where(blue < 0, blue)
    blue = np.ma.masked_where(mask != 0, blue)

    green = np.ma.masked_where(green < 0, green)
    green = np.ma.masked_where(mask != 0, green)

    red = np.ma.masked_where(red < 0, red)
    red = np.ma.masked_where(mask != 0, red)

    nir = np.ma.masked_where(nir < 0, nir)
    nir = np.ma.masked_where(mask != 0, nir)
    
    theta_v = np.ma.masked_where(mask != 0, theta_v)
    theta_w = np.ma.masked_where(mask != 0, theta_w)

    # Mask the other arrays
    bathy = np.ma.masked_where(mask != 0, bathy)
    zones = np.ma.masked_where(mask != 0, zones)

    """
    print ('Number of pixels in each zone:')
    print ('1:', len(np.where(zones == 1)[0]))
    print ('2:', len(np.where(zones == 2)[0]))
    print ('3:', len(np.where(zones == 3)[0]))
    print ('4:', len(np.where(zones == 4)[0]))
    print ('5:', len(np.where(zones == 5)[0]))
    """


    return blue, green, red, nir, bathy, mask, zones, theta_w, theta_v, pixelWidth, (geotransform, inDs)


###################################################################################
###################################################################################
def extract_tide(tides, bathy, this_time):

    # then extract the wind speed and tide level (used in the weather figure file)
    post = datetime(this_time.year, this_time.month, this_time.day, this_time.hour+1)
    pre = datetime(this_time.year, this_time.month, this_time.day, this_time.hour)
    tides_now = tides.loc[pre:post]

    # add each component as a single float
    mytide = np.nanmean(np.asarray(tides_now))

    H_unit = mytide * np.ones(bathy.shape, dtype = float)

    return H_unit

###################################################################################
###################################################################################
def extract_tidewind(tides, wind, bathy, this_time):

    # then extract the wind speed and tide level (used in the weather figure file)
    post = datetime(this_time.year, this_time.month, this_time.day, this_time.hour+1)
    pre = datetime(this_time.year, this_time.month, this_time.day, this_time.hour)
    tides_now = tides.loc[pre:post]


    post = datetime(this_time.year, this_time.month, this_time.day)
    pre = datetime(this_time.year, this_time.month, this_time.day - 1)
    wind_now = wind.loc[pre:post]

    # add each component as a single float
    mytide = np.nanmean(np.asarray(tides_now))
    mywind = np.nanmean(np.asarray(wind_now))

    H_unit = mytide * np.ones(bathy.shape, dtype = float)
    W_unit = mywind * np.ones(bathy.shape, dtype = float)

    return H_unit, W_unit


###################################################################################
###################################################################################
def format_one(mask, features, target, binary = True):
    if binary == True:
        target[target>0] = 1

    # Make them into a RF-friendly format
    #mask [mask == 5.] = 0 # convert channels to possible seagrass

    # Ravel key arrays: mask, aband with potential nodata, target array
    # Ravel instead of flatten because no change will occur (allows faster, non-copy method)
    # we now enter a realm of flat arrays
    rm = mask.ravel()
    rf1 = features[0].ravel()
    rt = target.ravel()

    # Where is the mask favourable and where is there data?
    w = np.where(np.logical_and(rm == 0, rf1 != -9999., rt >= 0))
    W = w[0]

    # no subsampling here

    # create the features 2D array
    X = []
    for f in features:
        X.append(f.ravel()[W])
    X = np.asarray(X).T
    
    # create the target arrays for training and testing
    Y = target.ravel()[W]

    return X, Y, W





###################################################################################
###################################################################################
def format_self(mask, features, target, binary = True, n_folds = 5):

    if binary == True:
        target[target>0] = 1

    # Make them into a RF-friendly format
    mask [mask == 5.] = 0 # convert channels to possible seagrass

    # Ravel key arrays: mask, aband with potential nodata, target array
    # Ravel instead of flatten because no change will occur (allows faster, non-copy method)
    # we now enter a realm of flat arrays
    rm = mask.ravel()
    rf1 = features[0].ravel()
    rt = target.ravel()


    # Where is the mask favourable and where is there data?
    # this lengthy thing replaces the numpy logical where more than 2 arrays are compared
    w1 = np.zeros(rm.shape, dtype = int); w1[rm == 0] = 1
    w2 = np.zeros(rf1.shape, dtype = int); w2[rf1 > 0.] = 1
    w3 = np.zeros(rt.shape, dtype = int); w3[rt >= 0.] = 1
    wsum = w1+w2+w3
    W = np.where(wsum == 3)[0]


    # subsample indices
    train_idx = np.random.choice(W, int((n_folds-1)*len(W)/n_folds), replace=False)
    test_idx = np.asarray([ele for ele in W if ele not in train_idx])

    # create the features 2D array
    Xtrain = []; Xtest = []
    for f in features:
        Xtrain.append(f.ravel()[train_idx])
        Xtest.append(f.ravel()[test_idx])
    Xtrain = np.asarray(Xtrain).T
    Xtest = np.asarray(Xtest).T
    
    # create the target arrays for training and testing
    Ytrain, Ytest = target.ravel()[train_idx], target.ravel()[test_idx]


    return Xtrain, Ytrain, Xtest, Ytest, W, train_idx, test_idx






###################################################################################
###################################################################################
def format_trainthreshtest (mask, features, target, W = np.asarray([None]), train_idx = np.asarray([None]), test_idx = np.asarray([None]), thresh_idx = np.asarray([None]), prop_train = 0.7, binary = True):

    if binary == True:
        target[target>0] = 1

    # Ravel key arrays: mask, aband with potential nodata, target array
    # Ravel instead of flatten because no change will occur (allows faster, non-copy method)
    # we now enter a realm of flat arrays
    rm = mask.ravel()
    rf1 = features[0].ravel()
    rt = target.ravel()

    #print (rm.shape)
    #print (rf1.shape)
    #print (rt.shape)

    if W.any() == None: # if we haven't yet defined the location indices
        # Where is the mask favourable and where is there data?
        # this lengthy thing replaces the numpy logical where more than 2 arrays are compared
        w1 = np.zeros(rm.shape, dtype = int); w1[rm == 0.] = 1
        w2 = np.zeros(rf1.shape, dtype = int); w2[rf1 >= 0.] = 1
        w3 = np.zeros(rt.shape, dtype = int); w3[rt >= 0.] = 1
        wsum = w1+w2+w3
        W = np.where(wsum >= 3)[0]

    if train_idx.any() == None:
        # subsample indices
        train_idx =np.asarray(sorted(np.random.choice(W,int(prop_train*len(W)),replace=False)))
        not_train_idx = np.asarray([ele for ele in W if ele not in train_idx])
        test_idx = np.random.choice(not_train_idx, int(0.5*len(not_train_idx)), replace=False)
        thresh_idx = np.asarray([ele for ele in not_train_idx if ele not in test_idx])

    # create the features 2D array
    Xtrain = []; Xtest = []; Xthresh = []
    for f in features:
        Xtrain.append(f.ravel()[train_idx])
        Xtest.append(f.ravel()[test_idx])
        Xthresh.append(f.ravel()[thresh_idx])

    Xtrain = np.asarray(Xtrain).T
    Xtest = np.asarray(Xtest).T
    Xthresh = np.asarray(Xthresh).T
    
    # create the target arrays for training and testing
    Ytrain, Ytest, Ythresh = target.ravel()[train_idx], target.ravel()[test_idx], target.ravel()[thresh_idx]


    # Normalise by zone !!!!!!!
    Z = Xtrain[:,-1]
    for z in range(1,6):
        # zone indices
        zidx = np.where(Z == z)[0]

        for f in range(len(Xtrain[0,:-1])):
            M = np.nanmax(Xtrain[zidx,f])
            m = np.nanmin(Xtrain[zidx,f])

            # normalise
            Xtrain[zidx,f] = (Xtrain[zidx,f] - m) / (M-m)

    # Normalise by zone !!!!!!!
    Z = Xthresh[:,-1]
    for z in range(1,6):
        # zone indices
        zidx = np.where(Z == z)[0]

        for f in range(len(Xthresh[0,:-1])):
            M = np.nanmax(Xthresh[zidx,f])
            m = np.nanmin(Xthresh[zidx,f])

            # normalise
            Xthresh[zidx,f] = (Xthresh[zidx,f] - m) / (M-m)


    # Normalise by zone !!!!!!!
    Z = Xtest[:,-1]
    for z in range(1,6):
        # zone indices
        zidx = np.where(Z == z)[0]

        for f in range(len(Xtest[0,:-1])):
            M = np.nanmax(Xtest[zidx,f])
            m = np.nanmin(Xtest[zidx,f])

            # normalise
            Xtest[zidx,f] = (Xtest[zidx,f] - m) / (M-m)

    """
    print (Xtrain.shape)
    A = Xtrain[:,-1]
    print ('Training set: number of pixels in each zone:')
    print ('1:', len(np.where(A == 1)[0]))
    print ('2:', len(np.where(A == 2)[0]))
    print ('3:', len(np.where(A == 3)[0]))
    print ('4:', len(np.where(A == 4)[0]))
    print ('5:', len(np.where(A == 5)[0]))

    A = Xthresh[:,-1]
    print ('Threshold set: number of pixels in each zone:')
    print ('1:', len(np.where(A == 1)[0]))
    print ('2:', len(np.where(A == 2)[0]))
    print ('3:', len(np.where(A == 3)[0]))
    print ('4:', len(np.where(A == 4)[0]))
    print ('5:', len(np.where(A == 5)[0]))

    A = Xtest[:,-1]
    print ('Testing set: number of pixels in each zone:')
    print ('1:', len(np.where(A == 1)[0]))
    print ('2:', len(np.where(A == 2)[0]))
    print ('3:', len(np.where(A == 3)[0]))
    print ('4:', len(np.where(A == 4)[0]))
    print ('5:', len(np.where(A == 5)[0]))
    """


    return Xtrain, Ytrain, Xtest, Ytest, Xthresh, Ythresh, W, train_idx, test_idx, thresh_idx






###################################################################################
###################################################################################
def format_zones (mask, features, W = np.asarray([None]), binary = True):

    #if binary == True:
    #    target[target>0] = 1

    # Ravel key arrays: mask, aband with potential nodata, target array
    # Ravel instead of flatten because no change will occur (allows faster, non-copy method)
    # we now enter a realm of flat arrays
    rm = mask.ravel()
    rf1 = features[0].ravel()
    rfz = features[-1].ravel()

    X_zonal = []; Zonal_idx = []
    for z in range(1,int(np.amax(rfz)+1)): # for each zone
        print (z)

        if W.any() == None: # if we haven't yet defined the location indices
            # Where is the mask favourable and where is there data?
            # this lengthy thing replaces the numpy logical where more than 2 arrays are compared
            w1 = np.zeros(rm.shape, dtype = int); w1[rm == 0] = 1
            w2 = np.zeros(rf1.shape, dtype = int); w2[rf1 >= 0.] = 1
            w3 = np.zeros(rfz.shape, dtype = int); w3[rfz == z] = 1
            wsum = w1+w2+w3
            W = np.where(wsum >= 3)[0]

        #print ('W', len(W))
        # subsample indices

        #zone_idx = np.where(features[-1].ravel() == z)[0] # find pixels in the zone

        #combo_idx = np.asarray([ele for ele in W if ele in zone_idx]) # combine zone and the safety W
        combo_idx = np.copy(W) # combine zone and the safety W
        W = np.asarray([None])
        #print (combo_idx)
        x_zonal = []
        for f in features:
            x_zonal.append(f.ravel()[combo_idx])
        
        x_zonal = np.asarray(x_zonal).T
        X_zonal.append(x_zonal)
        Zonal_idx.append(combo_idx)

    # Normalise by zone
    for z in range(len(X_zonal)):
        #print (z, X_zonal[z].shape)
        for f in range(len(X_zonal[z][0,:-1])):

            M = np.nanmax(X_zonal[z][:,f])
            m = np.nanmin(X_zonal[z][:,f])


            # normalise
            X_zonal[z][:,f] = (X_zonal[z][:,f] - m) / (M-m)
    #quit()

    return X_zonal, Zonal_idx

    #this should be improved for better speed






###################################################################################
###################################################################################
def RF_fit_binary (Xtrain, Ytrain, ne = 100, md = 30, mf = 2):
    """
    description: Creates and trains a random forest model with arguments and returns the model and time needed to fit it. This works for nonbinary problems but it will cause problems odwn the line

    arguments:
    Xtrain = 2D array of training features of shape (N_samples, N_features)
    Ytrain = 1D array of of training target of length (N_samples). Only 0 and 1 values.
    *ne = number of estimators (trees)
    *md = maximum tree depth

    returns:
    model = a RandomForestClassifier object fitted using Xtrain and Ytrain
    timer = a datetime object recording the time needed to fit the model
    
    author: GchGoodwin
    """
    
    model = RandomForestClassifier(n_estimators = ne, max_depth = md, max_features = mf) # Define the model
    timer_start = datetime.now() # Start the timer
    model.fit(Xtrain, Ytrain) # fit the model
    timer  = datetime.now() - timer_start #record fitting time


    return model, timer



###################################################################################
###################################################################################
def RF_2Dproba_binary (model, X, idx, shape):
    """
    description: Uses a fitted RandomForestClassifier object to predict the probability of being positive (value 1) according to X features. 

    arguments:
    model = a fitted RandomForestClassifier, produced by the function RF_fit_binary 
    X = 2D array of features of shape (N_samples, N_features).
    idx = the 1D indices at which the model was applied.
    shape = (rows, cols) tuple: the shape of the desired output array

    returns:
    Y2D = the probability of the target being positive (Y == 1). shape is shape
    
    author: GchGoodwin
    """
    #print (X.shape)
    Yproba = model.predict_proba(X) #the probability for each value
    #print (Yproba)
    #print (len(Yproba[0]) > 1) 
    if len(Yproba[0]) > 1:
        Ypp = Yproba[:,1] # the probability of having a value of 1.
    else: # you only have one choice
        Ypp = 1 - Yproba[:,0]

    Y2D = -9999. * np.ones((shape[0]*shape[1]), dtype = np.float)
    Y2D[idx] = Ypp
    Y2D = Y2D.reshape(shape)

    return Y2D, Ypp


###################################################################################
###################################################################################
def expand_proba(Y2D):# NO, you don't need this because the actual ??? OR DO YOU?

    return 1 



###################################################################################
###################################################################################
def evaluate_sepval(sep, Ypp, Ysep, target, w):

    Ysep[w] = sep

    Ydiff = Ypp - Ysep # create the diff between the proba and the sep value
    Yopp = Ysep - Ypp # do the opposite for reasons made clear below

    Ypred = -9999.*np.ones(Ypp.shape, dtype  = np.float)
    Ypred[Ydiff >= 0] = 1 # positive difference yields some seagrass
    Ypred[Yopp > 0] = 0 # negative difference yields a bare

    Ysum = Ypred + target # match values
    TP = len(Ysum[Ysum == 2])
    TN = len(Ysum[Ysum == 0])
    Ydiff = Ypred - target # match values
    FP = len(Ydiff[Ydiff == 1])
    FN = len(Ydiff[Ydiff == -1])

    # These are the values that represent our performance
    Sen = TP / (TP + FP + 0.0000000000000001)
    Pre = TP / (TP + FN + 0.0000000000000001)

    # and this is what we want to optimise
    Objective = (Sen + Pre) /2

    return Objective

###################################################################################
###################################################################################
def define_threshold_proba(Ypp, target, zones):
    """
    description: Compares the proba to the target to determine local separation value.
    This uses optimisation!
    WARNING: THE RISK HERE IS TO OVERFIT !!!!

    arguments:
    Ypp = the 2D probability of the target being positive (Y == 1). 
    target: the 2D target values

    returns:
    Ysep = the 2D separation values for proba to classify as seagrass or not
    
    author: GchGoodwin
    """

    target [Ypp < 0] = -9999. # only use target values where we calculated probability
    # If you have to separate pixels that are outside of the training, then use a mix of separation values within a kernel.
    #SG = len(target[target == 1]) # number of actual positives (SeaGrass)
    #TF = len(target[target == 0]) # number of actual negatives (TidalFlat)

    # Initial value of Ysep
    Ysep = 0.5 * np.ones(Ypp.shape, dtype = np.float)

    
    #1. initialise all zones at 1 to optimize the separation value for the whole map
    ini_zones = np.ones(Ypp.shape, dtype = np.float)
    ini_zone_values = np.arange(np.amin(ini_zones), np.amax(ini_zones)+1)

    # examine each zone
    for z in ini_zone_values:
        print ('zone', z)
        w = np.where(ini_zones == z)

        # optimise sepval
        X = []; myrange = np.arange(0.01,1.,0.01)
        for sep in myrange:
            X.append(evaluate_sepval(sep,Ypp,Ysep,target,w))

        #fig = plt.figure(1, facecolor='White',figsize=[18, 12])
        #axa = plt.subplot2grid((1,1), (0,0),colspan=1, rowspan=1)
        #axa.plot(myrange, X)
        #plt.tight_layout()
        #plt.show()
        
        xmax = np.amax(X)
        idmax = np.where(X == xmax)
        sepvalbase = myrange[idmax][0]

        Ysep[w] = sepvalbase

    print ('initial run: obj = ', xmax, 'sepval = ', sepvalbase)
    print()
    

     #2. Work with each zone
    zone_values = np.arange(np.amin(zones), np.amax(zones)+1)

    # examine each zone
    #fig = plt.figure(1, facecolor='White',figsize=[18, 12])
    #axa = plt.subplot2grid((1,1), (0,0),colspan=1, rowspan=1)

    for z in zone_values:
        print ('zone', z)
        w = np.where(zones == z)
        print ('    w = ', len(w[0]))

        # optimise sepval
        X = []; myrange = np.arange(0.01,1.,0.01)
        for sep in myrange:
            X.append(evaluate_sepval(sep,Ypp,Ysep,target,w))

        #axa.plot(myrange, X, lw = z)
        
        # Only change values if they bring an improvement
        if np.amax(X) > xmax:
            xmax = np.amax(X)
            idmax = np.where(X == xmax)
            sepval = myrange[idmax][0]
            Ysep[w] = sepval

            print ('    Zonal run: obj'+str(z)+' = ', xmax, 'sepval'+str(z)+' = ', sepval)
            print()

        else:
            Ysep[w] = sepvalbase


    #plt.tight_layout()
    #plt.show()

    #quit()

    """
    fig = plt.figure(1, facecolor='White',figsize=[18, 12])
    axa = plt.subplot2grid((1,1), (0,0),colspan=1, rowspan=1)

    axa.plot(myrange, X)

    plt.tight_layout()
    plt.show()
    quit()
    

   

    print (sepval[x][0], np.amax(Obj))

    fig = plt.figure(1, facecolor='White',figsize=[18, 12])
    axa = plt.subplot2grid((1,2), (0,0),colspan=1, rowspan=1)
    axb = plt.subplot2grid((1,2), (0,1),colspan=1, rowspan=1)

    axa.imshow(np.ma.masked_where(Ypp < 0, Ypp), cmap = plt.cm.viridis, vmin = 0, vmax = 1, interpolation = 'none')
    axb.imshow(np.ma.masked_where(Ysep < 0, Ysep), cmap = plt.cm.viridis, vmin = 0., vmax = 1, interpolation = 'none')

    plt.tight_layout()
    plt.show()
    quit()
        
    quit()
    """



    return Ysep, xmax




###################################################################################
###################################################################################
def singlezone_threshold_proba(Ypp, ytarget, zone_value):


    # Initial value of Ysep
    Ysep = 0.5 * np.ones(Ypp.shape, dtype = np.float)
    zones = zone_value * np.ones(Ypp.shape, dtype = np.float)

    # examine each zone
    print ('zone', zone_value)
    w = np.where(zones == zone_value)

    # optimise sepval
    X = []; myrange = np.arange(0.01,1.,0.01)
    for sep in myrange:
        X.append(evaluate_sepval(sep,Ypp,Ysep,ytarget,w))


    xmax = np.amax(X)
    idmax = np.where(X == xmax)
    sepvalbase = myrange[idmax][0]

    Ysep[w] = sepvalbase

    print ('initial run: obj = ', xmax, 'sepval = ', sepvalbase)
    print()
    




    return Ysep, xmax



###################################################################################
###################################################################################
def define_threshold_proba_old(Ypp, target, zones):
    """
    description: Compares the proba to the target to determine local separation value.
    This uses optimisation!
    WARNING: THE RISK HERE IS TO OVERFIT !!!!

    arguments:
    Ypp = the 2D probability of the target being positive (Y == 1). 
    target: the 2D target values

    returns:
    Ysep = the 2D separation values for proba to classify as seagrass or not
    
    author: GchGoodwin
    """

    target [Ypp < 0] = -9999. # only use target values where we calculated probability
    # If you have to separate pixels that are outside of the training, then use a mix of separation values within a kernel.
    SG = len(target[target == 1]) # number of actual positives (SeaGrass)
    TF = len(target[target == 0]) # number of actual negatives (TidalFlat)


    zones = np.ones(Ypp.shape, dtype = np.float)
    print (np.amin(zones))
    print (np.amax(zones))



    zone_values = np.arange(np.amin(zones), np.amax(zones)+1)
    print (zone_values)
    sepval = np.arange(0.01,1, 0.01)
    

    Ysep = np.zeros(Ypp.shape, dtype = np.float) # initialise the separation proba at 1

    for i in range(0,3):
        print ('iteration', i)
        for z in zone_values:
            print ('zone', z)
            w = np.where(zones == z)

            Obj = []

            for sep in sepval:

                Ysep[w] = sep
                

                Ydiff = Ypp - Ysep # create the diff between the proba and the sep value
                Yopp = Ysep - Ypp # do the opposite for reasons made clear below

                Ypred = -9999.*np.ones(Ypp.shape, dtype  = np.float)
                Ypred[Ydiff >= 0] = 1 # positive difference yields some seagrass
                Ypred[Yopp > 0] = 0 # negative difference yields a bare

                Ysum = Ypred + target # match values
                TP = len(Ysum[Ysum == 2])
                TN = len(Ysum[Ysum == 0])
                Ydiff = Ypred - target # match values
                FP = len(Ydiff[Ydiff == 1])
                FN = len(Ydiff[Ydiff == -1])

                # These are the values that represent our performance
                Sen = TP / (TP + FP)
                Pre = TP / (TP + FN)

                # and this is what we want to optimise
                Objective = (Sen + Pre) /2
                Obj.append(Objective)

            Obj = np.asarray(Obj)

            x = np.where(Obj == np.amax(Obj))

            Ysep[w] = sepval[x][0]

            print (sepval[x][0], np.amax(Obj))

    fig = plt.figure(1, facecolor='White',figsize=[18, 12])
    axa = plt.subplot2grid((1,2), (0,0),colspan=1, rowspan=1)
    axb = plt.subplot2grid((1,2), (0,1),colspan=1, rowspan=1)

    axa.imshow(np.ma.masked_where(Ypp < 0, Ypp), cmap = plt.cm.viridis, vmin = 0, vmax = 1, interpolation = 'none')
    axb.imshow(np.ma.masked_where(Ysep < 0, Ysep), cmap = plt.cm.viridis, vmin = 0., vmax = 1, interpolation = 'none')

    plt.tight_layout()
    plt.show()
    quit()
        
    quit()



    return Ysep, np.amax(Obj)


    """
    Ysep = baseline_sep * np.ones(Ypp.shape, dtype = np.float)
    Ydiff = Ypp - Ysep # create the diff between the proba and the sep value
    Yopp = Ysep - Ypp # do the opposite for reasons made clear below

    Ypred = -9999.*np.ones(Ypp.shape, dtype  = np.float)
    Ypred[Ydiff >= 0] = 1 # positive difference yields some seagrass
    Ypred[Yopp > 0] = 0 # negative difference yields a bare


    # FIX FALSE positive
    Ypred_diff = Ypred - target # match values
    FP = len(Ypred_diff[Ypred_diff == 1])
    FN = len(Ypred_diff[Ypred_diff == -1])

    Wfp_row = np.where(Ypred_diff == 1)[0] # this is where it is not the same even with the baseline sep
    Wfp_col = np.where(Ypred_diff == 1)[1] # this is where it is not the same even with the baseline sep

    for i in range(len (Wfp_row)):
        r = Wfp_row[i]; c = Wfp_col[i]
        Ysep = deform_kernel(Ysep, r, c, 15, deform_func, 1.1* -Ydiff[r,c])


    Wfn_row = np.where(Ypred_diff == -1)[0] # this is where it is not the same even with the baseline sep
    Wfn_col = np.where(Ypred_diff == -1)[1] # this is where it is not the same even with the baseline sep

    for i in range(len (Wfn_row)):
        r = Wfn_row[i]; c = Wfn_col[i]
        Ysep = deform_kernel(Ysep, r, c, 15, deform_func, 1.1* -Ydiff[r,c])




    Ydiff = Ypp - Ysep # create the diff between the proba and the sep value
    Yopp = Ysep - Ypp # do the opposite for reasons made clear below

    Ypred = -9999.*np.ones(Ypp.shape, dtype  = np.float)
    Ypred[Ydiff >= 0] = 1 # positive difference yields some seagrass
    Ypred[Yopp > 0] = 0 # negative difference yields a bare

    Ysum = Ypred + target # match values
    TP = len(Ysum[Ysum == 2])
    TN = len(Ysum[Ysum == 0])
    Ydiff = Ypred - target # match values
    FP = len(Ydiff[Ydiff == 1])
    FN = len(Ydiff[Ydiff == -1])

    # These are the values that represent our performance
    Sen = TP / (TP + FP)
    Pre = TP / (TP + FN)

    # and this is what we want to optimise
    Objective = (Sen + Pre) /2

    print (Objective)
    """

    #fig = plt.figure(1, facecolor='White',figsize=[18, 12])
    #axa = plt.subplot2grid((1,2), (0,0),colspan=1, rowspan=1)
    #axb = plt.subplot2grid((1,2), (0,1),colspan=1, rowspan=1)

    #axa.imshow(np.ma.masked_where(Ypp < 0, Ypp), cmap = plt.cm.viridis, vmin = 0, vmax = 1, interpolation = 'none')
    #axb.imshow(np.ma.masked_where(Ysep < 0, Ysep), cmap = plt.cm.viridis, vmin = 0, vmax = 1, interpolation = 'none')

    #plt.tight_layout()
    #plt.show()
    #quit()




    #quit()



    # Implement a spline change
    #https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html

    # It's gonna have something to do with the weight assigned to each point (function of difference with baseline_sep)

    # Process:
    # Identify the locations where the prediction and target don't match
    # Determine a deformation function inside a circular kernel
    # Find a good deformation function !!!!!!
    #http://people.duke.edu/~hpgavin/cee421/frame-finite-def.pdf
    # For each point:
    # Determine the difference between the pred and target
    # Determine a weight factor 
    # Deform the kernel (kernel size is determined by the function, difference and weight)
    # When all the points are deformed:
    # assess metrics:
    # if worse, decrease the weight factor (not necessarily linearly)
    # if better, increase the weight factor
    # Optimise the weight factor !!!!

    # Wait a sec, that solution is too local, it might make for a weird system.
    # UNLESS you make the kernel super wide and the function slope super shallow


    #quit()



    #fig = plt.figure(1, facecolor='White',figsize=[18, 12])
    #axa = plt.subplot2grid((1,1), (0,0),colspan=1, rowspan=1)
    #axa.imshow(np.ma.masked_where(Y2D < 0, Y2D), cmap = plt.cm.viridis, vmin = 0, vmax = 1, interpolation = 'none')
    #plt.tight_layout()
    #plt.show()
    #quit()






###################################################################################
###################################################################################
def classify_binary (Ypp,Ysep):
    Ydiff = Ypp - Ysep # create the diff between the proba and the sep value
    Yopp = Ysep - Ypp # do the opposite for reasons made clear below

    Ypred = -9999*np.ones(Ypp.shape, dtype  = np.uint8)
    Ypred[Ydiff >= 0] = 1 # positive difference yields some seagrass
    Ypred[Yopp > 0] = 0 # negative difference yields a bare

    return Ypred


###################################################################################
###################################################################################

def evaluate_binary(pred, actual, figname):

    mysum = actual + pred
    mydif = actual - pred

    diffmap = -9999. * np.ones(pred.shape, dtype = np.float)

    diffmap [mysum == 0] = 0 # TN
    diffmap [mysum == 2] = 1 # TP
    diffmap [mydif == 1] = -1 # FN
    diffmap [mydif == -1] = -2 # FP

    TN = len(np.where(diffmap == 0)[0])
    TP = len(np.where(diffmap == 1)[0])
    FN = len(np.where(diffmap == -1)[0])
    FP = len(np.where(diffmap == -2)[0])

    S = TP / (TP+FN+0.0000000000001)
    P = TP / (TP+FP+0.0000000000001)
    A = (TP+TN) / (TP+TN+FP+FN+0.0000000000001)

    #fignum = int(10000*np.random.rand())

    fig = plt.figure(1, facecolor='White',figsize=[18, 12])
    axa = plt.subplot2grid((1,1), (0,0),colspan=1, rowspan=1)
    
    axa.imshow(np.ma.masked_where(diffmap<0, diffmap), cmap = plt.cm.summer_r, vmin = 0, vmax = 1., interpolation = 'none')
    diffmap = np.ma.masked_where(diffmap >=0,diffmap)
    axa.imshow(np.ma.masked_where(diffmap<-2, diffmap), cmap = plt.cm.seismic, vmin = -2.5, vmax = -0.5, interpolation = 'none', alpha = 0.4)
    #axa.imshow(np.ma.masked_where(diffmap!=-2, diffmap), cmap = plt.cm.Greys_r, vmin = -2, vmax = 1, interpolation = 'none')
    
    axa.set_title(str(round(S,2)) + '//' + str(round(P,2)) + '//' + str(round(A,2)) )
    plt.tight_layout()
    plt.savefig(figname)
    plt.clf()
    #plt.close("all")


    return diffmap, S, P, A 








###################################################################################
###################################################################################
def kernel_circle (array, diameter, row_centre, col_centre):

    """
    This function defines a circular kernel within an array (array), centred on (x_centre, y_centre). It's radius is radius.
    Args:
        array (2D numpy array): a 2-D array.
        radius (float): the radius of the circle defining the size of the kernel. radius MUST be an ODD number to account for the central element.
        x_centre (int): The index of the element in the 1st dimension.
        y_centre (int): The index of the element in the 2nd dimension.

    Returns:
        kernel (2D numpy array): The kernel of selected elements.
        kernel_row (2D numpy array): The kernel of row indices of the selected elements.
        kernel_col (2D numpy array): The kernel of column indices of the selected elements.

    Author: GCHG
    """

    if (-1)**diameter < 0:
        row_to_0 = row_centre
        row_to_End = array.shape[0]-row_centre
        col_to_0 = col_centre
        col_to_End = array.shape[1]-col_centre

        width = np.floor(diameter/2)

        Lim_top = row_centre - min(width, row_to_0)
        Lim_bottom = row_centre + min(width+1, row_to_End)
        Lim_left = col_centre - min(width, col_to_0)
        Lim_right = col_centre + min(width+1, col_to_End)

        kernel = array [int(Lim_top):int(Lim_bottom), int(Lim_left):int(Lim_right)]
       
        size_x, size_y = kernel.shape


        if row_to_0 < row_to_End:
            if row_to_0 < width:
                Cx = np.floor((size_x-min(width, row_to_0))/2)
            else:
                Cx = np.floor(size_x/2)
        else:
            if row_to_End < width:
                Cx = np.floor((size_x+min(width+1, row_to_End))/2)
            else:
                Cx = np.floor(size_x/2)

        if col_to_0 < col_to_End:
            if col_to_0 < width:
                Cy = np.floor((size_y-min(width, col_to_0))/2)
            else:
                Cy = np.floor(size_y/2)
        else:
            if col_to_End < width:
                Cy = np.floor((size_y+min(width+1, col_to_End))/2)
            else:
                Cy = np.floor(size_y/2)

        k_centre = (int(Cx),int(Cy))

        mask = create_circle_mask(*np.ogrid[0:size_x, 0:size_y], center=k_centre, radius= np.floor(diameter / 2))
        
        idx = np.where(mask == True)[0]
        idy = np.where(mask == True)[1]

        idxx = int(Lim_top) + np.asarray(idx)
        idyy = int(Lim_left) + np.asarray(idy)


        k_indices = (idx,idy)
        indices = (idxx,idyy)


        kernel = np.ma.masked_where(mask == False, kernel)


    else:
        print ()
        print (" ... WARNING: you need to choose an odd kernel size, buddy")
        print ()
        pass

    return kernel, k_indices, indices, k_centre



###################################################################################
###################################################################################
# Function for deforming a kernel
def deform_kernel (array, row_centre, col_centre, diameter, function, pull):

    K, k_idx, idx, mask = kernel_circle (array, diameter, row_centre, col_centre)
    K_def = function(K, pull, mask)
    array [idx] = K_def[k_idx]

    return array


###################################################################################
###################################################################################
# Function to define how to deform the kernel
def deform_func (arr, pull, k_centre, W = 1):

    nx, ny = arr.shape
    x = np.arange(nx) - k_centre[0]
    y = np.arange(ny) - k_centre[1]
    X, Y = np.meshgrid(y, x)

    D = np.sqrt(X**2. + Y**2.)
    sigma = 0.2 * len(D)

    deformed_arr = arr - (W * pull / (sigma * np.sqrt(2*np.pi)) * np.exp(-D**2/(2*sigma**2)))

    deformed_arr[deformed_arr < 0] = 0
    deformed_arr[deformed_arr > 1] = 1

    return deformed_arr


###################################################################################
###################################################################################
# Function definition for creating masks
def create_circle_mask(X_arr, Y_arr, center, radius):
    c_x, c_y = center
    dists_sqrd = (X_arr - c_x)**2 + (Y_arr - c_y)**2
    return dists_sqrd <= radius**2



###################################################################################
###################################################################################
def SVC_fit_predict_binary (Xtrain, Ytrain, Xtest, ne = 100, md = 30):

    model = SVC()
    timer_start = datetime.now()
    model.fit(Xtrain, Ytrain)
    timer  = datetime.now() - timer_start
    print ('RF training time', timer)
    Ypred = model.predict(Xtest)
    #Yproba = model.predict_proba(Xtest)
    #Yproba_pos = Yproba[:,1]

    return Ypred, Ypred



###################################################################################
###################################################################################
def confusion_metrix (m):
    mysum = 0
    for r in range(len(m)):
        mysum += m[r,r]
        S = m[r,r] / np.sum(m[r,:])
        P = m[r,r] / np.sum(m[:,r])
    A = mysum/np.sum(m)

    return S,P,A


###################################################################################
###################################################################################
def RF_fit_predict_score(X_train,Y_train,X_val,Y_val, ne = 500, md = 15):
    timer_start = datetime.now() 

    model = RandomForestClassifier(n_estimators = ne, max_depth = md)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_val)

    timer  = datetime.now() - timer_start
    print ('RF fitted and predicted in:', timer)

    m = confusion_matrix(Y_val, Y_pred) # rows = pred value ; cols = actual value

    return model, Y_pred, m






################################################################################
################################################################################
def kernel_std(arr, mask):  
    # Add a roughness feature (5-pixel kernel stdev)
    karr = -9999.*np.ones(arr.shape, dtype = np.float)
    for x in range(len(arr)):
        for y in range(len(arr[0])):
            if mask[x,y] == 0 and blue[x,y] >= 0:
                kern,posx, posy = ml.kernel(arr, 5, x,y)
                karr[x,y] = np.nanstd(kern)

    return karr





################################################################################
################################################################################
def kernel (array, kernel_size, row_centre, col_centre):
    """
    This function defines a square kernel within an array (array), centred on (x_centre, y_centre). The is of a width of kernel_size.
    Args:
        array (2D numpy array): a 2-D array.
        kernel_size (float): the width of the square defining the size of the kernel. kernel_size MUST be an ODD number to account for the central element.
        x_centre (int): The index of the element in the 1st dimension.
        y_centre (int): The index of the element in the 2nd dimension.

    Returns:
        kernel (2D numpy array): The kernel of selected elements.
        kernel_row (2D numpy array): The kernel of row indices of the selected elements.
        kernel_col (2D numpy array): The kernel of column indices of the selected elements.

    Author: GCHG
    """

    if (-1)**kernel_size < 0:
        row_to_0 = row_centre
        row_to_End = array.shape[0]-row_centre
        col_to_0 = col_centre
        col_to_End = array.shape[1]-col_centre

        width = np.floor(kernel_size/2)

        Lim_top = row_centre - min(width, row_to_0)
        Lim_bottom = row_centre + min(width+1, row_to_End)
        Lim_left = col_centre - min(width, col_to_0)
        Lim_right = col_centre + min(width+1, col_to_End)

        kernel = array [int(Lim_top):int(Lim_bottom), int(Lim_left):int(Lim_right)]
        kernel_row = np.arange(int(Lim_top),int(Lim_bottom))
        kernel_col = np.arange(int(Lim_left),int(Lim_right))

    else:
        print ()
        print (" ... WARNING: you need to choose an odd kernel size, buddy")
        print ()
        pass

    return kernel, kernel_row, kernel_col




################################################################################
################################################################################
def map_features_target(blue, green, red, bathy, H_unique, cover, mask, i, Fig_dir, figname):
    fig = plt.figure(i, facecolor='White',figsize=[18, 12])
    axa = plt.subplot2grid((2,4), (0,0),colspan=1, rowspan=1)
    axb = plt.subplot2grid((2,4), (0,1),colspan=1, rowspan=1)
    axc = plt.subplot2grid((2,4), (0,2),colspan=1, rowspan=1)
    axd = plt.subplot2grid((2,4), (0,3),colspan=1, rowspan=1)

    axaa = plt.subplot2grid((2,4), (1,0),colspan=1, rowspan=1)
    axbb = plt.subplot2grid((2,4), (1,1),colspan=1, rowspan=1)
    axcc = plt.subplot2grid((2,4), (1,2),colspan=1, rowspan=1)
    axdd = plt.subplot2grid((2,4), (1,3),colspan=1, rowspan=1)


    mblue = np.ma.masked_where(mask != 0, blue)
    mblue = np.ma.masked_where(mblue < 0, mblue)

    mgreen = np.ma.masked_where(mask != 0, green)
    mgreen = np.ma.masked_where(mgreen < 0, mgreen)

    mred = np.ma.masked_where(mask != 0, red)
    mred = np.ma.masked_where(mred < 0, mred)

    depth = H_unique - bathy
    mdepth = np.ma.masked_where(mask != 0, depth)

    mdensity = np.ma.masked_where(mask != 0, cover)
    mdensity = np.ma.masked_where(mdensity <= 0, mdensity)

    #bandmax = 1


    axa.imshow(mblue, cmap = plt.cm.viridis, vmin = 0, vmax = np.percentile(mblue,90), interpolation = 'none')
    axaa.imshow(mblue, cmap = plt.cm.viridis, vmin = 0, vmax = np.percentile(mblue,90), interpolation = 'none')
    axaa.imshow(mdensity, cmap = plt.cm.Reds, vmin = 0, vmax = 1, interpolation = 'none', alpha = 0.5)
    
    axb.imshow(mgreen, cmap = plt.cm.viridis, vmin = 0, vmax = np.percentile(mgreen,90), interpolation = 'none')
    axbb.imshow(mgreen, cmap = plt.cm.viridis, vmin = 0, vmax = np.percentile(mgreen,90), interpolation = 'none')
    axbb.imshow(mdensity, cmap = plt.cm.Reds, vmin = 0, vmax = 1, interpolation = 'none', alpha = 0.5)
    
    axc.imshow(mred, cmap = plt.cm.viridis, vmin = 0, vmax = np.percentile(mred,90), interpolation = 'none')
    axcc.imshow(mred, cmap = plt.cm.viridis, vmin = 0, vmax = np.percentile(mred,90), interpolation = 'none')
    axcc.imshow(mdensity, cmap = plt.cm.Reds, vmin = 0, vmax = 1, interpolation = 'none', alpha = 0.5)

    axd.imshow(mdepth, cmap = plt.cm.viridis, vmin = 0, vmax = 10, interpolation = 'none')
    axdd.imshow(mdepth, cmap = plt.cm.viridis, vmin = 0, vmax = 10, interpolation = 'none')
    axdd.imshow(mdensity, cmap = plt.cm.Reds, vmin = 0, vmax = 1, interpolation = 'none', alpha = 0.5)

    plt.tight_layout()
    plt.savefig(Fig_dir + figname)


################################################################################
################################################################################
def map_features_srfbot(blue, Blue, green, Green, red, Red, bathy, H_unique, mask, i, Fig_dir, figname):
    fig = plt.figure(i, facecolor='White',figsize=[18, 12])
    axa = plt.subplot2grid((2,4), (0,0),colspan=1, rowspan=1)
    axb = plt.subplot2grid((2,4), (0,1),colspan=1, rowspan=1)
    axc = plt.subplot2grid((2,4), (0,2),colspan=1, rowspan=1)
    axd = plt.subplot2grid((2,4), (0,3),colspan=1, rowspan=1)

    axaa = plt.subplot2grid((2,4), (1,0),colspan=1, rowspan=1)
    axbb = plt.subplot2grid((2,4), (1,1),colspan=1, rowspan=1)
    axcc = plt.subplot2grid((2,4), (1,2),colspan=1, rowspan=1)
    axdd = plt.subplot2grid((2,4), (1,3),colspan=1, rowspan=1)


    mblue = np.ma.masked_where(mask != 0, blue)
    mblue = np.ma.masked_where(mblue < 0, mblue)

    mgreen = np.ma.masked_where(mask != 0, green)
    mgreen = np.ma.masked_where(mgreen < 0, mgreen)

    mred = np.ma.masked_where(mask != 0, red)
    mred = np.ma.masked_where(mred < 0, mred)

    mBlue = np.ma.masked_where(mask != 0, Blue)
    mBlue = np.ma.masked_where(mBlue < 0, mBlue)
    mBlue = np.ma.masked_where(mBlue > 0.5, mBlue)

    mGreen = np.ma.masked_where(mask != 0, Green)
    mGreen = np.ma.masked_where(mGreen < 0, mGreen)
    mGreen = np.ma.masked_where(mGreen > 0.5, mGreen)

    mRed = np.ma.masked_where(mask != 0, Red)
    mRed = np.ma.masked_where(mRed < 0, mRed)
    mRed = np.ma.masked_where(mRed > 0.5, mRed)

    depth = H_unique - bathy
    mdepth = np.ma.masked_where(mask != 0, depth)

    #bandmax = 1

    axa.imshow(mblue, cmap = plt.cm.viridis, vmin = 0, vmax = 1000, interpolation = 'none')
    axaa.imshow(mBlue, cmap = plt.cm.viridis, vmin = 0, vmax = 0.15, interpolation = 'none')
    
    axb.imshow(mgreen, cmap = plt.cm.viridis, vmin = 0, vmax = 1000, interpolation = 'none')
    axbb.imshow(mGreen, cmap = plt.cm.viridis, vmin = 0, vmax = 0.15, interpolation = 'none')
    
    axc.imshow(mred, cmap = plt.cm.viridis, vmin = 0, vmax = 1000, interpolation = 'none')
    axcc.imshow(mRed, cmap = plt.cm.viridis, vmin = 0, vmax = 0.15, interpolation = 'none')

    axd.imshow(mdepth, cmap = plt.cm.viridis, vmin = 0, vmax = 10, interpolation = 'none')
    axdd.imshow(mdepth, cmap = plt.cm.viridis, vmin = 0, vmax = 10, interpolation = 'none')

    plt.tight_layout()
    plt.savefig(Fig_dir + figname)



################################################################################
################################################################################
def map_RF_res(blue, Ytest, Ypred, Yproba, test_idx, S, P, A, Fig_dir, figname):
    fig = plt.figure(2, facecolor='White',figsize=[18, 12])
    axa = plt.subplot2grid((1,2), (0,0),colspan=1, rowspan=1)
    axb = plt.subplot2grid((1,2), (0,1),colspan=1, rowspan=1)
    #axc = plt.subplot2grid((1,3), (0,2),colspan=1, rowspan=1)

    maptest = -9999. * np.ones(blue.shape, dtype  = np.float); maptest = maptest.ravel()
    mappred = -9999. * np.ones(blue.shape, dtype  = np.float); mappred = mappred.ravel()
    mapproba = -9999 * np.ones(blue.shape, dtype  = np.float); mapproba = mapproba.ravel()

    maptest[test_idx] = Ytest
    mappred[test_idx] = Ypred
    mapproba[test_idx] = Yproba[:,1]

    maptest = maptest.reshape(blue.shape)
    mappred = mappred.reshape(blue.shape)
    mapdiff = maptest - mappred # 1 => seagrass missed ; -1 => seagrass invented
    mapsum = maptest + mappred # 2 => seagrass OK ; 0 => tifal flat OK
    mapproba = mapproba.reshape(blue.shape)

    maptest = np.ma.masked_where(blue < 0, maptest)
    mapdiff = np.ma.masked_where(blue < 0, mapdiff)
    mapsum = np.ma.masked_where(blue < 0, mapsum)
    mapproba = np.ma.masked_where(blue < 0, mapproba)


    axa.imshow(np.ma.masked_where(maptest < 0, mapsum), cmap = plt.cm.summer_r, vmin = 0, vmax = 2, interpolation = 'none')
    axa.imshow(np.ma.masked_where(mapdiff != 1, mapdiff), cmap = plt.cm.cool, vmin = -1, vmax = 1, interpolation = 'none')
    axa.imshow(np.ma.masked_where(mapdiff != -1, mapdiff), cmap = plt.cm.autumn, vmin = -1, vmax = 1, interpolation = 'none')
    axb.imshow(np.ma.masked_where(mapproba < 0, mapproba), cmap = plt.cm.plasma, vmin = 0, vmax = 1, interpolation = 'none')

    #axc.hist(mapproba[mapdiff == 1], bins = np.arange(0,1,0.05), facecolor = plt.cm.cool(255), alpha = 0.5, lw = 0)
    #axc.hist(mapproba[mapdiff == -1], bins = np.arange(0,1,0.05), facecolor = plt.cm.autumn(0), alpha = 0.5, lw = 0)

    axb.set_title('S:' + str(round(100*S,1)) + ' // P:' + str(round(100*P,1)) + ' // A:' + str(round(100*A,1)))

    plt.savefig(Fig_dir + figname)


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
        #print ("%s opened successfully" %file_name)

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

    #print 'WOOO'
    #print envidata
    #print original_geotransform
    #print inDs
    #print inDs.GetProjection()

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