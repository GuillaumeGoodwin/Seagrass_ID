################################################################################
################################################################################
"""Import Python packages"""
################################################################################
################################################################################

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

from scipy.optimize import minimize

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

from scipy import ndimage
from scipy.ndimage.measurements import label

from sklearn.cluster import KMeans


#import Lee_model as lee
#import Functions as fn







###############################################################################
###############################################################################
def Apply_Random_Forest_full (features, model, Sepvals, exist_model, mask, zones, K_means = False):
    print ('Applying Random Forest model')

    X_zonal, Zonal_idx = format_zones (mask, features)

    minzone = 1; maxzone = int(np.amax(zones))
    
    counter = 0
    pred_t = np.zeros((mask.size))
    proba_t = np.zeros((mask.size))

    print ('starting predictions')
    for z in range(minzone,maxzone+1):
        print ('Zone is ', z)

        pred_z = np.zeros(mask.size)
        proba_z = np.zeros(mask.size)

        if  exist_model[counter] == 0:
            pred_t [Zonal_idx[counter]] = 0
            proba_t [Zonal_idx[counter]] = 0
            

        if  exist_model[counter] == 1:

            chosen_model = model[counter]
            chosen_sepval = Sepvals[counter]

            print (X_zonal[counter][:,:-1].shape)

            Xapply = X_zonal[counter][:,:-1]

            # Include Kmeans as a feature to improve?
            if K_means == True:
                xKmeans = KMeans(n_clusters = 2); xKmeans.fit(Xapply)
                xKmeans_labelled = np.asarray([xKmeans.labels_])

                print (xKmeans_labelled.shape)

                Xapply = np.concatenate((Xapply, xKmeans_labelled.T), axis = 1)

                print (Xapply.shape)



            Y2D, Ypp = RF_2Dproba_binary (chosen_model, Xapply, Zonal_idx[counter], mask.shape)
            pred_z = classify_binary (Y2D,chosen_sepval)

            pred_t [Zonal_idx[counter]] = pred_z.ravel()[Zonal_idx[counter]]
            proba_t [Zonal_idx[counter]] = Y2D.ravel()[Zonal_idx[counter]]

        counter+=1

    pred_t = pred_t.reshape(mask.shape)
    proba_t = proba_t.reshape(mask.shape)
    proba_t = (100*proba_t).astype(np.uint8)


    return proba_t, pred_t


###############################################################################
###############################################################################
def Train_Random_Forest_full (features, target, mask, zones, max_depth=30, n_trees=100, max_features=3, K_means = False):
    print ('formatting features and target for Random Forest')

    # Format the data
    Xtrain, Ytrain, Xtest, Ytest, Xthresh, Ythresh, W, Train_idx, Test_idx, Thresh_idx = format_trainthreshtest(mask, features, target)

    # Indices to keep
    idx0 = W.copy(); idx1 = Train_idx.copy(); idx2 = Test_idx.copy(); idx3 = Thresh_idx.copy()


    # Construct a model
    minzone = 1; maxzone = int(np.amax(zones))

    Zone_models = []; Zone_sepvals = []; exist_zonemodel = []

    print ('building RF model and thresholds for each zone')

    for z in range(minzone,maxzone+1):
        print ('Zone', z)

        trainzones = Xtrain[:,-1] #; print (trainzones)
        indazone = np.where(trainzones == z) #; print (indazone)

        xtrain = Xtrain[indazone] #; print (xtrain) # subsample x
        xtrain = xtrain[:,:-1] #; print (xtrain) # get rid of zone feature

        print (xtrain.shape)

        # Include Kmeans as a feature to improve?
        if K_means == True:
            xKmeans = KMeans(n_clusters = 2); xKmeans.fit(xtrain)
            xKmeans_labelled = np.asarray([xKmeans.labels_])

            print (xKmeans_labelled.shape)


            xtrain = np.concatenate((xtrain, xKmeans_labelled.T), axis = 1)
            print (xtrain.shape)


        ytrain = Ytrain[indazone] #;print (ytrain) # subsample y


        # Construct the model.
        # But only if there are pixels to construct it on ...
        if xtrain.shape[0] <= 10:
            print ('OOOOOOPSSSSSSS')
            skipped +=1
            Zone_models.append(0)
            Zone_sepvals.append(0)
            exist_zonemodel.append(0)

        elif xtrain.shape[0] > 10:
            RF_model, timer = RF_fit_binary (xtrain, ytrain, max_depth, n_trees, max_features)
            
            Zone_models.append(RF_model)
            exist_zonemodel.append(1)

            # Now define threshold proba for this model
            # also subsample the test!
            threshzones = Xthresh[:,-1]#; print (threshzones)
            indazone = np.where(threshzones == z) #;    print (indazone)
            xthresh = Xthresh[indazone]
            xthresh = xthresh[:,:-1]#; print (xthresh)

            # Include Kmeans as a feature to improve?
            if K_means == True:
                xKmeans = KMeans(n_clusters = 2); xKmeans.fit(xthresh)
                xKmeans_labelled = np.asarray([xKmeans.labels_])

                xthresh = np.concatenate((xthresh, xKmeans_labelled.T), axis = 1)

            ythresh = Ythresh[indazone]
            thresh_idx = Thresh_idx[indazone]#; print (thresh_idx)

            
            # Make the separation proba

            # Define indices for this scene
            x = xthresh; y = ythresh; idx = thresh_idx
            # Calculate proba positive in the threshold section
            Y2D, Ypp = RF_2Dproba_binary (RF_model, x, idx, mask.shape)
            # Make the threshold proba
            ysep, perf = singlezone_threshold_proba(Ypp, y, z)#; Ysep.append(ysep)

            # The final separation value is :
            ysep_avg = ysep[0] # dimensions should be irrelevant here
            #ysep_avg = 0.5 # dimensions should be irrelevant here
            Zone_sepvals.append(ysep_avg)


            # Test on the remaining unused pixels
            testzones = Xtest[:,-1]#; print (testzones)
            indazone = np.where(testzones == z)#; print (indazone)
            xtest = Xtest[indazone]
            xtest = xtest[:,:-1]#; print (xtest)

            # Include Kmeans as a feature to improve?
            if K_means == True:
                xKmeans = KMeans(n_clusters = 2); xKmeans.fit(xtest)
                xKmeans_labelled = np.asarray([xKmeans.labels_])

                xtest = np.concatenate((xtest, xKmeans_labelled.T), axis = 1)


            ytest = Ytest[indazone]
            test_idx = Test_idx[indazone]#; print (test_idx)


            # Define indices for this scene
            x = xtest; y = ytest; idx = test_idx

            y2D = -9999*np.ones(mask.shape)
            y2D = y2D.ravel()
            y2D[idx] = y
            y2D = y2D.reshape(mask.shape)

            print ('calculating proba')
            # Calculate proba positive in the threshold section
            proba, Ypp = RF_2Dproba_binary (RF_model, x, idx, mask.shape)
            pred = classify_binary (proba,ysep_avg)

            diffmap, S, P, A, QFN , QFP = fn.make_zonewise_perf(pred, proba, target, zones)


    return Zone_models, Zone_sepvals, exist_zonemodel, S, P, A



###############################################################################
################################################################################
# Contiguous areas

def getContiguousAreas(data, value, min_pixels = 1, contiguity = 'queen'):
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

    ###
    cover = cover[:-1] # get rid of the last row of mask ... FIX THIS ANOTHER WAY! HOW????
    density = density[:-1] # get rid of the last row of mask ... FIX THIS ANOTHER WAY! HOW????
    ###

    cover[cover < thresh_cover] = 0
    cover[cover >= thresh_cover] = 1

    cover = np.ma.masked_where(mask != 0, cover)
    density = np.ma.masked_where(mask != 0, density)



    return cover, density, pixelWidth, (geotransform, inDs)


###################################################################################
###################################################################################
def load_digitised_vegetation_data(thisdir, this_time, mask, thresh_cover = 0.5): 
    # Load the vegetation cover and density
    cover, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(thisdir+str(this_time.year)+'_cover_vetted.bil')
    density, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(thisdir+str(this_time.year)+'_density_vetted.bil')

    ###
    cover = cover[:-1] # get rid of the last row of mask ... FIX THIS ANOTHER WAY! HOW????
    density = density[:-1] # get rid of the last row of mask ... FIX THIS ANOTHER WAY! HOW????
    ###

    cover[cover < thresh_cover] = 0
    cover[cover >= thresh_cover] = 1

    cover = np.ma.masked_where(mask != 0, cover)
    density = np.ma.masked_where(mask != 0, density)



    return cover, density, pixelWidth, (geotransform, inDs)



###################################################################################
###################################################################################
def load_vetted_vegetation_data(veg_dir, this_time, mask, thresh_cover = 0.5): 
    print (veg_dir+str(this_time.year)+'/'+str(this_time.year)+'_cover_vetted.bil') 
    # Load the vegetation cover and density
    cover, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(veg_dir+str(this_time.year)+'/'+str(this_time.year)+'_cover_vetted.bil')
    density, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_3d_array(veg_dir+str(this_time.year)+'/'+str(this_time.year)+'_density_vetted.bil')

    ###
    cover = cover[:-1] # get rid of the last row of mask ... FIX THIS ANOTHER WAY! HOW????
    density = density[:-1] # get rid of the last row of mask ... FIX THIS ANOTHER WAY! HOW????
    ###

    cover[cover < thresh_cover] = 0
    cover[cover >= thresh_cover] = 1

    cover = np.ma.masked_where(mask != 0, cover)
    density = np.ma.masked_where(mask != 0, density)



    return cover, density, pixelWidth, (geotransform, inDs)


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
def makerowcol_deprecated(blue):
    loc_rows = np.zeros(blue.shape, dtype = np.float); loc_cols = np.zeros(blue.shape, dtype = np.float)
    for k in range(len(loc_rows)): loc_rows[k,:] = 50 * np.round(k/50,0)
    for k in range(len(loc_cols[0])): loc_cols[:,k] = 50 * np.round(k/50,0)

    return loc_rows, loc_cols

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

    theta_v = np.arcsin(np.sin(theta_v) * n_air / n_water) # Snell's Law
    theta_w = np.arcsin(np.sin(theta_w) * n_air / n_water) # Snell's Law



    #fig = plt.figure(0, facecolor='White',figsize=[23, 20])
    #ax = plt.subplot2grid((1,2), (0,0), rowspan=1, colspan=1)
    #ax2 = plt.subplot2grid((1,2), (0,1), rowspan=1, colspan=1)
    #ax.imshow(theta_w, cmap = plt.cm.jet)
    #ax2.imshow(theta_w_conv, cmap = plt.cm.jet)
    #plt.show()
    #quit()


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

    # Make them into a RF-friendly format. NO
    #mask [mask == 5.] = 0 # convert channels to possible seagrass

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
def normalise (X, method = 'minmax', quantile = 5):
    """
    normalises values of X in each zone
    options for the method are 'minmax' for actual maxima and 'quantile' for percentiles
    quantile value gives the difference from the extreme value
    """

    Z = X[:,-1]
    for z in range(1,4):
        # zone indices
        zidx = np.where(Z == z)[0]

        for f in range(len(X[0,:-1])):
            if method == 'minmax':
                M = np.nanmax(X[zidx,f])
                m = np.nanmin(X[zidx,f])
            
            elif method == 'quantile':
                M = np.percentile(X[zidx,f], 100-quantile)
                m = np.percentile(X[zidx,f], quantile)
            
            X[zidx,f] = (X[zidx,f] - m) / (M-m)

        #X[X[zidx] > 1] = 1
        #X[X[zidx] < 0] = 0

    X[:,:-1][X[:,:-1] > 1] = 1
    X[:,:-1][X[:,:-1] < 0] = 0


    return X

###################################################################################
###################################################################################
def zonal_normalise (X, method = 'minmax', quantile = 5):
    """
    normalises values of X in each zone
    options for the method are 'minmax' for actual maxima and 'quantile' for percentiles
    quantile value gives the difference from the extreme value
    """

    for z in range(len(X)):
        for f in range(len(X[z][0,:-1])):
            if method == 'minmax':
                M = np.nanmax(X[z][:,f])
                m = np.nanmin(X[z][:,f])
            
            elif method == 'quantile':
                M = np.percentile(X[z][:,f], 100-quantile)
                m = np.percentile(X[z][:,f], quantile)
                
            X[z][:,f] = (X[z][:,f] - m) / (M-m)

        X[z][:,:-1][X[z][:,:-1] > 1] = 1
        X[z][:,:-1][X[z][:,:-1] < 0] = 0


    return X

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
    Xtrain = normalise (Xtrain, method = 'quantile', quantile = 5)
    Xthresh = normalise (Xthresh, method = 'quantile', quantile = 5)
    Xtest = normalise (Xtest, method = 'quantile', quantile = 5)

    return Xtrain, Ytrain, Xtest, Ytest, Xthresh, Ythresh, W, train_idx, test_idx, thresh_idx


###################################################################################
###################################################################################
def format_zones (mask, features, W = np.asarray([None]), binary = True):
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
    X_zonal = zonal_normalise (X_zonal, method = 'quantile', quantile = 5)


    return X_zonal, Zonal_idx

    #this should be improved for better speed






###################################################################################
###################################################################################
def RF_fit_binary (Xtrain, Ytrain, md, nt, mf):

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
    
    model = RandomForestClassifier(n_estimators = nt, max_depth = md, max_features = mf) # Define the model
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
    #Objective = (Sen + Pre) /2
    if Sen+Pre > 0:
        Objective = 2*(Sen*Pre/(Sen+Pre)) 
    else:
        Objective = 0

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