"""
Intertidal_groundcover_main.py

This is the main script in a research project to extract ground cover (vegetation, sediment type) in intertidal settings such as lagoons, bays and estuaries. This work differs from typical ground cover determination because it combines detection methods for emerged AND submerged environments

Author: CchGoodwin

"""


"""
Soooo it works if you give it a whole year of data.
This might make it difficult to estimate seasonal growth.
So it might be worth looking at the performance with a single image.
Otherwise, you could use composites of 2-3 images to have an idea of seasonal extent.
This is an effective option. We also have good results only using bathymetry and surface reflectance and not bottom reflectance. Maybe it's not worth doing all the hydrodynamic calculations, because they take time, space and a well-calibrated model which not everyone has.

We found out that trying to determine density classes was a bit useless on it own, but worked well as a nested approach. Or does it? Let's get our metrics straight!

"""

import os
import pickle
import numpy as np
import copy
import pandas as pd
import geopandas as gpd
import tarfile as tar
import geopandas as gpd
import subprocess as sp
import matplotlib.pyplot as plt

from itertools import product as pdt
from scipy.optimize import minimize
from datetime import datetime, timedelta

from osgeo import gdal, ogr, osr
from shapely.geometry import Polygon
import rasterio
import fiona
import math
from rasterstats.io import Raster


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, quantile_transform
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
from sklearn.metrics import confusion_matrix, r2_score


#import Functions as fn
#import ML_functions as ml
#import Lee_model as lee



###################################################################################
###################################################################################
def makerowcol(blue):
    loc_rows = np.zeros(blue.shape, dtype = np.float); loc_cols = np.zeros(blue.shape, dtype = np.float)
    for k in range(len(loc_rows)): loc_rows[k,:] = k
    for k in range(len(loc_cols[0])): loc_cols[:,k] = k

    return loc_rows, loc_cols


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














###################################################################################
###################################################################################
def train_model(imgdir, maskraster, zoneraster, vegdict, k, ooi_rasters, SAC = False):

	# load the masking rasters
	mask = rasterio.open(maskraster).read(1)
	zones = rasterio.open(zoneraster).read(1)

	# load the ooi rasters
	ooi = []
	for i in range(len(ooi_rasters)):
		ooi.append( rasterio.open(ooi_rasters[i]).read(1) )

	# for each training option
	mykeys = vegdict.keys()


	# load the vegetation
	trainveg = rasterio.open(vegdict[mykeys[k]]).read(1)

	# load the associated image
	trainraster = imgdir + mykeys[k] + '_rgbn_clip.bil'
	trainblu = rasterio.open(trainraster).read(1)
	traingrn = rasterio.open(trainraster).read(2)
	trainred = rasterio.open(trainraster).read(3)
	trainnir = rasterio.open(trainraster).read(4)

	# group the rasters into features
	trainfeatures = [trainblu, traingrn, trainred, trainnir]
	trainfeatures.append(ooi)
	if SAC == True:
		loc_r, loc_c = makerowcol(trainblu)
		trainfeatures.append([loc_r, loc_c])
	trainfeatures.append(zones)

	#Define the target
	Target = trainveg

	# Make a RF model and its thresholds
	RF_models, sepvals, exist_model, S_self, P_self, A_self = Train_Random_Forest_full (features, Target, mask, zones, max_depth=30, n_trees=100, max_features=3)


	return RF_models, sepvals, exist_model, S_self, P_self, A_self


##################################################################
##################################################################
def rate_model():

	testveg = rasterio.open(vegdict[mykeys[k]]).read(1)

	proba, pred = apply_model(features, RF_models, sepvals, exist_model, mask, zones)


	#Define the target
	Target = testveg


	diffmap, s,p,a,xx,yy = make_zonewise_perf(pred_m, proba, density_m, zones)
	s1 = s[0]; s2 = s[1]; s3 = s[2]; s = s[-1]
	p1 = p[0]; p2 = p[1]; p3 = p[2]; p = p[-1]
	a1 = a[0]; a2 = a[1]; a3 = a[2]; a = a[-1]

	count += 1

	"""
	else:
		s,s1,s2,s3 = 0,0,0,0
		p,p1,p2,p3 = 0,0,0,0
		a,a1,a2,a3 = 0,0,0,0

	S.append(s); P.append(p); A.append(a)
	S1.append(s1); P1.append(p1); A1.append(a1)
	S2.append(s2); P2.append(p2); A2.append(a2)
	S3.append(s3); P3.append(p3); A3.append(a3)

	fdf = fdf.append({'features':features_txt, 'model_scene':scene,'S':S,'P':P,'A':A,'S1':S1,'P1':P1,'A1':A1,'S2':S2,'P2':P2,'A2':A2,'S3':S3,'P3':P3,'A3':A3}, ignore_index = True)
	"""

	return metrics







##################################################################
##################################################################
def apply_model(imgdir, maskraster, zoneraster, vegdict, k, ooi_rasters, SAC = False):

	# load the associated image
	testraster = imgdir + mykeys[k] + '_rgbn_clip.bil'
	testblu = rasterio.open(testraster).read(1)
	testgrn = rasterio.open(testraster).read(2)
	testred = rasterio.open(testraster).read(3)
	testnir = rasterio.open(testraster).read(4)

	# group the rasters into features
	testfeatures = [testblu, testgrn, testred, testnir]
	testfeatures.append(ooi)
	if SAC == True:
		loc_r, loc_c = makerowcol(testblu)
		testfeatures.append([loc_r, loc_c])
	testfeatures.append(zones)

	proba, pred = Apply_Random_Forest_full (features, RF_models, sepvals, exist_model, mask, zones)


	return models







def XXX ():
	#plt.show()



	#############################################

	Models.append(RF_models)
	Model_scenes.append(scene)
	Model_times.append(this_time)
	Sepvals.append(sepvals)
	Existence.append(exist_model)


	fig = plt.figure(t, facecolor='White',figsize=[43, 40])


	#ax = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)
	#ax.imshow(zones, cmap = plt.cm.Greys, vmin = 0, vmax = 3)
	#plt.show()
	#quit()


	count = 0
	# Now apply the random forest
	for u in range(len(train_scenes)):

		# This is our scene
		appl_scene = train_scenes[u]

		if appl_scene != scene:
			
			print ()
			print ('Training scene')
			print (scene)
			print ('Application scene')
			print (appl_scene)


			appl_time = ml.gettime(overpass_times, appl_scene) 

			# Load/make feature rasters
			satnumber = scene.split('_')[0][-1]
			bands_dict, wavelength_dict, aw_dict = fn.satnumber_dicts (satnumber)
			blue, green, red, nir, bathy, mask, zones, theta_w, theta_v, pixelWidth, (geotransform, inDs) = ml.load_rasters_no_WWTM(dwld_dir, appl_scene, bathyraster, maskraster, zoneraster)
			H_unique = ml.extract_tide(tides, bathy, appl_time)
			loc_rows, loc_cols = ml.makerowcol(blue)

			# Calculate the bottom reflectance
			#Blue = lee.calculate_bottom_reflectance_known_SPM (20, blue, wavelength_dict['blue'], H_unique - bathy, 3, theta_v, theta_w, aw_dict['blue'])
			#Green = lee.calculate_bottom_reflectance_known_SPM (20, green, wavelength_dict['green'], H_unique - bathy, 3, theta_v, theta_w, aw_dict['green'])
			#Red = lee.calculate_bottom_reflectance_known_SPM (20, red, wavelength_dict['red'], H_unique - bathy, 3, theta_v, theta_w, aw_dict['red'])
			



			##################################
			# Define features
			# Case 1: no seagrass location
			features = np.array([blue, green, red, nir, bathy-H_unique, D50,zones])
			features_txt = 'b_g_r_n_d_d50' 

			# Case 2: the model knows initial seagrass location
			#features = np.array([blue, green, red, nir, bathy-H_unique, D50, loc_rows,loc_cols,zones])
			#features_txt = 'b_g_r_n_d_d50_loc' 

			#################################

			proba, pred = ml.Apply_Random_Forest_full (features, RF_models, sepvals, exist_model, mask, zones)

			pred[blue < 0] = 255
			proba[blue < 0] = 0
			pred[mask != 0] = 255
			proba[mask != 0] = 0

			hs = fn.make_hillshade_arr (proba)


			# Load the testing target
			cover, density, pixelWidth, (geotransform, inDs) = ml.load_vetted_vegetation_data(veg_dir, appl_time, mask)
			density[cover < 0.5] = 0; density[cover < 0] = -9999.
			density[density >= 1] = 1

			# Plot stuff
			pred_m = np.ma.masked_where(blue < 0,pred)
			density_m = np.ma.masked_where(blue <0 ,density)

			diffmap, s,p,a,xx,yy = fn.make_zonewise_perf(pred_m, proba, density_m, zones)
			print (s)
			print (p)
			print ()
			s1 = s[0]; s2 = s[1]; s3 = s[2]; s = s[-1]
			p1 = p[0]; p2 = p[1]; p3 = p[2]; p = p[-1]
			a1 = a[0]; a2 = a[1]; a3 = a[2]; a = a[-1]



			pred_m = np.ma.masked_where(pred !=1 ,pred_m)
			density_m = np.ma.masked_where(density_m <1 ,density_m)

			ax = plt.subplot2grid((1,4), (0,count), rowspan=1, colspan=1)
			ax.imshow(hs, cmap = plt.cm.Greys, vmin = 100, vmax = 150)
			ax.imshow(pred_m,cmap = plt.cm.viridis, vmin = 0, vmax = 1, alpha = 0.8)
			ax.imshow(density_m,cmap = plt.cm.seismic, vmin = 0, vmax = 1, alpha = 0.5)

			ax.set_title(datetime.strftime(appl_time, '%Y_%m_%d')+ ': '+ str(np.round(s,2)) + ' ' + str(np.round(p,2))+ ' ' + str(np.round(a,2)))
			count += 1

		else:
			s,s1,s2,s3 = 0,0,0,0
			p,p1,p2,p3 = 0,0,0,0
			a,a1,a2,a3 = 0,0,0,0

		S.append(s); P.append(p); A.append(a)
		S1.append(s1); P1.append(p1); A1.append(a1)
		S2.append(s2); P2.append(p2); A2.append(a2)
		S3.append(s3); P3.append(p3); A3.append(a3)

	fdf = fdf.append({'features':features_txt, 'model_scene':scene,'S':S,'P':P,'A':A,'S1':S1,'P1':P1,'A1':A1,'S2':S2,'P2':P2,'A2':A2,'S3':S3,'P3':P3,'A3':A3}, ignore_index = True)

	plt.savefig('/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Feature_tests/Best_so_far/5models/'+ features_txt + '_' +  datetime.strftime(this_time, '%Y_%m_%d')+'_q5_n3_md30_ne100.png')
	

fdf.to_csv('/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Feature_tests/Best_so_far/5models/Features_'+features_txt+'_q5_n3_md30_ne100.csv')
pickle.dump(fdf, open( '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Feature_tests/Best_so_far/5models/Features_'+features_txt+'_q5_n3_md30_ne100.pkl', "wb" ))

pickle.dump( (Models, Model_scenes, Model_times, Sepvals, Existence), open( Fig_dir2 + '/5Modelsdata_'+features_txt+'_q5_n3_md30_ne100.pkl', "wb" ) )

quit()











#############################################################
#############################################################
# We now have a model trained on this scene, with the corresponding thresholds


for t in range(len(prediction_scenes)):
	scene = prediction_scenes[t]
	# get time
	this_time = ml.gettime(overpass_times, scene) 

	# Load/make relevant feature rasters
	blue, green, red, nir, bathy, mask, zones, theta_w, theta_v, pixelWidth, (geotransform, inDs) = ml.load_rasters_no_WWTM(dwld_dir, scene, bathyraster, maskraster, zoneraster)

	H_unique = ml.extract_tide(tides, bathy, this_time)
	loc_rows, loc_cols = ml.makerowcol(blue)
	# Define array of feature arrays
	features = np.array([blue, green, red, nir, bathy-H_unique, loc_rows, loc_cols, zones]) #zones = temp

	# No need for a target here!
	# Format the features
	print ('formatting zones')
	print (features.shape)

	X_zonal, Zonal_idx = ml.format_zones (mask, features)
	print ('Done! moving on')
	minzone = 1; maxzone = int(np.amax(zones))
	
	print ()
	counter = 0
	pred_t = np.zeros((blue.size))
	proba_t = np.zeros((blue.size))

	print ('starting predictions')


	for z in range(minzone,maxzone+1):
		print ('zone is ', z)

		pred_z = np.zeros(blue.size)
		proba_z = np.zeros(blue.size)

		if  exist_modelused[counter] == 0:

			pred_t [Zonal_idx[counter]] = 0
			proba_t [Zonal_idx[counter]] = 0
			

		if  exist_modelused[counter] == 1:

			chosen_model = modelused[counter]
			chosen_sepval = Sepvals[m][counter]

			Y2D, Ypp = ml.RF_2Dproba_binary (chosen_model, X_zonal[counter][:,:-1], Zonal_idx[counter], blue.shape)
			pred_z = ml.classify_binary (Y2D,chosen_sepval)

			pred_t [Zonal_idx[counter]] = pred_z.ravel()[Zonal_idx[counter]]
			proba_t [Zonal_idx[counter]] = Y2D.ravel()[Zonal_idx[counter]]

		counter+=1

	pred_t = pred_t.reshape(blue.shape)
	proba_t = proba_t.reshape(blue.shape)
	
	blue_fill =  blue.filled(-1)

	pred_t[blue_fill < 0.] = 255
	pred_t = pred_t.astype(np.uint8)
	
	proba_t = 100*proba_t
	proba_t[blue_fill < 0.] = 255
	proba_t = proba_t.astype(np.uint8)

quit()







fig = plt.figure(0, facecolor='White',figsize=[23, 20])
ax = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)
ax.imshow(Y2D, cmap = plt.cm.viridis, vmin = 0, vmax = 0.1)
plt.show()
quit()

		









quit()

Model_times.append(this_time)
Model_scenes.append(scene)
Models.append(Zone_models)
Sepvals.append(Zone_sepvals)
Existence.append(exist_zonemodel)

# OK, now we've determined the separation thresholds and self-tested
print (Models)
print(Sepvals)
print (Existence)
print (skipped)




pickle.dump( (Models, Model_scenes, Model_times, Sepvals, Existence), open( Fig_dir + '/Modeldata.pkl', "wb" ) )



"""
else:
	(Models, Model_scenes, Model_times, Sepvals, Existence) = pickle.load( open( Fig_dir + '/Modeldata.pkl', "rb" ) )

	#print (Models)
	#print (Model_scenes)
	#print (Model_times)
	#print (Sepvals)
	#print (Existence)
"""

# Now we can run the predictions on everything else!
for m in range(len(Model_scenes)):

	print (m)

	modelscene = Model_scenes[m]
	modeltime = Model_times[m]
	modelused = Models[m]
	exist_modelused = Existence[m]

	print ('this is the model time', modeltime)

	# Initiate predictions and probabilities
	pred_maps = []; proba_maps = []; pred_times = []; pred_satnumbers = []


	for t in range(len(prediction_scenes)):
		scene = prediction_scenes[t]
		# get time

	gcs_path = "/home/willgoodwin/Software/anaconda3/envs/coastalsat/lib/python3.7/site-packages/fiona/gdal_data/"


	timer_start  = datetime.now()


	#############################################################################
	# Define paths
	#############################################################################
	LandSat_dir = "/home/willgoodwin/PostDoc/Padova/Data/Landsat/"
	dwld_dir = LandSat_dir + 'Scenes/All/'

	train_file = dwld_dir + 'TRAIN_SCENES.csv'
	prediction_file = dwld_dir + 'PREDICTION_SCENES.csv'

	overpass_file = LandSat_dir + "Landsat_scenes_overpass.csv"
	veg_dir = "/home/willgoodwin/PostDoc/Padova/Data/VL_Vegetation/"

	Fig_dir = '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/TimeZonal/'
	Fig_dir2 = '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/New_figures/'


	bathyraster = "/home/willgoodwin/PostDoc/Padova/Data/VL_Bathymetry/bathy_2012_3003_nearest_laguna3003.bil"
	D50raster = "/home/willgoodwin/PostDoc/Padova/Data/VL_D50/Campioni_D50_3003.bil"
	maskraster = "/home/willgoodwin/PostDoc/Padova/Data/Basemaps/basemap_laguna.bil"
	zoneraster = "/home/willgoodwin/PostDoc/Padova/Data/Basemaps/zonemap_laguna.bil"


	mask, pixelWidth, (geotransform, inDs) = ml.ENVI_raster_binary_to_3d_array(maskraster)
	mask = mask[:-1] # get rid of the last row of mask ... FIX THIS ANOTHER WAY! HOW????

	zones, pixelWidth, (geotransform, inDs) = ml.ENVI_raster_binary_to_3d_array(zoneraster)
	zones = zones[:-1]

	D50, pixelWidth, (geotransform, inDs) = ml.ENVI_raster_binary_to_3d_array(D50raster)


	#############################################################################
	# load data for multiple scenes: scenes, overpass times, tides, wind
	#############################################################################
	train_scenes, datetime_list = fn.landsat_scenes_to_panda(train_file)
	train_scenes = list(train_scenes['scene'])

	prediction_scenes, datetime_list = fn.landsat_scenes_to_panda(prediction_file)
	prediction_scenes = list(prediction_scenes['scene'])


	#test_scenes, datetime_list = fn.landsat_scenes_to_panda(test_scenes_csv)
	overpass_times = pd.read_csv(overpass_file)

	tides = fn.extract_station_record ('Livello_(m)', ["chioggiacitta", "saline", "piattaforma"])
	#wind = fn.extract_station_record ('Velocita_vento_(m/s)', ["chioggiaporto", "saline", "piattaforma"])


	#############################################################################
	# Run RF: training and testing can be on the same and different sets
	#############################################################################
	print('Launching Random Forest')


	###################################################################
	#APPLICATION ZONE STARTS HERE
	###################################################################

	skip = False

	"""
	Contenders: GdoRdloc 2009/2010 and same with surface refl for the less deep areas
	Add blue and nir to see effects  
	"""



	fdf = pd.read_csv('/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Feature_tests/Features_3.csv')



	#areas, ids = fn.getContiguousAreas(mask, 0, min_pixels = 3, contiguity = 'queen')




	Models = []; Sepvals = []; Model_scenes = []; Model_times = []; Model_zones = []
	Existence = []
	skipped = 0

	for t in range(len(train_scenes)):

		# This is our scene
		scene = train_scenes[t]; print (scene)
		this_time = ml.gettime(overpass_times, scene) 

		# Load/make feature rasters
		satnumber = scene.split('_')[0][-1]
		bands_dict, wavelength_dict, aw_dict = fn.satnumber_dicts (satnumber)
		blue, green, red, nir, bathy, mask, zones, theta_w, theta_v, pixelWidth, (geotransform, inDs) = ml.load_rasters_no_WWTM(dwld_dir, scene, bathyraster, maskraster, zoneraster)
		H_unique = ml.extract_tide(tides, bathy, this_time)
		
		loc_rows, loc_cols = ml.makerowcol(blue)


		# Calculate the bottom reflectance
		#Blue = lee.calculate_bottom_reflectance_known_SPM (20, blue, wavelength_dict['blue'], H_unique - bathy, 3, theta_v, theta_w, aw_dict['blue'])
		#Green = lee.calculate_bottom_reflectance_known_SPM (20, green, wavelength_dict['green'], H_unique - bathy, 3, theta_v, theta_w, aw_dict['green'])
		#Red = lee.calculate_bottom_reflectance_known_SPM (20, red, wavelength_dict['red'], H_unique - bathy, 3, theta_v, theta_w, aw_dict['red'])
			

		# Load the target
		cover, density, pixelWidth, (geotransform, inDs) = ml.load_vetted_vegetation_data(veg_dir, this_time, mask)
		# Define the limit to consider vegetated. Half the pixel for now
		density[cover < 0.5] = 0; density[cover < 0] = -9999.


		# Metrics for all, and zones by number
		S = []; S1 = []; S2 = []; S3 = []
		P = []; P1 = []; P2 = []; P3 = []
		A = []; A1 = []; A2 = []; A3 = []

		# Define features
		###################################
		# Case 1: no seagrass location
		features = np.array([blue, green, red, nir, bathy-H_unique, D50,zones])
		features_txt = 'b_g_r_n_d_d50' 

		# Case 2: the model knows initial seagrass location
		#features = np.array([blue, green, red, nir, bathy-H_unique, D50, loc_rows,loc_cols,zones])
		#features_txt = 'b_g_r_n_d_d50_loc'  


		###################################
		# Define target
		Target = density.astype(int)

		# Make a RF model and its thresholds
		RF_models, sepvals, exist_model, S_self, P_self, A_self = ml.Train_Random_Forest_full (features, Target, mask, zones, max_depth=30, n_trees=100, max_features=3)


		#############################################
		# look at importance
		#for m in range(len(RF_models)):
		#	importances = RF_models[m].feature_importances_
		#	print (importances)
		#quit()

		#sorted_indices = np.argsort(importances)[::-1]
					 
		#plt.title('Feature Importance')
		#plt.bar(range(X_train.shape[1]), importances[sorted_indices], align='center')
		#plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_indices], rotation=90)
		#plt.tight_layout()
			this_time = ml.gettime(overpass_times, scene) 

			# Load/make relevant feature rasters
			blue, green, red, nir, bathy, mask, zones, theta_w, theta_v, pixelWidth, (geotransform, inDs) = ml.load_rasters_no_WWTM(dwld_dir, scene, bathyraster, maskraster, zoneraster)

			H_unique = ml.extract_tide(tides, bathy, this_time)
			loc_rows, loc_cols = ml.makerowcol(blue)
			# Define array of feature arrays
			features = np.array([blue, green, red, nir, bathy-H_unique, loc_rows, loc_cols, zones]) #zones = temp

			# No need for a target here!
			# Format the features
			print ('formatting zones')
			print (features.shape)

			X_zonal, Zonal_idx = ml.format_zones (mask, features)
			print ('Done! moving on')
			minzone = 1; maxzone = int(np.amax(zones))
			
			print ()
			counter = 0
			pred_t = np.zeros((blue.size))
			proba_t = np.zeros((blue.size))

			print ('starting predictions')


			for z in range(minzone,maxzone+1):
				print ('zone is ', z)

				pred_z = np.zeros(blue.size)
				proba_z = np.zeros(blue.size)

				if  exist_modelused[counter] == 0:

					pred_t [Zonal_idx[counter]] = 0
					proba_t [Zonal_idx[counter]] = 0
					

				if  exist_modelused[counter] == 1:

					chosen_model = modelused[counter]
					chosen_sepval = Sepvals[m][counter]

					Y2D, Ypp = ml.RF_2Dproba_binary (chosen_model, X_zonal[counter][:,:-1], Zonal_idx[counter], blue.shape)
					pred_z = ml.classify_binary (Y2D,chosen_sepval)

					pred_t [Zonal_idx[counter]] = pred_z.ravel()[Zonal_idx[counter]]
					proba_t [Zonal_idx[counter]] = Y2D.ravel()[Zonal_idx[counter]]

				counter+=1

			pred_t = pred_t.reshape(blue.shape)
			proba_t = proba_t.reshape(blue.shape)
			
			blue_fill =  blue.filled(-1)

			pred_t[blue_fill < 0.] = 255
			pred_t = pred_t.astype(np.uint8)
			
			proba_t = 100*proba_t
			proba_t[blue_fill < 0.] = 255
			proba_t = proba_t.astype(np.uint8)


			
			fig = plt.figure(1, facecolor='White',figsize=[10, 8])
			ax0 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)
			#ax0.imshow(np.ma.masked_where(pred_t==255,pred_t), cmap = plt.cm.summer_r, vmin = 0, vmax = 1)
			ax0.imshow(np.ma.masked_where(proba_t==255,proba_t), cmap = plt.cm.summer_r, vmin = 0, vmax = 100)
			plt.tight_layout()
			plt.show()
			

			proba_maps.append(proba_t)
			pred_maps.append(pred_t)
			pred_times.append(this_time)
			#pred_satnumbers.append(satnumber)

			#if t > 2:
			#	break

		pred_maps = np.asarray(pred_maps)
		pred_maps = np.asarray(pred_maps)
		print (pred_maps.shape)


		pickle.dump( pred_maps, open( Fig_dir + '/Model_'+modelscene+'_predmaps.pkl', "wb" ) )
		pickle.dump( proba_maps, open( Fig_dir + '/Model_'+modelscene+'_probamaps.pkl', "wb" ) )
	pickle.dump( pred_times, open( Fig_dir + '/predtimes.pkl', "wb" ) )
	#pickle.dump( pred_satnumbers, open( Fig_dir + '/predsatnumber.pkl', "wb" ) )

	#Seems to be working!
	#Next, we see if this produces realistic results
	#Next, we implement time-based corrections

	# DONE!

	# Amend for mask values as 255
	# sort out storage
	# make sure you don't have too many copies around
	# and RUN







	quit()
	###################################################################
	# APPLICATION ZONE ENDS HERE
	###################################################################


	###################################################################
	#TEST ZONE STARTS HERE
	###################################################################

	shape = (10,10)

	mask = np.zeros(shape)

	zones = np.ones(shape)
	zones[int(shape[0]/3):, :] = 2
	zones[-4:, :] = 3
	zones = zones.astype(int)

	time = [0,1,2,3]
	time = [0,1]


	for t in range(len(time)):
		blue = np.random.random(shape)
		green = np.random.random(shape)

		features = np.asarray([blue,green, zones]) # the zones are a temporary feature

		

		# Combine in time
		if t == 0:

			Target = np.round(np.random.random(shape),0)
			Target = Target.astype(int)

			Xtrain, Ytrain, Xtest, Ytest, Xthresh, Ythresh, W, Train_idx, Test_idx, Thresh_idx = ml.format_trainthreshtest(mask, features, Target)

			idx0 = W.copy() 
			idx1 = Train_idx.copy() 
			idx2 = Test_idx.copy() 
			idx3 = Thresh_idx.copy()

		else:

			target = np.round(np.random.random(shape),0)
			target = target.astype(int)


			xtrain, ytrain, xtest, ytest, xthresh, ythresh, w, train_idx, test_idx, thresh_idx = ml.format_trainthreshtest(mask, features, target, idx0, idx1, idx2, idx3)

			Xtrain = np.concatenate([Xtrain,xtrain], axis = 0)
			Xtest = np.concatenate([Xtest,xtest], axis = 0)
			Xthresh = np.concatenate([Xthresh,xthresh], axis = 0)
			
			Ytrain = np.concatenate([Ytrain,ytrain], axis = 0)
			Ytest = np.concatenate([Ytest,ytest], axis = 0)
			Ythresh = np.concatenate([Ythresh,ythresh], axis = 0)

			W = np.concatenate([W,w], axis = 0)

			Train_idx = np.concatenate([Train_idx,train_idx], axis = 0)
			Test_idx = np.concatenate([Test_idx,test_idx], axis = 0)
			Thresh_idx = np.concatenate([Thresh_idx,thresh_idx], axis = 0)
			
			Target = np.concatenate([Target,target], axis = 0)

		print (W)
		print (Train_idx)
		print (Test_idx)
		print (Thresh_idx)
		print(Xtrain)
		print(Ytrain)
		print()

	print (Xtrain)
	print (Ytrain)
	print()

	# Raaahhhh need to think about indexing! or do you?
	minzone = 1; maxzone = np.amax(zones)
	Models = []; Sepvals = []
	for z in range(minzone,maxzone+1):
		print (z)

		trainzones = Xtrain[:,-1]

		print (trainzones)

		indazone = np.where(trainzones == z)

		print (indazone)


		xtrain = Xtrain[indazone] # subsample x
		xtrain = xtrain[:,:-1] # get rid of zone feature
		ytrain = Ytrain[indazone] # subsample y

		print (xtrain)
		print (ytrain)

		print (xtrain.shape)
		print()

		# Run the model on self. Ypred and Yproba have the same shape as Ytest
		RF_model, timer = ml.RF_fit_binary (xtrain, ytrain)
		Models.append(RF_model)
		print ('trained model in:', timer)
		print()

		######
		# Now define threshold proba
		######


		# also subsample the test!
		threshzones = Xthresh[:,-1]
		print (threshzones)
		indazone = np.where(threshzones == z)
		print (indazone)
		xthresh = Xthresh[indazone]
		xthresh = xthresh[:,:-1]
		ythresh = Ythresh[indazone]
		thresh_idx = Thresh_idx[indazone]

		print (xthresh)
		print (thresh_idx)

		L = int(len(thresh_idx)/len(time))

		# reshape the things to work with 1 image per time frame
		thresh_idx = thresh_idx.reshape(len(time),L)
		xthresh = xthresh.reshape(len(time),L,len(xthresh[0]))
		ythresh = ythresh.reshape(len(time),L)

		print (L)
		print (thresh_idx)
		print (xthresh)


		Ysep = []
		for t in range(len(time)):
			x = xthresh[t]
			y = ythresh[t]
			idx = thresh_idx[t]

			print ('calculating proba')
			# Calculate proba positive in the threshold section
			Y2D, Ypp = ml.RF_2Dproba_binary (RF_model, x, idx, blue.shape)

			#print (Y2D)
			print (Ypp)
			print (y)

			#DONE
			ysep, perf = ml.singlezone_threshold_proba(Ypp, y, z)
			Ysep.append(ysep)

		print (Ysep)

		# average Ysep? Let's try ...
		ysep_avg = np.mean(Ysep) # dimensions should be irrelevant here
		Sepvals.append(ysep_avg)
		print (ysep_avg)





		# What now?
		# Test on the remaining unused pixels

		# also subsample the test!
		testzones = Xtest[:,-1]
		print (testzones)
		indazone = np.where(testzones == z)
		print (indazone)
		xtest = Xtest[indazone]
		xtest = xtest[:,:-1]
		ytest = Ytest[indazone]
		test_idx = Test_idx[indazone]

		print (xtest)
		print (test_idx)

		L = int(len(test_idx)/len(time))

		# reshape the things to work with 1 image per time frame
		test_idx = test_idx.reshape(len(time),L)
		xtest = xtest.reshape(len(time),L,len(xtest[0]))
		ytest = ytest.reshape(len(time),L)

		print (L)
		print (test_idx)
		print (xtest)


		Ysep = []
		for t in range(len(time)):
			x = xtest[t]
			y = ytest[t]
			idx = test_idx[t]

			y2D = -9999*np.ones(blue.shape)
			y2D = y2D.ravel()
			y2D[idx] = y
			y2D = y2D.reshape(blue.shape)

			print ('calculating proba')
			# Calculate proba positive in the threshold section
			Y2D, Ypp = ml.RF_2Dproba_binary (RF_model, x, idx, blue.shape)

			pred = ml.classify_binary (Y2D,ysep_avg)

			ml.evaluate_binary(pred, y2D, Fig_dir + 'Zonal/t'+str(t)+'_z'+str(z)+'.png')

			#print (Y2D)
			print (Y2D)
			print (pred)
			print (y)

	# OK, now we've determined the separation thresholds and self-tested
	print (Models)
	print(Sepvals)

	# Now we can run the tests on everything else!
	newtimes = [2,3,4]

	pred_all = []

	for t in range(len(newtimes)):
		# Generate some features for each time frame
		blue = np.random.random(shape)
		green = np.random.random(shape)
		features = np.asarray([blue,green,zones]) # the zones are a temporary feature

		target = 2*np.random.random(shape)
		target = target.astype(np.uint8)

		# Format the features
		X_zonal, Y_zonal, Zonal_idx = ml.format_zones (mask, features, target)


		minzone = 1; maxzone = np.amax(zones)
		counter = 0
		pred_t = np.zeros((blue.size))
		print (pred_t)
		for z in range(minzone,maxzone+1):
			print (X_zonal[counter].shape)
			Y2D, Ypp = ml.RF_2Dproba_binary (Models[counter], X_zonal[counter][:,:-1], Zonal_idx[counter], blue.shape)
			pred_z = ml.classify_binary (Y2D,Sepvals[counter])

			print (pred_z)
			print (pred_z.shape)

			pred_t [Zonal_idx[counter]] = pred_z.ravel()[Zonal_idx[counter]]

			print (pred_t)


			counter+=1

		pred_t = pred_t.reshape(shape)
		pred_t = pred_t.astype(np.uint8)

		print (pred_t)

		pred_all.append(pred_t)

	pred_all = np.asarray(pred_all)



	print (pred_all.shape)
	pickle.dump( pred_all, open( Fig_dir + 'Zonal/predictions.pkl', "wb" ) )



	# DONE!

	# Amend for mask values as 255
	# sort out storage
	# make sure you don't have too many copies around
	# and RUN







	quit()
	###################################################################
	# TEST ZONE ENDS HERE
	###################################################################



	if not os.path.isfile('/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Vetted/Zonal/Models.p'):


		Models = []
		Model_times = []
		Ysep = []
		Perf = []

		zones, pixelWidth, (geotransform, inDs) = ml.ENVI_raster_binary_to_3d_array(zoneraster)
		zones = zones[:-1]


		for i in range(len(train_scenes)): # the training scene

			print (i)
			
			#Define the scene ID and paths
			scene = train_scenes['scene'].iloc[i]; scene_dir= dwld_dir+scene+'/'
			processed_dir = scene_dir + 'Level2/'

			# get the datetime
			this_time = overpass_times[overpass_times['scene'] == scene]['datetime'].values[0]
			this_time = datetime.strptime(this_time, '%Y-%m-%d %H:%M:%S')

			# Load vegetation cover data. Make sure only pixels full of seagrass are taken
			cover, density, pixelWidth, (geotransform, inDs) = ml.load_vetted_vegetation_data(veg_dir, this_time, mask)


			# Load other rasters
			satnumber = scene.split('_')[0][-1]
			bands_dict, wavelength_dict, aw_dict = fn.satnumber_dicts (satnumber)
			blue, green, red, nir, bathy, mask, zones, theta_w, theta_v, pixelWidth, (geotransform, inDs) = ml.load_rasters_no_WWTM(scene_dir, processed_dir, scene, bands_dict, bathyraster, maskraster, zoneraster)

			print (blue.size)
			print (blue.size * 165)
			quit()


			# select the tide and wind data
			H_unique = ml.extract_tide(tides, bathy, this_time)
			
			# Location is divided in zones
			loc_rows = np.zeros(blue.shape, dtype = np.float); loc_cols = np.zeros(blue.shape, dtype = np.float)
			for k in range(len(loc_rows)):
				loc_rows[k,:] = k
			for k in range(len(loc_cols[0])):
				loc_cols[:,k] = k

			features = np.array([blue, green, red, nir, bathy - H_unique, loc_rows, loc_cols, zones])

			target = density


			# Format and subsample data for modelling
			xtrain, ytrain, xtest, ytest, w, train_idx, test_idx = ml.format_self(mask, features, target)

			# Combine in time
			if i == 0:
				Xtrain, Ytrain, Xtest, Ytest, W, Train_idx, Test_idx = xtrain, ytrain, xtest, ytest, w, train_idx, test_idx 

				Target = target

			else:
				Xtrain = np.concatenate([Xtrain,xtrain], axis = 0)
				Xtest = np.concatenate([Xtest,xtest], axis = 0)
				Ytrain = np.concatenate([Ytrain,ytrain], axis = 0)
				Ytest = np.concatenate([Ytest,ytest], axis = 0)
				W = np.concatenate([W,w], axis = 0)
				Train_idx = np.concatenate([Train_idx,train_idx], axis = 0)
				Test_idx = np.concatenate([Test_idx,test_idx], axis = 0)

				Target = np.concatenate([Target,target], axis = 0)

	quit()

	minzone = 1; maxzone = 5

	for z in range(minzone,maxzone+1):
		print (z)

		trainzones = Xtrain[:,-1]

		indazone = np.where(trainzones == z)

		xtrain = Xtrain[indazone] # subsample x
		xtrain = xtrain[:,:-1] # get rid of zone feature
		ytrain = Ytrain[indazone] # subsample y

		print (xtrain.shape)

		# Run the model on self. Ypred and Yproba have the same shape as Ytest
		RF_model, timer = ml.RF_fit_binary (xtrain, ytrain)
		print ('trained model in:', timer)


		# Calculate porba positive
		Ypp = ml.RF_2Dproba_binary (RF_model, Xtest, test_idx, blue.shape)


		# Look at feature importance
		print (RF_model.feature_importances_)


		#DONE
		ysep, perf = ml.define_threshold_proba(Ypp, target, zones)



	quit()













	Models.append(RF_model)
	Model_times.append(this_time)
	Ysep.append(ysep)
	Perf.append(perf)


	"""
	Ypred, Yproba = ml.RF_fit_predict_proba_binary (Xtrain, Ytrain, Xtest)

	# Score the model
	confmat = confusion_matrix(Ytest, Ypred)
	S,P,A = ml.confusion_metrix (confmat)
	print('self-training score:', S,P,A)

	#ax1 =  plt.subplot2grid((len(train_scenes),len(test_scenes)),(i,i),colspan=1, rowspan=1)
	#ax1.set_title ('S:'+str(int(100*S))+'; P:'+str(int(100*P)))
	#ax1.pie(confmat.ravel(), explode = explode, startangle=90, colors = colours)
	#ax1.axis('equal')

	#ax1.annotate(str(this_time.year)+'_'+str(this_time.month)+'_'+str(this_time.day), xy =(0.,0.3) , xycoords = 'axes fraction', rotation = 90, fontsize = 10)
	"""

	timer = datetime.now()
	Ypp = []
	Times = []
	for i in range(len(train_scenes)): # the training scene
		#Define the scene ID and paths
		scene_i = train_scenes['scene'].iloc[i]
		scene_dir_i= dwld_dir+scene+'/'
		processed_dir_i = scene_dir + 'Level2/'

		print (i)

		for j in range(len(test_scenes)): # the testing scene
			print (j)
			#Define the scene ID and paths
			scene_j = test_scenes['scene'].iloc[j]
			scene_dir_j= dwld_dir+scene+'/'
			processed_dir_j = scene_dir + 'Level2/'

			# DO NOT SELF-TEST
			if scene_i != scene_dir_j:

				# get the datetime
				this_time = overpass_times[overpass_times['scene'] == scene_j]['datetime'].values[0]
				this_time = datetime.strptime(this_time, '%Y-%m-%d %H:%M:%S')

				# Load vegetation cover data
				#cover, density, pixelWidth, (geotransform, inDs) = ml.load_vetted_vegetation_data(veg_dir, this_time) 
				cover, density, pixelWidth, (geotransform, inDs) = ml.load_vetted_vegetation_data(veg_dir, this_time, mask) 
				#Make sure only pixels full of seagrass are taken
				density[cover < 0.1] = 0; density[cover < 0] = -9999.

				# Load other rasters
				satnumber = scene.split('_')[0][-1]
				bands_dict, wavelength_dict, aw_dict = fn.satnumber_dicts (satnumber)

				# load rasters and keep all in mind and memory
				bblue, bgreen, bred, bnir, bathy, mask, zones, theta_w, theta_v, pixelWidth, (geotransform, inDs) = ml.load_rasters_no_WWTM(scene_dir, processed_dir, scene, bands_dict, bathyraster, maskraster, zoneraster)

				# select the tide and wind data
				H_unique = ml.extract_tide(tides, bathy, this_time)
				D = H_unique - bathy

				# Add TWO location features
				loc_rows = np.zeros(blue.shape, dtype = np.float); loc_cols = np.zeros(blue.shape, dtype = np.float)
				for k in range(len(loc_rows)):
					loc_rows[k,:] = k
				for k in range(len(loc_cols[0])):
					loc_cols[:,k] = k

				#Blue = lee.calculate_bottom_reflectance_known_SPM (20, bblue, wavelength_dict['blue'], H_unique - bathy, 3, theta_v, theta_w, aw_dict['blue'])
				#Green = lee.calculate_bottom_reflectance_known_SPM (20, bgreen, wavelength_dict['green'], H_unique - bathy, 3, theta_v, theta_w, aw_dict['green'])
				#Red = lee.calculate_bottom_reflectance_known_SPM (20, bred, wavelength_dict['red'], H_unique - bathy, 3, theta_v, theta_w, aw_dict['red'])
				#Nir = lee.calculate_bottom_reflectance_known_SPM (20, bnir, wavelength_dict['nir'], H_unique - bathy, 3, theta_v, theta_w, aw_dict['nir'])

				# Define the features and target
				features_test = np.array([bblue, bgreen, bred, bnir, bathy - H_unique, zones])
				#features_test = np.array([bblue, bgreen, bred, bnir, bathy - H_unique, loc_rows, loc_cols, zones])
				#features_test = np.array([bblue, bgreen, bred, bnir, bathy - H_unique])
				target_test = cover
				Xtest, Ytest, test_idx = ml.format_one(mask, features_test, target_test)
				
				# Run the RF model on the test data			
				ypp = ml.RF_2Dproba_binary (Models[i], Xtest, test_idx, blue.shape)
				ypp [blue < 0] = -9999.
				Ypp.append(ypp)
				Times.append(this_time)




					

		timer = datetime.now() - timer
		print ('did all predictions in:', timer)

		print ('Pickling')

		pickle.dump( Ysep, open( '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Vetted/TimeSeries/Ysep.p', "wb" ) )
		pickle.dump( Model_times, open( '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Vetted/TimeSeries/Mtimes.p', "wb" ) )
		pickle.dump( Ypp, open( '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Vetted/TimeSeries/Ypp.p', "wb" ) )
		pickle.dump( Times, open( '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Vetted/TimeSeries/Times.p', "wb" ) )



	else:

		Ysep = pickle.load( open( '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Vetted/TimeSeries/Ysep.p', "rb" ) )
		Ypp = pickle.load( open( '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Vetted/TimeSeries/Ypp.p', "rb" ) )
		Times = pickle.load( open( '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Vetted/TimeSeries/Times.p', "rb" ) )
		Model_times = pickle.load( open( '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Vetted/TimeSeries/Mtimes.p', "rb" ) )



	#############################################################################
	# Elaborate predictions from the RF results
	#############################################################################

	"""
	You now have a collection of probabilities and separation probabilities.
	What to do with them?


	The answer to this is an efficient ordering system


	Here, we run for each model and each application:
		Define ypp, ysep
		Define target

		if time == mtime:
			Attribute target values to prediction 
		if time > mtime (moving forward in time):
			Fill nodata using the previous prediction (or target if the previous time is mtime)

			# Tentative step:
			# Modulate probability of presence based on previous prediction (or target if the previous time is mtime) and neighbours!!!!!
			# How to do that?
			# 

		Predict remaining unknown presence based on proba

	We will evaluate the results by:
		matching the presence predicted in yr1 using the model of trained on targets of yr0

	"""

	maskraster = "/home/willgoodwin/PostDoc/Padova/Data/Basemaps/basemap_laguna.bil"
	mask, blah, blabla = ml.ENVI_raster_binary_to_3d_array(maskraster); mask = mask[:-1]

	Ypp = np.asarray(Ypp)
	Ypp = Ypp.reshape(len(Model_times), int(len(Times)/len(Model_times)), Ypp[0].shape[0], Ypp[0].shape[1])

	Times = np.asarray(Times)
	Times = Times.reshape(len(Model_times), int(len(Times)/len(Model_times)))





	if not os.path.isfile('/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Vetted/TimeSeries/Pred.p'):

		# determine all based on raw proba o start with
		# This works and is nice and clean and simple
		Pred = []
		for i in range(len(Model_times)): # the training scene
			mtime = Model_times[i]

			for j in range(len(Times[i])):
				time = Times[i,j]

				if time == mtime: # If we are looking at the training scene
					# Prediction is the target
					cover, density, pixelWidth, (geotransform, inDs) = ml.load_vetted_vegetation_data(veg_dir, mtime, mask) 
					density[cover < 0.1] = 0; density[cover < 0] = -9999.; density[density > 1] = 1
					pred = np.ones(density.shape, dtype = np.float)
					pred[density < 1.] = 0.; pred [mask != 0] = -9999.

				else:
					# Predict based on proba
					pred = ml.classify_binary(Ypp[i,j], Ysep[i])
					pred[Ypp[i,j] < 0] = -9999. # Delete nodata
				
				# Append the prediction
				Pred.append(pred)

		# Reformat and reshape
		Pred = np.asarray(Pred)
		Pred = Pred.reshape(len(Model_times), int(len(Pred)/len(Model_times)), Pred[0].shape[0], Pred[0].shape[1])

		# Save as pickle file
		#pickle.dump( Pred, open( '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Vetted/TimeSeries/Pred.p', "wb" ) )



	# If you've calculated predictions, load them
	else:
		Pred = pickle.load( open( '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Vetted/TimeSeries/Pred.p', "rb" ) )



	#############################################################################
	# Stabilisation process through elimination of momentary switches
	#############################################################################

	for i in range(len(Pred)):

		for j in range(len(Pred[i])):

			cover, density, pixelWidth, (geotransform, inDs) = ml.load_vetted_vegetation_data(veg_dir, Times[i,j]); density[cover < 0.1] = 0; density[cover < 0] = -9999.; density[density > 1] = 1

			diffmap, S, P, A = ml.evaluate_binary(Pred[i,j], density, '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Vetted/TimeSeries/' + 'Perf'+str(i)+'_'+ str(j) + '.png')


		Pconv = Pred[i][1:] - Pred[i][:-1]# This is the derivative-ish. any non-zero is a change
		Pconv [Pconv == -1] = 1 # Now all 1s mean change!

		
		Psucc = Pconv[1:] + Pconv[:-1] # succesive changes. 2 is a change that follows a change
		Psucc[Psucc !=2] = 0
		Psucc = Psucc / 2

		# these are the time intervals of change
		Tconv = Times[i][1:] - Times[i][:-1]
		Tsucc = Tconv[1:].copy()

		# Make Pconv the interval of time preceding a change
		for j in range(len(Tconv)):
			print(Tconv[j])
			Tconv[j] =  Tconv[j].total_seconds() / (3600*24)
			Pconv[j,:,:] = Pconv[j,:,:] * Tconv[j] # This is the time interval during which conversion occurred

		
		# Make Psucc the interval of time preceding a second change
		for j in range(len(Tsucc)):
			print(Tsucc[j])
			Tsucc[j] =  Tsucc[j].total_seconds() / (3600*24)
			Psucc[j,:,:] = Psucc[j,:,:] * (Tsucc[j]+Tconv[j]) # This is the time interval during which a conversion occurred AND WAS THEN REVERSED!

		# These are the numbers of changes and successive changes (change reversals)
		Nconv = np.count_nonzero(Pconv, axis = 0)
		Nsucc = np.count_nonzero(Psucc, axis = 0)

		# This is the average time during which conversion happened and was reversed (resp)
		mean_convtime = np.nanmean(np.ma.masked_where(Pconv<=0, Pconv), axis = 0)
		mean_succtime = np.nanmean(np.ma.masked_where(Psucc<=0, Psucc), axis = 0)

		# MAYBE YOU SHOULD DO THIS FOR EVERY MAP!
		# eg: If you are a 1 surrounded by 0s and the reversal time is less than 90 days (for example)

		# Judge using spikes
		for j in range(len(Psucc)): # loop over all from the second
			# NB: pixels on either time of a spike are considered unstable?
			# Or is it just the spike that is unstable?
			# It's more likely that the spike is unstable right?
			
			w0 = np.where(Psucc[j] == 0.)
			w90 = np.where(Psucc[j] < 90) # includes w0

			#define
			pred = Pred[i,j-1]
			# Switch !
			#pred[w90] = 1 - pred[w90]
			damean = np.round(np.nanmean(Pred[i,j-2:j], axis = 0),0)
			pred[w90] = damean[w90]
			# correct !
			#pred[w0] = 1 - pred[w0]
			pred[w0] = Pred[i,j][w0]
			pred[Pred[i,j] == -9999.] = -9999.

			# reassign
			Pred[i,j-1] = pred

			# Done! Check if it works

			# NOTE TO SELF: LOOKING AT CHANNEL HEAD GEOMETRIES IS INTERESTING FOR THE FUTURE

	print (Pred.shape)

	for i in range(len(Pred)):
		for j in range(len(Pred[i])):

			cover, density, pixelWidth, (geotransform, inDs) = ml.load_vetted_vegetation_data(veg_dir, Times[i,j]); density[cover < 0.1] = 0; density[cover < 0] = -9999.; density[density > 1] = 1

			print (i,j)

			diffmap, S, P, A = ml.evaluate_binary(Pred[i,j], density, '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Vetted/TimeSeries/' + 'Stab_Perf'+str(i)+'_'+ str(j) + '.png')


	quit()

		#############################################################################
		# Fill nodata with the average between previous and next things
		# This should happen at the end, when we have a decent idea of cover
		#############################################################################

	for i in range(len(Pred)):
		for j in range(1,len(Pred[i])-1): # only fill for those that have two neighbours

			prevnext = (Pred[i,j-1] + Pred[i,j+1]) / 2 # 1 is seagrass
			prevnext[prevnext < 0] = -9999. # reestablish nodata
			prevnext = np.round(prevnext,0) # 0 is no seagrass

			this = Pred[i,j]

			w = np.where (this == -9999.)

			this[w] = prevnext[w]



			# Fill no data with previous or target? 
			pred[pred == -9999.] = Pred[0][pred == -9999.]



		print (Pred[i].shape)
		quit()




		fig = plt.figure(1, facecolor='White',figsize=[18, 12])
		axa = plt.subplot2grid((1,2), (0,0),colspan=1, rowspan=1)
		axb = plt.subplot2grid((1,2), (0,1),colspan=1, rowspan=1)

		axa.scatter(Nconv.ravel(), mean_convtime.ravel())
		axb.scatter(Nsucc.ravel(), mean_succtime.ravel())

		plt.tight_layout()
		plt.show()


		quit()

		shortest_conv = np.amin(np.ma.masked_where(Pconv<=0, Pconv), axis = 0)
		mean_conv = np.nanmean(np.ma.masked_where(Pconv<=0, Pconv), axis = 0)
		sum_conv = np.sum(Pconv, axis = 0)

		print (Nsucc.shape)
		quit()


		for j in range(1,len(Pred[i])-1):
			print ('wo')


	# Review this to make sure it actually does what it says !!!!!!

	# It definitely decreases the overshoot but maybe a bit too much.
	# This is probably due to the nodata filling generating loads of residual positives that end up causing instability ...


	for mt in range(LMT):
		mtime = Model_times[mt]
		print (mtime)

		# select the predictions to work on
		this_Pred = []
		this_time = []
		for t in range(mt*int(LT/LMT), (mt+1) * int(LT/LMT)):
			print (t)
			time = Times.ravel()[t]
			if time != mtime:
				this_Pred.append(Pred[t])
				this_time.append(time)
		this_Pred = np.asarray(this_Pred)
		this_time = np.asarray(this_time)
		print (this_Pred.shape)


		# Measure the number of conversions for each pixel
		Pconv = this_Pred[1:] - this_Pred[:-1] # This is the derivative-ish. any non-zero is a change
		Pconv [Pconv == -1] = 1 # Now all 1s mean change!

		# We can assume that if the average conversion rate is above 1 every 6 months, it's unstable
		# Or can we?

		#Convsum = np.sum(Pconv, axis = 0) # convsum is the number of changes
		#dt = this_time[-1]-this_time[0]
		#dt_days = dt.total_seconds() / (3600*24)
		#A = Convsum / dt_days


		# This is not precise enough. We need to find the shortest time between changes
		Tconv = this_time[1:] - this_time[:-1]

		for i in range(len(Tconv)):
			Tconv[i] =  Tconv[i].total_seconds() / (3600*24)
			Pconv[i,:,:] = Pconv[i,:,:] * Tconv[i]

		Nchanges = np.count_nonzero(Pconv, axis = 0)
		shortest_change = np.amin(np.ma.masked_where(Pconv<=0, Pconv), axis = 0)
		mean_change = np.nanmean(np.ma.masked_where(Pconv<=0, Pconv), axis = 0)
		sum_change = np.sum(Pconv, axis = 0)

		# Try this:
		# If you have more than 3 changes
		# AND
		# your mean change time is smaller than the change time for the mx change number
		# you are not seagrass
		# This is still a bit simplistic but it reduces the positives.

		max_time_for_deletion = np.amax(mean_change[Nchanges == np.amax(Nchanges)])
		for i in range(len(this_Pred)):
			this_Pred[i,Nchanges >= 3] = 0
			this_Pred[i,mean_change <= max_time_for_deletion] = 0

			cover, density, pixelWidth, (geotransform, inDs) = ml.load_vetted_vegetation_data(veg_dir, this_time[i]); density[cover < 0.9] = 0
			density[cover < 0] = -9999.; density[density >= 1] = 1

			# Now evaluate the correspondance to known vegetation
			diffmap, S, P, A = ml.evaluate_binary(this_Pred[i], density, '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Vetted/TimeSeries/' + 'Stab_Perf'+str(mt)+'_'+ datetime.strftime(this_time[i], '%Y%m%d') + '.png')












	#############################################################################
	# Fill nodata with the average between previous and next things
	# This should happen at the end, when we have a decent idea of cover
	#############################################################################

	for i in range(len(Pred)):
		for j in range(1,len(Pred[i])-1): # only fill for those that have two neighbours





			# Fill no data with previous or target? 
			pred[pred == -9999.] = Pred[0][pred == -9999.]









		quit()



	idx = np.where(Times[i] == mtime)[0][0] # this is the date index of our training image


	# Go backward in time from the training image
	for j in range(idx-1, -1, -1):
		time = Times[i,j]

		# Predict based on proba
		pred = ml.classify_binary(Ypp[i,j], Ysep[i])
		pred[Ypp[i,j] < 0] = -9999. # Delete nodata
		Pred.insert(0, pred)

		#cover, density, pixelWidth, (geotransform, inDs) = ml.load_vetted_vegetation_data(veg_dir, time) 
		#density[cover < 0.9] = 0; density[cover < 0] = -9999.; density[density > 1] = 1

		# Now evaluate the correspondance to known vegetation
		#diffmap, S, P, A = ml.evaluate_binary(pred, density, '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Vetted/TimeSeries/' + 'Perf'+str(i)+'_'+ str(j) + '.png')
	cover, density, pixelWidth, (geotransform, inDs) = ml.load_vetted_vegetation_data(veg_dir, mtime) 
	density[cover < 0.9] = 0; density[cover < 0] = -9999.; density[density > 1] = 1

	pred = np.ones(density.shape, dtype = np.float)
	pred[density < 1.] = 0.; pred [mask != 0] = -9999.

	Pred.append(pred)

	# Fill no data with previous or target? 
	pred[pred == -9999.] = Pred[0][pred == -9999.]


	LMT = len(Model_times)
	LT = len(Times.ravel())

	#############################################################################
	# Stabilisation process
	#############################################################################

	# Review this to make sure it actually does what it says !!!!!!

	# It definitely decreases the overshoot but maybe a bit too much.
	# This is probably due to the nodata filling generating loads of residual positives that end up causing instability ...


	for mt in range(LMT):
		mtime = Model_times[mt]
		print (mtime)

		# select the predictions to work on
		this_Pred = []
		this_time = []
		for t in range(mt*int(LT/LMT), (mt+1) * int(LT/LMT)):
			print (t)
			time = Times.ravel()[t]
			if time != mtime:
				this_Pred.append(Pred[t])
				this_time.append(time)
		this_Pred = np.asarray(this_Pred)
		this_time = np.asarray(this_time)
		print (this_Pred.shape)


		# Measure the number of conversions for each pixel
		Pconv = this_Pred[1:] - this_Pred[:-1] # This is the derivative-ish. any non-zero is a change
		Pconv [Pconv == -1] = 1 # Now all 1s mean change!

		# We can assume that if the average conversion rate is above 1 every 6 months, it's unstable
		# Or can we?

		#Convsum = np.sum(Pconv, axis = 0) # convsum is the number of changes
		#dt = this_time[-1]-this_time[0]
		#dt_days = dt.total_seconds() / (3600*24)
		#A = Convsum / dt_days


		# This is not precise enough. We need to find the shortest time between changes
		Tconv = this_time[1:] - this_time[:-1]

		for i in range(len(Tconv)):
			Tconv[i] =  Tconv[i].total_seconds() / (3600*24)
			Pconv[i,:,:] = Pconv[i,:,:] * Tconv[i]

		Nchanges = np.count_nonzero(Pconv, axis = 0)
		shortest_change = np.amin(np.ma.masked_where(Pconv<=0, Pconv), axis = 0)
		mean_change = np.nanmean(np.ma.masked_where(Pconv<=0, Pconv), axis = 0)
		sum_change = np.sum(Pconv, axis = 0)

		# Try this:
		# If you have more than 3 changes
		# AND
		# your mean change time is smaller than the change time for the mx change number
		# you are not seagrass
		# This is still a bit simplistic but it reduces the positives.

		max_time_for_deletion = np.amax(mean_change[Nchanges == np.amax(Nchanges)])
		for i in range(len(this_Pred)):
			this_Pred[i,Nchanges >= 3] = 0
			this_Pred[i,mean_change <= max_time_for_deletion] = 0

			cover, density, pixelWidth, (geotransform, inDs) = ml.load_vetted_vegetation_data(veg_dir, this_time[i]); density[cover < 0.9] = 0
			density[cover < 0] = -9999.; density[density >= 1] = 1

			# Now evaluate the correspondance to known vegetation
			diffmap, S, P, A = ml.evaluate_binary(this_Pred[i], density, '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Vetted/TimeSeries/' + 'Stab_Perf'+str(mt)+'_'+ datetime.strftime(this_time[i], '%Y%m%d') + '.png')





		#fig = plt.figure(1, facecolor='White',figsize=[18, 12])
		#axa = plt.subplot2grid((1,3), (0,0),colspan=1, rowspan=1)
		#axb = plt.subplot2grid((1,3), (0,1),colspan=1, rowspan=1)
		#axc = plt.subplot2grid((1,3), (0,2),colspan=1, rowspan=1)
		#axa.imshow(np.ma.masked_where(this_Pred[0]<=0, this_Pred[0]), cmap = plt.cm.plasma, vmin = 0, vmax = 1, interpolation = 'none')
		#axb.imshow(np.ma.masked_where(this_Pred[0]<=0, this_Pred[0]), cmap = plt.cm.plasma, vmin = 0, vmax = 1, interpolation = 'none')

		#A = np.asarray([Nchanges.ravel(), sum_change.ravel()])
		#axc.scatter(Nchanges.ravel(), mean_change.ravel())


		#plt.tight_layout()
		#plt.show()


		#quit()	

	quit()


	# Make sure you can check the performance
	for mt in range(LMT):

		mtime = Model_times[mt]
		print (mtime)

		for t in range(mt*int(LT/LMT), (mt+1) * int(LT/LMT)):
			print (t)
			time = Times[t]
			if time in Model_times and time != mtime:
				print(time, 'wooo')	
				pred = Pred[t]		
				# Load vegetation cover data. Make sure only pixels full of seagrass are taken
				cover, density, pixelWidth, (geotransform, inDs) = ml.load_vetted_vegetation_data(veg_dir, time); density[cover < 0.9] = 0
				density[cover < 0] = -9999.; density[density >= 1] = 1

				# Now evaluate the correspondance to known vegetation
				diffmap, S, P, A = ml.evaluate_binary(pred, density, '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/Vetted/TimeSeries/' + 'Perf'+str(mt)+'_'+ datetime.strftime(time, '%Y%m%d') + '.png')


	quit()






	"""
	fig = plt.figure(1, facecolor='White',figsize=[18, 12])
	axa = plt.subplot2grid((1,2), (0,0),colspan=1, rowspan=1)
	axb = plt.subplot2grid((1,2), (0,1),colspan=1, rowspan=1)

	axa.imshow(blue, cmap = plt.cm.plasma, vmin = 0, vmax = 0.1, interpolation = 'none')
	axb.imshow(blue - Blue, cmap = plt.cm.seismic, vmin = -0.1, vmax = 0.1, interpolation = 'none')

	for x in range(len(blue)):
		for y in range(len(blue[0])):
			axb.scatter(Dblue-Blue )

	plt.tight_layout()
	plt.show()
	quit()
	"""




	#features_test = np.array([bblue, bgreen, bred, bnir, bathy - H_unique])
	#features_test = np.array([Blue, Green, Red, bnir, bathy - H_unique, loc_rows, loc_cols])
	features_test = np.array([bblue, bgreen, bred, bnir, bathy - H_unique, loc_rows, loc_cols])
	#features_test = np.array([bblue, bgreen, bred, bnir, dblue, dgreen, dred, dnir, bathy - H_unique])

	#features_test = np.array([blue, green, loc_rows, loc_cols])
	target_test = cover



	# separate!
	#print(features.shape)

	# Create the test data
	Xtest, Ytest, test_idx = ml.format_one(mask, features_test, target_test)


	# Run the model. Ypred and Yproba have the same shape as Ytest
	Ypred, Yproba = ml.RF_fit_binary (Xtrain, Ytrain)
	quit()
	Ypred, Yproba = ml.RF_fit_predict_proba_binary (Xtrain, Ytrain, Xtest)

	# Score the model
	confmat = confusion_matrix(Ytest, Ypred)
	S,P,A = ml.confusion_metrix (confmat)
	print('foreign-training score:', S,P,A)

	# Make some maps
	ml.map_RF_res(target_test, Ytest, Ypred, Yproba, test_idx, S, P, A, Fig_dir, 'Vetted/10pct_thresh/fulltest/RF_'+str(i)+str(j)+'_srf_map.png')









	#ax1 =  plt.subplot2grid((len(train_scenes),len(test_scenes)),(i,j),colspan=1, rowspan=1)
	#if S > 0 and P > 0 and A > 0:
	#	ax1.set_title ('S:'+str(int(100*S))+'; P:'+str(int(100*P)))
	#else:
	#	ax1.set_title ('wrong!')
	#ax1.pie(confmat.ravel(), explode = explode, startangle=90, colors = colours)
	#ax1.axis('equal')
	#ax1.annotate(str(this_time.year)+'_'+str(this_time.month)+'_'+str(this_time.day), xy =(0.,0.3) , xycoords = 'axes fraction', rotation = 90, fontsize = 10)


	quit()
	#plt.tight_layout()
	#plt.savefig(Fig_dir + '2004/RF_rgbnbathyHspm.png')

	quit()













	for i in [0,1]:

		#########################################################################
		#########################################################################
		# TESTING ZONE
		print()

		# mask array
		m = np.array([[2,2,1],[2,2,2],[1,3,2]])

		# feature arrays
		f1 = np.random.randint(0,9, size=(3,3))
		f2 = np.random.randint(0,9, size=(3,3))

		# Insert nodata
		f1[0,2] = -9999.; f1[1,1] = -9999.

		# target array
		t = np.array([[0,1,0],[1,1,0],[0,0,1]])

		# Ravel key arrays: mask, aband with potential nodata, target array
		# Ravel instead of flatten because no change will occur (allows faster, non-copy method)
		rm = m.ravel()
		rf1 = f1.ravel()
		rt = t.ravel()

		# Where is the mask favourable and where is there data?
		w = np.where(np.logical_and(rm == 2, rf1 != -9999.))
		W = w[0]

		# subsample indices
		train_idx = np.random.choice(W, int(len(W)/2), replace=False)
		test_idx = np.asarray([ele for ele in W if ele not in train_idx])

		# create the features 2D array
		features = [f1,f2]
		Xtrain = []; Xtest = []
		for f in features:
			Xtrain.append(f.ravel()[train_idx])
			Xtest.append(f.ravel()[test_idx])
		Xtrain = np.asarray(Xtrain).T
		Xtest = np.asarray(Xtest).T
		
		Ytrain, Ytest = t.ravel()[train_idx], t.ravel()[test_idx]

		Ypred = np.zeros(t.ravel().shape, dtype= int)
		Ypred[test_idx] = np.random.randint(1,9)
		Yfinal = Ypred.reshape(3,3)

		print (m.ravel())
		print (f1.ravel())
		print (f2.ravel())

		print()

		print(train_idx)
		print(test_idx)

		print()

		print(Xtrain)
		print(Ytrain)
		print(Ytest)
		print(Ypred)
		print(Yfinal)

		

		quit()

		print(W)
		print(train_idx)
		print(test_idx)

		print(rm)
		print(rm[train_idx])
		print(rm[test_idx])

		rm[test_idx] = 8

		print(rm)





		quit()

		arrw = arr[w].ravel()
		aw = a[w].ravel() + 1


		print (arr)
		print (a)
		print (w)
		print(arrw)
		print(aw)

		a[w] = aw

		print(a)



		print()
		quit()
		#########################################################################
		#########################################################################



















		# For this, you need to find indices where:
		# mask == 0 AND blue > 0 (not nodata)
		w_exist_sg = np.where(np.logical_and(mask == 0, blue > 0))
		print(w_exist_sg)

		quit()

		# THIS IS NOT GOOD YET! USE LINEAR INDEXING! !!!!!!

		# subsample!
		train_size = int(4 * mask.size / 5)
		train_idx = np.random.choice(mask, train_size, replace=False)

		print(possible_sg)
		print(data_exist)
		print(train_idx)

		quit()






		arr = np.array([[1,2],[2,3]])
		a = np.array([[0,0],[0,0]])

		w = np.where(arr == 2)

		arrw = arr[w].ravel()
		aw = a[w].ravel() + 1


		print (arr)
		print (a)
		print (w)
		print(arrw)
		print(aw)

		a[w] = aw

		print(a)



		quit()




		# test various hypotheses ^^
		

		#this_datetime = preselected[0]
		quit()
		
		# adjust for satellite number


	print (preselected)

	quit()









	arr = np.array([[1,2],[2,3]])
	a = np.array([[0,0],[0,0]])

	w = np.where(arr == 2)

	arrw = arr[w].ravel()
	aw = a[w].ravel() + 1


	print (arr)
	print (a)
	print (w)
	print(arrw)
	print(aw)

	a[w] = aw

	print(a)



	quit()







	##############################################################################
	# Open the datasets
	##############################################################################
	print ('opening the calibration pickle:', LandSat_dir + '2004_calibdata_clean.pkl')
	data04 = pd.read_pickle(LandSat_dir + '2004_prepdata_clean.pkl')

	print ('opening the test pickle:', LandSat_dir + '2012_calibdata_clean.pkl')
	data12 = pd.read_pickle(LandSat_dir + '2012_prepdata_clean.pkl')

	print ('opening the test pickle:', LandSat_dir + '2015_calibdata_clean.pkl')
	data15 = pd.read_pickle(LandSat_dir + '2015_prepdata_clean.pkl')




	##########################################################################
	# Now attempt to express the variability in some new columns
	##########################################################################
	useful_bands = ['blue_t', 'green_t', 'red_t', 'nir_t']

	useful_angles = ['solar_zenith_t', 'sensor_zenith_t']

	useful = useful_bands + useful_angles

	to_drop = []

	# Get rid of obviously useless rows

	# the emersed non-urban areas
	data04 = data04[data04['mask'] != 1]
	data12 = data12[data12['mask'] != 1]
	data15 = data15[data15['mask'] != 1]

	# the adriatic sea. Wait, this could be a useful reference ... Keep it or later
	data04 = data04[data04['mask'] != 6]
	data12 = data12[data12['mask'] != 6]
	data15 = data15[data15['mask'] != 6]

	# Make metrics for summer images
	data04 = fn.make_metrics (data04, useful, to_drop, [3,5])
	data12 = fn.make_metrics (data12, useful, to_drop, [3,5])
	data15 = fn.make_metrics (data15, useful, to_drop, [1,3])

	# Get rid of naughty negative values
	data04 = data04[data04['blue_mean'] >= 0]
	data12 = data12[data12['blue_mean'] >= 0]
	data15 = data15[data15['blue_mean'] >= 0]

	# that's enough metrics for now
	timer  = datetime.now() - timer_start
	print ('preprocessing time', timer)
	timer_start = datetime.now()




	##########################################################################
	# Show the differences in spectral response in areas devoid of seagrasses
	##########################################################################
	"""
	# subsample the training data to fit the area of the test data
	idx = data12.index.values
	data04 = data04.loc[data04.index.intersection(idx)]
	idx = data04.index.values
	data12 = data12.loc[data12.index.intersection(idx)]


	idx = data12.index.values
	data044 = data04.loc[data04.index.intersection(idx)]
	idx = data044.index.values
	data15 = data15.loc[data15.index.intersection(idx)]


	built04 = data04[data04['mask'] == 2]
	built044 = data044[data044['mask'] == 2]
	built12 = data12[data12['mask'] == 2]
	built15 = data15[data15['mask'] == 2]
	print(built04.shape, built044.shape, built12.shape, built15.shape)

	canal04 = data04[data04['mask'] == 5]
	canal044 = data044[data044['mask'] == 5]
	canal12 = data12[data12['mask'] == 5]
	canal15 = data15[data15['mask'] == 5]
	print(canal04.shape, canal044.shape, canal12.shape, canal15.shape)


	fig = plt.figure(1, facecolor='White',figsize=[14, 14])

	i = 0
	colours = ['b', 'g', 'r', 'k']
	for band in ['blue_mean', 'green_mean', 'red_mean', 'nir_mean']:
		ax1 = plt.subplot2grid((4,2),(i,0),colspan=1, rowspan=1)
		ax2 = plt.subplot2grid((4,2),(i,1),colspan=1, rowspan=1)

		axes = [ax1,ax2]
		B = [built04[band], canal04[band]]
		A = [built12[band], canal12[band]]

		for axis,a,b in zip(axes,A,B):
			axis.hist2d(a, b, cmap = plt.cm.viridis, bins = np.arange(0,0.5,0.01), vmin = 0, vmax = 100)

			model = np.polyfit(a,b,2)
			fitline = np.poly1d(model)
			
			axis.scatter(a,fitline(a), marker = '.', s = 5, facecolor = 'r')
			axis.annotate('y = sum(Kn*x^n)', xy = (0.05, 0.95), xycoords = 'axes fraction', c = 'r')
			
			for k in range(len(model)):
				axis.annotate('K' + str(len(model)-k)+ '='+ str(round(model[k],3)), xy = (0.05, 0.95-0.05*(k+1)), xycoords = 'axes fraction', c = 'r')

			R2 = r2_score(b,fitline(a))
			axis.annotate('R2 = ' + str(round(R2,3)), xy = (0.8, 0.05), xycoords = 'axes fraction', c = 'r')


		#ax1.set_ylabel(b+ ' test dataset response')
		ax1.set_ylabel(band+ ' training dataset reflectance')
		ax1.set_xlabel(band+ ' test dataset response')
		ax2.set_xlabel(band+ ' test dataset response')
		ax1.set_title('urban areas')
		ax2.set_title('canals')

		i+=1

	plt.tight_layout()
	plt.savefig(Fig_dir + 'transform_2d.png')

	quit()
	"""

	##########################################################################
	# Define a bunch of scenarii
	##########################################################################
	bands = ['blue', 'green','red', 'nir']

	metrics = ['min', 'max', 'mean', 'std']
	mm = ['min', 'max']
	stats = [ 'mean', 'std']

	angles = ['solar_zenith', 'sensor_zenith']

	bathy = ['bathy']
	mask = ['mask']
	classif = ['density_class']


	# Optimised surface scenarii
	scenari = [[ bathy+[i+'_'+j for i,j in pdt(bands,metrics)]+classif , 'SA000'],
			   [ bathy+[i+'_'+j for i,j in pdt(bands,mm)]+classif , 'SM000'],
			   [ bathy+[i+'_'+j for i,j in pdt(bands,stats)]+classif , 'SS000'],
			   [ bathy+[i+'_'+j for i,j in pdt(bands+angles,metrics)]+classif , 'SA00A'],
			   [ bathy+[i+'_'+j for i,j in pdt(bands+angles,mm)]+classif , 'SM00A'],
			   [ bathy+[i+'_'+j for i,j in pdt(bands+angles,stats)]+classif , 'SS00A']
			   ]



	##########################################################################
	# Now simulate loads of random classifiers
	##########################################################################


	fig = plt.figure(1, facecolor='White',figsize=[6, 14])

	labels = ['TN', 'FP', 'FN', 'TP']
	colours = [plt.cm.spring_r(0), plt.cm.YlOrRd(50), plt.cm.YlGn(150), plt.cm.brg_r(0)]
	explode = [0,0.1,0.1,0.]

	for o in range(len(scenari)):

		ax1 =  plt.subplot2grid((len(scenari),3),(o,0),colspan=1, rowspan=1)
		ax2 =  plt.subplot2grid((len(scenari),3),(o,1),colspan=1, rowspan=1)
		ax3 =  plt.subplot2grid((len(scenari),3),(o,2),colspan=1, rowspan=1)
		axes = [ax1,ax2,ax3]

		dati = [data04, data12,data15]

		cols = scenari[o][0]; name = scenari[o][1]

		ax1.annotate(name, xy =(0.,0.3) , xycoords = 'axes fraction', rotation = 90, fontsize = 16)

		for a in range(len(axes)):
			if a == 0:
				X_train, X_val, Y_train, Y_val, Xnb_train, Xnb_val, Ynb_train, Ynb_val, wherev_train, wherev_val = fn.nested_train_test_self (data04, cols)
			else:
				X_train, X_val, Y_train, Y_val, Xnb_train, Xnb_val, Ynb_train, Ynb_val, wherev_train, wherev_val = fn.nested_train_test_other (data04, dati[a], cols)

			# find the distribution peaks and align them
			selectrain = np.in1d(range(X_train.shape[0]), wherev_train)
			selecval = np.in1d(range(X_val.shape[0]), wherev_val)

			Xt = np.copy(X_train)
			Xv = np.copy(X_val)

			"""
			for s in range(len(Xv[0])):
				step = 0.05
				histt,binst = np.histogram(Xt[:,s], bins  = np.arange(0,1,step))
				histv,binsv = np.histogram(Xv[:,s], bins  = np.arange(0,1,step))

				#A = np.where(histt == np.amax(histt))[0]
				#B = np.where(histv == np.amax(histv))[0]

				#A5 = np.where(histt >= np.percentile(histt, 10))[0][0]
				#A95 = np.where(histt <= np.percentile(histt, 90))[0][-1]

				#B5 = np.where(histv >= np.percentile(histv, 10))[0][0]
				#B95 = np.where(histv <= np.percentile(histv, 90))[0][-1]

				#print (A-B)
				#print (A95,B95)
				#print()

				before = np.copy(X_val[:,s])
				#X_val[:,s] = ( X_val[:,s] + step*(A5-B5) ) * (A95/B95)

				At = np.percentile(Xt[:,s],25)
				Av = np.percentile(Xv[:,s],25)
				
				Bt = np.percentile(Xt[:,s],75)
				Bv = np.percentile(Xv[:,s],75)

				X_val[:,s] = ( before + step*(At-Av) ) * (Bt/Bv)
				
				#print (np.mean(before - X_val[:,s]), At-Av, Bt/Bv)
				#print()
				"""

			# run a quantile transform on both the training and validation datasets
			#X_train = quantile_transform(X_train, axis=0, output_distribution='normal')
			#X_val = quantile_transform(X_val, axis=0, output_distribution='normal')
				
			# now predict the results
			model, Y_pred, m = fn.RF_fit_predict_score(X_train,Y_train,X_val,Y_val)
			S,P,A = fn.confusion_metrix (m)

			axes[a].set_title ('S:'+str(int(100*S))+'; P:'+str(int(100*P))+'; A:'+str(int(100*A)))
			axes[a].pie(m.ravel(), explode = explode, startangle=90, colors = colours)
			axes[a].axis('equal')



			"""
			fig = plt.figure(2, facecolor='White',figsize=[6, 14])

			for b in range(len(bands)):

				#print (np.histogram(Xt[:,b*len(bands)+2],bins = np.arange(0,1,0.05)))

				axa =  plt.subplot2grid((len(bands),3),(b,a),colspan=1, rowspan=1)
				axa.hist(X_train[:,b*len(bands)+2], bins = np.arange(0,1,0.05), facecolor = 'r', alpha = 0.5, density = True)
				axa.hist(X_val[:,b*len(bands)+2], bins = np.arange(0,1,0.05), facecolor = 'b', alpha = 0.5, density = True)
				axa.set_title(bands[b])


		plt.savefig(Fig_dir + 'Hist_mean_all_meanshift2.png')
		quit()
		"""
		break
		
		


	plt.tight_layout()
	plt.savefig(Fig_dir + 'TEST_srf_canaltrans_noqt_addcanals.png')
	quit()


	# https://machinelearningmastery.com/quantile-transforms-for-machine-learning/







































	#########################################################################
	# Try a nested approach to get density classes
	#########################################################################
	"""
	To do a nested approach:
	1. train a model on all the data
	2. train a second model on actual positives
	3. predict veg presence on all validation set with the 1st model
	4. predict veg density on predicted positives with the 2nd model
	5. evaluate perf of 1st and 2nd models on the full dataset
	"""


	# prepare to plot effects of veg cover
	fig = plt.figure(1, facecolor='White',figsize=[14, 14])
	ax1 =  plt.subplot2grid((2,2),(0,0),colspan=1, rowspan=1)
	ax2 =  plt.subplot2grid((2,2),(0,1),colspan=1, rowspan=1)
	ax3 =  plt.subplot2grid((2,2),(1,0),colspan=1, rowspan=1)
	ax4 =  plt.subplot2grid((2,2),(1,1),colspan=1, rowspan=1)


	# select the data based on the fraction cover
	empty_data = combodata[combodata['fraction_veg_cover'] == 0]
	full_data = combodata[combodata['fraction_veg_cover'] > 0.9]
	short_data = empty_data.append(full_data)

	# get data into array
	X = np.asarray(short_data.loc[:,['bathy']+useful_bands+classif])

	# get rid of nan-containing rows in array
	X = X[~np.isnan(X).any(axis=1), :]

	# separate the features (X) from classification columns (y)
	Y = X[:,-1]; X = X[:,:-1]

	# normalise features
	for x in range(len(X[0])):
		X[:,x] = (X[:,x] - min(X[:,x])) / ( max(X[:,x]) - min(X[:,x]) )

	# Make binary classif. Keep a copy of Y for further non-binary classif
	Ynb = copy.deepcopy(Y)
	Y[Y > 0] = 1

	# split the data
	kf = KFold(n_splits=5, shuffle = True)
	kf.get_n_splits(X)


	M = []; Mnb = []; Mall = []


	count = 0
	for train_idx, test_idx in kf.split(X):
		# create training and validation datasets for model 1
		X_train, X_val = X[train_idx], X[test_idx]
		Y_train, Y_val = Y[train_idx], Y[test_idx]

		# this is for later
		Yall_train, Yall_val = Ynb[train_idx], Ynb[test_idx]
		Ynb_train, Ynb_val = Ynb[train_idx], Ynb[test_idx]

		# make non-binary classif among positives 
		Xnb_train, Xnb_val = X_train[Ynb_train >= 1], X_val[Ynb_val >= 1]
		Ynb_train, Ynb_val = Ynb_train[Ynb_train >= 1], Ynb_val[Ynb_val >= 1]


		# create the binary RF, fit and predict veg presence
		print ('looking for vegetation') 
		model = RandomForestClassifier(n_estimators = 100, max_depth = 30)
		timer_start = datetime.now()
		model.fit(X_train, Y_train); print ('fitting complete')
		timer  = datetime.now() - timer_start
		print ('RF fitting time', timer)
		Y_pred = model.predict(X_val); print ('prediction complete')

		# create the non-binary nested RF, fit and predict veg density
		print ('looking for vegetation density') 
		modelnb = RandomForestClassifier(n_estimators = 100, max_depth = 30)
		timer_start = datetime.now()
		modelnb.fit(Xnb_train, Ynb_train); print ('fitting complete')
		timer  = datetime.now() - timer_start
		print ('RF fitting time', timer)
		Ynb_pred = modelnb.predict(Xnb_val); print ('prediction complete')


		# create the non-binary RF, fit and predict veg density and presence
		print ('I want it all') 
		modelall = RandomForestClassifier(n_estimators = 100, max_depth = 30)
		timer_start = datetime.now()
		modelall.fit(X_train, Yall_train); print ('fitting complete')
		timer  = datetime.now() - timer_start
		print ('RF fitting time', timer)
		Yall_pred = modelall.predict(X_val); print ('prediction complete')


		# Build a confusion matrix for each model
		m = confusion_matrix(Y_val, Y_pred) # rows = pred value ; cols = actual value
		mnb = confusion_matrix(Ynb_val, Ynb_pred) # rows = pred value ; cols = actual value
		mall = confusion_matrix(Yall_val, Yall_pred) # rows = pred value ; cols = actual value

		M.append(m)
		Mnb.append(mnb)
		Mall.append(mall)

		count += 1
		#if count > 0:
		#	break

	# Here are our confusion matrices
	M = np.asarray(M); Mnb = np.asarray(Mnb); Mall = np.asarray(Mall)
	M = np.mean(M,axis = 0); Mnb = np.mean(Mnb,axis = 0); Mall = np.mean(Mall,axis = 0)

	# Make an extra one to compare apples with apples
	Mallveg = Mall[1:,1:]

	Mallcopy = np.copy(Mall)

	Mallcopy[0,1] = np.sum(Mallcopy[0,1:])
	Mallcopy[1,0] = np.sum(Mallcopy[1:,0])
	Mallcopy[1,1] = np.sum(Mallcopy[1:,1:])

	Mallbin = Mallcopy[:2,:2]


	sn.heatmap(M, ax = ax1, cmap = plt.cm.viridis, annot=True, fmt='g', cbar = False)
	sn.heatmap(Mnb, ax = ax2, cmap = plt.cm.viridis, annot=True, fmt='g', cbar = False)
	sn.heatmap(Mallbin, ax = ax3, cmap = plt.cm.viridis, annot=True, fmt='g', cbar = False)
	sn.heatmap(Mallveg, ax = ax4, cmap = plt.cm.viridis, annot=True, fmt='g', cbar = False)

	ax2.set_xticklabels([1,2,3,4])
	ax4.set_xticklabels([1,2,3,4])
	ax2.set_yticklabels([1,2,3,4])
	ax4.set_yticklabels([1,2,3,4])


	ax3.set_xlabel('Predicted');ax4.set_xlabel('Predicted') 
	ax1.set_ylabel('Actual'), ax3.set_ylabel('Actual')


	axes = [ax1,ax2,ax3,ax4]
	matrices = [M,Mnb,Mallbin,Mallveg]

	for m in range(len(matrices)):
		mtx = matrices[m]
		mysum = 0
		for r in range(len(mtx)):
			mysum += mtx[r,r]
			S = mtx[r,r] / np.sum(mtx[r,:])
			P = mtx[r,r] / np.sum(mtx[:,r])

			axes[m].annotate('Sen:'+str(round(100*S,2))+'%', xy = [len(mtx),r+0.7], xycoords = 'data', rotation = -90, color = 'r')
			axes[m].annotate('Pre:'+str(round(100*P,2))+'%', xy = [r+0.3,0.], xycoords = 'data', color= 'r')

		axes[m].set_title('Accuracy:'+str(round(100*(mysum/np.sum(mtx)),2))+'%')

	"""
	# plot average class-wise metrics
	for i in range(len(Precision[0])):
		kPre = np.mean(Precision[:,i])
		kSen = np.mean(Sensitivity[:,i])
		paint = i/len(Precision[0])

		ax1.annotate('Pre:'+str(round(100*kPre,2))+'%', xy = [i,0], xycoords = 'data')
		ax1.annotate('Sen:'+str(round(100*kSen,2))+'%', xy = [0,i+1], xycoords = 'data', rotation = 90)


	# plot average class-wise metrics
	for i in range(len(Precisionnb[0])):
		kPre = np.mean(Precisionnb[:,i])
		kSen = np.mean(Sensitivitynb[:,i])
		paint = i/len(Precisionnb[0])

		ax2.annotate('Pre:'+str(round(100*kPre,2))+'%', xy = [i,0], xycoords = 'data')
		ax2.annotate('Sen:'+str(round(100*kSen,2))+'%', xy = [0,i+1], xycoords = 'data', rotation = 90)

	# plot average class-wise metrics
	for i in range(len(Precisionall[0])):
		kPre = np.mean(Precisionall[:,i])
		kSen = np.mean(Sensitivityall[:,i])
		paint = i/len(Precisionall[0])

		ax3.annotate('Pre:'+str(round(100*kPre,2))+'%', xy = [i,0], xycoords = 'data')
		ax3.annotate('Sen:'+str(round(100*kSen,2))+'%', xy = [0,i+1], xycoords = 'data', rotation = 90)

	"""

	plt.tight_layout()
	plt.savefig('Figures/2004/RFc_confusion_td30_summer_nested.png')
	quit()


	#########################################################################
	#########################################################################


	"""
	# prepare to plot effects of veg cover
	fig = plt.figure(1, facecolor='White',figsize=[14, 10])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

	count = 0
	for cov in np.arange(0.1,1,0.1):


		print (cov)

		# select the data based on the fraction cover
		empty_data = combodata[combodata['fraction_veg_cover'] == 0]
		full_data = combodata[combodata['fraction_veg_cover'] > cov]
		short_data = empty_data.append(full_data)

		# get data into array
		X = np.asarray(short_data.loc[:,useful_bands+classif])

		# get rid of nan-containing rows in array
		X = X[~np.isnan(X).any(axis=1), :]

		# separate the features (X) from classification columns (y)
		Y = X[:,-1]; X = X[:,:-1]

		# normalise features
		for x in range(len(X[0])):
			X[:,x] = (X[:,x] - min(X[:,x])) / ( max(X[:,x]) - min(X[:,x]) )

		# Make classif binary. Keep a copy of the original
		Ynb = copy.deepcopy(Y)
		Y[Y > 0] = 1

		# split the data
		kf = KFold(n_splits=5, shuffle = True)
		kf.get_n_splits(X)

		Precision = []; Sensitivity = []

		for train_idx, test_idx in kf.split(X):
			X_train, X_val = X[train_idx], X[test_idx]
			Y_train, Y_val = Y[train_idx], Y[test_idx]

			model = RandomForestClassifier(n_estimators = 100, max_depth = 30)
			timer_start = datetime.now()
			model.fit(X_train, Y_train); print ('fitting complete')
			timer  = datetime.now() - timer_start
			print ('RF fitting time', timer)
			Y_pred = model.predict(X_val); print ('prediction complete')


			M = confusion_matrix(Y_val, Y_pred) # rows = pred value ; cols = actual value

			# class-wise metrics
			sen = []; pre = []
			for i in range(len(M)):
				sen.append(M[i,i] / np.sum(M[i,:])) # num correct preds / all preds to be this value
				pre.append(M[i,i] / np.sum(M[:,i])) # num correct preds / all actual values

			Precision.append(pre)
			Sensitivity.append(sen)
			

		Precision = np.asarray(Precision)
		Sensitivity = np.asarray(Sensitivity)

		# plot average class-wise metrics
		for i in range(len(Precision[0])):
			kPre = np.mean(Precision[:,i])
			kSen = np.mean(Sensitivity[:,i])
			paint = i/len(Precision[0])

			ax1.scatter(count, kPre, marker = 'o', facecolor = plt.cm.jet(paint), lw = 0, alpha = 0.8, label = 'class'+str(i))
			ax1.annotate(' '+ str(round(100*kPre,2)), xy = (count, kPre), xycoords = 'data', rotation = 0)

			ax1.scatter(count+0.1, kSen, marker = 'D', facecolor = plt.cm.jet(paint), lw = 0, alpha = 0.8)
			ax1.annotate(' '+ str(round(100*kSen,2)), xy = (count+0.1, kSen), xycoords = 'data', rotation = 0)
			
		ax1.annotate(timer,  xy = (count, 1.01), xycoords = 'data', rotation = 45)

		count += 1
		
		print()

	ax1.set_ylim(0.4,1.10)
	ax1.set_xlabel('Min. cover to consider pixel vegetated')
	#ax1.legend()
	ax1.grid(True)

	plt.savefig('Figures/2004/RFc_performance_td30_summer_full_cover.png')
	"""



	# prepare to plot effects of veg cover and density class

	# select the data based on the fraction cover
	empty_data = combodata[combodata['fraction_veg_cover'] == 0]
	full_data = combodata[combodata['fraction_veg_cover'] > 0.05]
	short_data = empty_data.append(full_data)

	# get data into array
	X = np.asarray(short_data.loc[:,['bathy']+useful_bands+classif+cover])

	# get rid of nan-containing rows in array
	X = X[~np.isnan(X).any(axis=1), :]

	# separate the features (X) from classification columns (Y) and vcover (C)
	Y = X[:,-2]; C = X[:,-1]; X = X[:,:-2]

	# normalise features
	for x in range(len(X[0])):
		X[:,x] = (X[:,x] - min(X[:,x])) / ( max(X[:,x]) - min(X[:,x]) )


	fig = plt.figure(1, facecolor='White',figsize=[14, 10])
	ax1 =  plt.subplot2grid((3,2),(1,0),colspan=1, rowspan=1)
	ax11 =  plt.subplot2grid((3,2),(1,1),colspan=1, rowspan=1)
	ax111 =  plt.subplot2grid((3,2),(2,0),colspan=1, rowspan=1)
	ax1111 =  plt.subplot2grid((3,2),(2,1),colspan=1, rowspan=1)
	ax11111 =  plt.subplot2grid((3,2),(0,0),colspan=1, rowspan=1)
	ax2 =  plt.subplot2grid((3,2),(0,1),colspan=1, rowspan=1)

	# split the data
	kf = KFold(n_splits=5, shuffle = True)
	kf.get_n_splits(X)

	# Make classif binary. Keep a copy of the original
	Ynb = copy.deepcopy(Y)
	Y[Y > 0] = 1
	for train_idx, test_idx in kf.split(X):
		X_train, X_val = X[train_idx], X[test_idx]
		Y_train, Y_val = Y[train_idx], Y[test_idx]
		Ynb_val = Ynb[test_idx]
		C_val = C[test_idx]

		model = RandomForestClassifier(n_estimators = 100, max_depth = 30)
		timer_start = datetime.now()
		model.fit(X_train, Y_train); print ('fitting complete')
		timer  = datetime.now() - timer_start
		print ('RF fitting time', timer)

		# votes for vegetation!
		Proba = model.predict_proba(X_val)
		PP = Proba[:,1]
		Y_pred = model.predict(X_val)

		Y_pred = model.predict(X_val); print ('prediction complete')

		M = confusion_matrix(Y_val, Y_pred) # rows = pred value ; cols = actual value

		print ('confusion matrix')
		print (M)
		print()


		ax1.scatter(C_val[Ynb_val==1], PP[Ynb_val==1], marker = '.', s = 8, facecolor = 'k', alpha = 0.8)
		ax11.scatter(C_val[Ynb_val==1], PP[Ynb_val==1], marker = '.', s = 8, facecolor = 'k', alpha = 0.8)
		ax111.scatter(C_val[Ynb_val==2], PP[Ynb_val==2], marker = '.', s = 8, facecolor = 'k', alpha = 0.8)
		ax1111.scatter(C_val[Ynb_val==3], PP[Ynb_val==3], marker = '.', s = 8, facecolor = 'k', alpha = 0.8)
		#ax11111.scatter(C_val[Ynb_val==0], PP[Ynb_val==0], marker = '.', s = 8, facecolor = 'k', alpha = 0.8)
		


		ax2.scatter(C_val, PP, marker = '.', s = 8, facecolor = plt.cm.jet(Y_pred), alpha = 0.8)
		ax11111.scatter(C_val, PP, marker = '.', s = 8, facecolor = plt.cm.jet(Y_val), alpha = 0.5)


	ax1.set_xlim(-0.05,1)
	ax11.set_xlim(-0.05,1)
	ax111.set_xlim(-0.05,1)
	ax1111.set_xlim(-0.05,1)
	ax11111.set_xlim(-0.05,1)
	ax2.set_xlim(-0.05,1)
	ax1.set_ylabel('Probability of being vegetated')
	ax1.set_xlabel('Min. cover to consider pixel vegetated')
	ax2.set_xlabel('Min. cover to consider pixel vegetated')

	ax1.grid(True)

	plt.savefig('Figures/2004/RFc_performance_td30_summer_full_coverclass.png')


	# Make this a heatmap but I don't think it will improve the situation much ...
	# Also, WTF is wrong with these percentage covers that are >> 1 ? Look this up !!!!!

	# Next stage, validate on spectral data + polygons from another dataset
	# OR, improve the classification by playing around with probabilities and fraction cover (if the heatmap is conclusive).
	# Another way to improve the classification of densities could work by training a second model with the true positives. This might work out the best.

	# Start thinking about an efficient way to:
	#1. do calibration on the CORILA dataset
	#2. run the model on arrays for full-lagoon predictions
	#3. save the resulting arrays to rasters.
	#4. HOW DO WE VALIDATE WINTER PREDICTIONS?

	quit()



	##########################################################################
	# Attempt to express variables for a single image
	##########################################################################
	"""
	useful_bands = ['blue_t', 'green_t', 'red_t', 'nir_t',
					'blue_bot_t', 'green_bot_t', 'red_bot_t', 'nir_bot_t']
	bands_to_drop = ['swir_t', 'swir_bot_t', 'blue_synth_t', 'green_synth_t', 'red_synth_t', 'nir_synth_t', 'swir_synth_t']

	for b in useful_bands:
		X = np.asarray(combodata.loc[:,b])
		X = np.asarray(list(map(lambda m_ls: np.asarray(m_ls), X)), dtype = np.float)

		X[X == -9999.] = np.nan
		if 'bot' in b:
			X[X > 0.5] = np.nan

		for x in range(len(X[0])):
			combodata[b[:-2]+'_val'+str(x)] = list(X[:,x]) 

		# discard the legnthy column
		combodata = combodata.drop(b, axis = 1) 

	# discard useless bands
	for b in bands_to_drop:
		combodata = combodata.drop(b, axis = 1) 


	# that's enough metrics for now
	timer  = datetime.now() - timer_start
	print ('preprocessing time', timer)
	timer_start = datetime.now()
	"""

	##########################################################################
	# Now try some classifiers
	##########################################################################
	"""
	bands = ['blue', 'green','red', 'nir']
	bands_bot = ['blue_bot', 'green_bot','red_bot', 'nir_bot']

	mydates =['val0','val1','val2','val3','val4','val5','val6','val7']

	cover = ['fraction_veg_cover']
	classif = ['density_class']


	scenarii = []; sc_names = []
	scenarii_bot = []; sc_bot_names = []

	for k in range(len(mydates)):
		useful_bands = [i+'_'+mydates[k] for i in bands]
		useful_bands_bot = [i+'_'+mydates[k] for i in bands_bot]

		scenarii.append(['bathy']+useful_bands)
		scenarii_bot.append(['bathy']+useful_bands_bot)

		D = list(combodata.loc[(175,993), 'survey_t'])

		sc_names.append('srf_'+ D[k])
		sc_bot_names.append('bot_'+ D[k])


	scenarii = scenarii+scenarii_bot
	sc_names = sc_names+sc_bot_names

	# reset the index ???
	combodata = combodata.reset_index()

	# prepare to plot effects of ...
	fig = plt.figure(1, facecolor='White',figsize=[14, 10])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

	count = 0
	for b in range(len(scenarii)):

		print (sc_names[count])

		# select the data based on the fraction cover
		empty_data = combodata[combodata['fraction_veg_cover'] == 0]
		full_data = combodata[combodata['fraction_veg_cover'] > 0.9]
		short_data = empty_data.append(full_data)

		
		print (scenarii[b]+classif)

		# get data into array
		X = np.asarray(short_data.loc[:,scenarii[b]+classif])

		# get rid of nan-containing rows in array
		X = X[~np.isnan(X).any(axis=1), :]

		# separate the features (X) from classification columns (y)
		Y = X[:,-1]; X = X[:,:-1]

		# normalise features
		for x in range(len(X[0])):
			X[:,x] = (X[:,x] - min(X[:,x])) / ( max(X[:,x]) - min(X[:,x]) )

		# Make classif binary
		#Y[Y > 0] = 1

		# get rid of the continuous y values
		for i in range(len(Y)):
			Y[i] = int(Y[i])

		# split the data
		from sklearn.model_selection import KFold
		kf = KFold(n_splits=5, shuffle = True)
		kf.get_n_splits(X)

		Precision = []; Sensitivity = []

		for train_idx, test_idx in kf.split(X):
			X_train, X_val = X[train_idx], X[test_idx]
			Y_train, Y_val = Y[train_idx], Y[test_idx]

			model = RandomForestClassifier(n_estimators = 100, max_depth = 30)
			timer_start = datetime.now()
			model.fit(X_train, Y_train); print ('fitting complete')
			timer  = datetime.now() - timer_start
			print ('RF fitting time', timer)
			Y_pred = model.predict(X_val); print ('prediction complete')

			M = confusion_matrix(Y_val, Y_pred) # rows = pred value ; cols = actual value

			# class-wise metrics
			sen = []; pre = []
			for i in range(len(M)):
				sen.append(M[i,i] / np.sum(M[i,:])) # num correct preds / all preds to be this value
				pre.append(M[i,i] / np.sum(M[:,i])) # num correct preds / all actual values

			Precision.append(pre)
			Sensitivity.append(sen)
			break

		Precision = np.asarray(Precision)
		Sensitivity = np.asarray(Sensitivity)

		# plot average class-wise metrics
		for i in range(len(Precision[0])):
			kPre = np.mean(Precision[:,i])
			kSen = np.mean(Sensitivity[:,i])
			paint = i/len(Precision[0])

			ax1.scatter(count, kPre, marker = 'o', facecolor = plt.cm.jet(paint), lw = 0, alpha = 0.8, label = 'class'+str(i))
			ax1.annotate(' '+ str(round(100*kPre,2)), xy = (count, kPre), xycoords = 'data', rotation = 0)

			ax1.scatter(count+0.1, kSen, marker = 'D', facecolor = plt.cm.jet(paint), lw = 0, alpha = 0.8)
			ax1.annotate(' '+ str(round(100*kSen,2)), xy = (count+0.1, kSen), xycoords = 'data', rotation = 0)
			
			ax1.annotate(' '+sc_names[count], xy = (count, 0.4), xycoords = 'data', rotation = 45)

		
		ax1.annotate(timer,  xy = (count, 1.05), xycoords = 'data', rotation = 0)
		count += 1
		
		print()

	ax1.set_ylim(0.4,1.10)
	#ax1.legend()
	ax1.grid(True)

	plt.savefig('Figures/2004/RFc_performance_td30_singleimage.png')

	quit()
	"""






	##########################################################################
	# Attempt to express variables for groups of images
	##########################################################################

	useful_bands = ['blue_t', 'green_t', 'red_t', 'nir_t',
					'blue_bot_t', 'green_bot_t', 'red_bot_t', 'nir_bot_t']
	bands_to_drop = ['swir_t', 'swir_bot_t', 'blue_synth_t', 'green_synth_t', 'red_synth_t', 'nir_synth_t', 'swir_synth_t']

	for b in useful_bands:
		X = np.asarray(combodata.loc[:,b])
		X = np.asarray(list(map(lambda m_ls: np.asarray(m_ls), X)), dtype = np.float)

		X[X == -9999.] = np.nan
		if 'bot' in b:
			X[X > 0.5] = np.nan

		# Only use a summer trio
		X = X[:,3:5]

		# Make the relevant metrics
		combodata[b[:-2]+'_max'] = list(np.nanmax(X, axis = 1)) 
		combodata[b[:-2]+'_min'] = list(np.nanmin(X, axis = 1)) 
		combodata[b[:-2]+'_avg'] = list(np.nanmean(X, axis = 1)) 
		combodata[b[:-2]+'_std'] = list(np.nanstd(X, axis = 1))

		# discard the legnthy column
		combodata = combodata.drop(b, axis = 1) 

	# discard useless bands
	for b in bands_to_drop:
		combodata = combodata.drop(b, axis = 1) 


	# that's enough metrics for now
	timer  = datetime.now() - timer_start
	print ('preprocessing time', timer)
	timer_start = datetime.now()


	##########################################################################
	# Now try some classifiers
	##########################################################################

	bands = ['blue', 'green','red', 'nir']
	bands_bot = ['blue_bot', 'green_bot','red_bot', 'nir_bot']

	mydates =['val0','val1','val2','val3','val4','val5','val6','val7']

	cover = ['fraction_veg_cover']
	classif = ['density_class']


	scenarii = []; sc_names = []
	scenarii_bot = []; sc_bot_names = []

	for k in range(len(mydates)):
		useful_bands = [i+'_'+mydates[k] for i in bands]
		useful_bands_bot = [i+'_'+mydates[k] for i in bands_bot]

		scenarii.append(['bathy']+useful_bands)
		scenarii_bot.append(['bathy']+useful_bands_bot)

		D = list(combodata.loc[(175,993), 'survey_t'])

		sc_names.append('srf_'+ D[k])
		sc_bot_names.append('bot_'+ D[k])


	scenarii = scenarii+scenarii_bot
	sc_names = sc_names+sc_bot_names

	# reset the index ???
	combodata = combodata.reset_index()

	# prepare to plot effects of ...
	fig = plt.figure(1, facecolor='White',figsize=[14, 10])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

	count = 0
	for b in range(len(scenarii)):

		print (sc_names[count])

		# select the data based on the fraction cover
		empty_data = combodata[combodata['fraction_veg_cover'] == 0]
		full_data = combodata[combodata['fraction_veg_cover'] > 0.9]
		short_data = empty_data.append(full_data)

		
		print (scenarii[b]+classif)

		# get data into array
		X = np.asarray(short_data.loc[:,scenarii[b]+classif])

		# get rid of nan-containing rows in array
		X = X[~np.isnan(X).any(axis=1), :]

		# separate the features (X) from classification columns (y)
		Y = X[:,-1]; X = X[:,:-1]

		# normalise features
		for x in range(len(X[0])):
			X[:,x] = (X[:,x] - min(X[:,x])) / ( max(X[:,x]) - min(X[:,x]) )

		# Make classif binary
		#Y[Y > 0] = 1

		# get rid of the continuous y values
		for i in range(len(Y)):
			Y[i] = int(Y[i])

		# split the data
		from sklearn.model_selection import KFold
		kf = KFold(n_splits=5, shuffle = True)
		kf.get_n_splits(X)

		Precision = []; Sensitivity = []

		for train_idx, test_idx in kf.split(X):
			X_train, X_val = X[train_idx], X[test_idx]
			Y_train, Y_val = Y[train_idx], Y[test_idx]

			model = RandomForestClassifier(n_estimators = 100, max_depth = 30)
			timer_start = datetime.now()
			model.fit(X_train, Y_train); print ('fitting complete')
			timer  = datetime.now() - timer_start
			print ('RF fitting time', timer)
			Y_pred = model.predict(X_val); print ('prediction complete')

			M = confusion_matrix(Y_val, Y_pred) # rows = pred value ; cols = actual value

			# class-wise metrics
			sen = []; pre = []
			for i in range(len(M)):
				sen.append(M[i,i] / np.sum(M[i,:])) # num correct preds / all preds to be this value
				pre.append(M[i,i] / np.sum(M[:,i])) # num correct preds / all actual values

			Precision.append(pre)
			Sensitivity.append(sen)
			break

		Precision = np.asarray(Precision)
		Sensitivity = np.asarray(Sensitivity)

		# plot average class-wise metrics
		for i in range(len(Precision[0])):
			kPre = np.mean(Precision[:,i])
			kSen = np.mean(Sensitivity[:,i])
			paint = i/len(Precision[0])

			ax1.scatter(count, kPre, marker = 'o', facecolor = plt.cm.jet(paint), lw = 0, alpha = 0.8, label = 'class'+str(i))
			ax1.annotate(' '+ str(round(100*kPre,2)), xy = (count, kPre), xycoords = 'data', rotation = 0)

			ax1.scatter(count+0.1, kSen, marker = 'D', facecolor = plt.cm.jet(paint), lw = 0, alpha = 0.8)
			ax1.annotate(' '+ str(round(100*kSen,2)), xy = (count+0.1, kSen), xycoords = 'data', rotation = 0)
			
			ax1.annotate(' '+sc_names[count], xy = (count, 0.4), xycoords = 'data', rotation = 45)

		
		ax1.annotate(timer,  xy = (count, 1.05), xycoords = 'data', rotation = 0)
		count += 1
		
		print()

	ax1.set_ylim(0.4,1.10)
	#ax1.legend()
	ax1.grid(True)

	plt.savefig('Figures/2004/RFc_performance_td30_summergroups.png')

	quit()

















































	# prepare to plot effects of ...
	fig = plt.figure(1, facecolor='White',figsize=[14, 10])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

	for md in range(10,100,20): # effect of tree depth

		print ('tree depth:', md)

		# Prepare the scoring metrics
		Precision = []; Sen = []; Spec = []

		# split the data
		from sklearn.model_selection import KFold
		kf = KFold(n_splits=5, shuffle = True)
		kf.get_n_splits(X)


		for train_idx, test_idx in kf.split(X):
			X_train, X_val = X[train_idx], X[test_idx]
			Y_train, Y_val = Y[train_idx], Y[test_idx]

			model = RandomForestClassifier(n_estimators = 100, max_depth = md)
			model.fit(X_train, Y_train); print ('fitting complete')
			Y_pred = model.predict(X_val); print ('prediction complete')

			# ths only works for binary classif
			SG = Y_val[Y_val >= 1]; SG_pred = Y_pred[Y_val >= 1]
			RTP = 100*sum(SG_pred)/len(SG) # recall: rate of true positives among actual positives

			NV = Y_val[Y_val < 1]; NV_pred = Y_pred[Y_val < 1]
			RTN = 100*(1- sum(NV_pred)/len(NV) )# rate of true negatives

			P = len(Y_pred[Y_pred >= 1]); TP = sum(SG_pred)
			pre = 100 * TP/P # precision: rate of positives that are true

			Precision.append(pre); Sen.append(RTP); Spec.append(RTN)

		if md == 10:
			ax1. scatter(md, np.mean(Precision), marker = 'o', facecolor = 'b', label = 'Precision')
			ax1. scatter(md, np.mean(Sen), marker = 'D', facecolor = 'g', label = 'Sensitivity')
			ax1. scatter(md, np.mean(Spec), marker = 'X', facecolor = 'r', label = 'Specificity')
		else:
			ax1. scatter(md, np.mean(Precision), marker = 'o', facecolor = 'b')
			ax1. scatter(md, np.mean(Sen), marker = 'D', facecolor = 'g')
			ax1. scatter(md, np.mean(Spec), marker = 'X', facecolor = 'r')

		ax1.legend(loc = 1)


	plt.savefig('Figures/2004/RFc_performance_'+sc_names[count]+'_treedepth.png')

	count += 1

	#VotingClassifier(estimators=[('Gaussian NB', GaussianNB()), ('Random Forest', RandomForestClassifier(n_estimators = 100, max_depth = 100))], 			                       voting='hard', weights=[2,1])


	"""
	AA = Y_val[Y_val >= 1]
	aa = Y_pred[Y_val >= 1]

	BB = Y_val[Y_val < 1]
	bb = Y_pred[Y_val < 1]

	print (sc_names[count-1])

	print ('percentage seagrasses correctly identified', 100*sum(aa)/len(AA))
	print ('percentage unveg correctly identified', 100*(1-sum(bb)/len(BB)))
	"""

	timer  = datetime.now() - timer_start
	print ('RF time', timer)
	timer_start = datetime.now()
	print()

	quit()


	# things to test:
	# tree depth effect
	# N_estimators effect
	# effect of removing bands
	# capacity of voting process to predict density

















































	for D in range(len(uniquedates)):
		model_data = data[data['survey'] == uniquedates[D]]
		str_date = uniquedates[D].strftime("%Y%m%d")
		print (str_date)
		Year = str_date[:4]

		# make a figure of the bathymetry of the seagrass
		fig = plt.figure(1, facecolor='White',figsize=[14, 10])
		ax1 =  plt.subplot2grid((3,4),(0,0),colspan=1, rowspan=1)
		ax2 =  plt.subplot2grid((3,4),(0,1),colspan=1, rowspan=1)
		ax3 =  plt.subplot2grid((3,4),(0,2),colspan=1, rowspan=1)
		ax4 =  plt.subplot2grid((3,4),(0,3),colspan=1, rowspan=1)
		ax5 =  plt.subplot2grid((3,4),(1,0),colspan=4, rowspan=1)
		ax6 =  plt.subplot2grid((3,4),(2,0),colspan=4, rowspan=1)
		axes = [ax1,ax2,ax3,ax4]

		for i in range(len(axes)):
			short_data = model_data[model_data['density_class'] == i+1]
			axes[i].scatter(short_data.loc[:,'bathy'], short_data.loc[:,'fraction_veg_cover'], marker  = '.', facecolor = 'k', alpha = 0.4, s = 5, lw = 0)
			axes[i].set_xlabel('bathymetry (m)')
			axes[i].set_title('Density class:'+ str(i+1))
			axes[i].set_xlim(-11.5,1.5)
			axes[i].set_ylim(0,1)

		noveg = model_data[model_data['fraction_veg_cover'] < 0.01]
		ax5.hist(noveg.loc[:,'bathy'], bins = 200, facecolor = 'y', alpha = 0.5, edgecolor = 'k', lw = 0.7)
		ax6.hist(noveg.loc[:,'depth'], bins = 200, facecolor = 'y', alpha = 0.5, edgecolor = 'k', lw = 0.7)

		lessveg = model_data[model_data['fraction_veg_cover'] > 0.01]
		lessveg = lessveg[lessveg['fraction_veg_cover'] <= 0.5]
		ax5.hist(lessveg.loc[:,'bathy'], bins = 200, facecolor = 'g', alpha = 0.3, edgecolor = 'k', lw = 0.7)
		ax6.hist(lessveg.loc[:,'depth'], bins = 200, facecolor = 'g', alpha = 0.3, edgecolor = 'k', lw = 0.7)

		partveg = model_data[model_data['fraction_veg_cover'] > 0.5]
		partveg = partveg[partveg['fraction_veg_cover'] <= 0.99]
		ax5.hist(partveg.loc[:,'bathy'], bins = 200, facecolor = 'g', alpha = 0.5, edgecolor = 'k', lw = 0.7)
		ax6.hist(partveg.loc[:,'depth'], bins = 200, facecolor = 'g', alpha = 0.5, edgecolor = 'k', lw = 0.7)

		fullveg = model_data[model_data['fraction_veg_cover'] > 0.99]
		ax5.hist(fullveg.loc[:,'bathy'], bins = 200, facecolor = 'g', alpha = 0.8, edgecolor = 'k', lw = 0.7)
		ax6.hist(fullveg.loc[:,'depth'], bins = 200, facecolor = 'g', alpha = 0.8, edgecolor = 'k', lw = 0.7)

		ax1.set_ylabel('Fraction of pixel covered by vegetation')
		ax5.set_xlabel('Bathymetry (m)')
		ax5.set_ylabel('Distribution')
		ax5.set_xlim(-10, 1)
		ax6.set_xlim(-0.5, 10)
		plt.savefig('Figures/2004/Vegetation_bathy_coverage_'+str_date+'.png')

		



		# make a figure of the Spectral response density of the seagrass and ground
		for x in np.arange(-4,0.5,0.25):
			short_data = combodata[combodata['bathy'] <=x+0.25]
			short_data = combodata[combodata['bathy'] >=x]

			fig = plt.figure(2, facecolor='White',figsize=[14, 10])
			
			bands = ['blue', 'green','red', 'nir', 'swir']
			for i in range(len(bands)):

				ax1 = plt.subplot2grid((5,4),(i,0),colspan=1, rowspan=1)
				ax2 = plt.subplot2grid((5,4),(i,1),colspan=1, rowspan=1)
				ax3 = plt.subplot2grid((5,4),(i,2),colspan=1, rowspan=1)
				ax4 = plt.subplot2grid((5,4),(i,3),colspan=1, rowspan=1)

				ax1.set_ylabel(bands[i]+' band')

				if i == 0:
					ax2.set_title('date: ' + str_date)
					ax3.set_title('bathymetry: ' + str(100*x) + ' to ' + str(100*x+25) + ' cm')
				#if i < len(bands) -1:
				#	ax1.set_xticklabels([])
				#	ax2.set_xticklabels([])
				#	ax3.set_xticklabels([])
				#	ax4.set_xticklabels([])

				#ax1.set_xlim(0.,0.2); ax2.set_xlim(0.,0.2); ax3.set_xlim(0.,0.2); ax4.set_xlim(0.,0.2)
				ax1.set_ylim(0,1000); ax2.set_ylim(0,1000); ax3.set_ylim(0,1000); ax4.set_ylim(0,1000); 

				#short_data = short_data[short_data[bands[i]+'_bot_t_min'] < 0.5 ]
				short_data = short_data[short_data[bands[i]+'_bot_t_max'] < 0.5 ]
				short_data = short_data[short_data[bands[i]+'_bot_t_min'] >= 0]

				noveg = short_data[short_data['fraction_veg_cover'] < 0.01]
				N = 20
				ax1.hist(noveg.loc[:,bands[i]+'_t_dmax']/10000, bins = N, facecolor = 'y', alpha = 0.5, edgecolor = 'k', lw = 0.)
				ax2.hist(noveg.loc[:,bands[i]+'_bot_t_dmax'], bins = N, facecolor = 'y', alpha = 0.5, edgecolor = 'k', lw = 0.)

				ax3.hist(noveg.loc[:,bands[i]+'_t_range']/10000, bins = N, facecolor = 'y', alpha = 0.5, edgecolor = 'k', lw = 0.)
				ax4.hist(noveg.loc[:,bands[i]+'_bot_t_range'], bins = N, facecolor = 'y', alpha = 0.5, edgecolor = 'k', lw = 0.)

				fullveg = short_data[short_data['fraction_veg_cover'] > 0.99]
				ax1.hist(fullveg.loc[:,bands[i]+'_t_dmax']/10000, bins = N, facecolor = 'g', alpha = 0.5, edgecolor = 'k', lw = 0.)
				ax2.hist(fullveg.loc[:,bands[i]+'_bot_t_dmax'], bins = N, facecolor = 'g', alpha = 0.5, edgecolor = 'k', lw = 0.)

				ax3.hist(fullveg.loc[:,bands[i]+'_t_range']/10000, bins = N, facecolor = 'g', alpha = 0.5, edgecolor = 'k', lw = 0.)
				ax4.hist(fullveg.loc[:,bands[i]+'_bot_t_range'], bins = N, facecolor = 'g', alpha = 0.5, edgecolor = 'k', lw = 0.)

			ax1.set_xlabel('MaxChange in Surface Reflectance')
			ax3.set_xlabel('MaxChange in Surface Reflectance')
			ax2.set_xlabel('MaxChange in Bottom Reflectance')
			ax4.set_xlabel('MaxChange in Bottom Reflectance')

			plt.savefig('Figures/Spectral_pties/'+str_date+'_bathy'+str(int(-100*x))+'cm_change.png')
		quit()
		continue


















		for x in np.arange(-4,0.5,0.25):
			short_data = combodata[combodata['bathy'] <=x+0.25]
			short_data = combodata[combodata['bathy'] >=x]

			fig = plt.figure(2, facecolor='White',figsize=[14, 10])
			
			bands = ['blue', 'green','red', 'nir', 'swir']
			for i in range(len(bands)):

				# Read sonia's bare soil analysis
				#baresoil_ref = np.array([[0.0500735,0.0646339,0.0160046,0.0179186],
				#						[0.0705193,0.0861473,0.0227438,0.0248304],
				#						[0.0782468,0.0938513,0.0242921,0.0262757],
				#						[0.100798,0.121318,0.0267086,0.0353639]])

				ax1 = plt.subplot2grid((5,4),(i,0),colspan=1, rowspan=1)
				ax2 = plt.subplot2grid((5,4),(i,1),colspan=1, rowspan=1)
				ax3 = plt.subplot2grid((5,4),(i,2),colspan=1, rowspan=1)
				ax4 = plt.subplot2grid((5,4),(i,3),colspan=1, rowspan=1)

				ax1.set_ylabel(bands[i]+' band')

				if i == 0:
					ax2.set_title('date: ' + str_date)
					ax3.set_title('bathymetry: ' + str(100*x) + ' to ' + str(100*x+25) + ' cm')
				if i < len(bands) -1:
					ax1.set_xticklabels([])
					ax2.set_xticklabels([])
					ax3.set_xticklabels([])
					ax4.set_xticklabels([])

				#ax1.scatter(short_data.loc[:,bands[i]]/10000, short_data.loc[:,'fraction_veg_cover'], marker  = '.', facecolor = 'k', alpha = 0.4, s = 5, lw = 0)
				#ax2.scatter(short_data.loc[:,bands[i]+'_bot'], short_data.loc[:,'fraction_veg_cover'], marker  = '.', facecolor = 'k', alpha = 0.4, s = 5, lw = 0)

				ax1.set_xlim(0.,0.2); ax2.set_xlim(0.,0.2); ax3.set_xlim(0.,0.2); ax4.set_xlim(0.,0.2)
				ax1.set_ylim(0,1); ax2.set_ylim(0,1)

				short_data = short_data[short_data[bands[i]+'_t_dmax'] < 2000 ]
				short_data = short_data[short_data[bands[i]+'_bot_t_dmax'] >= 0]

				noveg = short_data[short_data['fraction_veg_cover'] < 0.01]
				ax3.hist(noveg.loc[:,bands[i]+'_t_dmax']/10000, bins = 200, facecolor = 'y', alpha = 0.5, edgecolor = 'k', lw = 0.)
				ax4.hist(noveg.loc[:,bands[i]+'_bot_t_dmax'], bins = 200, facecolor = 'y', alpha = 0.5, edgecolor = 'k', lw = 0.)
				#if i < 4:
				#	ax4.scatter(np.mean(baresoil_ref[i]), 0, marker = 'D', s = 25, facecolor = 'r')


				fullveg = short_data[short_data['fraction_veg_cover'] > 0.99]
				ax3.hist(fullveg.loc[:,bands[i]+'_t_dmax']/10000, bins = 200, facecolor = 'g', alpha = 0.5, edgecolor = 'k', lw = 0.)
				ax4.hist(fullveg.loc[:,bands[i]+'_bot_t_dmax'], bins = 200, facecolor = 'g', alpha = 0.5, edgecolor = 'k', lw = 0.)

			ax1.set_xlabel('MaxChange in Surface Reflectance')
			ax3.set_xlabel('MaxChange in Surface Reflectance')
			ax2.set_xlabel('MaxChange in Bottom Reflectance')
			ax4.set_xlabel('MaxChange in Bottom Reflectance')

			plt.savefig('Figures/Spectral_pties/'+str_date+'_bathy'+str(int(-100*x))+'cm_dmax.png')
			quit()
		continue

		# Classes are virtually inseparable. But are they for slices of bathy? Yes
		# What about for changes in refl?
		
		"""
		# Make a classification column. Try binary first
		CC = np.zeros(len(model_data))
		pct_tolerance = 0.1
		CC[model_data['fraction_veg_cover'] > pct_tolerance] = -1
		CC[model_data['fraction_veg_cover'] >= 1 - pct_tolerance] = 1
		#combodata = combodata.assign(bin_class = CC)
		model_data = model_data.assign(bin_class = CC)

		# select the binary values
		#combodata = combodata[combodata['bin_class'] >= 0.]
		model_data = model_data[model_data['bin_class'] >= 0.]
		"""


		# Now select the desired columns for the combined data
		#mycols_combo = ['bathy', 'blue_t_max', 'blue_t_min', 'blue_t_dmax', 'blue_t_dmin', 'green_t_max', 'green_t_min', 'green_t_dmax', 'green_t_dmin', 'red_t_max', 'red_t_min', 'red_t_dmax', 'red_t_dmin', 'nir_t_max', 'nir_t_min', 'nir_t_dmax', 'nir_t_dmin', 'swir_t_max', 'swir_t_min', 'swir_t_dmax', 'swir_t_dmin', 'bin_class']


		#mycols_surf = ['blue', 'green', 'bin_class']
		#mycols_bot = ['blue_bot', 'green_bot', 'bin_class']

		#model_combodata = combodata[mycols_combo]
		#model_data = model_data[mycols]


		# Get the range of vegetation bathys
		veg_data =  model_data[model_data['fraction_veg_cover'] > 0.]
		veg_bathy =  veg_data.loc[:,'bathy']
		lo_bathy = np.percentile(veg_bathy, 5)
		hi_bathy = np.percentile(veg_bathy, 95)


		# filter the data
		"""
		CC = np.zeros(len(model_data))
		pct_tolerance = 0.1
		CC[model_data['fraction_veg_cover'] > pct_tolerance] = -1
		CC[model_data['fraction_veg_cover'] >= 1 - pct_tolerance] = 1
		#combodata = combodata.assign(bin_class = CC)
		model_data = model_data.assign(bin_class = CC)
		# select the binary values
		#combodata = combodata[combodata['bin_class'] >= 0.]
		model_data = model_data[model_data['bin_class'] >= 0.]
		model_data = model_data.drop('bin_class', axis = 1)
		"""
		
		# filter by bathy
		model_data = model_data[model_data['bathy'] >= lo_bathy]
		model_data = model_data[model_data['bathy'] < hi_bathy]

		# Now select the desired columns (first instance selection)
		mycols_surf = ['fraction_veg_cover', 'depth', 'blue', 'green','red', 'nir', 'swir', 'density_class']
		mycols_bot = ['fraction_veg_cover', 'depth', 'blue_bot', 'green_bot','red_bot', 'nir_bot', 'swir_bot', 'density_class']

		bot_data = model_data[mycols_bot]
		surf_data = model_data[mycols_surf]

		# get rid of nodata and outliers
		for band in range(len(mycols_surf)-1):
			surf_data = surf_data[surf_data[mycols_surf[band]] >= 0]
		for band in range(len(mycols_bot)-1):
			bot_data = bot_data[bot_data[mycols_bot[band]] >= 0]
			if mycols_bot[band] not in ['fraction_veg_cover', 'depth']:
				bot_data = bot_data[bot_data[mycols_bot[band]] < 0.5]


		# make a figure
		for md in np.arange(2,20,2):
			fig=plt.figure(1, facecolor='White',figsize=[14, 10])
			ax1 =  plt.subplot2grid((2,1),(0,0),colspan=1, rowspan=1)
			ax2 =  plt.subplot2grid((2,1),(1,0),colspan=1, rowspan=1)

			# compare bottom and surface reflectance 
			L = [surf_data,bot_data]
			axes = [ax1,ax2]
			for l in range(len(L)):
				print(l)
				# Split-out validation dataset
				array = L[l].values
				# split the features and targets
				X = array[:,:-1]; y = array[:,-1]

				# normalise the feature values
				print (L[l])
				print (X[0])
				for k in range(len(X[0])):
					X[:,k] = X[:,k]/np.amax(X[:,k])

				# make your training and testing sets. You should do a KFolds here
				X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

				frac_train, frac_val = X_train[:,0], X_val[:,0]
				X_train, X_val = X_train[:,1:], X_val[:,1:]

				from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
				from sklearn.neighbors import KNeighborsRegressor

				Reg = RandomForestRegressor(max_depth = md)


				#Reg = AdaBoostRegressor() # this yields really interesting stuff, but is it useful ???
				#Reg = KNeighborsRegressor() 
				Reg.fit(X_train,Y_train)
				Y_pred = Reg.predict(X_val)
				# do a violin
				for k in range(5):
					idx = np.where(Y_val == k)
					fn.cool_violin_1D(k, Y_pred[idx], step = 0.05, axis = axes[l], colour = plt.cm.jet(k/4))
				#print(np.multiply(Y_val,frac_val))
				#print(np.multiply(Y_pred,frac_val))

				axes[l].scatter(Y_val, Y_pred, marker = '.', alpha = 0.5, facecolor = plt.cm.jet(Y_val/4), lw = 0.)
				#axes[l].scatter(np.multiply(Y_val,frac_val), np.multiply(Y_pred,frac_val), marker = '.', alpha = 0.5, facecolor = plt.cm.jet(Y_val/4), lw = 0)
				

				axes[l].plot([-1,5], [-1,5], 'r', lw = 2)

			plt.savefig('Figures/ML/RFR_'+str(md)+'png')
		quit()

		# Important: consider pixel interactions
		# Simulated labelled datasets! Deep Simulated Learning => look at skimage feature extraction
		# Generative Adversarial Networks: use a short set of labelled data to generate synthetic training imagery

		#https://towardsdatascience.com/handling-imbalanced-datasets-in-machine-learning-7a0e84220f28

		# 











		
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

		"""
		h = .02  # step size in the mesh

		names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
		         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
		         "Naive Bayes", "QDA"]
		# , "Gaussian Process"

		classifiers = [
		    KNeighborsClassifier(3),
		    SVC(kernel="linear", C=0.025),
		    SVC(gamma=2, C=1),
		    DecisionTreeClassifier(max_depth=5),
		    RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1),
		    MLPClassifier(alpha=1, max_iter=1000),
		    AdaBoostClassifier(),
		    GaussianNB(),
		    QuadraticDiscriminantAnalysis()]
		# GaussianProcessClassifier(1.0 * RBF(1.0)),

		figure = plt.figure(figsize=(27, 9))
		i = 1
		# iterate over datasets
		# for ds_cnt, ds in enumerate(datasets):
		# preprocess dataset, split into training and test part
		array = surf_data.values
		X = array[:,:-1]
		y = array[:,-1]

		X = StandardScaler().fit_transform(X)
		X_train, X_test, y_train, y_test = \
		    train_test_split(X, y, test_size=.4, random_state=42)

		x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
		y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
		                     np.arange(y_min, y_max, h))

		# just plot the dataset first
		cm = plt.cm.RdBu
		cm_bright = ListedColormap(['#FF0000', '#0000FF'])
		ax=plt.subplot2grid((1,len(classifiers)+1),(0,0),colspan=1, rowspan=1)
		#if ds_cnt == 0:
		ax.set_title("Input data")
		# Plot the training points
		ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
		           edgecolors='k', marker = '.')
		# Plot the testing points
		ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
		           edgecolors='k', marker = '.')
		ax.set_xlim(xx.min(), xx.max())
		ax.set_ylim(yy.min(), yy.max())
		ax.set_xticks(())
		ax.set_yticks(())
		i += 1

		print ('got here')
		
		j = 0
		# iterate over classifiers
		for name, clf in zip(names, classifiers):
			print (name)
			ax=plt.subplot2grid((1,len(classifiers)+1),(0,j+1),colspan=1, rowspan=1)
			
			try:
				clf.fit(X_train, y_train)
				print ('managed to fit')
				score = clf.score(X_test, y_test)

				# Plot the decision boundary. For that, we will assign a color to each
				# point in the mesh [x_min, x_max]x[y_min, y_max].
				if hasattr(clf, "decision_function"):
				    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
				else:
				    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

				print ('got there')
				# Put the result into a color plot
				Z = Z.reshape(xx.shape)
				print ('did the reshape')
				ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

				# Plot the training points
				ax.scatter(X_train[:, 0], X_train[:, 1], marker = '.', c=y_train, cmap=cm_bright,
				           edgecolors='k')
				# Plot the testing points
				ax.scatter(X_test[:, 0], X_test[:, 1], marker = '.', c=y_test, cmap=cm_bright,
				           edgecolors='k', alpha=0.6)

				ax.set_xlim(xx.min(), xx.max())
				ax.set_ylim(yy.min(), yy.max())
				ax.set_xticks(())
				ax.set_yticks(())
				#if ds_cnt == 0:
				ax.set_title(name)
				ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')
			except:
				print ('oops, something went wrong')
				continue
			j += 1
			#break

			
			plt.tight_layout()
			plt.savefig('Figures/ML/'+'Multimodel_'+str_date+'_bathyrange.png')
			"""
			#plt.show()
			#quit()






			# Try the logistic regression algo. Default RF for now.
			#log_reg = RandomForestClassifier(n_estimators = 100, max_depth = 100)
			#log_reg = BernoulliNB()

		from numpy import mean
		from numpy import std
		from sklearn.datasets import make_classification
		from sklearn.model_selection import cross_val_score
		from sklearn.model_selection import RepeatedStratifiedKFold
		from sklearn.linear_model import LogisticRegression
		from sklearn.neighbors import KNeighborsClassifier
		from sklearn.tree import DecisionTreeClassifier
		from sklearn.svm import SVC
		from sklearn.naive_bayes import GaussianNB
		from sklearn.ensemble import StackingClassifier
		from matplotlib import pyplot
		 
		# get a list of models to evaluate
		def get_models():
			models = dict()
			models['lr'] = LogisticRegression()
			models['knn'] = KNeighborsClassifier()
			#models['cart'] = DecisionTreeClassifier()
			models['svm'] = SVC()
			models['bayes'] = GaussianNB()
			return models
		 
		# evaluate a given model using cross-validation
		def evaluate_model(model, X, y):
			cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
			scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
			return scores

		# get a stacking ensemble of models
		def get_stacking():
			# define the base models
			level0 = list()
			level0.append(('lr', LogisticRegression()))
			level0.append(('knn', KNeighborsClassifier()))
			#level0.append(('cart', DecisionTreeClassifier()))
			level0.append(('svm', SVC()))
			level0.append(('bayes', GaussianNB()))
			# define meta learner model
			level1 = LogisticRegression()
			# define the stacking ensemble
			model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
			return model
		 
		# get a list of models to evaluate
		def get_models():
			models = dict()
			models['lr'] = LogisticRegression()
			models['knn'] = KNeighborsClassifier()
			#models['cart'] = DecisionTreeClassifier()
			models['svm'] = SVC()
			models['bayes'] = GaussianNB()
			models['stacking'] = get_stacking()
			return models
		 
		# evaluate a give model using cross-validation
		def evaluate_model(model, X, y):
			cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
			scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=cv, n_jobs=-1, error_score='raise')

			return scores
		# get the models to evaluate
		models = get_models()
		# evaluate the models and store results
		results, names = list(), list()
		for name, model in models.items():
			scores = evaluate_model(model, X, y)
			results.append(scores)
			names.append(name)
			print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
		# plot model performance for comparison
		pyplot.boxplot(results, labels=names, showmeans=True)
		pyplot.show()
		 


		quit()


		#from sklearn.ensemble import VotingClassifier
		#log_reg=VotingClassifier(estimators=[('Gaussian NB', GaussianNB()), ('Random Forest', RandomForestClassifier(n_estimators = 100, max_depth = 100))], 			                       voting='hard', weights=[2,1])
		log_reg.fit(X_train, Y_train)
		print ('managed to fit')

		# the results
		Y_pred = log_reg.predict(X_val)
		print ('managed to predict')



		A = Y_val + Y_pred # 2 is a true pos, 0 is a true neg
		B = Y_val - Y_pred # -1 is a False pos, 1 is a False neg

		TP = len(A[A == 2.])
		TN = len(A[A == 0.])
		FN = len(B[B>0])
		FP = len(B[B<0])

		Acc = (TP+TN) / len(A)
		Rpos = TP/(TP+FN)
		Rneg = TN/(TN+FP)

		#ax1.scatter(bot_data.loc[:,'blue_bot'], bot_data.loc[:,'green_bot'])
		
		if z == 0:
			noveg = model_data[model_data['fraction_veg_cover'] <= 0.0]
			ax1.hist(noveg.loc[:,'bathy'], bins = 200, facecolor = 'y', alpha = 0.5)

			fullveg = model_data[model_data['fraction_veg_cover'] >= 1]
			ax1.hist(fullveg.loc[:,'bathy'], bins = 200, facecolor = 'g', alpha = 0.5)

		w = bathyrange_up[z]-bathyrange_down[z] + (1-l)/4
		ax2.bar(bathyrange_down[z]+l/4, TP, w, lw = 0.8, facecolor = 'g', edgecolor = 'k', align = 'edge')
		ax2.bar(bathyrange_down[z]+l/4, FN, w, bottom=TP, lw = 0.3, facecolor = 'g', edgecolor = 'k', alpha  = 0.3, align = 'edge')
		ax2.bar(bathyrange_down[z]+l/4, TN, w, bottom=TP+FN, lw = 0.8, facecolor = 'y', edgecolor = 'k', align = 'edge')
		ax2.bar(bathyrange_down[z]+l/4, FP, w, bottom=TP+TN+FN, lw = 0.3, facecolor = 'y', edgecolor = 'k', alpha = 0.3, align = 'edge')

		# the text
		Text = ' A:'+str(round(Acc*100,1)) + '%; R+:' +str(round(Rpos*100,1)) + '%; R-:' +str(round(Rneg*100,1))+'%'

		ax2.annotate(Text, xycoords = 'data', xy = (bathyrange_down[z]+l/4+0.1, TP+FN), rotation = 90)	

		ax1.set_xlabel('bathymetry (m)')

		#plt.show()

		plt.savefig('Figures/ML/'+str_date+'_ensemble1_bathyrange.png')
		print()

		quit()


		


	print ()

	quit()


	# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py

	# https://scikit-learn.org/stable/glossary.html#term-random-state

	# https://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf

	# https://medium.com/turo-engineering/how-not-to-use-random-forest-265a19a68576

	# https://datascience.stackexchange.com/questions/8820/unbalanced-classes-how-to-minimize-false-negatives


	"""
	IDEAS TO MAKE THIS SHITE WORK!

	- compare and tune multiple models

	- fine-tune the sampling subset

	- 

	"""

























	# Spot Check Algorithms
	models = []
	models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))
	#models.append(('SVM', SVC(gamma='auto')))
	# evaluate each model in turn
	results = []
	names = []
	for name, model in models:
		kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
		cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
		print (cv_results)
		quit()
		results.append(cv_results)
		names.append(name)
		print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
	# Compare Algorithms
	plt.boxplot(results, labels=names)
	plt.title('Algorithm Comparison')
	plt.show()


	quit()
	#############################################################################
	# Plot evolution of signatures in time
	#############################################################################

	"""
	minvegrange = np.arange(0,1,0.1)
	for v in minvegrange:
		print ('plotting range:', v)
		minveg = v
		fig_width = 14; fig_height = 10
		fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])

		# the bands axes
		ax1 =  plt.subplot2grid((5,3),(0,0),colspan=1, rowspan=1)
		ax2 =  plt.subplot2grid((5,3),(1,0),colspan=1, rowspan=1)
		ax3 =  plt.subplot2grid((5,3),(2,0),colspan=1, rowspan=1)
		ax4 =  plt.subplot2grid((5,3),(3,0),colspan=1, rowspan=1)
		ax5 =  plt.subplot2grid((5,3),(4,0),colspan=1, rowspan=1)

		ax1n =  plt.subplot2grid((5,3),(0,1),colspan=1, rowspan=1)
		ax2n =  plt.subplot2grid((5,3),(1,1),colspan=1, rowspan=1)
		ax3n =  plt.subplot2grid((5,3),(2,1),colspan=1, rowspan=1)
		ax4n =  plt.subplot2grid((5,3),(3,1),colspan=1, rowspan=1)
		ax5n =  plt.subplot2grid((5,3),(4,1),colspan=1, rowspan=1)

		# the hydrodynamic axes
		axa =  plt.subplot2grid((5,3),(0,2),colspan=1, rowspan=1)
		axb =  plt.subplot2grid((5,3),(1,2),colspan=1, rowspan=1)
		axc =  plt.subplot2grid((5,3),(2,2),colspan=1, rowspan=1)
		axd =  plt.subplot2grid((5,3),(3,2),colspan=1, rowspan=1)
		axe =  plt.subplot2grid((5,3),(4,2),colspan=1, rowspan=1)

		colors = [plt.cm.terrain(200), plt.cm.Greens(150)]

		nvdata = data[data['vegetation'] == 'Nonveg']
		vdata = data[data['vegetation'] != 'Nonveg']
		datalist = [nvdata, vdata]

		for i in range(len(datalist)):
			print ('checking for this vegetation status')
			numax = [ax1,ax2,ax3,ax4,ax5]
			numnax = [ax1n,ax2n,ax3n,ax4n,ax5n]
			numband = ['blue','green','red','nir','swir']
			letax = [axa,axb,axc,axd,axe]
			letband = ['depth','waveheight','spm','fraction_veg_cover','density_class']

			for j in range(len(numax)):
				print ('plotting this axis', numband[j])
				dates = list(datalist[i]['survey'])
				uniquedates = [] 
				for k in dates: 
				    if k not in uniquedates: 
				        uniquedates.append(k) 

				for k in uniquedates:
					print ('plotting this date:', k)
					subset = datalist[i][datalist[i]['survey'] == k]
					if i != 0:
						subset = subset[subset['fraction_veg_cover']>=minveg]

					numy = np.asarray(subset[numband[j]])
					lety = np.asarray(subset[letband[j]])

					#clean the data
					numy = numy [numy < 1]
					numy = numy [numy > 0]
					#lety = lety [lety < 1]

					datime = (k-datetime(2012,1,1)).days

					if i == 0:
						fn.cool_violin_1D(datime, numy, 0.01, numnax[j], colour = colors[i])
					if i == 1:
						fn.cool_violin_1D(datime, numy, 0.01, numax[j], colour = colors[i])

					if not lety.all() == 0.:
						fn.cool_violin_1D(datime, lety, 0.05, letax[j], colour = 'k')

				numax[j].set_ylim(0,0.2)
				numnax[j].set_ylim(0,0.2)
				numax[j].set_ylabel(numband[j])
				letax[j].set_ylabel(letband[j])

				numax[j].grid(True)
				numnax[j].grid(True)
				letax[j].grid(True)

				if j < len(numax)-1:
					numax[j].set_xticklabels('')
					numnax[j].set_xticklabels('')
					letax[j].set_xticklabels('')
				else:
					numax[j].set_xlabel('days since jan 1st')
					numnax[j].set_xlabel('days since jan 1st')
					letax[j].set_xlabel('days since jan 1st')

		plt.savefig('Figures/'+'spectralevolution_sup'+str(int(minveg*100))+'pctveg.png')

	quit()
	"""


	#############################################################################
	# Plot spectral response at a given time
	#############################################################################
	"""
	dates = list(data['survey'])
	uniquedates = [] 
	for d in dates: 
	    if d not in uniquedates: 
	        uniquedates.append(d) 


	for d in range(len(uniquedates)):
		print('plotting date:', uniquedates[d])

		subset = data[data['survey'] == uniquedates[d]]

		fig_width = 14; fig_height = 10
		fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])

		# the axes
		ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

		# the wavelengths, etc
		satnumber = '7'
		bands_dict, wavelength_dict, aw_dict = fn.satnumber_dicts (satnumber)

		# the bands we are looking at
		numband = ['blue','green','red','nir','swir1']
		numband_pd = ['blue','green','red','nir','swir']
		wl = []
		for i in range(len(numband)):
			wl.append(wavelength_dict[numband[i]])

		# the range of vegetation cover
		coverange = [-0.1]+list(np.arange(0,1.1,0.1))

		for c in range(len(coverange)-1):
			subsub = subset[subset['fraction_veg_cover'] > coverange[c]]
			finalsub = subsub[subsub['fraction_veg_cover'] <= coverange[c+1]]

			bands_subset = np.asarray(finalsub[numband_pd])

			subset_avg = np.mean(bands_subset, axis = 0)
			subset_std = np.std(bands_subset, axis = 0)

			ax1.plot(wl, subset_avg, 'o-', c = plt.cm.summer_r(coverange[c]), lw = 1.5)
			#ax1.fill_between(wl, subset_avg-0.01*subset_std, subset_avg+0.01*subset_std, facecolor = plt.cm.summer_r(coverange[c]), lw = 0, alpha = 0.5)

			ax1.plot(wl, subset_avg, 'o-', c = plt.cm.summer_r(coverange[c]), lw = 1.5)

			str_date = uniquedates[d].strftime("%Y%m%d")

			ax1.set_ylim(0,0.5)

		plt.savefig('Figures/'+'spectral_response_'+ str_date +'.png')
	"""
		

	#############################################################################
	# Plot spectral response at a given time for given hydrodynamic conditions
	#############################################################################
	"""
	dates = list(data['survey'])
	uniquedates = [] 
	for d in dates: 
	    if d not in uniquedates: 
	        uniquedates.append(d) 


	for d in range(len(uniquedates)):
		print('plotting date:', uniquedates[d])

		# the date
		subset = data[data['survey'] == uniquedates[d]]

		# start a figure for that date
		fig_width = 14; fig_height = 10
		fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])

		# the bands we are looking at
		numband_pd = ['blue','green','red','nir','swir']

		for b in range(len(numband_pd)):

			ax0 = plt.subplot2grid((5,3),(b,0),colspan=1, rowspan=1)
			ax1 = plt.subplot2grid((5,3),(b,1),colspan=1, rowspan=1)
			ax2 = plt.subplot2grid((5,3),(b,2),colspan=1, rowspan=1)

			bandset = subset[ [numband_pd[b]] +['fraction_veg_cover', 'depth', 'waveheight', 'spm']]	
			# the range of vegetation cover
			coverange = [-0.01, 0.]
			coverange = [-0.01, 0., 0.5]
			coverange = [-0.01, 0., 0.5, 0.9]
			coverange = [-0.01, 0., 0.5, 0.9, 1]
			coverange = [-0.01, 0., 0.5, 0.9, 1]

			for c in range(len(coverange)-1):
				subsub = bandset[bandset['fraction_veg_cover'] > coverange[c]]
				finalsub = subsub[subsub['fraction_veg_cover'] <= coverange[c+1]]

				color_value = (coverange[c] + coverange[c+1]) / 2

				ax0.scatter(finalsub['depth'], finalsub[numband_pd[b]], s = 8, marker = 'o', alpha = 0.5, facecolor = plt.cm.summer_r(color_value), lw = 0)
				ax1.scatter(finalsub['waveheight'], finalsub[numband_pd[b]], s = 8, marker = 'o', alpha = 0.5, facecolor = plt.cm.summer_r(color_value), lw = 0)
				ax2.scatter(finalsub['spm'], finalsub[numband_pd[b]], s = 8, marker = 'o', alpha = 0.5, facecolor = plt.cm.summer_r(color_value), lw = 0)


			str_date = uniquedates[d].strftime("%Y%m%d")

			ax0.set_ylim(0,0.25); ax0.set_xlim (-0,0.6); ax0.set_ylabel('Rbot '+numband_pd[b]); ax0.grid(True)
			ax1.set_ylim(0,0.25); ax1.grid(True)
			ax2.set_ylim(0,0.25); ax2.grid(True)

			if b == len(numband_pd)-1:
				 ax0.set_xlabel('Depth(m)'); ax1.set_xlabel('Wave Height(m)'); ax2.set_xlabel('SPM(mg/L)')

		plt.savefig('Figures/'+'Hydro_conditions_'+ str_date +'_totalveg.png')
	"""


	#############################################################################
	# Plot spectral response extremes to visualise separation
	#############################################################################
	"""
	dates = list(data['survey'])
	uniquedates = [] 
	for d in dates: 
	    if d not in uniquedates: 
	        uniquedates.append(d) 

	for d in range(len(uniquedates)):
		print('plotting date:', uniquedates[d])
		str_date = uniquedates[d].strftime("%Y%m%d")
		print (str_date)

		# the date
		subset = data[data['survey'] == uniquedates[d]]

		# Option: select shallow areas where the seagrasses are
		subset = subset[subset['depth'] < 0.7]

		# start a figure for that date
		fig_width = 12; fig_height = 14
		fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])

		# the bands we are looking at
		numband_pd = ['blue','green','red','nir','swir']
		combinations = [[p,q] for p,q in product(numband_pd, numband_pd) if p[0] < q[0]]

		for c in range(len(combinations)):
			x = combinations[c][0]; y = combinations[c][1]
			noveg = subset[subset['fraction_veg_cover'] < 0.1]
			fullveg = subset[subset['fraction_veg_cover'] > 0.9]

			X = np.asarray(noveg[x]); Y = np.asarray(noveg[y])
			Xv = np.asarray(fullveg[x]); Yv = np.asarray(fullveg[y])


			if c < len(combinations)/2:
				axis = plt.subplot2grid((5,2),(c,0),colspan=1, rowspan=1)
			else:
				axis = plt.subplot2grid((5,2),(int(c-len(combinations)/2),1),colspan=1, rowspan=1)
				
			axis.set_xlabel('Rbot '+x); axis.set_ylabel('Rbot '+y)
			axis.set_xlim(0,0.25); axis.set_ylim(0,0.25)
			axis.grid(True)


			axis.scatter(X, Y, marker = 'o', s = 8, facecolor = plt.cm.summer_r(0), alpha = 0.5, lw = 0)
			#axis.scatter(Xv, Yv, marker = 'o', s = 8, facecolor = plt.cm.summer_r(255), alpha = 0.5, lw = 0)
		plt.tight_layout()
		plt.savefig('Figures/'+'Cloud_sep_'+str_date+'_shallow_noveg.png')
	quit()
	"""


	#############################################################################
	# Try a PCA
	#############################################################################
	"""
	dates = list(data['survey'])
	uniquedates = [] 
	for d in dates: 
	    if d not in uniquedates: 
	        uniquedates.append(d) 

	for d in range(len(uniquedates)):
		print('plotting date:', uniquedates[d])
		D = uniquedates[d]
		str_date = D.strftime("%Y%m%d")
		subset = data.loc[data['survey'] == D]

		# Make a classification column. Try binary first
		CC = np.zeros(len(subset))
		CC[subset['fraction_veg_cover']>0.1] = 0.5
		CC[subset['fraction_veg_cover']>0.9] = 1
		subset = subset.assign(bin_class = CC)
		subset = subset[subset['bin_class'] != 0.5]

		# option: work with shallow areas. This will need some calibration: what is deep?
		subset = subset.loc[subset['depth'] < 1]

		# Choose your spectral bands 
		bands = ['blue','green','red','nir','swir']

		# Normalise the values and get rid of outliers bigger than 1
		for b in bands:
			subset = subset.loc[subset[b] < 0.6] # get rid of outliers
			subset.loc[:,b] = (subset.loc[:,b]-min(subset.loc[:,b])) / (max(subset.loc[:,b])-min(subset.loc[:,b])) # normalise


		# Separating out the features (x) from the target (y)
		x = subset.loc[:, bands].values
		y = subset.loc[:,'bin_class'].values

		#  Select the number of components you want
		n = 3
		cols = list(np.arange(1,n+1))
		cols_hdr = ['PC '+str(i) for i in cols]

		# start the PCA
		from sklearn.decomposition import PCA
		
		pca = PCA(n_components = n)
		principalComponents = pca.fit_transform(x)
		principalDf = pd.DataFrame(data = principalComponents, columns = cols_hdr)
		Proj = principalDf.assign(bin_class = y) 

		# plot the results
		fig = plt.figure(1, facecolor='White',figsize=[20, 10])
		ax = plt.subplot2grid((1,2),(0,0),colspan=1, rowspan=1, projection='3d')
		ax_pca = plt.subplot2grid((1,2),(0,1),colspan=1, rowspan=1, projection='3d')
		
		ax.set_xlabel('Blue', fontsize = 12)
		ax.set_ylabel('Green', fontsize = 12)
		ax.set_zlabel('Red', fontsize = 12)
		#ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_zlim(0,1);
		#ax.set_title('Visible spectrum response for date ' + str_date, fontsize = 16)

		ax_pca.set_xlabel('PC1', fontsize = 12)
		ax_pca.set_ylabel('PC2', fontsize = 12)
		ax_pca.set_zlabel('PC3', fontsize = 12)
		#ax_pca.set_title('PC response projection over 5 bands for date ' + str_date, fontsize = 16)

		X = x[y == 0.]
		ax.scatter(X[:,0], X[:,1], X[:,2], c = 'y', s = 6, marker = 'D', alpha = 0.05, lw = 0)

		X = x[y == 1.]
		ax.scatter(X[:,0], X[:,1], X[:,2], c = 'g', s = 6, marker = 'o', alpha = 0.4, lw = 0)


		ID = Proj.loc[:,'bin_class'] == 0
		ax_pca.scatter(Proj.loc[ID, 'PC 1'], Proj.loc[ID, 'PC 2'], Proj.loc[ID, 'PC 3'], c = 'y', s = 8, marker = '.', alpha = 0.05)
		
		ID = Proj.loc[:,'bin_class'] == 1
		ax_pca.scatter(Proj.loc[ID, 'PC 1'], Proj.loc[ID, 'PC 2'], Proj.loc[ID, 'PC 3'], c = 'g', s = 8, marker = 'o', alpha = 0.4)
		
		#ax.legend(targets, title = 'Vegetation presence')
		ax.grid()
		ax_pca.grid()

		for i in range(0,360,10):
			ax.view_init(elev=15., azim=i)
			ax_pca.view_init(elev=15., azim=i)

			plt.tight_layout(True)
			#plt.show()
			if not os.path.isdir('Figures/'+str_date+'/'):
				os.system('mkdir ' + 'Figures/'+str_date+'/')
			plt.savefig('Figures/'+str_date+'/'+'BinaryPCA_'+str_date+'_'+str(n)+'axes_azimuth'+str(i)+'_shallow.png')
		#quit()
	"""



	#############################################################################
	# Incorporate
	#############################################################################



	quit()

	quit()



	"""
	More ideas! 
	- measure the change in reflectance over time and characterise by trajectory
	- Separate the nonveg in deep and shallow waters. This sort of works but eliminates a lot of our options

	"""










