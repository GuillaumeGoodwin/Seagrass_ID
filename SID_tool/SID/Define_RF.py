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
   
   # Format the data 
    print ('formatting features and target for Random Forest')
    Xtrain, Ytrain, Xtest, Ytest, Xthresh, Ythresh, W, Train_idx, Test_idx, Thresh_idx = format_trainthreshtest(mask, features, target)

    # Indices to keep
    idx0 = W.copy(); idx1 = Train_idx.copy(); idx2 = Test_idx.copy(); idx3 = Thresh_idx.copy()

    # Construct a model
    minzone = 1; maxzone = int(np.amax(zones))
    Zone_models = []; Zone_sepvals = []; exist_zonemodel = []
    skipped = 0

    print ('building RF model and thresholds for each zone')
    for z in range(minzone,maxzone+1):
        print ('Zone', z)

        trainzones = Xtrain[:,-1] #; print (trainzones)
        indazone = np.where(trainzones == z) #; print (indazone)

        xtrain = Xtrain[indazone] #; print (xtrain) # subsample x
        xtrain = xtrain[:,:-1] #; print (xtrain) # get rid of zone feature
        print (xtrain.shape)

        ytrain = Ytrain[indazone] #;print (ytrain) # subsample y

        # Construct the model.
        # But only if there are pixels to construct it on ...
        if xtrain.shape[0] <= 10:
            print ('WE HAVE NO PIXELS')
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
            ysep_avg = 0.5 # dimensions should be irrelevant here
            Zone_sepvals.append(ysep_avg)


            # Test on the remaining unused pixels
            testzones = Xtest[:,-1]#; print (testzones)
            indazone = np.where(testzones == z)#; print (indazone)
            xtest = Xtest[indazone]
            xtest = xtest[:,:-1]#; print (xtest)

            ytest = Ytest[indazone]
            test_idx = Test_idx[indazone]#; print (test_idx)

            # Define indices for this scene
            x = xtest; y = ytest; idx = test_idx

            y2D = -9999*np.ones(mask.shape)
            y2D = y2D.ravel()
            y2D[idx] = y
            y2D = y2D.reshape(mask.shape)

            print ('Self-testing the model')
            # Calculate proba positive in the threshold section
            proba, Ypp = RF_2Dproba_binary (RF_model, x, idx, mask.shape)
            pred = classify_binary (proba,ysep_avg)

            diffmap, S, P, A = make_zonewise_perf(pred, proba, target, zones)



    return Zone_models, Zone_sepvals, exist_zonemodel, S, P, A


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

        print (sep, evaluate_sepval(sep,Ypp,Ysep,ytarget,w))


    xmax = np.amax(X)
    idmax = np.where(X == xmax)
    sepvalbase = myrange[idmax][0]

    Ysep[w] = sepvalbase

    print ('initial run: obj = ', xmax, 'sepval = ', sepvalbase)
    print()
    

    return Ysep, xmax



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
    	#This is F1
        Objective = 2*(Sen*Pre/(Sen+Pre)) 
    else:
        Objective = 0

    return Objective



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
	rm = mask.ravel()
	rf1 = features[0].ravel()
	rt = target.ravel()

	print (mask.shape, features[0].shape, target.shape)
	print (len(rm), len(rf1), len(rt))


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


	#fig = plt.figure(0, facecolor='White',figsize=[18, 12])
	#axa = plt.subplot2grid((1,1), (0,0),colspan=1, rowspan=1)
	#axa.imshow(y2Dtrain)
	#axa.scatter(Xtrain[:,0], Ytrain)


	# Normalise by zone !!!!!!!
	Xtrain = normalise (Xtrain, method = 'quantile', quantile = 5)
	Xthresh = normalise (Xthresh, method = 'quantile', quantile = 5)
	Xtest = normalise (Xtest, method = 'quantile', quantile = 5)

	#axa.scatter(Xtrain[:,0], Ytrain, marker = 'D')

	#plt.tight_layout()
	#plt.show()
	#quit()



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
    for z in range(1,int(np.amax(Z))+1):
        # zone indices
        zidx = np.where(Z == z)[0]

        if len(zidx)>0:

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
    print (X)
    print (X.shape)

    X[np.isnan(X)] = 0

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

    lendiff = len(survey) - len(proba)
    lenodiff = len(survey[0]) - len(proba[0])

    if lendiff >= 0 and lenodiff >= 0:
    	survey = survey[:len(proba),:len(proba[0])]
    else:
    	print ('feature was badly rasterised')
    	quit()


    mask = np.ma.getmask(proba)
    survey = np.ma.masked_where(mask == True, survey)
    survey = np.ma.masked_where(proba > 100, survey)

    mysum = survey + pred
    mydif = survey - pred

    diffmap = np.zeros(pred.shape, dtype = np.float)
    diffmap [mysum == 0] = 1. # TN
    diffmap [mysum == 2] = 2. # TP
    diffmap [mydif == 1] = -1. # FN
    diffmap [mydif == -1] = -2. # FP

    diffmap[survey < 0] = 0
    diffmap[survey > 1] = 0
    diffmap[pred < 0] = 0
    diffmap[pred > 1] = 0


    S,P,A = [],[],[]

    for z in range(1,int(np.amax(zones)+1)):

    	print ('zone', z)

    	wz = np.where(zones == z)
    	if len(wz[0]) > 0.1 * zones.size:

	        print (' evaluating this zone')

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

    S = np.asarray(S)
    P = np.asarray(P)
    A = np.asarray(A)


    return diffmap, S, P, A






###################################################################################
###################################################################################
def zonal_normalise (X, method = 'minmax', quantile = 5):
    """
    normalises values of X in each zone
    options for the method are 'minmax' for actual maxima and 'quantile' for percentiles
    quantile value gives the difference from the extreme value
    """

    for z in range(len(X)):
    	if len(X[z]) > 0:
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

        print (Sepvals)

        pred_z = np.zeros(mask.size)
        proba_z = np.zeros(mask.size)

        if  exist_model[counter] == 0 or model[counter] == 0:
            pred_t [Zonal_idx[counter]] = 0
            proba_t [Zonal_idx[counter]] = 0
            
        elif  exist_model[counter] == 1:
            chosen_model = model[counter]
            chosen_sepval = Sepvals[counter]
            #print (X_zonal[counter][:,:-1].shape)

            Xapply = X_zonal[counter][:,:-1]

            # Include Kmeans as a feature to improve?
            if K_means == True:
                xKmeans = KMeans(n_clusters = 2); xKmeans.fit(Xapply)
                xKmeans_labelled = np.asarray([xKmeans.labels_])
                #print (xKmeans_labelled.shape)

                Xapply = np.concatenate((Xapply, xKmeans_labelled.T), axis = 1)
                #print (Xapply.shape)

            print(chosen_model)

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
def make_depth(ooi, tidelist, zones, time):

	bathy = rasterio.open(ooi).read(1)

	dt  = datetime.strptime(time, '%Y%m%d%H%M')

	for z in range(1,max(np.unique(zones))):
		w = np.where (zones == z)

		#print (z, len(w[0]) > 0.01*zones.size)

		if len(w[0]) > 0.01*zones.size:

			tide = tidelist[z-1]

			col = [f for f in tide.columns.values if f.startswith('Livello')][0]

			idx = tide.index[np.logical_and(tide.index >  dt - timedelta(0,1800), tide.index <  dt + timedelta(0,1800))]

			try:
				E = np.float(tide[col].loc[idx])

			except:
				E = 0

			bathy[w] = E - bathy[w]


	return bathy



###################################################################################
###################################################################################
def train_model(imgdir, maskraster, zoneraster, vegdict, k, ooi_rasters, tidelist, SAC = False, md = 30, nt = 100, mf = 3):

	# load the masking rasters
	mask = rasterio.open(maskraster).read(1)
	zones = rasterio.open(zoneraster).read(1)

	# load the vegetation
	mykeys = list(vegdict.keys())
	trainveg = rasterio.open(vegdict[mykeys[k]]).read(1)

	# load the ooi rasters
	ooi = []
	for i in range(len(ooi_rasters)):
		if i == 0:
			O = make_depth(ooi_rasters[i], tidelist, zones, mykeys[k])
		else:
			O = rasterio.open(ooi_rasters[i]).read(1)
		ooi.append( rasterio.open(ooi_rasters[i]).read(1) )


	# load the associated image
	trainraster = imgdir + mykeys[k] + '_rgbn_clip.bil'
	trainblu = rasterio.open(trainraster).read(1).astype(float)
	traingrn = rasterio.open(trainraster).read(2).astype(float)
	trainred = rasterio.open(trainraster).read(3).astype(float)
	trainnir = rasterio.open(trainraster).read(4).astype(float)

	print ('Training model on ', vegdict[mykeys[k]], ' with image ', trainraster)

	# group the rasters into features
	trainfeatures = [trainblu, traingrn, trainred, trainnir]
	for o in ooi:
		trainfeatures.append(o)
	if SAC == True:
		loc_r, loc_c = makerowcol(trainblu)
		trainfeatures.append(loc_r)
		trainfeatures.append(loc_c)
	trainfeatures.append(zones)


	#Make sure everyone has the correct shape
	#print (mask.shape)
	#print (zones.shape)
	#print (trainblu.shape)
	#print (ooi[0].shape)
	#print (ooi[1].shape)
	#print (trainveg.shape)
	#print ()

	for f in range(1,len(trainfeatures)):
		lendiff = len(trainfeatures[f]) - len(trainblu)
		lenodiff = len(trainfeatures[f][0]) - len(trainblu[0])

		if lendiff >= 0 and lenodiff >= 0:
			trainfeatures[f] = trainfeatures[f][:len(trainblu),:len(trainblu[0])]
		else:
			print ('feature was badly rasterised')
			quit()
		#print (trainfeatures[f].shape)


	lendiff = len(trainveg) - len(trainblu)
	lenodiff = len(trainveg[0]) - len(trainblu[0])

	if lendiff >= 0 and lenodiff >= 0:
		trainveg = trainveg[:len(trainblu),:len(trainblu[0])]
	else:
		print ('feature was badly rasterised')
		quit()


	# Make a RF model and its thresholds
	RF_model, sepvals, exist_model, S_self, P_self, A_self = Train_Random_Forest_full (trainfeatures, trainveg, mask, zones, max_depth=md, n_trees=nt, max_features=mf)

	return RF_model, sepvals, exist_model, S_self, P_self, A_self




##################################################################
##################################################################
def choose_optimal_model():
	print ('boo')


##################################################################
##################################################################
def rate_model(vegdict, k, pred, proba, zones):

	zonearr = rasterio.open(zones).read(1)
	testveg = rasterio.open(vegdict[list(vegdict.keys())[k]]).read(1)

	testveg[testveg>=0.5] = 1
	testveg[testveg<0.5] = 0

	diffmap, s,p,a = make_zonewise_perf(pred, proba, testveg, zonearr)

	F1 = 2* (s*p)/(s+p+0.0000000000001)

	return F1





##################################################################
##################################################################
def apply_model(imgdir, RF_model, sepvals, exist_model, vegdict, ooi_rasters, tidelist, maskraster, zoneraster, SAC = False, times = None):

	timeseries = []; timeprobas = []; timestamps = []

	# load the masking rasters
	mask = rasterio.open(maskraster).read(1)
	zones = rasterio.open(zoneraster).read(1)


	# make a list of rasters from the images directory
	rasterlist = [f for f in os.listdir(imgdir) if f.endswith('_clip.bil')]

	# Select a single file to predict
	if times != None:
		rasterlist = [f for f in rasterlist if times in f]
		#print (rasterlist)


	for testraster in rasterlist:
		time = testraster.split('_')[0]

		# get the time
		timetxt = testraster

		# load the ooi rasters
		ooi = []
		for i in range(len(ooi_rasters)):
			if i == 0:
				O = make_depth(ooi_rasters[i], tidelist, zones, time)
			else:
				O = rasterio.open(ooi_rasters[i]).read(1)
			ooi.append( rasterio.open(ooi_rasters[i]).read(1) )

		# load the associated image
		img = rasterio.open(imgdir+testraster)

		testblu = img.read(1).astype(float)
		testgrn = img.read(2).astype(float)
		testred = img.read(3).astype(float)
		testnir = img.read(4).astype(float)

		img.close()


		# group the rasters into features
		testfeatures = [testblu, testgrn, testred, testnir]
		for o in ooi:
			testfeatures.append(o)
		if SAC == True:
			loc_r, loc_c = makerowcol(testblu)
			testfeatures.append(loc_r)
			testfeatures.append(loc_c)
		testfeatures.append(zones)

		# Make sure to clip the edges of the arrays so that everyone has the same shape.
		# This might be awkward if you have more than one or two pixels difference

		"""
		testfeatures = np.asarray(testfeatures)

		for o in testfeatures:
			print (type(o), o.shape)
		print (mask.shape)
		print (zones.shape)

		X = []
		for f in testfeatures:
			X.append(f.ravel())
		X = np.asarray(X).T

		# NOrmalise
		print (X)
		print (X.shape)


		features = normalise(X)
		"""


		#Make sure everyone has the correct shape
		print (mask.shape)
		print (zones.shape)
		print (testblu.shape)
		print (ooi[0].shape)
		print (ooi[1].shape)
		#print (trainveg.shape)
		#print ()


		print ('Applying model to datetime:', testraster.split('_')[0])
		proba, pred = Apply_Random_Forest_full (testfeatures, RF_model, sepvals, exist_model, mask, zones)

		timeseries.append(pred); timeprobas.append(proba); timestamps.append(testraster.split('_')[0])

		print (np.count_nonzero(pred), np.sum(pred))

		"""
		fig = plt.figure(1, facecolor='White',figsize=[18, 12])
		axa = plt.subplot2grid((1,1), (0,0),colspan=1, rowspan=1)

		testgrn[testgrn > 500] = 500
		axa.imshow(testgrn, cmap = plt.cm.Greys, alpha = 1)
		predm = np.ma.masked_where(pred == 0, pred)
		
		axa.imshow(np.ma.masked_where(mask == 0, mask), cmap = plt.cm.Greys, alpha = 1)

		axa.imshow(predm, alpha = 0.5)
		plt.tight_layout()
		plt.show()

		print (timeseries)
		#print (timestamps)
		#quit()
		"""
		

	timeseries = np.asarray(timeseries, dtype = np.uint8)
	timeprobas = np.asarray(timeprobas, dtype = np.uint8)


	return timeseries, timeprobas, timestamps






