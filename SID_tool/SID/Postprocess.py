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
import rasterio.plot
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


def correct_sg(imgdir, sg_timeseries, timestamps, maskraster):


	# open the mask raster to save time
	masker = rasterio.open(maskraster)
	mask = masker.read(1)
	masker.close()



	#SG = np.copy(sg_timeseries)
	
	
	SG = []
	for i in range(len(timestamps)):
		print (timestamps[i])

		# Load the bands and seagrass
		imgraster = imgdir + timestamps[i] + '_rgbn_clip.bil'
		blu = rasterio.open(imgraster).read(1).astype(float)

		sg = sg_timeseries[i]
		sg[blu <= 0] = 255


		sg_smooth = smooth_gaps(sg)

		SG.append(sg_smooth)


	SG = np.asarray(SG, dtype = np.uint8)

	print (sg_timeseries.shape)
	print (SG.shape)


	# Correct the absence of data
	for i in range(len(SG[0])):
		for j in range(len(SG[0,0])):
			print (i,'/', len(SG[0]) , j,'/', len(SG[0,0]))
			

			SG[:,i,j] = correct_nodata(SG[:,i,j])

			SG[:,i,j] = correct_shortswitch(SG[:,i,j])

	print (SG)
	print (SG.shape)

	

	return SG




#############################################################
#############################################################
def smooth_gaps(C):


	conti_ID = np.zeros(C.shape, dtype = np.uint8)
					
	contiguous_area, veg_id = getContiguousAreas(C, 1, min_pixels = 4)
	veg_id[veg_id != 0] = 1

	location_id = np.copy(veg_id)

	contiguous_area, bare_id = getContiguousAreas(C, 0, min_pixels = 4)
	location_id[np.logical_and(bare_id == 0,C==0)] = 1
	location_id[C == 255] = 255			

	C_conti = location_id

	return C_conti




###########################
def correct_nodata (C):

	print ('Filling nodata')

	Ccor = np.copy(C)

	#print (C.shape)
	#print (C)


	# do the correction for nodata
	wveg = np.where(C == 1)[0]
	wbar = np.where(C == 0)[0]
	wnul = np.where(C == 255)[0]

	#print ('starting')

	print (wnul)
	#print (Ccor)
	skipped = 0

	while len(wnul) > 0:
		#print (len(wnul))
		idx = wnul[0]

		if idx > 0 and idx < len(C)-2:
			prv = idx-1
			cp = C[prv]
			nxt = idx+1
			cn = C[nxt]

			while cn == 255 and nxt < len(C)-1:
				nxt+=1
				#print (nxt)
				cn = C[nxt]
		
			if cn == cp: # same classif on before and after nodata
				Ccor[idx:nxt] = cp # this is the temp classif that will be used to do the longevity again. 
			wnul = np.delete(wnul, np.arange(0,nxt-idx))
		elif idx >= len(C)-2 or idx == 0:
			skipped = 1
			break

		#print (wnul)
		#print (Ccor)



	#if skipped == 0:
	#	quit()

	return Ccor
			
#######################################################
def correct_shortswitch (C):

	print ('correcting short switches')

	Ccor = np.copy(C)
	Eph = 255*np.ones(C.shape, dtype = np.uint8)

	wveg = np.where(C == 1)[0]
	wbar = np.where(C == 0)[0]
	wnul = np.where(C == 255)[0]

	print (wveg)
	print (wbar)



	# Ephemeral positives
	for n in range(len(wveg)):
		#print ('Algae? ', n, '/', len(wveg))
		idx = wveg[n]

		if idx > 1 and idx < np.amax(wveg)-2:
			#print (Ccor[idx-2:idx+3], np.sum(Ccor[idx-2:idx+3]))

			if np.sum(Ccor[idx-2:idx+3]) <=1.01:
				Ccor[idx] = 0 # 255 if declassified, 0 if switched
				Eph[idx] = 1
	
	# Ephemeral negatives
	for n in range(len(wbar)):
		#print ('Sediment plume', n, '/', len(wbar))
		idx = wbar[n]

		if idx > 1 and idx < np.amax(wbar)-2:

			if np.sum(Ccor[idx-2:idx+3]) >=3.99 and np.amax(Ccor[idx-2:idx+3]) < 250:
				Ccor[idx] = 1 # 255 if declassified, 1  if switched
				Eph[idx] = 0


	return Ccor




#######################################################
#######################################################
def make_longevity (T,C):



	fig = plt.figure(0, facecolor='White',figsize=[13, 20])
	ax = plt.subplot2grid((1,2), (0,0), rowspan=1, colspan=1)
	ax2 = plt.subplot2grid((1,2), (0,1), rowspan=1, colspan=1)
	ax.imshow(C[130], cmap = plt.cm.viridis, vmin = 0, vmax = 2)
	plt.show()
	quit()


	Longevity = 255*np.ones(T.shape, dtype = np.uint8)
	Longevity_wks = 10000*np.ones(T.shape, dtype = np.uint16)

	wveg = np.where(C == 1)[0]
	wbar = np.where(C == 0)[0]
	wnul = np.where(C == 255)[0]

	if len(wveg) == 0 or len(wbar) == 0:
		Longevity[:] = len(T)
		Longevity_wks[:] = int((T[-1] - T[0]).days / 7)

		Longevity[C == 255] = 255
		Longevity_wks[C == 255] = 10000

	else:
		while len(wveg) > 0 and len(wbar) > 0:

			if len(wveg) > 0: mv = min(wveg)
			else: mv = mb

			if len(wbar) > 0: mb = min(wbar)
			else: mb = mv

			if len(wnul) > 0: mn = min(wnul)
			else: mn = max(mv,mb)

			if mv > mn and mb > mn :
				wnul = np.delete(wnul, [0])
				Longevity[mn] = 255
				Longevity_wks[mn] = 10000

			elif mb > mn and mv > mn:
				wnul = np.delete(wnul, [0])
				Longevity[mn] = 255
				Longevity_wks[mn] = 10000
			
			else:
				if mv < mb :
					Longevity[mv:min(mb,mn)] = min(mb,mn)-mv
					wveg = np.delete(wveg, np.arange(min(mb,mn)-mv))

					Longevity_wks[mv:min(mb,mn)] = int((T[min(mb,mn)] - T[mv]).days / 7)

				else:
					Longevity[mb:min(mv,mn)] = min(mv,mn)-mb
					wbar = np.delete(wbar, np.arange(min(mv,mn)-mb))

					Longevity_wks[mb:min(mv,mn)] = int((T[min(mv,mn)] - T[mb]).days / 7)


		if len(wveg) > 0:
			Longevity[wveg] = len(wveg)
			Longevity_wks[wveg] = int((T[wveg[-1]] - T[wveg[0]]).days / 7)
		if len(wbar) > 0:
			Longevity[wbar] = len(wbar)
			Longevity_wks[wbar] = int((T[wbar[-1]] - T[wbar[0]]).days / 7)


	return Longevity, Longevity_wks






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


def seasonal_frequency (Ccor, seasons_4, seasons_2):

	SF4 = 255*np.ones(Ccor.shape, dtype = np.uint8)
	SF2 = 255*np.ones(Ccor.shape, dtype = np.uint8)

	U_4 = np.unique(seasons_4)
	U_2 = np.unique(seasons_2)

	for i in range(len(U_4)):
		U = U_4[i]
		w = np.where(seasons_4 == U)[0]

		C = Ccor[w]
		SF = int(100*len(C[C == 1]) / len(C))
		SF4[w] = SF



	for i in range(len(U_2)):
		U = U_2[i]
		w = np.where(seasons_2 == U)[0]

		C = Ccor[w]
		SF = int(100*len(C[C == 1]) / len(C))
		SF2[w] = SF


	return SF4, SF2


##################################################
##################################################
def biannual_stats (proba, pred,pred_times, s):

	proba = np.asarray(proba, dtype = np.float)

	S = 100*s

	proba_avg = np.zeros(len(proba), dtype = np.uint8)
	proba_std = np.zeros(len(proba), dtype = np.uint8)
	pos_freq = np.zeros(len(proba), dtype = np.uint8)

	pred_times_ord = np.asarray([i.toordinal() for i in pred_times])

	start_time = pred_times[0]; start_ordinal = start_time.toordinal()
	end_time = pred_times[-1]; end_ordinal = end_time.toordinal()

	# loop across the times
	for i in range(len(proba)):
		lim_pre = max(start_ordinal,pred_times[i].toordinal()-90) # 3 months before
		lim_post = min(end_ordinal,pred_times[i].toordinal()+90) # 3 months after

		start_idx = np.where(pred_times_ord >= lim_pre)[0][0]
		end_idx = np.where(pred_times_ord <= lim_post)[0][-1]

		#	print (np.where(pred_times_ord >= lim_pre), start_idx)
		#	print (np.where(pred_times_ord <= lim_post), end_idx)

		pp = proba[start_idx:end_idx]
		pp = pp [pp != 255]

		if len(pp) > 0:
			p = np.nanmean(pp - S)

			if p >= 0:
				proba_avg[i] = int(100 * p / (100-S))
			else:
				proba_avg[i] = int(100 * (-p / S))

			proba_std[i] = np.nanstd(proba[start_idx:end_idx])
			


		Csample = pred[start_idx:end_idx]
		#print (Csample)

		numscenes = end_idx - start_idx
		
		#pos_freq[i] = np.sum(Csample[Csample <= 1])
		pos_freq[i] = int(100*np.sum(Csample[Csample == 1]) / numscenes)

		#	print (proba[start_idx:end_idx])
		#	quit()

	return proba_avg, proba_std, pos_freq








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







