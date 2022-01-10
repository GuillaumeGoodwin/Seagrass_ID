"""
Intertidal_groundcover_main.py

This is the main script in a research project to extract ground cover (vegetation, sediment type) in intertidal settings such as lagoons, bays and estuaries. This work differs from typical ground cover determination because it combines detection methods for emerged AND submerged environments

Author: CchGoodwin

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
from matplotlib.patches import Rectangle


from itertools import product as pdt
from scipy.optimize import minimize
from datetime import datetime, timedelta

from osgeo import gdal, ogr, osr
from shapely.geometry import Polygon
import rasterio
import fiona
import math
from rasterstats.io import Raster

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import scikitplot as skplt
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
import seaborn as sn
from sklearn.metrics import confusion_matrix, r2_score

import scipy

from PIL import Image, ImageFilter, ImageOps
from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
)


import Functions as fn
import ML_functions as ml
import Lee_model as lee

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
maskraster = "/home/willgoodwin/PostDoc/Padova/Data/Basemaps/basemap_laguna.bil"
zoneraster = "/home/willgoodwin/PostDoc/Padova/Data/Basemaps/zonemap_laguna.bil"

mask, pixelWidth, (geotransform, inDs) = ml.ENVI_raster_binary_to_3d_array(maskraster)
mask = mask[:-1] # get rid of the last row of mask ... FIX THIS ANOTHER WAY! HOW????

zones, pixelWidth, (geotransform, inDs) = ml.ENVI_raster_binary_to_3d_array(zoneraster)
zones = zones[:-1]

# load an additional mask for the aoi
aoi, pixelWidth, (geotransform, inDs) = ml.ENVI_raster_binary_to_3d_array('/home/willgoodwin/PostDoc/Padova/Data/Basemaps/Inlets_AoI_laguna.bil')
aoi = aoi[:-1]

#############################################################################
# load data for multiple scenes: scenes, overpass times, tides, wind
#############################################################################
train_scenes, datetime_list = fn.landsat_scenes_to_panda(train_file)
train_scenes = list(train_scenes['scene'])

prediction_scenes, datetime_list = fn.landsat_scenes_to_panda(prediction_file)
prediction_scenes = list(prediction_scenes['scene'])

overpass_times = pd.read_csv(overpass_file)

tides = fn.extract_station_record ('Livello_(m)', ["chioggiacitta", "saline", "piattaforma"])


#############################################################################
# These are the scenes with special stuff
#############################################################################

Digiscenes = ["LE07_L1TP_192028_19990906_20170217_01_T1", #d
			  "LE07_L1TP_192028_20000316_20170213_01_T1", #d
			  "LE07_L1TP_192028_20001127_20170209_01_T1", #d
			  "LE07_L1TP_192028_20010911_20170203_01_T1", #d
			  "LE07_L1TP_192028_20020117_20170201_01_T1", #d
			  "LE07_L1TP_192028_20020322_20191106_01_T1", #d
			  "LE07_L1TP_192028_20020914_20170128_01_T1", #d
			  "LE07_L1TP_192028_20030325_20170214_01_T1", #d
			  "LE07_L1TP_192028_20040428_20170121_01_T1", #d
			  "LE07_L1TP_192028_20050720_20170113_01_T1", #d
			  "LE07_L1TP_192028_20060213_20170110_01_T1", #d
			  "LE07_L1TP_192028_20070216_20170104_01_T1", #d
			  "LE07_L1TP_192028_20090104_20161223_01_T1", #d
			  "LE07_L1TP_192028_20100328_20161215_01_T1", #d
			  "LE07_L1TP_192028_20110502_20161210_01_T1", #d
			  "LE07_L1TP_192028_20120214_20161203_01_T1", #d
			  "LE07_L1TP_192028_20130216_20161126_01_T1", #d
			  "LC08_L1TP_192028_20140331_20170424_01_T1", #d
			  "LC08_L1TP_192028_20150113_20170414_01_T1", #d
			  "LC08_L1TP_192028_20160116_20170405_01_T1", #d
			  "LC08_L1TP_192028_20160928_20170321_01_T1", #d
			  "LC08_L1TP_192028_20170408_20180523_01_T1", #d
			  "LC08_L1TP_192028_20170830_20170914_01_T1", #d
			  "LC08_L1TP_192028_20180326_20180404_01_T1", #d
			  "LC08_L1TP_192028_20180630_20180716_01_T1", #d
			  "LC08_L1TP_192028_20190225_20190309_01_T1", #d
			  "LC08_L1TP_192028_20190905_20190917_01_T1"] #d

Inletscenes = ["LE07_L1TP_192028_20080610_20161228_01_T1",
			   "LE07_L1TP_192028_20090901_20161218_01_T1",
			   "LE07_L1TP_192028_20100904_20161212_01_T1",
			   "LE07_L1TP_192028_20111009_20161206_01_T1",
			   "LE07_L1TP_192028_20120909_20161129_01_T1",
			   "LC08_L1TP_192028_20130904_20170502_01_T1",
			   "LC08_L1TP_192028_20140806_20170420_01_T1",
			   "LC08_L1TP_192028_20150809_20170406_01_T1"]

Deletescenes = ["LE07_L1TP_192028_20060909_20170107_01_T1",
				"LE07_L1TP_192028_20060301_20170111_01_T1",
				"LE07_L1TP_192028_20070608_20170103_01_T1",
				"LE07_L1TP_192028_20080219_20161230_01_T1",
				"LE07_L1TP_192028_20071217_20161231_01_T1",
				"LE07_L1TP_192028_20090221_20161222_01_T1",
				"LE07_L1TP_192028_20100920_20161212_01_T1",
				"LE07_L1TP_192028_20120113_20161203_01_T1",
				"LC08_L1TP_192028_20170118_20170311_01_T1",
				"LC08_L1TP_192028_20190124_20190205_01_T1"]
###



#############################################################################
# load pickled models, their probas and all the prediction times
#############################################################################
# Case 1: no seagrass location
#features = np.array([blue, green, red, nir, bathy-H_unique, D50,zones])
features_txt = 'b_g_r_n_d_d50' 

# Case 2: the model knows initial seagrass location
#features = np.array([blue, green, red, nir, bathy-H_unique, D50, loc_rows,loc_cols,zones])
#features_txt = 'b_g_r_n_d_d50_loc' 

Models, Model_scenes, Model_times, Sepvals, Existence = pickle.load( open( Fig_dir2 + '/5Modelsdata_' + features_txt + '_q5_n3_md30_ne100.pkl', "rb" ) )



pred_times = pickle.load( open( Fig_dir2 + '/predtimes.pkl', "rb" ) )


predmaps = pickle.load( open( Fig_dir2 + 'ContiID_mixed_b_g_r_n_d_d50.pkl', "rb" ) )


#############################################################################
# Start evaluation of contrast
#############################################################################

Contrast = []

# Loop through scenes
counter = 0
for scene in prediction_scenes:



	fig = plt.figure(0, facecolor='White',figsize=[15, 15])
	ax = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)

	ax.imshow(predmaps[counter], vmin = 0, vmax = 2)

	plt.show()

	counter+=1

	continue



	#print (counter)
	# If it is not the training scene and we have info on its vegetation
	#if scene in Model_scenes+Inletscenes+Digiscenes and scene != modelscene:
	if scene not in Deletescenes:
		print (counter, scene)

		time = ml.gettime(overpass_times, scene) 

		# Load/make feature rasters
		satnumber = scene.split('_')[0][-1]
		bands_dict, wavelength_dict, aw_dict = fn.satnumber_dicts (satnumber)
		blue, green, red, nir, bathy, mask, zones, theta_w, theta_v, pixelWidth, (geotransform, inDs) = ml.load_rasters_no_WWTM(dwld_dir, scene, bathyraster, maskraster, zoneraster)
		H_unique = ml.extract_tide(tides, bathy, time)
		D = H_unique - bathy


		# Load the prediction
		C = predmaps[counter]

		C_temp = np.copy(C)

		C_temp[C_temp == 255] = 0

		C_edges = fn.make_classif_edges_arr (C_temp)

		C_edges [mask != 0] = 0
		C_edges [C_edges < 1] = 0

		##############################################################
		# Make the tidal flat contrast metric
		
		band_txt = ['blue', 'green', 'red','nir']
		band = 0
		for arr in [blue]:
			contrast = np.zeros(arr.shape, dtype = np.float32)

			# Make the tidal flat unique indices
			cont, tf_id = fn.getContiguousAreas(mask, 0, min_pixels = 3, contiguity = 'queen')
			# Make the bare unique indices
			cont, ba_id = fn.getContiguousAreas(C, 0, min_pixels = 3, contiguity = 'queen')
			# Make the vegetated unique indices
			cont, ve_id = fn.getContiguousAreas(C, 1, min_pixels = 3, contiguity = 'queen')


			for v in range(1,np.amax(ve_id)):
				tfi = tf_id[ve_id == v][0]
				print (tfi)

				if len(arr[np.logical_and(tf_id == tfi,C == 0)]) > 0:
					sp_bare = np.nanmean(arr[np.logical_and(tf_id == tfi,C == 0)])
					std_bare = np.nanstd(arr[np.logical_and(tf_id == tfi,C == 0)])

					sp_vege = np.nanmean(arr[np.logical_and(ve_id == v,C == 1)])
					std_vege = np.nanstd(arr[np.logical_and(ve_id == v,C == 1)])
				
					A = sp_bare-sp_vege
					if A >= 0:
						colour = 'g'
						B = (sp_bare-std_bare) - (sp_vege+std_vege)
					elif A < 0:
						colour = 'k'
						B = (sp_vege-std_vege) - (sp_bare+std_bare)


					if A >= 0 and B >= 0: #good separation in the right way
						contrast[ve_id == v] = 100*B
					elif A >= 0 and B < 0: #bad separation in the right way
						contrast[ve_id == v] = 100*B
					elif A < 0 and B >= 0: #good separation in the wrong way
						contrast[ve_id == v] = 0
					else: # not much sep in the wrong way
						contrast[ve_id == v] = 0


			"""
			fig = plt.figure(0, facecolor='White',figsize=[13, 7])
			ax = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)
			#ax.imshow(arr, plt.cm.Greys_r)
			#ax.imshow(tf_id)
			#ax.imshow(ve_id, cmap = plt.cm.Reds, alpha = 0.5)
			ax.imshow(contrast, cmap = plt.cm.seismic)#, vmin = -2, vmax = 2)
			ax.imshow(np.ma.masked_where(C_edges < 1, C_edges), plt.cm.Greys_r, vmax = 2)

			plt.show()
			quit()
			"""
			

			##############################################################
			# Make the border contrast metric

			start = datetime.now()
			for r in range(len(C)):
			#for r in range(400,600):
				print (r)
				for c in range(len(C[0])):
					if C_edges[r,c] >= 1 and C[r,c]==1:
						# Make a kernel
						sp_kernel, k_indices, indices, spk_centre = fn.kernel_circle (arr, 9, r, c)
						c_kernel, k_indices, indices, ck_centre = fn.kernel_circle (C, 9, r, c)


						X = np.ma.count_masked(sp_kernel)
						XX = np.ma.count_masked(sp_kernel[c_kernel != 1])
						L = len(sp_kernel[c_kernel != 1])

						if XX < L:
							sp_in = np.nanmean(sp_kernel[c_kernel == 1])
							std_in = np.nanstd(sp_kernel[c_kernel == 1])
							sp_out= np.nanmean(sp_kernel[c_kernel != 1])
							std_out= np.nanstd(sp_kernel[c_kernel != 1])
							#print (sp_in, sp_out)

							# Make contrast
							#contrast[r,c] = np.nanstd(kernel)
							#contrast[r,c] = int(5000*(sp_out - sp_in))
							contrast[r,c] = (sp_out - sp_in) / std_out

			#contrast = np.ma.masked_where(mask != 0, contrast)
			#contrast = np.ma.masked_where(arr <= 0, contrast)

			contrast [contrast > 127] = 127
			contrast [contrast < -128] = -128

			timer = datetime.now()-start
			print(timer)

			fig = plt.figure(0, facecolor='White',figsize=[13, 7])
			ax = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)
			ax.imshow(arr, plt.cm.Greys_r)
			#ax.imshow(C, plt.cm.Greys, vmin = 0, vmax = 2, alpha = 1)
			#ax.imshow(contrast, cmap = plt.cm.seismic, vmin = -2, vmax = 2, alpha = 0.8)
			ax.imshow(np.ma.masked_where(contrast==0,contrast), cmap = plt.cm.seismic, vmin = -1, vmax = 1, alpha = 0.8)
			plt.show()
			quit()

			band+=1

			Contrast.append(contrast)


			
	counter+=1

pickle.dump( Contrast, open( Fig_dir2 + 'b_3px_contrast.pkl', "wb" ) )

	

quit()

		