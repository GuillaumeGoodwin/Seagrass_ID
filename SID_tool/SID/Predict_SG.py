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
D50raster = "/home/willgoodwin/PostDoc/Padova/Data/VL_D50/Campioni_D50_3003.bil"
maskraster = "/home/willgoodwin/PostDoc/Padova/Data/Basemaps/basemap_laguna.bil"
zoneraster = "/home/willgoodwin/PostDoc/Padova/Data/Basemaps/zonemap_laguna.bil"

mask, pixelWidth, (geotransform, inDs) = ml.ENVI_raster_binary_to_3d_array(maskraster)
mask = mask[:-1] # get rid of the last row of mask ... FIX THIS ANOTHER WAY! HOW????

zones, pixelWidth, (geotransform, inDs) = ml.ENVI_raster_binary_to_3d_array(zoneraster)
zones = zones[:-1]

D50, pixelWidth, (geotransform, inDs) = ml.ENVI_raster_binary_to_3d_array(D50raster)


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
# load pickled models and all the prediction times
#############################################################################

Case = 1

if Case == 1:
	Models, Model_scenes, Model_times, Sepvals, Existence = pickle.load( open( Fig_dir2 + '/5Modelsdata_b_g_r_n_d_d50_q5_n3_md30_ne100.pkl', "rb" ) )
	Z_bestmodel = [3,3,0]
elif Case == 2:
	Models, Model_scenes, Model_times, Sepvals, Existence = pickle.load( open( Fig_dir2 + '/5Modelsdata_b_g_r_n_d_d50_loc_q5_n3_md30_ne100.pkl', "rb" ) )
	Z_bestmodel = [3,3,0]


pred_times = pickle.load( open( Fig_dir2 + '/predtimes.pkl', "rb" ) )



#############################################################################
# Select model for each zone
#############################################################################

print ('selecting models')

Ranges = []

for M in [0,1,2,3,4]:
	modelscene = Model_scenes[M]
	modeltime = Model_times[M]
	model = Models[M]
	exist_model = Existence[M]
	sepvals = Sepvals[M]

	print (sepvals)

	ranges = []

	# Loop through scenes
	counter = 0
	for scene in Model_scenes:
		print (counter)
		# If it is not the training scene and we have info on its vegetation
		if scene != modelscene:

			print (counter, scene)

			appl_time = ml.gettime(overpass_times, scene) 

			# Load/make feature rasters
			satnumber = scene.split('_')[0][-1]
			bands_dict, wavelength_dict, aw_dict = fn.satnumber_dicts (satnumber)
			blue, green, red, nir, bathy, mask, zones, theta_w, theta_v, pixelWidth, (geotransform, inDs) = ml.load_rasters_no_WWTM(dwld_dir, scene, bathyraster, maskraster, zoneraster)
			H_unique = ml.extract_tide(tides, bathy, appl_time)
			loc_rows, loc_cols = ml.makerowcol(blue)



			##############################################
			# Define the features
			if Case == 1:
				features = np.array([blue, green, red, nir, bathy-H_unique, D50,zones])
				features_txt = 'b_g_r_n_d_d50' 
			elif Case == 2:
				features = np.array([blue, green, red, nir, bathy-H_unique, D50, loc_rows,loc_cols,zones])
				features_txt = 'b_g_r_n_d_d50_loc'  

			##############################################

			# Create the probamap and predictions
			proba, pred = ml.Apply_Random_Forest_full (features, model, sepvals, exist_model, mask, zones)
			proba[blue < 0] = 255


			cover, density, pixelWidth, (geotransform, inDs) = ml.load_vetted_vegetation_data(veg_dir, appl_time, mask); cover[cover> 0.5] = 1; cover[cover <= 0.5] = 0

			diffmap, S, P, A, QFN , QFP = fn.make_zonewise_perf (pred, proba, cover, zones)

			S = np.asarray(S); P = np.asarray(P)
			min_coeff = P; max_coeff = 1 + (P-S*P)/S
			R = max_coeff - min_coeff

			F1 = 2*S*P/(S+P)

			ranges.append(F1)	

		else:
			ranges.append(np.zeros(4))

	Ranges.append(ranges)

Ranges = np.asarray(Ranges)




"""
# The winning model for each zone is the one with the larges median F1
Z_bestmodel = []
for z in range(0,3):
	AVG = []
	for m in range(len(Ranges)):
		avg = np.percentile(Ranges[m,:,z],50)
		AVG.append(avg)

	AVG = np.asarray(AVG)
	X = np.where(AVG == max(AVG))[0]
	if len(X) == 0:
		Z_bestmodel.append(0)
	else:
		Z_bestmodel.append(X[0])

print (Z_bestmodel)quit()
"""
#############################################################################
# Start predictions
#############################################################################


probamaps = 255 * np.ones((len(prediction_scenes), len(blue),len(blue[0])), dtype = np.uint8)

#Z_bestmodel = [3,3,0]

print (probamaps.shape)


# Loop through scenes to predict
counter = 0
for scene in prediction_scenes:
	print (counter)

	# pick from the best models
	for z in range(len(Z_bestmodel)):
		
		M = Z_bestmodel[z]

		modelscene = Model_scenes[M]
		modeltime = Model_times[M]
		model = Models[M]
		exist_model = Existence[M]
		sepvals = Sepvals[M]


		print (counter, scene)

		appl_time = ml.gettime(overpass_times, scene) 

		# Load/make feature rasters
		satnumber = scene.split('_')[0][-1]
		bands_dict, wavelength_dict, aw_dict = fn.satnumber_dicts (satnumber)
		blue, green, red, nir, bathy, mask, zones, theta_w, theta_v, pixelWidth, (geotransform, inDs) = ml.load_rasters_no_WWTM(dwld_dir, scene, bathyraster, maskraster, zoneraster)
		H_unique = ml.extract_tide(tides, bathy, appl_time)
		loc_rows, loc_cols = ml.makerowcol(blue)

		# Calculate the bottom reflectance
		#Blue = lee.calculate_bottom_reflectance_known_SPM (20, blue, wavelength_dict['blue'], H_unique - bathy, 3, theta_v, theta_w, aw_dict['blue'])
		#Green = lee.calculate_bottom_reflectance_known_SPM (20, green, wavelength_dict['green'], H_unique - bathy, 3, theta_v, theta_w, aw_dict['green'])
		#Red = lee.calculate_bottom_reflectance_known_SPM (20, red, wavelength_dict['red'], H_unique - bathy, 3, theta_v, theta_w, aw_dict['red'])



		##############################################
		# Define the features
		if Case == 1:
			features = np.array([blue, green, red, nir, bathy-H_unique, D50,zones])
			features_txt = 'b_g_r_n_d_d50' 
		elif Case == 2:
			features = np.array([blue, green, red, nir, bathy-H_unique, D50, loc_rows,loc_cols,zones])
			features_txt = 'b_g_r_n_d_d50_loc'  

		##############################################

		# Create the probamap and predictions
		proba, pred = ml.Apply_Random_Forest_full (features, model, sepvals, exist_model, mask, zones)
		proba[blue < 0] = 255

		probamaps[counter,np.where(zones == z+1)[0],np.where(zones == z+1)[1]] = proba[zones == z+1]


	counter += 1


pickle.dump(probamaps, open( Fig_dir2 + '/probamap_mixed_'+features_txt+'_n3_md30_ne100.pkl', "wb" ) )

print (Z_bestmodel)


quit()
















































































# Loop through models to apply
#for M in range(len(Models)):
for M in [0,1,2,3,4]:
	modelscene = Model_scenes[M]
	modeltime = Model_times[M]
	model = Models[M]
	exist_model = Existence[M]
	sepvals = Sepvals[M]

	probamap = []

	# Loop through scenes
	counter = 0
	for scene in prediction_scenes:
		print (counter)
		# If it is not the training scene and we have info on its vegetation
		#if scene in Model_scenes+Inletscenes+Digiscenes and scene != modelscene:
		if scene != '1':

			#if counter >= 10:
			#	continue

			print (counter, scene)

			appl_time = ml.gettime(overpass_times, scene) 

			# Load/make feature rasters
			satnumber = scene.split('_')[0][-1]
			bands_dict, wavelength_dict, aw_dict = fn.satnumber_dicts (satnumber)
			blue, green, red, nir, bathy, mask, zones, theta_w, theta_v, pixelWidth, (geotransform, inDs) = ml.load_rasters_no_WWTM(dwld_dir, scene, bathyraster, maskraster, zoneraster)
			H_unique = ml.extract_tide(tides, bathy, appl_time)
			loc_rows, loc_cols = ml.makerowcol(blue)

			# Calculate the bottom reflectance
			#Blue = lee.calculate_bottom_reflectance_known_SPM (20, blue, wavelength_dict['blue'], H_unique - bathy, 3, theta_v, theta_w, aw_dict['blue'])
			#Green = lee.calculate_bottom_reflectance_known_SPM (20, green, wavelength_dict['green'], H_unique - bathy, 3, theta_v, theta_w, aw_dict['green'])
			#Red = lee.calculate_bottom_reflectance_known_SPM (20, red, wavelength_dict['red'], H_unique - bathy, 3, theta_v, theta_w, aw_dict['red'])



			##############################################
			# Define the features
			# Case 1: no seagrass location
			features = np.array([blue, green, red, nir, bathy-H_unique, D50,zones])
			features_txt = 'b_g_r_n_d_d50' 

			# Case 2: the model knows initial seagrass location
			#features = np.array([blue, green, red, nir, bathy-H_unique, D50, loc_rows,loc_cols,zones])
			#features_txt = 'b_g_r_n_d_d50_loc'  

			##############################################

			# Create the probamap and predictions
			proba, pred = ml.Apply_Random_Forest_full (features, model, sepvals, exist_model, mask, zones)
			proba[blue < 0] = 255


			probamap.append(proba)

		else:
			probamap.append(np.zeros(mask.shape, dtype = np.uint8))

		counter+=1




	pickle.dump( probamap, open( Fig_dir2 + '/probamap_'+str(modeltime.year)+'_'+features_txt+'_n3_md30_ne100.pkl', "wb" ) )

	#pickle.dump( probamap, open( Fig_dir2 + '/probamap_'+str(modeltime.year)+'_bdog.pkl', "wb" ) )


		