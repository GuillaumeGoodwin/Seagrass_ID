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

Digiscenes = ["LE07_L1TP_192028_19990906_20170217_01_T1", #dd
			  "LE07_L1TP_192028_20000316_20170213_01_T1", #dd
			  "LE07_L1TP_192028_20001127_20170209_01_T1", #dd
			  "LE07_L1TP_192028_20010911_20170203_01_T1", #dd
			  "LE07_L1TP_192028_20020117_20170201_01_T1", #dd
			  "LE07_L1TP_192028_20020322_20191106_01_T1", #dd
			  "LE07_L1TP_192028_20020914_20170128_01_T1", #dd
			  "LE07_L1TP_192028_20030325_20170214_01_T1", #dd
			  "LE07_L1TP_192028_20040428_20170121_01_T1", #dd
			  "LE07_L1TP_192028_20050720_20170113_01_T1", #dd
			  "LE07_L1TP_192028_20060213_20170110_01_T1", #dd
			  "LE07_L1TP_192028_20070216_20170104_01_T1", #dd
			  "LE07_L1TP_192028_20090104_20161223_01_T1", #dd
			  "LE07_L1TP_192028_20100328_20161215_01_T1", #dd
			  "LE07_L1TP_192028_20110502_20161210_01_T1", #dd
			  "LE07_L1TP_192028_20120214_20161203_01_T1", #dd
			  "LE07_L1TP_192028_20130216_20161126_01_T1", #dd
			  "LC08_L1TP_192028_20140331_20170424_01_T1", #dd
			  "LC08_L1TP_192028_20150113_20170414_01_T1", #dd
			  "LC08_L1TP_192028_20160116_20170405_01_T1", #dd
			  "LC08_L1TP_192028_20160928_20170321_01_T1", #dd
			  "LC08_L1TP_192028_20170408_20180523_01_T1", #dd
			  "LC08_L1TP_192028_20180326_20180404_01_T1", #dd
			  "LC08_L1TP_192028_20180630_20180716_01_T1", #dd
			  "LC08_L1TP_192028_20190225_20190309_01_T1", #dd
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
#Models, Model_scenes, Model_times, Sepvals, Existence = pickle.load( open( Fig_dir2 + '/5Modelsdata_bgdor_n3_md30_ne100.pkl', "rb" ) )
Models, Model_scenes, Model_times, Sepvals, Existence = pickle.load( open( Fig_dir2 + '/5Modelsdata_b_g_dor_Rr_n_d_avloc_q5_n3_md30_ne100.pkl', "rb" ) )
#Models, Model_scenes, Model_times, Sepvals, Existence = pickle.load( open( Fig_dir2 + '/Modeldata_bgx_n2_md90_ne100.pkl', "rb" ) )
#Models, Model_scenes, Model_times, Sepvals, Existence = pickle.load( open( Fig_dir2 + '/Modeldata_bdogdorBbGgRrndloc.pkl', "rb" ) )

probafiles = [Fig_dir2+'probamap_2002_b_g_dor_n_d_n3_md30_ne100.pkl',
			  Fig_dir2+'probamap_2004_b_g_dor_n_d_n3_md30_ne100.pkl',
			  Fig_dir2+'probamap_2009_b_g_dor_Rr_n_d_avloc_n3_md30_ne100.pkl',
			  Fig_dir2+'probamap_2010_b_g_dor_n_d_n3_md30_ne100.pkl',
			  Fig_dir2+'probamap_2017_b_g_dor_n_d_n3_md30_ne100.pkl']


pred_times = pickle.load( open( Fig_dir2 + '/predtimes.pkl', "rb" ) )






PRED = pickle.load( open( Fig_dir2 + 'PERF/PRED.pkl', "rb" ) )
CONF = pickle.load( open( Fig_dir2 + 'PERF/CONF.pkl', "rb" ) )

SF4_c = pickle.load( open( Fig_dir2 + 'SF4_all.pkl', "rb" ) )
SF2_c = pickle.load( open( Fig_dir2 + 'SF2_all.pkl', "rb" ) )


seasons_4, seasons_2 = pickle.load( open( Fig_dir2 + 'seasons.pkl', "rb" ) )
U_4 = np.unique(seasons_4)
U_2 = np.unique(seasons_2)



###################################################################################
# Plot an intellegible way of comparing the probabilities to the presence/ density
###################################################################################
"""
# Make a time_based performance figure for one model
fig = plt.figure(0, facecolor='White',figsize=[14, 9])


S = -1

x = 0; y = 0

for num in range(len(prediction_scenes)):

	print (num)
	testscene = prediction_scenes[num]
	testidx = np.where(np.asarray(prediction_scenes) == testscene)[0][0]
	testime = pred_times[testidx]
	testseason_4, testseason_2 = fn.count_seasons([testime])
	Season_idx = np.where(seasons_2 == testseason_2)
	SF_2_idx = np.where(U_2 == testseason_2)[0][0]

	print (testidx, SF_2_idx, Season_idx)



	if SF_2_idx != S:
		S = SF_2_idx

		axis = plt.subplot2grid((6,7), (x,y), rowspan=1, colspan=1)

		axis.imshow(np.ma.masked_where(PRED[num,1250:1335,320:450] != 0, CONF[num,1250:1335,320:450]), cmap = plt.cm.Greys, vmin = 0, vmax = 100)
		axis.imshow(np.ma.masked_where(PRED[num,1250:1335,320:450] != 1, CONF[num,1250:1335,320:450]), cmap = plt.cm.summer, vmin = 0, vmax = 100)


		if testime.month in [4,5,6,7,8,9]:
			axis.set_title(str(testime.year)+'-growth')
		elif testime.month in [1,2,3]:
			axis.set_title(str(testime.year)+'-decay')
		elif testime.month in [10,11,12]:
			axis.set_title(str(testime.year)+'-decay')

		y += 1
		if y == 7:
			y = 0
			x += 1

		axis.set_yticklabels([])
		axis.set_xticklabels([])

plt.tight_layout()

Figname = 'ZOOM/zoomed.png'
#plt.show()
plt.savefig(Fig_dir2+Figname)
#plt.show()
#quit()
"""





#################################################################################""
#################################################################################""
# Make a time_based performance figure for one model


S = -1

x = 0; y = 0

count = 0

for num in range(len(prediction_scenes)):

	print (num)
	testscene = prediction_scenes[num]
	testidx = np.where(np.asarray(prediction_scenes) == testscene)[0][0]
	testime = pred_times[testidx]
	testseason_4, testseason_2 = fn.count_seasons([testime])
	Season_idx = np.where(seasons_2 == testseason_2)
	SF_2_idx = np.where(U_2 == testseason_2)[0][0]

	print (testidx, SF_2_idx, Season_idx)


	
	if SF_2_idx != S:
		S = SF_2_idx

		fig = plt.figure(0, facecolor='White',figsize=[20, 30])
		axis = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)

		A = axis.imshow(np.ma.masked_where(PRED[num] != 0, CONF[num]), cmap = plt.cm.Greys, vmin = 0, vmax = 100)
		B = axis.imshow(np.ma.masked_where(PRED[num] != 1, CONF[num]), cmap = plt.cm.summer_r, vmin = 0, vmax = 100)
		axis.imshow(np.ma.masked_where(mask == 0, mask), cmap = plt.cm.Greys, vmin = 0, vmax = 10)
		axis.imshow(np.ma.masked_where(zones >= 1, zones), cmap = plt.cm.Greys, vmin = 0, vmax = 1)

		if testime.month in [4,5,6,7,8,9]:
			axis.set_title(str(testime.year)+'-growth', fontsize = 30)
			figname = str(testime.year)+'-growth'
		elif testime.month in [1,2,3]:
			axis.set_title(str(testime.year)+'-decay', fontsize = 30)
			figname = str(testime.year)+'-decay'
		elif testime.month in [10,11,12]:
			axis.set_title(str(testime.year)+'-decay', fontsize = 30)
			figname = str(testime.year)+'-decay'

		y += 1
		if y == 7:
			y = 0
			x += 1


		from mpl_toolkits.axes_grid1 import make_axes_locatable
		divider = make_axes_locatable(axis)

		cax = divider.new_vertical(size="5%", pad=0.2, pack_start=True)
		fig.add_axes(cax)
		cbar = fig.colorbar(A, cax=cax, orientation="horizontal")
		cbar.set_label('Confidence (negative)', fontsize = 24)

		cax = divider.new_vertical(size="5%", pad=0.7, pack_start=True)
		fig.add_axes(cax)
		cbar = fig.colorbar(B, cax=cax, orientation="horizontal")
		cbar.set_label('Confidence (positive)', fontsize = 24)

		#cbar = plt.colorbar(A, orientation = 'horizontal', fraction = 0.1)
		#cbar.set_label('Confidence (negative)')

		#cbar = plt.colorbar(B, orientation = 'horizontal', fraction = 0.1)
		#cbar.set_label('Confidence (positive)')


		#fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)


		axis.set_yticklabels([])
		axis.set_xticklabels([])

		plt.tight_layout()

		plt.savefig(Fig_dir2+'ZOOM/'+str(count)+'.png')
		count += 1
		#quit()
#plt.show()
#quit()








quit()



#Figname = 'Performance_bdog.png'
zonecol = plt.cm.gist_rainbow



Area = []
Area_c = []

AreaL = []
AreaL_c = []



counter = 0

AZ = np.zeros((len(prediction_scenes),len(np.arange(1,6))), dtype = np.float32)
AZL = np.zeros((len(prediction_scenes),len(np.arange(1,6))), dtype = np.float32)

for scene in prediction_scenes:

	if scene != '1':#modelscene:

		##########################################################
		# Load all necessary info

		time = pred_times[counter]
		txt_time = datetime.strftime(time, '%Y%m%d')
		
		#classif = np.copy(predmaps[counter])
		classif = fn.proba2pred(probamaps[counter],sepvals,zones,mask)
		pred = np.copy(classif)


		#pred_c = predmaps_c[counter]
		#L_c = Long_wks_c[counter]
		L = Long_wks[counter]

		proba = np.copy(probamaps[counter])
		pred[proba == 255] = 255
		#pred_c[proba == 255] = 255


		############################################################
		# Begin area calculations

		# the full area just so we can "ignore" the L7 blank strips
		AoI = len(np.where(np.logical_and(proba != 255, mask==0))[0])

		if scene in Model_scenes:
			# Load the survey
			kso, kso_density, kso_poly, pxw, geodata = fn.load_vetted_vegetation_data(veg_dir, time, mask)
		
			# remove masks
			kso [kso == 255] = 0
			kso[np.ma.getmask(kso) == True] = 0

			A_ksoprev = 0
			for z in np.arange(1,6):
				z_kso = kso[zones == z]
				A_kso = np.sum(z_kso)
				A_kso = 100 * A_kso / AoI

				ax.bar(time,A_kso, width = 100, bottom = A_ksoprev, facecolor = zonecol((z-1)/7), lw = 2, edgecolor = 'k', alpha = 1)

				A_ksoprev += A_kso


		for z in np.arange(1,6):
			z_pred = pred[zones == z]
			A_pred = np.sum(z_pred[z_pred == 1])
			A_pred = 100 * A_pred / AoI
			AZ[counter,z-1] = A_pred

			#ax.bar(time,A_pred, width = 50, bottom = A_predprev, facecolor = zonecol((z-1)/6), lw = 0, alpha = 0.6)

		

		##A = np.sum(pred[pred == 1])
		##A_c = np.sum(pred_c[pred_c == 1])

		##AL = np.sum(pred[np.logical_and(pred == 1, L >= 52)])
		##AL_c = np.sum(pred_c[np.logical_and(pred_c == 1, L_c >= 52)])


		
		#print (MO)

		

		#Area.append(A/MO)
		#Area_c.append(A_c/MO)


		#AreaL.append(AL/MO)
		#AreaL_c.append(AL_c/MO)



		#if counter >= 10:
			#Am = np.mean(Area[counter-10:counter])
			#Am_c = np.mean(Area_c[counter-10:counter])

			#ax.scatter(pred_times[counter], Am, facecolor = 'k')
			#ax.scatter(pred_times[counter], Am_c, facecolor = 'b')


		counter += 1

for z in np.arange(1,6):
	ax.plot(pred_times, np.sum(AZ[:,0:z], axis = 1), c = zonecol((z-1)/7), lw = 0.67*np.sqrt(np.amax(AZ[:,z-1])), alpha = 0.5)

for z in np.arange(1,6):
	X = np.sum(AZ[:,0:z], axis = 1); Xm = []
	for l in np.arange(10,len(pred_times)):
		Xm.append(np.mean(X[l-10:l]))
	ax.plot(pred_times[10:], Xm, c = zonecol((z-1)/7), lw = 0.67*np.sqrt(np.amax(AZ[:,z-1])))
	ax.plot(pred_times[10:], Xm, '--', c = 'k', lw = 0.4*np.sqrt(np.amax(AZ[:,z-1])))


#ax.plot(pred_times, Area_c, 'b')

#ax.plot(pred_times, AreaL, '--k')
#ax.plot(pred_times, AreaL_c, '--b')

#ax.plot(pred_times, Area_kso, 'or')



Figname = 'Area_bgdor_m3_md30_ne100_noloc.png'
#plt.show()
plt.savefig(Fig_dir2+Figname)
quit()
	















# Load the actual final maps that matter
Pro,Sep = pickle.load( open( Fig_dir + '/tri_combomodel_nofillnodigi_6.pkl', "rb" ) )

if not os.path.isfile(Fig_dir + '/tri_patchexp_classif_nofillnodigi.pkl'):
	print ('making a Classification')
	Classif = np.zeros(Pro.shape, dtype = np.uint8)
	for i in range(len(Classif)):
		Classif[i] = fn.proba2pred(Pro[i],Sep,zones,mask)
		Classif[i][Pro[i]>=255] = 255

weights = pickle.load( open( Fig_dir + '/tri_opt_weights_nofillnodigi_6.pkl', "rb" ) )



"""
fig = plt.figure(0, facecolor='White',figsize=[23, 20])
ax = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)
ax.imshow(Pro[50], cmap = plt.cm.viridis, vmin = 0, vmax = 255)
plt.show()
quit()
"""

#


testing_modelscenes = []
for i in range(len(Model_scenes)):
    if weights[i] == 0.:
        testing_modelscenes.append(Model_scenes[i])



####

sgl_preds = ['/Model_LE07_L1TP_192028_20040903_20170119_01_T1_predmaps.pkl', '/Model_LE07_L1TP_192028_20090901_20161218_01_T1_predmaps.pkl', '/Model_LE07_L1TP_192028_20100904_20161212_01_T1_predmaps.pkl', '/Model_LC08_L1TP_192028_20170830_20170914_01_T1_predmaps.pkl']


sgl_probas = ['/Model_LE07_L1TP_192028_20040903_20170119_01_T1_probamaps.pkl', '/Model_LE07_L1TP_192028_20090901_20161218_01_T1_probamaps.pkl', '/Model_LE07_L1TP_192028_20100904_20161212_01_T1_probamaps.pkl', '/Model_LC08_L1TP_192028_20170830_20170914_01_T1_probamaps.pkl']


# Make a time_based performance figure for one model
fig = plt.figure(0, facecolor='White',figsize=[23, 20])
# Define the axes
axS = plt.subplot2grid((3,5), (0,0), rowspan=1, colspan=1)
axP = plt.subplot2grid((3,5), (1,0), rowspan=1, colspan=1)
axA = plt.subplot2grid((3,5), (2,0), rowspan=1, colspan=1)
#axFN = plt.subplot2grid((5,5), (3,0), rowspan=1, colspan=1)
#axFP = plt.subplot2grid((5,5), (4,0), rowspan=1, colspan=1)

for x in range(4):
	sgl_pred = pickle.load( open( Fig_dir + sgl_preds[x], "rb" ) )
	sgl_proba = pickle.load( open( Fig_dir + sgl_probas[x], "rb" ) )

	axS1 = plt.subplot2grid((3,5), (0,x+1), rowspan=1, colspan=1)
	axP1 = plt.subplot2grid((3,5), (1,x+1), rowspan=1, colspan=1)
	axA1 = plt.subplot2grid((3,5), (2,x+1), rowspan=1, colspan=1)
	#axFN1 = plt.subplot2grid((5,5), (3,x+1), rowspan=1, colspan=1)
	#axFP1 = plt.subplot2grid((5,5), (4,x+1), rowspan=1, colspan=1)


	# Run a figure for all the scenes
	for i in range(len(prediction_scenes)):

		# Define the scene and time
		scene = prediction_scenes[i]
		time = pred_times[i]
		txt_time = str(time.year)+str(time.month).rjust(2,'0')+str(time.day).rjust(2,'0')
		print (scene, txt_time)

		# define prediction and probability rasters
		pred = np.ma.masked_where(Classif[i]==255, Classif[i])
		proba = np.ma.masked_where(Pro[i]==255, Pro[i])

		# define prediction and probability rasters
		pred0 = np.ma.masked_where(sgl_pred[i]==255, sgl_pred[i])
		proba0 = np.ma.masked_where(sgl_proba[i]==255, sgl_proba[i])


		exist_survey = True
		# Here comes the tricky bit: Choosing the survey to compare to and load the aoi
		if scene in testing_modelscenes:
			print ('Model Scene', scene)

			# This is the vegetation state at kso_idx
			kso, kso_density, pxw, geodata = ml.load_vetted_vegetation_data(veg_dir, time, mask)

			edge_w = 3; mark = 'D'

		elif scene in Inletscenes:
			print ('Inlet Scene', scene)
			kso, kso_density, pxw, geodata = ml.load_vegetation_data(veg_dir, time, mask)
			# load an additional mask for the aoi
			aoi, pixelWidth, (geotransform, inDs) = ml.ENVI_raster_binary_to_3d_array('/home/willgoodwin/PostDoc/Padova/Data/Basemaps/Inlets_AoI_laguna.bil')
			aoi = aoi[:-1]

			kso[aoi < 1] = 255
			kso_density[aoi < 1] = 255

			edge_w = 2; mark = 'o'
			
		elif scene in Digiscenes:
			print ('Digitised Scene', scene)
			kso, kso_density, pxw, geodata = ml.load_digitised_vegetation_data(dwld_dir+scene+'/Digi/', time, mask)
			# load an additional mask for the aoi
			aoi, pixelWidth, (geotransform, inDs) = ml.ENVI_raster_binary_to_3d_array('/home/willgoodwin/PostDoc/Padova/Data/Basemaps/Digi_AoI_laguna.bil')
			aoi = aoi[:-1]

			kso[aoi < 1] = 255
			kso_density[aoi < 1] = 255

			edge_w = 1; mark = 'x'
			
		else:
			kso = None; kso_density = None; aoi = None
			exist_survey = False

		
		# plot things
		if exist_survey == True:
			print ('evaluating model here')
			# Make the difference map and performance metrics
			S, P, A, QFN , QFP = fn.make_zonewise_perf(pred, proba, kso, zones)
			S0, P0, A0, QFN , QFP = fn.make_zonewise_perf(pred0, proba0, kso, zones)

			for z in range(1,4):
				if x == 0:
					axS.scatter(time, S[z], marker=mark, facecolor = plt.cm.jet(z/5), alpha = 0.5, lw = edge_w, edgecolor = 'k')
					axP.scatter(time, P[z], marker=mark, facecolor = plt.cm.jet(z/5), alpha = 0.5, lw = edge_w, edgecolor = 'k')
					axA.scatter(time, A[z], marker=mark, facecolor = plt.cm.jet(z/5), alpha = 0.5, lw = edge_w, edgecolor = 'k')

					#axFN.scatter(time, QFN[z][1], marker='o', facecolor = plt.cm.jet(z/5), alpha = 0.5, lw = edge_w, edgecolor = 'k' )
					#axFN.plot([time,time], [QFN[z][0],QFN[z][2]], c = 'k', alpha = 0.5, lw = edge_w)

					#axFP.scatter(time, QFP[z][1], marker='D', facecolor = plt.cm.jet(z/5), alpha = 0.5, lw = edge_w, edgecolor = 'k' )
					#axFP.plot([time,time], [QFP[z][0],QFP[z][2]], c = 'k', alpha = 0.5, lw = edge_w)

				axS1.scatter(time, S[z]-S0[z], marker=mark, facecolor = plt.cm.jet(z/5), alpha = 0.5, lw = edge_w, edgecolor = 'k')
				axP1.scatter(time, P[z]-P0[z], marker=mark, facecolor = plt.cm.jet(z/5), alpha = 0.5, lw = edge_w, edgecolor = 'k')
				axA1.scatter(time, A[z]-A0[z], marker=mark, facecolor = plt.cm.jet(z/5), alpha = 0.5, lw = edge_w, edgecolor = 'k')

				#axFN1.scatter(time, QFN[z][1], marker='o', facecolor = plt.cm.jet(z/5), alpha = 0.5, lw = 0.5, edgecolor = 'k' )
				#axFN1.plot([time,time], [QFN[z][0],QFN[z][2]], c = 'k', alpha = 0.5, lw = edge_w)

				#axFP1.scatter(time, QFP[z][1], marker='D', facecolor = plt.cm.jet(z/5), alpha = 0.5, lw = 0.5, edgecolor = 'k' )
				#axFP1.plot([time,time], [QFP[z][0],QFP[z][2]], c = 'k', alpha = 0.5, lw = edge_w)

			if x == 0:
				axS.scatter(time, S[-1], marker=mark, facecolor = None, alpha = 0.5, lw = 2*edge_w, edgecolor = 'r')
				axP.scatter(time, P[-1], marker=mark, facecolor = None, alpha = 0.5, lw = 2*edge_w, edgecolor = 'r')
				axA.scatter(time, A[-1], marker=mark, facecolor = None, alpha = 0.5, lw = 2*edge_w, edgecolor = 'r')

			axS1.scatter(time, S[-1]-S0[-1], marker=mark, facecolor = None, alpha = 0.5, lw = 2*edge_w, edgecolor = 'r')
			axP1.scatter(time, P[-1]-P0[-1], marker=mark, facecolor = None, alpha = 0.5, lw = 2*edge_w, edgecolor = 'r')
			axA1.scatter(time, A[-1]-A0[-1], marker=mark, facecolor = None, alpha = 0.5, lw = 2*edge_w, edgecolor = 'r')

		else:
			continue

	for ax in [axS,axP,axA,axS1,axP1,axA1]:
		ax.grid()
		ax.plot(pred_times, np.zeros(len(pred_times)), 'k', lw = 2)

	axS.set_ylabel('S')
	axP.set_ylabel('P')
	axA.set_ylabel('A')
	#axFN.set_ylabel('FN')
	#axFP.set_ylabel('FP')

	#axFN.set_ylim(0,100)
	#axFP.set_ylim(0,100)
	
	axS1.set_ylim(-0.8,0.8)
	axP1.set_ylim(-0.8,0.8)
	axA1.set_ylim(-0.8,0.8)
	

plt.savefig('Figures/TimeZonal/Analysis/'+Figname)









#########################################################################

# PICK UP HERE!!!!!

# For all our images we want one of these staple things

# Run a figure for all the scenes
for i in range(len(prediction_scenes)):

	# Define the scene and time
	scene = prediction_scenes[i]
	time = pred_times[i]
	txt_time = str(time.year)+str(time.month).rjust(2,'0')+str(time.day).rjust(2,'0')
	print (scene, txt_time)

	# define prediction and probability rasters
	pred = np.ma.masked_where(Classif[i]>=255, Classif[i])
	proba = np.ma.masked_where(Pro[i]>=255, Pro[i])
	hs = fn.make_hillshade_arr (proba) # A probability hillshade to make it look cool

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

	
	# plot things
	if exist_survey == True:

		
		# Try a figure with lines
		edges = fn.make_edges_arr (proba)
		ctr = fn.make_contour_arr (proba)
		# This is the figure. Hi figure!    
		fig = plt.figure(i+10, facecolor='White',figsize=[23, 20])
		axa = plt.subplot2grid((2,2), (0,0), rowspan=1, colspan=1)
		axb = plt.subplot2grid((2,2), (0,1), rowspan=1, colspan=1)
		axc = plt.subplot2grid((2,2), (1,0), rowspan=1, colspan=1)
		axd = plt.subplot2grid((2,2), (1,1), rowspan=1, colspan=1)

		axa.imshow(proba, cmap = plt.cm.jet, vmin = 0, vmax = 100)
		axa.imshow(np.ma.masked_where(proba>=255,edges), cmap = plt.cm.Greys, vmin = 0, vmax = 255, alpha = 0.5)
		axb.imshow(np.ma.masked_where(proba>=255,edges), cmap = plt.cm.Greys, vmin = 0, vmax = 255, alpha = 0.5)

		axc.imshow(np.ma.masked_where(proba>=255,kso), cmap = plt.cm.jet, vmin = 0, vmax = 1)
		axc.imshow(np.ma.masked_where(proba>=255,edges), cmap = plt.cm.Greys, vmin = 0, vmax = 255, alpha = 0.5)
		axd.imshow(np.ma.masked_where(proba>=255,ctr), cmap = plt.cm.Greys, vmin = 0, vmax = 255, alpha = 0.5)




		# Make the difference map and performance metrics
		diffmap, S, P, A = fn.make_diff(pred, kso)
		fn.make_staple_plot ('/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/TimeZonal/Analysis/Timeseries/Tri_'+txt_time+'_nofillnodigi.png', proba, pred, hs, zones, S, P, A, diffmap, kso_density, Sepvals)
		#quit()
	else:
		print ('nothing to see here')
		#make_staple_plot ('/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/TimeZonal/Analysis/Timeseries/'+txt_time+'_patch.png', proba, pred, hs)
	



quit()

# The next problem is: how to modify the prediction? Kernels around the individual patches.
# But should the patches be done indenpendantly?
# MAYBE USE ROUGHNESSSSSSSS!!!!







########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################





fig = plt.figure(0, facecolor='White',figsize=[20, 20])
ax =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

ax.imshow(diffmap, cmap = plt.cm.jet, vmin = -2, vmax = 2)

ax.set_xticklabels([])
ax.set_yticklabels([])
plt.tight_layout()
plt.show()
quit()



fig = plt.figure(0, facecolor='White',figsize=[20, 20])
ax =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

ax.imshow(kso, cmap = plt.cm.jet, vmin = 0, vmax = 255)

axis.set_xticklabels([])
axis.set_yticklabels([])
plt.tight_layout()
plt.show()
quit()

# Modelled binary classif
pred = np.ma.masked_where(pred_maps_3[-36]==255, pred_maps_3[-36]) 
# Modelled proba array
proba = np.ma.masked_where(proba_maps_3[-36]==255, proba_maps_3[-36])
# Surveyed binary classif
survey = kso


diffmap, S, P, A = make_diff(pred, kso)



make_staple_plot ('/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/TimeZonal/Analysis/staple.png', proba, pred, hs)
#make_staple_plot (proba, pred, hs, S, P, A, diffmap)






















####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################




























"""
BLUR
CONTOUR
DETAIL
EDGE_ENHANCE
EDGE_ENHANCE_MORE
EMBOSS
FIND_EDGES
SHARPEN
SMOOTH
SMOOTH_MORE
"""

quit()



axis.imshow(np.ma.masked_where(probarray[3,-36]==255, probarray[3,-36]), cmap = plt.cm.jet, vmin = 0, vmax = 100)
axis.set_xticklabels([])
axis.set_yticklabels([])
plt.tight_layout()
plt.show()
quit()

###################################################################################
# What to plot now? Try the inlet scenes
#################################################################################

# initiate a figure
fig = plt.figure(0, facecolor='White',figsize=[20, 20])


# these are the model indices
for M in [0,1,2,3]:

	# Define the properties linked to the model image
	modelscene = Model_scenes[M]
	modeltime = Model_times[M]
	modelused = Models[M]
	exist_modelused = Existence[M]
	sepvals = Sepvals[M]
	print (M, '// This is the model time:', modeltime)

	# Define the model axis
	ax =  plt.subplot2grid((4,1),(M,0),colspan=1, rowspan=1)

	pos = 0 # initial position
	for i in range(len(Inletscenes)):

		# define the scene
		scene = Inletscenes[i]
		# Find scene time
		this_time = ml.gettime(overpass_times, scene)

		# Put some text to indicate the year
		ax.annotate(str(this_time.year)+'-'+str(this_time.month), xy = (pos+1, 103), xycoords = 'data')

		kso, kso_density, pxw, geodata = ml.load_vegetation_data(veg_dir, this_time, mask) # careful, you have to load the inlet veg OK

		# load an additional mask for the aoi
		aoi, pixelWidth, (geotransform, inDs) = ml.ENVI_raster_binary_to_3d_array('/home/willgoodwin/PostDoc/Padova/Data/Basemaps/Inlets_AoI_laguna.bil')
		aoi = aoi[:-1]

		# Mask the kso with a raster of aoi
		kso_density = np.ma.masked_where(aoi < 1, kso_density)

		southernmost = 0
		# Look at each zone	
		for Z in Zone_vals:
			print ("Zone", Z)
			# Identify patches for each density
			for D in Density_vals:
				print ('finding contiguous patches')
				contiguous_area, patch_id = ml.getContiguousAreas(kso_density, D, min_pixels = 2)

				# For each patch
				for p in range(1,np.amax(patch_id)+1):
					# these are the patch indices
					pidx = np.where(patch_id == p)

					here_Z = zones[pidx][0]
					if here_Z == Z:
						print ('patch number:', p, 'of', np.amax(patch_id)+1)


						X = probarray[M,Surv_idx[M]]
						XX = X[pidx]

						Q25 = np.percentile(XX,25)
						Q50 = np.percentile(XX,50)
						Q75 = np.percentile(XX,75)

						# plot the sepval
						# plot the sepval
						if D != 0:
							ax.fill_between ([pos-1,pos+1], 0, [100*sepvals[Z-1],100*sepvals[Z-1]], facecolor='k', lw=0, alpha = 0.05)
						else:
							ax.fill_between ([pos-1,pos+1], [100*sepvals[Z-1],100*sepvals[Z-1]], 100, facecolor='k', lw=0, alpha = 0.05)
						# plot the quartiles
						ax.plot ([pos,pos], [Q25,Q75], c = plt.cm.jet(D/4), lw = 0.7, alpha = 0.6)
						# plot the median
						ax.scatter (pos, Q50, facecolor = plt.cm.jet(D/4), marker = 'o', lw = 0, s = 10 + 2*np.sqrt(len(pidx[0])), alpha = 0.6)
						# Move one step right
						pos += 1
				# Change density value
				pos += 5
			# Change zone and plot a big line
			pos += 10
			if Z != 5:
				ax.plot ([pos-5,pos-5], [-5,105], c = 'k', lw = 1.5)

		# Put a line for each year
		pos += 10
		ax.plot([pos-5,pos-5], [-5,105], 'k', lw = 4.5)

		ax.set_xlim(-5,pos + 10)		

		#plt.show()
		#quit()

	# Change panel
	#counter += 1
	ax.set_ylim(-5,105)
	ax.set_xlim(-2,pos-8)

	ax.set_ylabel(' Inlet P_sg for model '+str(M))





plt.savefig('/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/TimeZonal/Analysis/Inlet'+str(M)+'_patch.png')
quit()



###################################################################################
# What to plot now? Try the digitised scenes
###################################################################################

# initiate a figure
fig = plt.figure(0, facecolor='White',figsize=[40, 20])


# these are the model indices
for M in [0,1,2,3]:


	# Define the properties linked to the model image
	modelscene = Model_scenes[M]
	modeltime = Model_times[M]
	modelused = Models[M]
	exist_modelused = Existence[M]
	sepvals = Sepvals[M]
	print (M, '// This is the model time:', modeltime)

	# Define the model axis
	ax =  plt.subplot2grid((4,1),(M,0),colspan=1, rowspan=1)

	pos = 0 # initial position
	for i in range(len(Digiscenes)):

		# define the scene
		scene = Digiscenes[i]
		# Find scene time
		this_time = ml.gettime(overpass_times, scene)

		# Put some text to indicate the year
		ax.annotate(str(this_time.year)+'-'+str(this_time.month), xy = (pos+1, 103), xycoords = 'data')

		kso, kso_density, pxw, geodata = ml.load_digitised_vegetation_data(dwld_dir+scene+'/Digi/', this_time, mask) # careful, you have to load the inlet veg OK

		# load an additional mask for the aoi
		aoi, pixelWidth, (geotransform, inDs) = ml.ENVI_raster_binary_to_3d_array('/home/willgoodwin/PostDoc/Padova/Data/Basemaps/Digi_AoI_laguna.bil')
		aoi = aoi[:-1]

		# Mask the kso with a raster of aoi
		kso_density = np.ma.masked_where(aoi < 1, kso_density)

		southernmost = 0
		# Look at each zone	
		for Z in Zone_vals:
			print ("Zone", Z)
			# Identify patches for each density
			for D in Density_vals:
				print ('finding contiguous patches')
				contiguous_area, patch_id = ml.getContiguousAreas(kso_density, D, min_pixels = 2)

				# For each patch
				for p in range(1,np.amax(patch_id)+1):
					# these are the patch indices
					pidx = np.where(patch_id == p)

					here_Z = zones[pidx][0]
					if here_Z == Z:
						print ('patch number:', p, 'of', np.amax(patch_id)+1)



						X = probarray[M,Surv_idx[M]]
						XX = X[pidx]

						Q25 = np.percentile(XX,25)
						Q50 = np.percentile(XX,50)
						Q75 = np.percentile(XX,75)

						# plot the sepval
						if D != 0:
							ax.fill_between ([pos-5,pos+5], 0, [100*sepvals[Z-1],100*sepvals[Z-1]], facecolor='k', lw=0, alpha = 0.2)
						else:
							ax.fill_between ([pos-5,pos+5], [100*sepvals[Z-1],100*sepvals[Z-1]], 100, facecolor='k', lw=0, alpha = 0.2)
						# plot the quartiles
						ax.plot ([pos,pos], [Q25,Q75], c = plt.cm.jet(D/4), lw = 0.7, alpha = 0.6)
						# plot the median
						ax.scatter (pos, Q50, facecolor = plt.cm.jet(D/4), marker = 'o', lw = 0, s = 10 + 2*np.sqrt(len(pidx[0])), alpha = 0.6)
						# Move one step right
						pos += 1
				# Change density value
				pos += 5
			# Change zone and plot a big line
			pos += 10
			if Z != 5:
				ax.plot ([pos-5,pos-5], [-5,105], c = 'k', lw = 1.5)

		# Put a line for each year
		pos += 10
		ax.plot([pos-5,pos-5], [-5,105], 'k', lw = 4.5)

		ax.set_xlim(-5,pos + 10)		

		#plt.show()
		#quit()

	# Change panel
	#counter += 1
	ax.set_ylim(-5,105)
	ax.set_xlim(-2,pos-8)

	ax.set_ylabel(' Digitised P_sg for model '+str(M))



	plt.savefig('/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/TimeZonal/Analysis/Digi'+str(M)+'.png')
	quit()




#########################################################################################
# Plot patch-wise probas per zone and per density for all models
#########################################################################################

###
# How to properly add the 
###

# these are the model indices
for M in [0,1,2,3]:

	# initiate a figure
	fig = plt.figure(M, facecolor='White',figsize=[20, 20])

	# Define the properties linked to the model image
	modelscene = Model_scenes[M]
	modeltime = Model_times[M]
	modelused = Models[M]
	exist_modelused = Existence[M]
	sepvals = Sepvals[M]


	print (M, '// This is the model time:', modeltime)

	#Define the Known State 0 (kso) index
	kso_idx = np.where(np.asarray(prediction_scenes) == modelscene)[0][0]

	# This is the vegetation state at kso_idx
	kso, kso_density, pxw, geodata = ml.load_vetted_vegetation_data(veg_dir, modeltime, mask)

	# Remove the current one from the test indices
	#Surv_idx.remove(kso_idx)

	counter = 0
	for m in [0,1,2,3]:
		if m != M:
			print (m)

			# Make one axis per tested model
			ax =  plt.subplot2grid((3,1),(counter,0),colspan=1, rowspan=1)
			if counter == 0:
				ax.set_title('Survey '+ str(M))

			# Look at each zone	
			pos = 0	
			for Z in Zone_vals:
				print ("Zone", Z)

				# Identify patches for each density
				for D in Density_vals:
					print ('finding contiguous patches')
					contiguous_area, patch_id = ml.getContiguousAreas(kso_density, D, min_pixels = 2)

					"""
					fig = plt.figure(M, facecolor='White',figsize=[20, 20])
					axis =  plt.subplot2grid((1,1),(counter,0),colspan=1, rowspan=1)
					axis.imshow(kso_density, cmap = plt.cm.jet, vmin = 0, vmax = 5)
					plt.show()
					quit()
					"""		

					# For each patch
					for p in range(1,np.amax(patch_id)+1):
						# these are the patch indices
						pidx = np.where(patch_id == p)

						here_Z = zones[pidx][0]
						if here_Z == Z:
							print ('patch number:', p, 'of', np.amax(patch_id)+1)

							X = probarray[m,Surv_idx[m]]
							XX = X[pidx]

							Q25 = np.percentile(XX,25)
							Q50 = np.percentile(XX,50)
							Q75 = np.percentile(XX,75)

							ax.plot ([pos-5,pos+5], [100*sepvals[Z-1],100*sepvals[Z-1]], c='k', lw=2)
							print (sepvals[Z-1])
							ax.plot ([pos,pos], [Q25,Q75], c = plt.cm.jet(D/4), lw = 0.7)
							ax.scatter (pos, Q50, facecolor = plt.cm.jet(D/4), marker = 'o', lw = 0, s = 10 + 2*np.sqrt(len(pidx[0])))
							# Move one step right
							pos += 1

				# Change zone
				pos += 10
				ax.plot ([pos-5,pos-5], [-5,105], c = 'k', lw = 2.5)


			# Change panel
			counter += 1
			ax.set_ylim(-5,105)
			ax.set_xlim(-2,pos-8)

			ax.set_ylabel('P_sg for model '+str(m))


	plt.savefig('/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/TimeZonal/Analysis/Comp2Surv'+str(M)+'.png')
	print ('Figure saved')

quit()

"""
Next steps

"""


for Z in Zone_vals:
	# Identify patches for each density
	
	for m in range(len(probarray)):
		ax =  plt.subplot2grid((4,5),(m,Z-1),colspan=1, rowspan=1)

		for D in Density_vals:

			print ('finding contiguous patches')
			contiguous_area, patch_id = ml.getContiguousAreas(kso_density, D, min_pixels = 2)
			# Look at each model's predictions
			

			# For each patch
			for p in range(1,np.amax(patch_id)+1):
				print ('patch number:', p, 'of', np.amax(patch_id)+1)
				# these are the patch indices
				pidx = np.where(patch_id == p)

				here_Z = zones[pidx][0]

				if here_Z == Z:

					# Now make some stats inside the patch
					Q25 = np.zeros(len(probarray[0]))
					Q50 = np.zeros(len(probarray[0]))
					Q75 = np.zeros(len(probarray[0]))

					Q25all = np.zeros(len(probarray[0]))
					Q50all = np.zeros(len(probarray[0]))
					Q75all = np.zeros(len(probarray[0]))

					for t in range(len(probarray[0])):
						X = probarray[m,t]
						XX = X[pidx]

						Xall = probarray[:,t]
						XXall = Xall[:,pidx[0], pidx[1]]

						Q25[t] = np.percentile(XX,25)
						Q50[t] = np.percentile(XX,50)
						Q75[t] = np.percentile(XX,75)

						Q25all[t] = np.percentile(XXall,25)
						Q50all[t] = np.percentile(XXall,50)
						Q75all[t] = np.percentile(XXall,75)

					# Make a 5 to 10-img moving average
					L = 10
					Q50 = np.convolve(Q50, np.ones(L), 'valid') / L
					Q25 = np.convolve(Q25, np.ones(L), 'valid') / L
					Q75 = np.convolve(Q75, np.ones(L), 'valid') / L

					Q50all = np.convolve(Q50all, np.ones(L), 'valid') / L
					Q25all = np.convolve(Q25all, np.ones(L), 'valid') / L
					Q75all = np.convolve(Q75all, np.ones(L), 'valid') / L

					#axes[m].plot (pred_times[L-1:], (Q75-Q25) / np.sqrt(len(pidx[0])), plt.cm.jet(p/np.amax(patch_id)+1), lw = 0.1 + 0.005* np.sqrt(len(pidx[0])), alpha = 0.5)
					ax.plot (pred_times[L-1:], Q50-Q50all, c = plt.cm.jet(D/4), lw = 0.2 + D/2, alpha = 0.7)


		ax.set_ylim(-30,30)
		#ax.set_ylim(0,100)
		ax.set_xlim(datetime(1999,1,1),datetime(2020,1,1))


plt.savefig('/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/TimeZonal/Analysis/PatchZone2004_comp.png')
quit()
#########################################################################
#########################################################################
#########################################################################
#########################################################################


#########################################################################################
# Plot pixel trajectory for all models and compare models
#########################################################################################

# Plot pixel trajectories through time
fig = plt.figure(1, facecolor='White',figsize=[20, 20])

for Z in Zone_vals:
	# Identify patches for each density
	
	for m in range(len(probarray)):
		ax =  plt.subplot2grid((4,5),(m,Z-1),colspan=1, rowspan=1)

		for D in Density_vals:

			print ('finding contiguous patches')
			contiguous_area, patch_id = ml.getContiguousAreas(kso_density, D, min_pixels = 2)
			# Look at each model's predictions
			

			# For each patch
			for p in range(1,np.amax(patch_id)+1):
				print ('patch number:', p, 'of', np.amax(patch_id)+1)
				# these are the patch indices
				pidx = np.where(patch_id == p)

				here_Z = zones[pidx][0]

				if here_Z == Z:

					# Now make some stats inside the patch
					Q25 = np.zeros(len(probarray[0]))
					Q50 = np.zeros(len(probarray[0]))
					Q75 = np.zeros(len(probarray[0]))

					Q25all = np.zeros(len(probarray[0]))
					Q50all = np.zeros(len(probarray[0]))
					Q75all = np.zeros(len(probarray[0]))

					for t in range(len(probarray[0])):
						X = probarray[m,t]
						XX = X[pidx]

						Xall = probarray[:,t]
						XXall = Xall[:,pidx[0], pidx[1]]

						Q25[t] = np.percentile(XX,25)
						Q50[t] = np.percentile(XX,50)
						Q75[t] = np.percentile(XX,75)

						Q25all[t] = np.percentile(XXall,25)
						Q50all[t] = np.percentile(XXall,50)
						Q75all[t] = np.percentile(XXall,75)

					# Make a 5 to 10-img moving average
					L = 10
					Q50 = np.convolve(Q50, np.ones(L), 'valid') / L
					Q25 = np.convolve(Q25, np.ones(L), 'valid') / L
					Q75 = np.convolve(Q75, np.ones(L), 'valid') / L

					Q50all = np.convolve(Q50all, np.ones(L), 'valid') / L
					Q25all = np.convolve(Q25all, np.ones(L), 'valid') / L
					Q75all = np.convolve(Q75all, np.ones(L), 'valid') / L

					#axes[m].plot (pred_times[L-1:], (Q75-Q25) / np.sqrt(len(pidx[0])), plt.cm.jet(p/np.amax(patch_id)+1), lw = 0.1 + 0.005* np.sqrt(len(pidx[0])), alpha = 0.5)
					ax.plot (pred_times[L-1:], Q50-Q50all, c = plt.cm.jet(D/4), lw = 0.2 + D/2, alpha = 0.7)


		ax.set_ylim(-30,30)
		#ax.set_ylim(0,100)
		ax.set_xlim(datetime(1999,1,1),datetime(2020,1,1))


plt.savefig('/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/TimeZonal/Analysis/PatchZone2004_comp.png')
quit()



"""
Safe to say there's a very poor relationship between density and proba ...

What's next?


Compare model proba to  agreement with survey results and density



"""



# Plot pixel trajectories through time
fig = plt.figure(1, facecolor='White',figsize=[20, 20])

for D in Density_vals:
	# Identify patches for each density
	print ('finding contiguous patches')
	contiguous_area, patch_id = ml.getContiguousAreas(kso_density, D, min_pixels = 2)

	for m in range(len(probarray)):
		# Look at each model's predictions
		ax =  plt.subplot2grid((4,4),(m,D-1),colspan=1, rowspan=1)

		# For each patch
		for p in range(1,np.amax(patch_id)+1):
			print ('patch number:', p, 'of', np.amax(patch_id)+1)
			# these are the patch indices
			pidx = np.where(patch_id == p)

			Z = zones[pidx][0]

			# Now make some stats inside the patch
			Q25 = np.zeros(len(probarray[0]))
			Q50 = np.zeros(len(probarray[0]))
			Q75 = np.zeros(len(probarray[0]))

			Q25all = np.zeros(len(probarray[0]))
			Q50all = np.zeros(len(probarray[0]))
			Q75all = np.zeros(len(probarray[0]))

			for t in range(len(probarray[0])):
				X = probarray[m,t]
				XX = X[pidx]

				Xall = probarray[:,t]
				XXall = Xall[:,pidx[0], pidx[1]]

				Q25[t] = np.percentile(XX,25)
				Q50[t] = np.percentile(XX,50)
				Q75[t] = np.percentile(XX,75)

				Q25all[t] = np.percentile(XXall,25)
				Q50all[t] = np.percentile(XXall,50)
				Q75all[t] = np.percentile(XXall,75)

			# Make a 5 to 10-img moving average
			L = 10
			Q50 = np.convolve(Q50, np.ones(L), 'valid') / L
			Q25 = np.convolve(Q25, np.ones(L), 'valid') / L
			Q75 = np.convolve(Q75, np.ones(L), 'valid') / L

			Q50all = np.convolve(Q50all, np.ones(L), 'valid') / L
			Q25all = np.convolve(Q25all, np.ones(L), 'valid') / L
			Q75all = np.convolve(Q75all, np.ones(L), 'valid') / L

			#axes[m].plot (pred_times[L-1:], (Q75-Q25) / np.sqrt(len(pidx[0])), plt.cm.jet(p/np.amax(patch_id)+1), lw = 0.1 + 0.005* np.sqrt(len(pidx[0])), alpha = 0.5)
			ax.plot (pred_times[L-1:], Q50, c = plt.cm.jet(Z/5), lw = 0.2 + 0.005* np.sqrt(len(pidx[0])), alpha = 1)



		#ax.set_ylim(-30,30)
		ax.set_ylim(0,100)
		ax.set_xlim(datetime(1999,1,1),datetime(2020,1,1))


plt.savefig('/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/TimeZonal/Analysis/PatchDensity2004.png')
quit()











####################################################################################
####################################################################################
####################################################################################
# Plot pixel trajectories through time
fig = plt.figure(1, facecolor='White',figsize=[10, 8])
ax0 =  plt.subplot2grid((4,5),(0,0),colspan=2, rowspan=2)
ax1 =  plt.subplot2grid((4,5),(0,2),colspan=2, rowspan=2)
ax2 =  plt.subplot2grid((4,5),(2,0),colspan=2, rowspan=2)
ax3 =  plt.subplot2grid((4,5),(2,2),colspan=2, rowspan=2)
axes = [ax0,ax1,ax2,ax3]


MQ25 = []
MQ50 = []
MQ75 = []


#Interquartile range!  and show differences between models


for m in range(len(probarray)):

	# Get rid of this step first (i.e. ignoring the annoying empty strips of L7)
	print ('ignoring null strips')
	for t in range(1,len(probarray[0])):
		Tprob = probarray[m,t]
		Tprev = probarray[m,t-1]
		W = np.where(Tprob == 255)
		Tprob[W] = Tprev[W]

		probarray[m,t] = Tprob



	# Define the properties linked to the model image
	modelscene = Model_scenes[m]

	modeltime = Model_times[m]
	# Force them same patches as those found in 2004
	modeltime = Model_times[0]

	modelused = Models[m]
	exist_modelused = Existence[m]

	print (m, '// This is the model time:', modeltime)

	#Define the Known State 0 (kso) index
	kso_idx = np.where(np.asarray(prediction_scenes) == modelscene)[0][0]

	# This is the vegetation state at kso_idx
	kso, kso_density, pxw, geodata = ml.load_vetted_vegetation_data(veg_dir, modeltime, mask)

	# Identify patches
	print ('finding contiguous patches')
	contiguous_area, patch_id = ml.getContiguousAreas(kso, 1, min_pixels = 2)

	# For each patch
	for p in range(1,np.amax(patch_id)+1):
		print ('patch number:', p, 'of', np.amax(patch_id)+1)
		# these are the patch indices
		pidx = np.where(patch_id == p)

		# Now make some stats inside the patch
		Q25 = np.zeros(len(probarray[0]))
		Q50 = np.zeros(len(probarray[0]))
		Q75 = np.zeros(len(probarray[0]))

		for t in range(len(probarray[0])):
			X = probarray[m,t]
			XX = X[pidx]

			Q25[t] = np.percentile(XX,25)
			Q50[t] = np.percentile(XX,50)
			Q75[t] = np.percentile(XX,75)

		# Make a 5 to 10-img moving average
		L = 10
		Q50 = np.convolve(Q50, np.ones(L), 'valid') / L
		Q25 = np.convolve(Q25, np.ones(L), 'valid') / L
		Q75 = np.convolve(Q75, np.ones(L), 'valid') / L
		
		#axes[m].plot (pred_times[L-1:], (Q75-Q25) / np.sqrt(len(pidx[0])), plt.cm.jet(p/np.amax(patch_id)+1), lw = 0.1 + 0.005* np.sqrt(len(pidx[0])), alpha = 0.5)
		axes[m].plot (pred_times[L-1:], Q50, plt.cm.jet(p/np.amax(patch_id)+1), lw = 0.1 + 0.005* np.sqrt(len(pidx[0])), alpha = 0.5)

		axes[m].set_ylim(0,100)
		axes[m].set_xlim(datetime(1999,1,1),datetime(2020,1,1))

		MQ25.append(Q25)
		MQ50.append(Q50)
		MQ75.append(Q75)

plt.savefig('/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/TimeZonal/Analysis/Model0_patches.png')

	
	











quit()
for i in range(len(probarray[0])):

	# this is already zone-specific
	thisproba = probarray[:,i,:,:]

	med = np.percentile(thisproba, 50, axis = 0)
	diff2med = probarray[:,i,:,:] - med

	
	fig = plt.figure(1, facecolor='White',figsize=[10, 8])
	ax0 =  plt.subplot2grid((4,5),(0,0),colspan=2, rowspan=2)
	ax1 =  plt.subplot2grid((4,5),(0,2),colspan=2, rowspan=2)
	ax2 =  plt.subplot2grid((4,5),(2,0),colspan=2, rowspan=2)
	ax3 =  plt.subplot2grid((4,5),(2,2),colspan=2, rowspan=2)

	axa = plt.subplot2grid((4,5),(0,4),colspan=1, rowspan=1)
	axb = plt.subplot2grid((4,5),(1,4),colspan=1, rowspan=1)
	axc = plt.subplot2grid((4,5),(2,4),colspan=1, rowspan=1)
	axd = plt.subplot2grid((4,5),(3,4),colspan=1, rowspan=1)

	ax0.imshow(np.ma.masked_where(med==255, diff2med[0]), cmap = plt.cm.seismic_r, vmin = -100, vmax = 100)
	ax1.imshow(np.ma.masked_where(med==255, diff2med[1]), cmap = plt.cm.seismic_r, vmin = -100, vmax = 100)
	ax2.imshow(np.ma.masked_where(med==255, diff2med[2]), cmap = plt.cm.seismic_r, vmin = -100, vmax = 100)
	ax3.imshow(np.ma.masked_where(med==255, diff2med[3]), cmap = plt.cm.seismic_r, vmin = -100, vmax = 100)

	for ax in [ax0,ax1,ax2,ax3]:
		ax.set_xticklabels([]); ax.set_yticklabels([])

	for z in range(2,5):
		rm = mask.ravel()
		rz = zones.ravel()

		rd0 = diff2med[0].ravel()
		rd1 = diff2med[1].ravel()
		rd2 = diff2med[2].ravel()
		rd3 = diff2med[3].ravel()

		rdz0 = rd0[rz == z]
		rdz1 = rd1[rz == z]
		rdz2 = rd2[rz == z]
		rdz3 = rd3[rz == z]

		axa.hist(np.ma.masked_where(rdz0==255, rdz0), facecolor = plt.cm.viridis(z/4), alpha = 0.5, bins = 50, density = False)
		axb.hist(np.ma.masked_where(rdz1==255, rdz1), facecolor = plt.cm.viridis(z/4), alpha = 0.5, bins = 50, density = False)
		axc.hist(np.ma.masked_where(rdz2==255, rdz2), facecolor = plt.cm.viridis(z/4), alpha = 0.5, bins = 50, density = False)
		axd.hist(np.ma.masked_where(rdz3==255, rdz3), facecolor = plt.cm.viridis(z/4), alpha = 0.5, bins = 50, density = False)

	for ax in [axa,axb,axc,axd]:
		ax.set_ylim(0,10000)
		ax.set_xlim(-100,100)
		ax.grid()

	plt.tight_layout()
	plt.show()
	


	quit()
	if i > 8:
		quit()

	#blah = 0







quit()

##################################################################################
#TEST ZONE
##################################################################################
"""
from PIL import Image, ImageFilter, ImageDraw

# This is our scene
scene = prediction_scenes[0]; print (scene)



b = proba_maps_0[0]
#b = np.ma.masked_where(b == 255, b)
b[b == 255] = 0

bg = np.gradient(b)
bg = np.asarray(bg)
bg = np.amax(abs(bg), axis = 0)
#bg [bg >1] = 0


bx = b**2*bg**(1/2)

#bg2 =  bg - np.mean(bg) 

print (bg)
print (bg.shape)



image = Image.fromarray(np.uint8(plt.cm.jet(bx/np.amax(bx))*255))

#image = im.filter(ImageFilter.FIND_EDGES) 
im = image.filter((ImageFilter.SMOOTH_MORE))
im = im.filter((ImageFilter.CONTOUR))
im = im.filter((ImageFilter.EMBOSS))
#im = im.filter((ImageFilter.FIND_EDGES))
# Saving the Image Under the name Edge_Sample.png 
#im.save(r"/home/willgoodwin/Example_Imgbis.png")

imp = im.convert('P')
arr = np.array(imp)


fig = plt.figure(1, facecolor='White',figsize=[10, 8])
ax0 =  plt.subplot2grid((2,4),(0,0),colspan=2, rowspan=1)
ax1 =  plt.subplot2grid((2,4),(0,2),colspan=2, rowspan=1)
ax2 =  plt.subplot2grid((2,4),(1,0),colspan=2, rowspan=1)
ax3 =  plt.subplot2grid((2,4),(1,2),colspan=2, rowspan=1)

ax0.imshow(b, cmap = plt.cm.jet)
ax1.imshow(bg, cmap = plt.cm.jet)
ax2.imshow(bx, cmap = plt.cm.jet)
ax3.imshow(im, cmap = plt.cm.jet)


plt.tight_layout()
plt.show()

quit()


b = np.ma.masked_where(b == 255, b)

bm = np.ma.array(b, fill_value=0)
bm = bm.filled()
  
im = Image.fromarray(np.uint8(plt.cm.jet(bm/np.amax(bm))*255))

im.save(r"/home/willgoodwin/Example_Img.png")


#image = im.filter(ImageFilter.FIND_EDGES) 
im = im.filter((ImageFilter.SMOOTH_MORE))
im = im.filter((ImageFilter.CONTOUR))
#im = im.filter((ImageFilter.FIND_EDGES))
# Saving the Image Under the name Edge_Sample.png 
im.save(r"/home/willgoodwin/Example_Imgbis.png")

imp = im.convert('P')
arr = np.array(imp)
print (arr.shape)


# Opening the image (R prefixed to 
# string in order to deal with '\' 
# in paths) 
#img = Image.open(R"sample.png") 
  
# Converting the image to RGB mode 
#img1 = img.convert("RGB")  
  
# Coordinates of the pixel whose value 
# would be used as seed 
seed = (700, 555) 
   
# Pixel Value which would be used for 
# replacement  
rep_value = (255, 255, 0) 
   
# Calling the floodfill() function and  
# passing it image, seed, value and  
# thresh as arguments 
#ImageDraw.floodfill(im, seed, rep_value, thresh=50) 

#im.show()

# https://www.programcreek.com/python/example/57117/PIL.ImageFilter.CONTOUR
# https://stackoverflow.com/questions/7350588/how-to-calculate-slope-in-numpy

imp2 = im.convert('P')
# Displaying the image 
#im2 = ImageDraw.floodfill(im)
arr2 = np.array(imp2)


#https://stackoverflow.com/questions/46083880/fill-in-a-hollow-shape-using-python-and-pillow-pil


fig = plt.figure(1, facecolor='White',figsize=[10, 8])
ax0 =  plt.subplot2grid((2,4),(0,0),colspan=2, rowspan=1)
ax1 =  plt.subplot2grid((2,4),(0,2),colspan=2, rowspan=1)
ax2 =  plt.subplot2grid((2,4),(1,0),colspan=2, rowspan=1)
ax3 =  plt.subplot2grid((2,4),(1,2),colspan=2, rowspan=1)

ax0.imshow(bm, cmap = plt.cm.jet)
ax1.imshow(arr, cmap = plt.cm.jet)
ax2.imshow(arr2, cmap = plt.cm.jet)
#ax2.imshow(bn_img, cmap = plt.cm.jet)


plt.tight_layout()
plt.show()






bm = proba_maps_0[0]
bm = np.ma.masked_where(bm == 255, bm)

img_sobel = scipy.ndimage.filters.sobel(bm)
bn_img = np.zeros([img_sobel.shape[0],img_sobel.shape[1]])
sbl_max = np.amax(abs(img_sobel))
bn_img = np.abs(img_sobel) >= (1.1)


fig = plt.figure(2, facecolor='White',figsize=[10, 8])
ax0 =  plt.subplot2grid((2,4),(0,0),colspan=2, rowspan=1)
ax1 =  plt.subplot2grid((2,4),(0,2),colspan=2, rowspan=1)
ax2 =  plt.subplot2grid((2,4),(1,0),colspan=2, rowspan=1)
ax3 =  plt.subplot2grid((2,4),(1,2),colspan=2, rowspan=1)

ax0.imshow(bm, cmap = plt.cm.jet)
ax1.imshow(img_sobel, cmap = plt.cm.jet)
ax2.imshow(bn_img, cmap = plt.cm.jet)


plt.tight_layout()
plt.show()




quit()

"""


#############################################################################
#############################################################################
#############################################################################
"""
bm = proba_maps_0[0]
bm = np.ma.masked_where(bm == 255, bm)

img_sobel = scipy.ndimage.filters.sobel(bm)
bn_img = np.zeros([img_sobel.shape[0],img_sobel.shape[1]])
sbl_max = np.amax(abs(img_sobel))
bn_img = np.abs(img_sobel) >= (1.1)


fig = plt.figure(1, facecolor='White',figsize=[10, 8])
ax0 =  plt.subplot2grid((2,4),(0,0),colspan=2, rowspan=1)
ax1 =  plt.subplot2grid((2,4),(0,2),colspan=2, rowspan=1)
ax2 =  plt.subplot2grid((2,4),(1,0),colspan=2, rowspan=1)
ax3 =  plt.subplot2grid((2,4),(1,2),colspan=2, rowspan=1)

ax0.imshow(bm, cmap = plt.cm.jet)
ax1.imshow(img_sobel, cmap = plt.cm.jet)
ax2.imshow(bn_img, cmap = plt.cm.jet)


plt.tight_layout()
plt.show()





quit()


# Opening the image (R prefixed to string 
# in order to deal with '\' in paths) 
image = Image.open(r"/home/willgoodwin/Pictures/12890357_1600x1200.jpg") 
  
# Converting the image to greyscale, as edge detection  
# requires input image to be of mode = Greyscale (L) 
image = image.convert("L") 

print(image)

quit()
  
# Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES 
image = image.filter(ImageFilter.FIND_EDGES) 
  
# Saving the Image Under the name Edge_Sample.png 
image.save(r"/home/willgoodwin/Edge_Sample.png")


PIL_image = Image.fromarray(np.uint8(numpy_image)).convert('RGB')

PIL_image = Image.fromarray(numpy_image.astype('uint8'), 'RGB')

from PIL import Image
from matplotlib import cm
im = Image.fromarray(np.uint8(cm.gist_earth(myarray)*255))

quit()

"""

#############################################################################
#############################################################################
#############################################################################


##################################################################################
# Test zone ends here
##################################################################################








#############################################################################
# Start the behaviour model
#############################################################################
# Now we can modify the predictions based on the behaviour model

skip = True

if skip == False:
	final_maps0 = []
	final_maps1 = []
	final_maps2 = []
	final_maps3 = []

	for m in range(len(Model_scenes)):
		# Define some variables
		modelscene = Model_scenes[m]
		modeltime = Model_times[m]
		modelused = Models[m]
		exist_modelused = Existence[m]

		print (m, '// This is the model time:', modeltime)

		#Define the Known State 0 (kso) index
		kso_idx = np.where(np.asarray(prediction_scenes) == modelscene)[0][0]

		# Initiate predictions and probabilities
		Pred = []; Proba = []



		# This is the initial state
		ini_state, density, pxw, geodata = ml.load_vetted_vegetation_data(veg_dir, modeltime, mask)
		if m == 0: final_maps0.append(ini_state)
		if m == 1: final_maps1.append(ini_state)
		if m == 2: final_maps2.append(ini_state)
		if m == 3: final_maps3.append(ini_state)


		#############################################################################
		# Start the timeseries analysis and prediction stabilisation


		# Keep track of degradation. Maybe do the same later for growth
		# This counts the days/scenes since a pixel changed
		daydeg = np.zeros(pred_maps_0[0].shape)
		scenedeg = np.zeros(pred_maps_0[0].shape)
		# This keeps in memory previous scenes, but is it necessary? Do it anyway for now
		memdeg = np.zeros(pred_maps_0[0].shape)

		# try something smaller
		#daydeg = np.zeros((3,3))
		#scenedeg = np.zeros((3,3))
		#memdeg = np.zeros((3,3))


		# run forward
		for i in range(kso_idx+1,len(prediction_scenes)):
			print (i)
			scene = prediction_scenes[i]
			this_time = ml.gettime(overpass_times, scene)

			deltat = this_time - modeltime; deltat = deltat.days

			# load the previous state kso
			if i == kso_idx+1:
				kso, density, pxw, geodata = ml.load_vetted_vegetation_data(veg_dir, this_time, mask)

				# try with something smaller:
				#kso = np.asarray([[1,1,1],[0,0,0],[0,1,1]])

				# if this is the first step, include it in the final preds
				if m == 0: final_maps0.append(kso)
				if m == 1: final_maps1.append(kso)
				if m == 2: final_maps2.append(kso)
				if m == 3: final_maps3.append(kso)

			else:
				if m == 0: kso = final_maps0[-1]
				if m == 1: kso = final_maps1[-1]
				if m == 2: kso = final_maps2[-1]
				if m == 3: kso = final_maps3[-1]

			# This is the temporary state i
			tsi = np.copy(kso)

			# These are the current state predictions 
			newpred = pred_maps_0[i]
			newproba = proba_maps_0[i]

			# Try something smaller
			#newpred = np.asarray([[1,1,1],[1,0,0],[0,0,0]])
			#newproba = np.asarray([[1,1,1],[1,0,0],[0,0,0]])

			# find the previous vegetated pixels and give them each an identifier
			contiguous_area, patch_id = ml.getContiguousAreas(kso, 1, min_pixels = 2)

			# For each patch, check for total disappearance
			for j in range(1,np.amax(patch_id)+1):
				#print ('patch number', j)

				# these are the patch indices
				idx = np.where(patch_id == j)

				# this is the new prediction for the patch
				predp = newpred[idx]

				# If they are all null, then get rid of patch in the memory of change
				if (predp==0).all():
					memdeg[idx] = 0
					daydeg[idx] += deltat
					scenedeg[idx] += 1

			# degradation change is confirmed if 3*30 <= ndays < 6 * 30 (AND nscenes >= 3 later?) 
			w = np.where(np.logical_and(daydeg >= 90, daydeg < 180))

			S = scenedeg[w]
			if len(S) >=1 :
				for s in (2,int(np.amax(S)+1)):
					Sw = np.where(S == s)[0]
					if len(Sw) > 1:
						Sww = np.array([w[0][Sw],w[1][Sw]])
						final_maps0[-s][Sww] = memdeg[Sww]
			
			print ()



			# get rid of all pixels that have a 0. That's a bit simplistic ...
			#idx = np.where(newpred == 0)
			#tsi[idx] = 0


			# to make a buffer, make kernels using only pixels at the edge of the shape
			#https://gis.stackexchange.com/questions/86033/how-to-buffer-raster-pixels-by-their-values

			# WHAT IF YOU SEPARATE THE LOSSES AND GAINS?
			# Loss seems fairly stable ...


			# The trick will be to successfully discard images



			"""
			# find the previous vegetated pixels and give them each an identifier
			contiguous_area, patch_id = ml.getContiguousAreas(tsi, 1, min_pixels = 3)
			# For each remaining patch, several things happen
			for j in range(1,np.amax(patch_id)+1):

				# these are the patch indices
				idx = np.where(patch_id == j)

				# this is the new prediction for the patch
				predp = newpred[idx]

				# If they are all null, then get rid of patch
				if (predp==0).all():
					tsi[idx] = 0
					daycount[idx] += deltat
		

			Pred.append(tsi)
			"""

			

			if m == 0: final_maps0.append(tsi)
			if m == 1: final_maps1.append(tsi)
			if m == 2: final_maps2.append(tsi)
			if m == 3: final_maps3.append(tsi)

			#if i > kso_idx +20:
				#print (np.asarray(final_maps0))
				#quit()
				#break



			
			
			"""
			fig = plt.figure(1, facecolor='White',figsize=[10, 8])
			ax0 =  plt.subplot2grid((1,2),(0,0),colspan=1, rowspan=1)
			ax1 =  plt.subplot2grid((1,2),(0,1),colspan=1, rowspan=1)

			ax0.imshow(ini_state, cmap = plt.cm.summer_r, vmax = 1)
			#ax1.imshow(newpred, cmap = plt.cm.summer_r, vmax = 1)
			ax1.imshow(tsi, cmap = plt.cm.summer_r, vmax = 1)
			plt.tight_layout()
			plt.show()
			"""
			#quit()

			





		# RUN BACKWARD
		for i in range(kso_idx-1,-1,-1):
			print (i)
			Pred.insert(0,np.zeros(kso.shape))

			if m == 0: final_maps0.insert(0,np.zeros(kso.shape))
			if m == 1: final_maps1.insert(0,np.zeros(kso.shape))
			if m == 2: final_maps2.insert(0,np.zeros(kso.shape))
			if m == 3: final_maps3.insert(0,np.zeros(kso.shape))

		break
		 










"""

Fix the dating of partial simulations so you can see results
Add a temporary array 3 scenes deep to keep temporary changes in memory to be confirmed later
Work more on "slopes" => make quick derivatives and use them to calculate "borders". Maybe instead use scimage object delineation

"""



#quit()


############################################################################################
# Plot the results and check performance
############################################################################################
# List of whole lagoon surveys
lagoon_sy = [2004, 2009, 2010, 2017]
inlets_sy = np.arange(2006,2016)


if skip == False:

	ty = []; tm = []
	for i in range(kso_idx, len(final_maps0)):

		amap0 = final_maps0[i]
		#amap1 = final_maps1[i]
		#amap2 = final_maps2[i]
		#amap3 = final_maps3[i]

		
		ty.append(pred_times[i].year + pred_times[i].month/12)
		tm.append(pred_times[i].month)

		#####################################################################
		# plot the correspondance with full lagoon surveys
		
		if int(ty[-1]) in lagoon_sy and int(tm[-1]) >= 6 and int(tm[-1]) < 10:
			print (int(ty[-1]))

			cover, density, pixelWidth, (geotransform, inDs) = ml.load_vetted_vegetation_data(veg_dir, pred_times[i], mask)

			print ('making the eval figure')
			diffmap, S,P,A = ml.evaluate_binary(amap0, cover, '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/TimeZonal/Maps/Perf0_'+str(pred_times[i].year)+str(pred_times[i].month).rjust(2,'0')+str(pred_times[i].day).rjust(2,'0')+'.png')
			#print ('making the eval figure')
			#diffmap, S,P,A = ml.evaluate_binary(amap1, cover, '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/TimeZonal/Maps/Perf1_'+str(pred_times[i].year)+str(pred_times[i].month).rjust(2,'0')+str(pred_times[i].day).rjust(2,'0')+'.png')
			#print ('making the eval figure')
			#diffmap, S,P,A = ml.evaluate_binary(amap2, cover, '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/TimeZonal/Maps/Perf2_'+str(pred_times[i].year)+str(pred_times[i].month).rjust(2,'0')+str(pred_times[i].day).rjust(2,'0')+'.png')
			#print ('making the eval figure')
			#diffmap, S,P,A = ml.evaluate_binary(amap3, cover, '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/TimeZonal/Maps/Perf3_'+str(pred_times[i].year)+str(pred_times[i].month).rjust(2,'0')+str(pred_times[i].day).rjust(2,'0')+'.png')

			#quit()
		
		#####################################################################
		# plot the correspondance with inlet surveys

		if int(ty[-1]) in inlets_sy and int(tm[-1]) >= 6 and int(tm[-1]) < 10:
			print (int(ty[-1]))

			cover, density, pixelWidth, (geotransform, inDs) = ml.load_vegetation_data(veg_dir, pred_times[i], mask)

			for j in [0,1,2]:
				if j == 0:
					a,b,c,d = 500,600,580,750
					#a,b,c,d = 0,900,0,1200
				elif j == 1:
					a,b,c,d = 830,1010,360,500
				else:
					a,b,c,d = 1260,1380,340,420

				scover = cover[a:b,c:d]
				smap0 = amap0[a:b,c:d]
				#smap1 = amap1[a:b,c:d]
				#smap2 = amap2[a:b,c:d]
				#smap3 = amap3[a:b,c:d]

				print ('making the eval figure')
				ml.evaluate_binary(smap0, scover, '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/TimeZonal/Maps/Inlet'+str(j)+'_Perf0_'+str(int(ty[-1]))+str(int(tm[-1])).rjust(2,'0')+str(pred_times[i].day).rjust(2,'0')+'.png')
				#ml.evaluate_binary(smap1, scover, '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/TimeZonal/Maps/Inlet'+str(j)+'_Perf1_'+str(pred_times[i].year)+str(pred_times[i].month).rjust(2,'0')+str(pred_times[i].day).rjust(2,'0')+'.png')
				#ml.evaluate_binary(smap2, scover, '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/TimeZonal/Maps/Inlet'+str(j)+'_Perf2_'+str(pred_times[i].year)+str(pred_times[i].month).rjust(2,'0')+str(pred_times[i].day).rjust(2,'0')+'.png')
				#ml.evaluate_binary(smap3, scover, '/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/TimeZonal/Maps/Inlet'+str(j)+'_Perf3_'+str(pred_times[i].year)+str(pred_times[i].month).rjust(2,'0')+str(pred_times[i].day).rjust(2,'0')+'.png')

			#quit()
		

		continue
#####################################################################
# Plot the map of sg
ty = []
tm = []
for i in range(0, len(pred_maps_0)):

	amap0 = pred_maps_0[i]
	amap1 = pred_maps_1[i]
	amap2 = pred_maps_2[i]
	amap3 = pred_maps_3[i]
	print (i, 'mapping seagrass scene')
	ty.append(pred_times[i].year + pred_times[i].month/12)
	tm.append(pred_times[i].month)

	fig = plt.figure(1, facecolor='White',figsize=[10, 8])
	ax0 =  plt.subplot2grid((2,5),(0,0),colspan=2, rowspan=1)
	ax1 =  plt.subplot2grid((2,5),(0,2),colspan=2, rowspan=1)
	ax2 =  plt.subplot2grid((2,5),(1,0),colspan=2, rowspan=1)
	ax3 =  plt.subplot2grid((2,5),(1,2),colspan=2, rowspan=1)
	axt = plt.subplot2grid((2,5),(0,4),colspan=1, rowspan=2)

	ax0.imshow(np.ma.masked_where(amap0==255, amap0), cmap = plt.cm.summer_r, vmin = 0, vmax = 1)
	ax1.imshow(np.ma.masked_where(amap1==255, amap1), cmap = plt.cm.summer_r, vmin = 0, vmax = 1)
	ax2.imshow(np.ma.masked_where(amap2==255, amap2), cmap = plt.cm.summer_r, vmin = 0, vmax = 1)
	ax3.imshow(np.ma.masked_where(amap3==255, amap3), cmap = plt.cm.summer_r, vmin = 0, vmax = 1)

	axt.scatter(tm, ty, facecolor = 'k')

	for ax in [ax0,ax1,ax2,ax3]:
		ax.set_xticklabels([]); ax.set_yticklabels([])

	axt.set_xlim(0,13); axt.set_ylim(1998,2021)

	plt.tight_layout()
	plt.savefig('/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/TimeZonal/Maps/'+str(pred_times[i].year)+str(pred_times[i].month).rjust(2,'0')+str(pred_times[i].day).rjust(2,'0')+'.png')


	
	##############################################################################
	amap0 = proba_maps_0[i]
	amap1 = proba_maps_1[i]
	amap2 = proba_maps_2[i]
	amap3 = proba_maps_3[i]

	fig = plt.figure(2, facecolor='White',figsize=[10, 8])
	ax0 =  plt.subplot2grid((2,5),(0,0),colspan=2, rowspan=1)
	ax1 =  plt.subplot2grid((2,5),(0,2),colspan=2, rowspan=1)
	ax2 =  plt.subplot2grid((2,5),(1,0),colspan=2, rowspan=1)
	ax3 =  plt.subplot2grid((2,5),(1,2),colspan=2, rowspan=1)
	axt = plt.subplot2grid((2,5),(0,4),colspan=1, rowspan=2)


	ax0.imshow(np.ma.masked_where(amap0==255, amap0), cmap = plt.cm.summer_r, vmin = 0, vmax = 100)
	ax1.imshow(np.ma.masked_where(amap1==255, amap1), cmap = plt.cm.summer_r, vmin = 0, vmax = 100)
	ax2.imshow(np.ma.masked_where(amap2==255, amap2), cmap = plt.cm.summer_r, vmin = 0, vmax = 100)
	ax3.imshow(np.ma.masked_where(amap3==255, amap3), cmap = plt.cm.summer_r, vmin = 0, vmax = 100)

	axt.scatter(tm, ty, facecolor = 'k')

	for ax in [ax0,ax1,ax2,ax3]:
		ax.set_xticklabels([])
		ax.set_yticklabels([])

	axt.set_xlim(0,13)
	axt.set_ylim(1998,2021)

	plt.tight_layout()
	plt.savefig('/home/willgoodwin/PostDoc/Padova/Scripts/Seagrass_ID_scripts/Figures/TimeZonal/Maps/'+str(pred_times[i].year)+str(pred_times[i].month).rjust(2,'0')+str(pred_times[i].day).rjust(2,'0')+'_proba.png')
	
	#if i == 1:
	#	quit()
	



quit()





# Now we can run the predictions on everything else!
for m in range(len(Model_scenes)):

	print (m)

	modelscene = Model_scenes[m]
	modeltime = Model_times[m]
	modelused = Models[m]
	exist_modelused = Existence[m]

	print ('this is the model time', modeltime)

	#Define the Known State 0 (kso) index
	kso_idx = np.where(np.asarray(prediction_scenes) == modelscene)[0][0]

	# Initiate predictions and probabilities
	Pred = []; Proba = []

	# RUN FORWARD
	for i in range(kso_idx+1,len(prediction_scenes)):

		# This is our scene
		scene = prediction_scenes[i]; print (scene)

		# get time
		this_time = ml.gettime(overpass_times, scene) 

		# Load/make relevant feature rasters
		blue, green, red, nir, bathy, mask, zones, theta_w, theta_v, pixelWidth, (geotransform, inDs) = ml.load_rasters_no_WWTM(dwld_dir, scene, bathyraster, maskraster, zoneraster)

		"""
		fig = plt.figure(1, facecolor='White',figsize=[10, 8])
		ax0 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)
		ax0.imshow(blue, cmap = plt.cm.summer_r, vmin = 0.05, vmax = .1)
		plt.tight_layout()
		plt.show()

		quit()
		"""

		H_unique = ml.extract_tide(tides, bathy, this_time)
		loc_rows, loc_cols = ml.makerowcol(blue)

		# define the features
		features = np.array([blue, green, red, nir, bathy-H_unique, loc_rows, loc_cols, zones]) #zones = temp

		# load the previous prediction
		if i == kso_idx+1:
			prev, density, pxw, geodata = ml.load_vetted_vegetation_data(veg_dir, this_time, mask)
		else:
			prev = Pred[-1]

		print (prev)
		print (np.amax(prev))





	# RUN BACKWARD
	for i in range(kso_idx-1,-1,-1):
		print (i)

	quit()
	


