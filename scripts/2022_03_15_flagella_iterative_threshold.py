#%% Import all necessary libraries
import sys
sys.path.insert(0, './modules')
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})
from sklearn.decomposition import PCA
import napari
import msd
from skimage import measure
from naparimovie import Movie
from pathlib import Path
import scipy.signal
from scipy import optimize
import cv2 
import os.path
import pickle
import time

# time settings in the light sheet
pxum = 0.115
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
exp3D_ms = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 
pxum = 0.115

# load raw images
# search_dir = Path(r"\\10.206.26.21\opm2\franky-sample-images")

setName = 'suc-70'

this_file_dir = os.path.join(os.path.dirname(os.path.abspath("./")),
                            'Dropbox (ASU)','Research')
loadFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Flagella-data', setName)

intensityFiles = list(Path(loadFolder).glob("*.npy"))
# thresholdFiles = list(Path(savingFolder).glob("*threshold.npy"))
whichFiles = intensityFiles[37]

imgs = da.from_npy_stack(whichFiles)
# imgs_thresh = da.from_npy_stack(thresholdFiles[0])
print(whichFiles)

#%% binarization, extract coordinates, and compute CM
blobBin = []
xb = []
xp = []
nt = len(imgs)
cm = np.zeros((nt,3))
n1s = np.zeros((nt, 3))
n2s = np.zeros((nt, 3))
n3s = np.zeros((nt, 3))
r_coms = np.zeros((nt, 3))
flagella_len = np.zeros(nt)
radial_dist_pt = np.zeros(nt)
thresh_size = np.zeros(nt)
threshold = np.zeros(nt)
num_iterations = np.zeros(nt)
# starting threshold guess as fraction of maximum pixel value
max_thresh_iterations = 30
min_thresh_size_diff = 50
thresh_value = 0.76
thresh_max_start = thresh_value + 0.15
thresh_min_start = thresh_value - 0.15

tstart = time.perf_counter()

total_frame = 1
total_frame = nt

for frame in range(total_frame):
    
    # grab current image
    img_now = np.array(imgs[frame])

    # median filter image
    img_now_med = scipy.signal.medfilt(img_now, kernel_size=(3,3,3))
        
    # iteratively threshold using binary search
    thresh_max = thresh_max_start
    thresh_min = thresh_min_start
    thresh_value_temp = thresh_value
    for ii in range(max_thresh_iterations):
        if ii > 0:
            thresh_constraint = thresh_size[0] - thresh_size_temp
            
            # todo: investigate case where bounces back and forth between
            # two values the whole time... probably related to thresholding
            # integer data
            
            # if constrain was met, break out of loop
            if np.abs(thresh_constraint) < min_thresh_size_diff:
                break
            
            if thresh_constraint > 0:
                
                # if not enough thresholded points, decrease threshold
                # since thres_value_temp is too high a threshold, update thresh_max
                thresh_max = thresh_value_temp
                
                # new threshold value guess
                thresh_value_temp = 0.5 * (thresh_value_temp + thresh_min)
                
            else:
                
                # if too many thresholded points, increase threshold
                # since thresh_value_temp is too low, update thresh_min
                thresh_min = thresh_value_temp
                thresh_value_temp = 0.5 * (thresh_value_temp + thresh_max)
        
        # threshold intensity image with the default value
        img_thresh_med = img_now_med > thresh_value_temp * np.max(img_now_med)
        
        # label and measure every clusters
        blobs = measure.label(img_thresh_med, background=0)
        labels = np.arange(1, blobs.max() + 1, dtype=int)
        sizes = np.array([np.sum(blobs == l) for l in labels])
        
        # keep only the largest cluster  
        max_ind = np.argmax(sizes)
        thresh_size_temp = sizes[max_ind]
        
        # mask showing which pixels ae in largest cluster
        blob = blobs == labels[max_ind]
    
        # print iteration number 
        print('frame = %d, iteration = %d, # thresholded pixels = %d, threshold = %.6f, elapsed time = %0.2fs'
              % (frame, ii, thresh_size_temp, thresh_value_temp, time.perf_counter() - tstart) )
        
        # if first frame, only do one iteration
        if frame == 0:
            break

    # track number of iterations run
    num_iterations[frame] = ii
    # size of thresholded region
    thresh_size[frame] = sizes[max_ind]
    # threshold value
    threshold[frame] = thresh_value_temp
    # store threshold/binarized image
    blobBin.append(blob)

blobBin = np.array(blobBin)

# Check thresholding
if total_frame == 1:
    viewer = napari.Viewer(ndisplay=3)      
    viewer.add_image(imgs[0], contrast_limits=[100,400],\
                     scale=[0.115,.115,.115],\
                     multiscale=False,colormap='gray',opacity=1)
    viewer.add_image(blobBin[0], contrast_limits=[0,1],\
                     scale=[0.115,.115,.115],\
                     multiscale=False,colormap='green',opacity=0.5)
    viewer.scale_bar.visible=True
    viewer.scale_bar.unit='um'
    viewer.scale_bar.position='top_right'
    viewer.axes.visible = True
    napari.run()

    #%% plot threshold size
plt.figure(dpi=600, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
plt.plot(thresh_size)
plt.xlabel('frame-num')
plt.ylabel('threshold array size')
plt.title(os.path.basename(whichFiles))
# plt.ylim([0,4000])
# plt.xlim([0,200])

#%% save to npy    
savingThreshold = os.path.join(os.path.dirname(whichFiles),'Threshold',
                  os.path.basename(whichFiles)[:-4] + '-threshold.npy')
np.save(savingThreshold, blobBin)

#%% save movie  
savingMovie_Time = os.path.join(os.path.dirname(whichFiles),'Movies',
                   os.path.basename(whichFiles)[:-4] + '-threshold-time.mov')
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(imgs, contrast_limits=[0,300],\
                    scale=[0.115,.115,.115],\
                    multiscale=False,colormap='gray',opacity=1)
viewer.add_image(blobBin, contrast_limits=[0,1],\
                    scale=[0.115,.115,.115],\
                    multiscale=False,colormap='green',opacity=0.5)
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
movie = Movie(myviewer=viewer)
movie.create_state_dict_from_script('./moviecommands/mcTime.txt')
movie.make_movie(savingMovie_Time,fps=1)

savingMovie_Rotate = os.path.join(os.path.dirname(whichFiles),'Movies',
                     os.path.basename(whichFiles)[:-6] + '-threshold-rotate.mov')
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(imgs, contrast_limits=[0,300],\
                    scale=[0.115,.115,.115],\
                    multiscale=False,colormap='gray',opacity=1)
viewer.add_image(blobBin, contrast_limits=[0,1],\
                    scale=[0.115,.115,.115],\
                    multiscale=False,colormap='green',opacity=0.5)
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
movie = Movie(myviewer=viewer)
movie.create_state_dict_from_script('./moviecommands/mcRotate.txt')
movie.make_movie(savingMovie_Rotate,fps=10)
