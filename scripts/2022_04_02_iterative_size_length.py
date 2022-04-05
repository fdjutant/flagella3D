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

setName = 'suc-40'

this_file_dir = os.path.join(os.path.dirname(os.path.abspath("./")),
                            'Dropbox (ASU)','Research')
loadFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Flagella-data', setName)

intensityFiles = list(Path(loadFolder).glob("*.npy"))
# thresholdFiles = list(Path(savingFolder).glob("*threshold.npy"))
whichFiles = intensityFiles[0]

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
thresh_length = np.zeros(nt)
thresh_size = np.zeros(nt)
threshold = np.zeros(nt)
num_iterations = np.zeros(nt)
# starting threshold guess as fraction of maximum pixel value
# volume_upper_bound = 
max_thresh_iterations = 30
min_length_diff = 3         # in px
thresh_value = 150          # initial threshold pixel value (determined by eyes)
thresh_max_start = thresh_value + 40
thresh_min_start = thresh_value - 40

tstart = time.perf_counter()

total_frame = 1
total_frame = nt

for frame in range(total_frame):
    
    # grab current image
    img_now = np.array(imgs[frame])

    # median filter image
    img_now_med = scipy.signal.medfilt(img_now, kernel_size=(3,3,3))
        
    # iteratively threshold using bi-section search
    thresh_max = thresh_max_start
    thresh_min = thresh_min_start
    thresh_value_temp = thresh_value
    for ii in range(max_thresh_iterations):
        if ii > 0:
            thresh_length_constraint = thresh_length[0] - current_length_temp
            thresh_size_constraint = thresh_size[0] - current_size
            
            # if constraint was met, break out of loop
            if np.abs(thresh_length_constraint) < min_length_diff \
               and np.abs(thresh_size_constraint) < 100:
                break
            
            if thresh_length_constraint > 0 or thresh_size_constraint > 300:
                
                # if not enough thresholded points, decrease threshold
                # since thres_value_temp is too high a threshold, update thresh_max
                thresh_max = thresh_value_temp
                
                # new threshold value guess
                thresh_value_temp = 0.5 * (thresh_value_temp + thresh_min)
                
            elif thresh_length_constraint < 0 or thresh_size_constraint < 300:
                # if too many thresholded points, increase threshold
                # since thresh_value_temp is too low, update thresh_min
                thresh_min = thresh_value_temp
                thresh_value_temp = 0.5 * (thresh_value_temp + thresh_max)
        
        # threshold intensity image with the default value
        img_thresh_med = img_now_med > thresh_value_temp
        
        # label and measure every clusters
        blobs = measure.label(img_thresh_med, background=0)
        labels = np.arange(1, blobs.max() + 1, dtype=int)
        sizes = np.array([np.sum(blobs == l) for l in labels])
        
        # keep only the largest cluster  
        max_ind = np.argmax(sizes)
        size_px = sizes[max_ind]
        
        # mask showing which pixels ae in largest cluster
        blob = blobs == labels[max_ind]
        
        # extract coordinates 
        X0 = np.argwhere(blob).astype('float') # coordinates 
        
        # compute center of mass
        CM1 = np.array([sum(X0[:,j]) for j in range(X0.shape[1])])/X0.shape[0]
        
        # determine axis n1 from PCA 
        coords = X0 - CM1 # shift all the coordinates into origin
        pca = PCA(n_components=1)
        pca.fit(coords)
        n1_temp = pca.components_[0]
        
        # rotate flagella on the principal axes
        dist_projected_along_n1 = n1_temp[0] * coords[:, 0] +\
                                  n1_temp[1] * coords[:, 1] +\
                                  n1_temp[2] * coords[:, 2]
                                  
        # determine the flagella length along the n1
        flagella_len_temp = np.max(dist_projected_along_n1) -\
                            np.min(dist_projected_along_n1)
                                
        # if the constraint repeats, then break (avoid being stucked in the loop)
        if ii > 0 and flagella_len_temp == current_length_temp:
            break
        if ii > 0 and size_px == current_size:
            break
        
        # update constraint
        current_length_temp = flagella_len_temp
        current_size = size_px
    
        # print iteration number 
        print('fr-# = %d, iteration = %d, # length = %.2f, threshold = %d, px-size = %d, time = %0.2fs'
              % (frame, ii, current_length_temp, thresh_value_temp, current_size, time.perf_counter() - tstart) )
        
        # if first frame, only do one iteration
        if frame == 0:
            break

    # track number of iterations run
    num_iterations[frame] = ii
    # size of thresholded region
    thresh_length[frame] = flagella_len_temp
    thresh_size[frame] = sizes[max_ind]
    # threshold value
    threshold[frame] = thresh_value_temp
    # store threshold/binarized image
    blobBin.append(blob)

blobBin = np.array(blobBin)

#%% save to npy    
savingThreshold = os.path.join(os.path.dirname(os.path.dirname(whichFiles)),
                               'threshold-length',
                  os.path.basename(whichFiles)[:-4] + '-threshold' + '.npy')
np.save(savingThreshold, blobBin)

#%% Check thresholding
napari_check = True
if napari_check == 1:
    viewer = napari.Viewer(ndisplay=3)      
    viewer.add_image(imgs, contrast_limits=[100,400],\
                     scale=[0.115,.115,.115],\
                     multiscale=False,colormap='gray',opacity=1)
    viewer.add_image(blobBin, contrast_limits=[0,1],\
                     scale=[0.115,.115,.115],\
                     multiscale=False,colormap='green',opacity=0.5)
    viewer.scale_bar.visible=True
    viewer.scale_bar.unit='um'
    viewer.scale_bar.position='top_right'
    viewer.axes.visible = True
    napari.run()

#%% plot threshold size
plot_Threshold = True
if plot_Threshold:
    plt.figure(dpi=600, figsize=(10,7))
    plt.rcParams.update({'font.size': 22})
    plt.plot(thresh_length)
    plt.xlabel('frame-num')
    plt.ylabel('threshold array size')
    plt.title(os.path.basename(whichFiles))
    # plt.ylim([0,4000])
    # plt.xlim([0,200])

