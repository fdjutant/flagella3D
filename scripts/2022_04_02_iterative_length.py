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
exp3D_sec = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 
pxum = 0.115

setName = 'suc-40'
this_file_dir = os.path.join(os.path.dirname(os.path.abspath("./")),
                            'Dropbox (ASU)','Research')
loadFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Flagella-data', setName)

intensityFiles = list(Path(loadFolder).glob("*.npy"))
# thresholdFiles = list(Path(savingFolder).glob("*threshold.npy"))
whichFiles = intensityFiles[14]

imgs = da.from_npy_stack(whichFiles)
# imgs_thresh = da.from_npy_stack(thresholdFiles[0])
print(whichFiles)
print('total-frame = %d' %len(imgs))

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
max_thresh_iterations = 10
min_length_diff = 10              # in px
thresh_init_value = 155          # initial threshold pixel value (determined by eyes)
thresh_max_start = thresh_init_value + 50
thresh_min_start = thresh_init_value - 50

tstart = time.perf_counter()

frame_start = 0
frame_final = nt

for frame in range(frame_start, frame_final):
    
    # grab current image
    img_now = np.array(imgs[frame])

    # median filter image
    img_now_med = scipy.signal.medfilt(img_now, kernel_size=(3,3,3))
        
    # iteratively threshold using bi-section search
    thresh_max = thresh_max_start
    thresh_min = thresh_min_start
    thresh_value_temp = thresh_init_value
    
    sizes_inside_iteration = np.zeros(max_thresh_iterations)
    thresh_value_temp_inside_iterations = np.zeros(max_thresh_iterations-1)
    for ii in range(max_thresh_iterations):
        if ii > 0:
            thresh_length_constraint = thresh_length[frame_start] - current_length_temp
            
            # if constraint was met, break out of loop
            if np.abs(thresh_length_constraint) < min_length_diff:
                break
            
            if thresh_length_constraint > 0:
                
                # if not enough thresholded points, decrease threshold
                # since thres_value_temp is too high a threshold, update thresh_max
                thresh_max = thresh_value_temp
                
                # new threshold value guess
                thresh_value_temp = 0.5 * (thresh_value_temp + thresh_min)
                
            elif thresh_length_constraint < 0:
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
        sizes_inside_iteration[ii] = size_px
        
        # cut the iteration if the size too small or blows up consequetively
        # if ii == max_thresh_iterations-1:
        #     min_ind_size = np.argmin(np.abs(thresh_size[frame_start] -
        #                                     sizes_inside_iteration))
        #     thresh_value_temp
        if ii> 0 and size_px > thresh_size[frame_start] * 2:
            break
        if ii> 0 and size_px < thresh_size[frame_start] * (1/2):
            break
        
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
        
        # update constraint
        current_length_temp = flagella_len_temp
    
        # print iteration number 
        print('fr-# = %d, iteration = %d, # length = %.2f, threshold = %d, px-size = %d, time = %0.2f mins'
              % (frame, ii, current_length_temp, thresh_value_temp, size_px, (time.perf_counter() - tstart)/60 ))
        
        # if first frame, only do one iteration
        if frame == frame_start:
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

#%% napari
viewer = napari.Viewer(ndisplay=3)      
viewer.add_image(imgs[frame_start:frame_final], contrast_limits=[100,400],\
                 scale=[0.115,.115,.115],\
                 multiscale=False,colormap='gray',opacity=1)
viewer.add_image(blobBin, contrast_limits=[0,1],\
                 scale=[0.115,.115,.115],\
                 multiscale=False,colormap='green',opacity=0.2)
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
napari.run()

#%% save to npy    
savingThreshold = os.path.join(os.path.dirname(os.path.dirname(whichFiles)),
                               'threshold-length',
                                os.path.basename(whichFiles)[:-4] +
                                '-threshold-' +
                                str(thresh_init_value) + '.npy')
np.save(savingThreshold, blobBin[100:200])
print(savingThreshold)

#%% plot threshold size
print('length-mean = %.2f, length-std = %.2f'
      %(np.mean(thresh_length[frame_start:frame_final]),
        np.std(thresh_length[frame_start:frame_final])))

plt.figure(dpi=600, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
plt.plot(thresh_length[frame_start:frame_final])
plt.xlabel('frame-num')
plt.ylabel('length (px)')
plt.title(os.path.basename(whichFiles))
# plt.ylim([0,4000])
# plt.xlim([0,200])

plt.figure(dpi=600, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
plt.plot(thresh_size[frame_start:frame_final], 'r')
plt.xlabel('frame-num')
plt.ylabel('threshold array size (px)')
plt.title(os.path.basename(whichFiles))