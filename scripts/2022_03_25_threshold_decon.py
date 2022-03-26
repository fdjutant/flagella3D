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
                          'Flagella-decon', setName)

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
thresh_value = 200
tstart = time.perf_counter()

total_frame = 1
total_frame = nt

for frame in range(total_frame):
    print('frame-#:', frame)
    
    # grab current image
    img_now = np.array(imgs[frame])

    # median filter image
    img_now_med = scipy.signal.medfilt(img_now, kernel_size=(3,3,3))
    # img_now_med = img_now
        
    # threshold intensity image with the default value
    fixed_intensity = thresh_value
    img_thresh_med = img_now_med > fixed_intensity
    
    # label and measure every clusters
    blobs = measure.label(img_thresh_med, background=0)
    labels = np.arange(1, blobs.max() + 1, dtype=int)
    sizes = np.array([np.sum(blobs == l) for l in labels])
    
    # keep only the largest cluster  
    max_ind = np.argmax(sizes)
    thresh_size_temp = sizes[max_ind]
    
    # mask showing which pixels ae in largest cluster
    blob = blobs == labels[max_ind]
    
    # store threshold/binarized image
    blobBin.append(blob)

blobBin = np.array(blobBin)

#%% save to npy    
savingThreshold = os.path.join(os.path.dirname(os.path.dirname(whichFiles)),
                               'threshold-decon',
                  os.path.basename(whichFiles)[:-4] + '-thresh-decon-' +
                  str(thresh_value) + '.npy')
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
    plt.plot(thresh_size)
    plt.xlabel('frame-num')
    plt.ylabel('threshold array size')
    plt.title(os.path.basename(whichFiles))
    # plt.ylim([0,4000])
    # plt.xlim([0,200])

