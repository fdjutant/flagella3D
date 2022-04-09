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
import scipy.signal
from scipy.optimize import least_squares
from naparimovie import Movie
from pathlib import Path
from scipy import optimize
import os.path
import pickle
import time
import tifffile

# time settings in the light sheet
pxum = 0.115
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
exp3D_sec = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 
pxum = 0.115

this_file_dir = os.path.join(os.path.dirname(os.path.abspath("./")),
                            'Dropbox (ASU)','Research')
thresholdFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Flagella-data', 'threshold-fixed-value')
intensityFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Flagella-data', 'TIF-files')

intensityFiles = list(Path(intensityFolder).glob("*.tif"))

whichFiles = 4
# imgs = da.from_npy_stack(intensityFiles[whichFiles])
imgs = tifffile.imread(intensityFiles[whichFiles])
print(intensityFiles[whichFiles].name)

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

tstart = time.perf_counter()

total_frame = 1
total_frame = 330

for frame in range(total_frame):
    
    print('frame: %d' %frame)
    
    # grab current image
    img_now = imgs[frame]

    # median filter image
    # img_now_med = scipy.signal.medfilt(img_now, kernel_size=(3,3,3))
    # img_now_med = img_now
        
    # threshold intensity image with the default value
    fixed_intensity = 150
    img_thresh_med = img_now > fixed_intensity
    
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
savingThreshold = os.path.join(os.path.dirname(whichFiles),'Threshold',
                  os.path.basename(whichFiles)[:-4] + '-threshold-' +
                  str(fixed_intensity) + '.npy')
np.save(savingThreshold, blobBin)

#%% Check thresholding
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

saving_Movie = os.path.join(thresholdFolder,
                  os.path.basename(intensityFiles[whichFiles])[:-4] + '-threshold-' +
                  str(fixed_intensity) + '.mov')
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
movie = Movie(myviewer=viewer)
movie.create_state_dict_from_script('./moviecommands/mcTime.txt')
movie.make_movie(saving_Movie,fps=10)

