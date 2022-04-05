#%% Import all necessary libraries
import sys
sys.path.insert(0, './modules')
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})
import napari
from skimage import measure
from pathlib import Path
import os.path
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
labkit_folder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Flagella-data', 'image-labKit')
intensityFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Flagella-data', 'suc-40')

labkit_files = list(Path(labkit_folder).glob("*.tif"))
intensityFiles = list(Path(intensityFolder).glob("*.npy"))

whichFiles = 0
imgs = da.from_npy_stack(intensityFiles[whichFiles])
img_labkit = tifffile.imread(labkit_files[whichFiles]).astype('bool')
nt = len(imgs)
print('filename = %s, with total-frame = %d' %(labkit_files[whichFiles].name, nt))

#%% binarization, extract coordinates, and compute CM
blobBin = []
xb = []
xp = []
cm = np.zeros((nt,3))
n1s = np.zeros((nt, 3))
n2s = np.zeros((nt, 3))
n3s = np.zeros((nt, 3))
r_coms = np.zeros((nt, 3))
flagella_len = np.zeros(nt)
radial_dist_pt = np.zeros(nt)
thresh_length = np.zeros(nt)
total_px = np.zeros(nt)
threshold = np.zeros(nt)

total_frame = 1
total_frame = nt

for frame in range(total_frame):
    
    # grab current image
    img_now = np.array(img_labkit[frame])

    # label and measure every clusters
    blobs = measure.label(img_now, background=0)
    labels = np.arange(1, blobs.max() + 1, dtype=int)
    sizes = np.array([np.sum(blobs == l) for l in labels])
    
    # keep only the largest cluster  
    max_ind = np.argmax(sizes)
    total_px[frame] = sizes[max_ind]
    
    # mask showing which pixels ae in largest cluster
    blob = blobs == labels[max_ind]
    
    # store threshold/binarized image
    blobBin.append(blob)
    
blobBin = np.array(blobBin)

#%% save to npy    
savingThreshold = os.path.join(labkit_files[whichFiles].parent.parent,
                   'threshold-labKit', labkit_files[whichFiles].name[:-4] +
                   '-threshold.npy')
np.save(savingThreshold, blobBin)

#%% Check thresholding
viewer = napari.Viewer(ndisplay=3)      
viewer.add_image(imgs, contrast_limits=[100,400],
                 scale=[0.115,.115,.115], blending='additive',
                 multiscale=False,colormap='gray',opacity=1)
viewer.add_image(img_labkit, contrast_limits=[0,1],
                 scale=[0.115,.115,.115], blending='additive',
                 multiscale=False,colormap='gray',opacity=1)
viewer.add_image(blobBin, contrast_limits=[0,1],
                 scale=[0.115,.115,.115], blending='additive',
                 multiscale=False,colormap='green',opacity=0.5)
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
napari.run()

#%% plot threshold size
plt.figure(dpi=600, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
plt.plot(total_px)
plt.xlabel('frame-num')
plt.ylabel('threshold array size')
plt.title(labkit_files[whichFiles].name)

