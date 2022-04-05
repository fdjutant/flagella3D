#%% Import all necessary libraries
import sys
sys.path.insert(0, './modules')
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})
import napari
from pathlib import Path
import os.path
import tifffile

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
intensityFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                            'Flagella-data', setName)
labKitFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                            'Flagella-data', setName)
thresholdFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                            'Flagella-data', 'threshold-iterative')
                            
intensityFiles = list(Path(intensityFolder).glob("*.npy"))
labKitFiles = list(Path(labKitFolder).glob("*-LabKit.tif"))
thresholdFiles = list(Path(thresholdFolder).glob("*threshold*.npy"))

whichFiles = 0
imgs = da.from_npy_stack(intensityFiles[whichFiles])
imgs_labkit = tifffile.imread(labKitFiles[whichFiles])
imgs_thresh = np.load(thresholdFiles[whichFiles])
print(intensityFiles[whichFiles])
print(thresholdFiles[whichFiles])

#%% save to TIF
fname_save_tiff = intensityFiles[whichFiles].with_suffix(".tif")
img_to_save = tifffile.transpose_axes(imgs, "TZYX", asaxes="TZCYXS")
tifffile.imwrite(fname_save_tiff, img_to_save, imagej=True)

#%% View image and threshold together
viewer = napari.Viewer(ndisplay=3)      
viewer.add_image(imgs, contrast_limits=[100,400],\
                 scale=[0.115,.115,.115],\
                 multiscale=False,colormap='gray',opacity=1, blending="additive")
viewer.add_image(imgs_labkit, contrast_limits=[100,400],\
                 scale=[0.115,.115,.115],\
                 multiscale=False,colormap='bop orange',opacity=1, blending="additive")
viewer.add_image(imgs_thresh, contrast_limits=[0,1],\
                 scale=[0.115,.115,.115],\
                 multiscale=False,colormap='green',opacity=0.25, blending="additive")
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
napari.run()