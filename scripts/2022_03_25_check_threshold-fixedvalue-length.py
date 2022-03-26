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
                           # 'Flagella-data', setName)
                            'Flagella-decon', setName)
thresholdFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                           # 'Flagella-data', 'threshold-length')
                           # 'Flagella-data', 'threshold-iterative')
                           # 'Flagella-data', 'threshold-fixed-value')
                            'Flagella-decon', 'threshold-decon')

intensityFiles = list(Path(intensityFolder).glob("*.npy"))
thresholdFiles = list(Path(thresholdFolder).glob("*threshold*.npy"))

whichFiles = 0
imgs = da.from_npy_stack(intensityFiles[whichFiles])
imgs_thresh = np.load(thresholdFiles[whichFiles])
print(intensityFiles[whichFiles])
print(thresholdFiles[whichFiles])

#%% View image and threshold together
viewer = napari.Viewer(ndisplay=3)      
viewer.add_image(imgs, contrast_limits=[100,400],\
                 scale=[0.115,.115,.115],\
                 multiscale=False,colormap='gray',opacity=1)
viewer.add_image(imgs_thresh, contrast_limits=[0,1],\
                 scale=[0.115,.115,.115],\
                 multiscale=False,colormap='green',opacity=0.5)
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
napari.run()