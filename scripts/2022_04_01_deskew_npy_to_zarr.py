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
import time
import zarr

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

for whichFiles in range(len(intensityFiles)):
# for whichFiles in range(0,1):
    
    # load images
    imgs = np.asarray(da.from_npy_stack(intensityFiles[whichFiles]))
    
    path_zarr = os.path.join(loadFolder,
                             intensityFiles[whichFiles])[:-4] + '.zarr'
    
    zdata = zarr.open(path_zarr, 'w')
    zdata.create_dataset("images", shape=imgs.shape,
                         dtype=imgs.dtype, compressor="none")
    zdata.images[:] = imgs