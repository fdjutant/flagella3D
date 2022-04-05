#%% Import all necessary libraries
import sys
sys.path.insert(0, './modules')
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})
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

setName = 'suc-70'

this_file_dir = os.path.join(os.path.dirname(os.path.abspath("./")),
                            'Dropbox (ASU)','Research')
intensityFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                            'Flagella-data', setName)
intensityFiles = list(Path(intensityFolder).glob("*.npy"))

#%% save to TIF
for whichFiles in range(len(intensityFiles)):

    imgs = da.from_npy_stack(intensityFiles[whichFiles])
    print(intensityFiles[whichFiles])
    
    fname_save_tiff = intensityFiles[whichFiles].with_suffix(".tif")
    img_to_save = tifffile.transpose_axes(imgs, "TZYX", asaxes="TZCYXS")
    tifffile.imwrite(fname_save_tiff, img_to_save, imagej=True)

