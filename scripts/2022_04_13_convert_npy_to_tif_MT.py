#%% Import all necessary libraries
import sys
sys.path.insert(0, './modules')
import dask.array as da
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})
from pathlib import Path
import os.path
import tifffile

setName = 'timelapse_2022_03_06-02_22_46'

this_file_dir = os.path.join(os.path.dirname(os.path.abspath("./")),
                            'Dropbox (ASU)','Research')
intensityFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Microtubule-data', setName)
intensityFiles = list(Path(intensityFolder).glob("*.npy"))

    #%% save to TIF
for whichFiles in range(len(intensityFiles)):

    imgs = da.from_npy_stack(intensityFiles[whichFiles])
    print(intensityFiles[whichFiles])
    
    fname_save_tiff = intensityFiles[whichFiles].with_suffix(".tif")
    img_to_save = tifffile.transpose_axes(imgs, "TZYX", asaxes="TZCYXS")
    tifffile.imwrite(fname_save_tiff, img_to_save, imagej=True)

