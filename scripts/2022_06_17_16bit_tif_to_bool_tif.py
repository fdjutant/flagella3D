#%% Import all necessary libraries
import sys
sys.path.insert(0, './modules')
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})
from pathlib import Path
import os.path
import tifffile

this_file_dir = os.path.join(os.path.dirname(os.path.abspath("./")),
                            'Dropbox (ASU)', 'BioMAN_Lab', 'Franky',
                            'Flagella-motor', 'Light-sheet-OPM',
                            'Result-data', 'Flagella-data')
thresholdFolder = os.path.join(this_file_dir,'threshold-labKit')
thresholdFiles = list(Path(thresholdFolder).glob("*-LabKit-*.tif"))

boolFolder = os.path.join(os.path.dirname(thresholdFolder),
                               'threshold-labKit-bool')

#%% save to TIF
for whichFiles in range(len(thresholdFiles)):

    imgs_thresh = tifffile.imread(thresholdFiles[whichFiles]).astype('bool')
    print(thresholdFiles[whichFiles].name)
    
    fname_save_tiff = os.path.join(boolFolder,
                           thresholdFiles[whichFiles].name[:-4] + '-bool.tif')
    img_to_save = tifffile.transpose_axes(imgs_thresh, "TZYX", asaxes="TZCYXS")
    tifffile.imwrite(fname_save_tiff, img_to_save)
