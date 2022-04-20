#%% To input the file
import napari
import dask.array as da
from dask.array import from_zarr
from pathlib import Path
import os.path
import tifffile
import pickle

dataName = 'timelapse_2022_03_06-02_22_46'
this_file_dir = os.path.join(os.path.dirname(os.path.abspath('./')),
                            'Dropbox (ASU)','Research')
folderName = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Snouty-data',
                          '2022-03-06', 'suc90_25um_3ms', dataName,
                          'deskew_output', 'zarr', 'OPM_data_zarr.zarr',
                          'opm_data')

# input Zarr and convert to dask
inputZarr = from_zarr(folderName)
stack = da.stack(inputZarr,axis=1)[0]

# # show in Napari    
# viewer = napari.Viewer(ndisplay=3)
# viewer.add_image(stack, contrast_limits=[100,200],colormap='gray')
# viewer.scale_bar.visible=True
# viewer.scale_bar.unit='px'
# viewer.scale_bar.position='top_right'
# napari.run()

#%% open pkl file
folder_pkl = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Snouty-data',
                          '2022-03-06', 'suc90_25um_3ms')
files_pkl = list(Path(folder_pkl).glob("*.pkl"))
with open(files_pkl[15], "rb") as f:
      data_loaded = pickle.load(f)
print(files_pkl[15].name)
print(data_loaded)      

#%% To check in small box
yC1 = data_loaded['yC1']; xC1 = data_loaded['xC1']

top = data_loaded['top']
bottom = data_loaded['bottom']
left = data_loaded['left']
right = data_loaded['right']

yout1 = stack[:,:, yC1-top:yC1+bottom, xC1-left:xC1+right]

viewer = napari.Viewer(ndisplay=3)      
viewer.add_image(yout1, contrast_limits=[100,200],
                 scale=[0.115,.115,.115],colormap='gray')
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
napari.run()
    
#%% To slice time steps
tL1 = data_loaded['tL1']
tR1 = data_loaded['tR1']
yout1 = stack[tL1:tR1,:, yC1-top:yC1+bottom, xC1-left:xC1+right]

# setup up result file path
savingFolder = os.path.join(os.path.dirname(os.path.abspath('./')),
                            'Dropbox (ASU)', 'Research',
                           'DNA-Rotary-Motor', 'Helical-nanotubes',
                           'Light-sheet-OPM', 'Result-data',
                           'Microtubule-data', 'TIF-files')

# save to TIF
fname_save_tiff = os.path.join(savingFolder,
                               'suc90-' + dataName + '-C' + '.tif')
img_to_save = tifffile.transpose_axes(yout1, "TZYX", asaxes="TZCYXS")
tifffile.imwrite(fname_save_tiff, img_to_save, imagej=True)
print('%s is saved' %dataName)
