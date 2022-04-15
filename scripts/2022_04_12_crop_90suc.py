#%% To input the file
import napari
import dask.array as da
from dask.array import from_zarr
import os.path
import tifffile

dataName = 'timelapse_2022_02_25-11_19_12'
dataName = 'timelapse_2022_02_25-11_08_56'
dataName = 'timelapse_2022_02_25-10_56_25'
dataName = 'timelapse_2022_02_25-10_49_43'
dataName = 'timelapse_2022_02_25-10_44_31'
dataName = 'timelapse_2022_02_25-06_29_14'
dataName = 'timelapse_2022_02_25-06_21_25'
dataName = 'timelapse_2022_02_25-06_06_18'
dataName = 'timelapse_2022_02_25-05_55_45'
dataName = 'timelapse_2022_02_25-05_26_06'
# this_file_dir = os.path.join(os.path.dirname(os.path.abspath('./')),
#                             'Dropbox (ASU)','Research')
# folderName = os.path.join(this_file_dir,
#                           'DNA-Rotary-Motor', 'Helical-nanotubes',
#                           'Light-sheet-OPM', 'Snouty-data',
#                           '2022-02-25', 'flagella_90suc', dataName,
#                           'deskew_output', 'zarr', 'OPM_data_zarr.zarr',
#                           'opm_data')
this_file_dir = os.path.join(os.path.dirname(os.path.abspath('./')),
                            'Desktop')
folderName = os.path.join(this_file_dir, dataName, 
                          # 'OPM_data.zarr')
                           'deskew_output', 'zarr',
                           'OPM_data_zarr.zarr', 'opm_data')

# input Zarr and convert to dask
inputZarr = from_zarr(folderName)
stack = da.stack(inputZarr,axis=1)[0]

# show in Napari    
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(stack, contrast_limits=[100,200],colormap='gray')
viewer.scale_bar.visible=True
viewer.scale_bar.unit='px'
viewer.scale_bar.position='top_right'
napari.run()

#%% To check in small box
yC1 = 156; xC1 = 962

top = 50; bottom = 50; left = 50; right = 50;
yout1 = stack[:,:, yC1-top:yC1+bottom, xC1-left:xC1+right];

viewer = napari.Viewer(ndisplay=3)      
viewer.add_image(yout1, contrast_limits=[100,200],
                 scale=[0.115,.115,.115],colormap='gray')
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
napari.run()
    
#%% To slice time steps
tL1 = 222; tR1 = 400;
yout1 = stack[tL1:tR1,:, yC1-top:yC1+bottom, xC1-left:xC1+right]

# setup up result file path
savingFolder = os.path.join(os.path.dirname(os.path.abspath('./')),
                            'Dropbox (ASU)', 'Research',
                           'DNA-Rotary-Motor', 'Helical-nanotubes',
                           'Light-sheet-OPM', 'Result-data',
                           'Flagella-data', 'TIF-files')

# save to TIF
fname_save_tiff = os.path.join(savingFolder,
                               'suc90-' + dataName[21:] + '-A' + '.tif')
img_to_save = tifffile.transpose_axes(yout1, "TZYX", asaxes="TZCYXS")
tifffile.imwrite(fname_save_tiff, img_to_save, imagej=True)
print('%s is saved' %dataName)
