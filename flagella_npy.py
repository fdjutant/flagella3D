#%% To input the file
import napari
import dask.array as da
from dask.array import from_zarr
from naparimovie import Movie
import numpy as np
import os.path
    
filepath = r"D:\Dropbox (ASU)\Research\DNA-Rotary-Motor\Helical-nanotubes\Light-sheet-OPM\Result-data"
folderName = "/2021-03-25/"
# C:\Users\labuser\Dropbox (ASU)\Research\DNA-Rotary-Motor\Helical-nanotubes\Light-sheet-OPM\Snouty-data\2021-10-22\suc70-h15\data-21\deskew_output\zarr\suc70-h15_zarr.zarr\opm_data
fileName = "flagella-25c-02a"

# input Zarr and convert to dask
stack = da.from_npy_stack(filepath + folderName + fileName + ".npy")

# show in Napari    
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(stack, contrast_limits=[110,300],
                 scale=[0.115,.115,.115],colormap='gray')
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
napari.run()