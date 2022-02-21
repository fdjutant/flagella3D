#%% To input the file
import napari
import dask.array as da
from dask.array import from_zarr
from naparimovie import Movie
import numpy as np
import os.path
    
# filepath = r"D:\Dropbox (ASU)\Research\DNA-Rotary-Motor\Helical-nanotubes\Light-sheet-OPM\Result-data"
# folderName = "/2021-03-25/"
filepath = r"C:\Users\labuser\Dropbox (ASU)\Research\DNA-Rotary-Motor\Helical-nanotubes\Light-sheet-OPM\Result-data"
folderName = "/20211018a_suc50_h15um/"

# C:\Users\labuser\Dropbox (ASU)\Research\DNA-Rotary-Motor\Helical-nanotubes\Light-sheet-OPM\Snouty-data\2021-10-22\suc70-h15\data-21\deskew_output\zarr\suc70-h15_zarr.zarr\opm_data
imgName = "suc50-h15-01-A"
thName = "run-03/"+ imgName + "-threshold"

# input Zarr and convert to dask
stackImg = da.from_npy_stack(filepath + folderName + imgName + ".npy")
stackTh = da.from_npy_stack(filepath + folderName + thName + ".npy")

# show in Napari    
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(stackImg, contrast_limits=[100,300],
                 scale=[0.115,.115,.115],colormap='gray',opacity=1)
viewer.add_image(stackTh, contrast_limits=[0,1],
                 scale=[0.115,.115,.115],colormap='blue',opacity=0.5)
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
napari.run()