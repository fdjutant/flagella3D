#%% To input the file
import napari
import dask.array as da
from dask.array import from_zarr
from naparimovie import Movie
import numpy as np
import os.path
    
dataNum = "09"

filepath = r"C:\\Users\\labuser\\Dropbox (ASU)\\Research\\DNA-Rotary-Motor\\Helical-nanotubes\\Light-sheet-OPM\\Snouty-data"
folderName = "/2021-12-03/MT-run-" +\
                dataNum + "/deskew_output/zarr"
# C:\Users\labuser\Dropbox (ASU)\Research\DNA-Rotary-Motor\Helical-nanotubes\Light-sheet-OPM\Snouty-data\2021-10-22\suc70-h15\data-21\deskew_output\zarr\suc70-h15_zarr.zarr\opm_data
fileName = "/MT-suc70"
resultPath = r"C:\Users\labuser\Dropbox (ASU)\Research\DNA-Rotary-Motor\Helical-nanotubes\Light-sheet-OPM\Result-data"

# input Zarr and convert to dask
inputZarr = from_zarr(filepath + folderName + fileName +\
                      "_zarr.zarr\\opm_data")
stack = da.stack(inputZarr,axis=1)

# show in Napari    
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(stack, contrast_limits=[100,200],colormap='gray')
napari.run()

#%% To check in small box
yC1 = 95; xC1 = 1164;
# yC1 = 108; xC1 = 716;

top = 40; bottom = 40; left = 40; right = 40;
yout1 = stack[0,:,:, yC1-top:yC1+bottom, xC1-left:xC1+right];

viewer = napari.Viewer(ndisplay=3)      
viewer.add_image(yout1, contrast_limits=[100,200],colormap='gray',opacity=1)
napari.run()
    
#%% To slice time steps
tL1 = 0; tR1 = 150;
# tL2 = 150; tR2 = 399;
# tL3 = 300; tR3 = 399;
# yout1 = np.concatenate((
#         stack[0,tL1:tR1,:, yC1-top:yC1+bottom, xC1-left:xC1+right],\
#         stack[0,tL2:tR2,:, yC1-top:yC1+bottom, xC1-left:xC1+right]),
#         axis=0);
yout1 = stack[0,tL1:tR1,:, yC1-top:yC1+bottom, xC1-left:xC1+right]
# viewer = napari.Viewer(ndisplay=3)      
# viewer.add_image(yout1, contrast_limits=[130,200],colormap='gray',opacity=1)
# napari.run()

# save movie  
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(yout1, contrast_limits=[100,200],
                 scale=[0.115,.115,.115],colormap='gray')
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
movie = Movie(myviewer=viewer)
movie.create_state_dict_from_script('./moviecommands/moviecommands5.txt')
movie.make_movie(resultPath + fileName + "-" +\
                dataNum + "-B.mov",fps=10)

# write to external file as Numpy array
da.to_npy_stack(resultPath + fileName + "-" +\
                dataNum + "-B.npy",yout1)