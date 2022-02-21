#%% To input the file
import napari
import dask.array as da
from dask.array import from_zarr
from naparimovie import Movie
import os.path

dataName = "MT-suc70-2nM-06"
this_file_dir = os.path.dirname(os.path.abspath("./"))
folderName = os.path.join(this_file_dir,
                          "DNA-Rotary-Motor", "Helical-nanotubes",
                          "Light-sheet-OPM", "Snouty-data",
                          "2021-12-22", dataName,
                          "timelapse_2021_12_17-08_54_33",
                          "deskew_output", "OPM_processed.zarr")
resultPath = os.path.join(this_file_dir,
                          "DNA-Rotary-Motor", "Helical-nanotubes",
                          "Light-sheet-OPM", "Result-data",
                          "20211221_suc70_MT")

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
yC1 = 212; xC1 = 759;

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
tL1 = 0; tR1 = 399;
# tL2 = 150; tR2 = 399;
# tL3 = 300; tR3 = 399;
# yout1 = np.concatenate((
#         stack[0,tL1:tR1,:, yC1-top:yC1+bottom, xC1-left:xC1+right],\
#         stack[0,tL2:tR2,:, yC1-top:yC1+bottom, xC1-left:xC1+right]),
#         axis=0);
yout1 = stack[tL1:tR1,:, yC1-top:yC1+bottom, xC1-left:xC1+right]
# viewer = napari.Viewer(ndisplay=3)      
# viewer.add_image(yout1, contrast_limits=[130,200],colormap='gray',opacity=1)
# napari.run()


# save movie  
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(stack, contrast_limits=[100,200],
                 scale=[0.115,.115,.115],colormap='gray')
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
movie = Movie(myviewer=viewer)
movie.create_state_dict_from_script('./moviecommands/moviecommands7.txt')
movie.make_movie(os.path.join(resultPath, dataName+"-ALL-2.mov"),fps=10)

# write to external file as Numpy array
da.to_npy_stack(os.path.join(resultPath, dataName+"-A.npy"),yout1)