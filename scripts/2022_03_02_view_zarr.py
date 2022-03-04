#%% To input the file
import napari
import dask.array as da
from dask.array import from_zarr
from naparimovie import Movie
import os.path
import pickle

setName = "20220227_franky_MT_suc99_20um"
# dataName = "timelapse_2022_02_28-03_03_30"
dataName = "timelapse_2022_02_28-03_17_17"
this_file_dir = os.path.dirname(os.path.abspath("./"))
folderName = os.path.join(this_file_dir,
                          "DNA-Rotary-Motor", "Helical-nanotubes",
                          "Light-sheet-OPM", "Snouty-data",
                          "2022-02-27", setName, dataName,
                          "decon_deskew_output", "zarr",
                          "OPM_data_zarr.zarr","opm_data")
savingFolder = os.path.join(this_file_dir,
                          "DNA-Rotary-Motor", "Helical-nanotubes",
                          "Light-sheet-OPM", "Result-data",
                          setName, dataName)

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
# yC1 = 268; xC1 = 680
# yC1 = 230; xC1 = 130;

yC1 = 260; xC1 = 1385

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
tL1 = 0; tR1 = 160;
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

# setup up result file path
if os.path.isdir(savingFolder) != True:
    os.mkdir(savingFolder) # create path 
savingFile = os.path.join(savingFolder, dataName) + '-B'

# write to external file as Numpy array
da.to_npy_stack(savingFile + '.npy', yout1)

# save movie  
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(stack, contrast_limits=[100,200],
                  scale=[0.115,.115,.115],colormap='gray')
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
movie = Movie(myviewer=viewer)
movie.create_state_dict_from_script('./moviecommands/mcTime.txt')
movie.make_movie(savingFile + '-Full.mov',fps=10)

# write cropPosition to a pickle file
data = {"yC1": yC1,
        "xC1": xC1,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "tL1": tL1,
        "tR1": tR1}
with open(savingFile + '.pkl', "wb") as f:
     pickle.dump(data, f)