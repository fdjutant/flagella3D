#%% To input the file
import napari
import dask.array as da
from dask.array import from_zarr
import os.path
import pickle

# setName = 'suc90_25um_3ms'
setName = ''
# dataName = 'timelapse_2022_03_06-01_47_30'
# dataName = 'timelapse_2022_03_06-01_52_30'
# dataName = 'timelapse_2022_03_06-02_16_09'
# dataName = 'timelapse_2022_03_06-02_50_31'

# dataName = 'timelapse_2022_03_06-01_15_20'
dataName = 'timelapse_2022_03_06-03_47_02'
this_file_dir = os.path.join(os.path.dirname(os.path.abspath('./')),
                            'Dropbox (ASU)','Research')
folderName = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Snouty-data',
                          '2022-03-07', setName, dataName,
                          'decon_deskew_output', 'zarr',
                          'OPM_data_zarr.zarr','opm_data')

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
yC1 = 248; xC1 = 1286       # timelapse_2022_03_06-01_15_20

# yC1 = 182; xC1 = 924       # timelapse_2022_03_06-01_15_20

# yC1 = 120; xC1 = 1629    # timelapse_2022_03_06-01_52_30
# yC1 = 200; xC1 = 1408    # timelapse_2022_03_06-01_52_30
# yC1 = 134; xC1 = 1436    # timelapse_2022_03_06-01_52_30

# yC1 = 180; xC1 = 1070       # timelapse_2022_03_06-02_16_09
# yC1 = 170; xC1 = 795       # timelapse_2022_03_06-02_16_09
# yC1 = 302; xC1 = 181       # timelapse_2022_03_06-02_16_09
# yC1 = 193; xC1 = 1678       # timelapse_2022_03_06-02_16_09
# yC1 = 223; xC1 = 1284       # timelapse_2022_03_06-02_16_09

# yC1 = 168; xC1 = 1160       # timelapse_2022_03_06-02_50_31

top = 60; bottom = 60; left = 60; right = 60
yout1 = stack[:,:, yC1-top:yC1+bottom, xC1-left:xC1+right]

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

# setup up result file path
setName = 'suc90_25um_2ms'
savingFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          '20220305_' + setName, dataName)
if os.path.isdir(savingFolder) != True:
    os.mkdir(savingFolder) # create path 
savingFile = os.path.join(savingFolder, dataName) + '-A'

# write to external file as Numpy array
da.to_npy_stack(savingFile + '.npy', yout1)

# save movie  
# from naparimovie import Movie
# viewer = napari.Viewer(ndisplay=3)
# viewer.add_image(stack, contrast_limits=[100,200],
#                   scale=[0.115,.115,.115],colormap='gray')
# viewer.scale_bar.visible=True
# viewer.scale_bar.unit='um'
# viewer.scale_bar.position='top_right'
# viewer.axes.visible = True
# movie = Movie(myviewer=viewer)
# movie.create_state_dict_from_script('./moviecommands/mcTime.txt')
# movie.make_movie(savingFile + '-Full.mov',fps=10)

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