#%% Import all necessary libraries
import sys
sys.path.insert(0, './modules')
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})
import napari
from movingHx import simulate_diff
from skimage.morphology import dilation
from naparimovie import Movie
import os.path
import tifffile
import pickle

# time settings in the light sheet
pxum = 0.115; 
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
# vol_exp = 1e-3/ (sweep_um/stepsize_nm * camExposure_ms) # in sec
vol_exp = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 

# input helix geometry
length_um = 8; radius_um = 0.5; pitchHx_um = 2.5 # all three in um

chirality = 1;  # left-handed: 1, right-handed: -1
resol = 300     # number of points/resolutions

# input helix motion
Nframes = 150
Dperp_um2sec = 0.10
Dpar_um2sec = 2 * Dperp_um2sec
Dpitch = 0.03
Droll = 1.5
Dyaw = 0.03

# input helix-and-noise level
hxInt = 200; hxVar = 0;
noiseInt = 50; noiseVar = 10;

# Generate diffusion
cmass, euler_angles, n1, n2, n3 = simulate_diff(Nframes, vol_exp,
                                                Dpar_um2sec, Dperp_um2sec,
                                                Dpitch, Droll, Dyaw)
cm_px = cmass.reshape(3,Nframes).T 
EuAng = euler_angles.reshape(3,Nframes).T
localAxes = np.zeros([Nframes,3,3])
for i in range(Nframes):
    localAxes[i] = np.array([ n1[i], n2[i], n3[i] ])

def createHx(phase):
    k = 2*np.pi / pitchHx
    points = []
    phase = 0        
    for i in range(resol):
        z = (-0.5 + i / (resol-1)) * length
        x = radius * np.cos(k*z + chirality*phase)
        y = radius * np.sin(k*z + chirality*phase)
        
        points.append([x,y,z])
    points = np.array(points)
    CM = sum(points)/len(points)
    return points-CM

def digitize(points,padding):
    points = points.astype('int')
    p = points.copy()

    if padding:
        for i in range(1, padding+1):
            for loc in range(1,27):
                shift = np.zeros(3,dtype='int')
                temp = loc
                for j in range(3):
                    for k in range(j+1):
                        temp = temp//3
                    shift[2-j] = temp%3
                shift += -1
                p = np.append(p,points+i*shift,axis=0)
    
    return np.unique(p,axis=0)

# convert to pixel units
length  = length_um  / pxum
radius  = radius_um  / pxum
pitchHx = pitchHx_um / pxum
Dpar  = Dpar_um2sec  / (pxum**2)
Dperp = Dperp_um2sec / (pxum**2)

# create the 3D time-series images
boxSize = int(length) + 300
intensity = np.zeros([Nframes,boxSize,boxSize,boxSize],dtype='uint8')

# generate 3D time-series images
for i in range(Nframes):
    print('frame: %d' %i)
    
    CM = cm_px[i]
    vectN = localAxes[i]
    
    img = np.zeros((boxSize,boxSize,boxSize),dtype='uint8')
    
    box = img.shape
    origin = np.array(box)/2
    
    # generate initial helix points:
    points = createHx(0)   

    # crete a 3D space
    n1 = vectN[0]
    n2 = vectN[1]
    n3 = vectN[2]
    axes = np.array([ n2, n3, n1 ])
    frame = CM + origin + np.matmul(axes.T,points.T).T
    frame = digitize(frame,0)
    for f in frame:
        x, y, z = f
        if hxVar == 0:
            img[x,y,z] = hxInt
        else:
            img[x,y,z] = np.random.uniform(hxInt - hxVar, hxInt + hxVar)

    img = dilation(img)
    img = dilation(img)
    
    # add salt-and-pepper noise 
    totalNum = img.shape[0]*img.shape[1]*img.shape[2]
    saltpepper = np.random.normal(noiseInt,noiseVar,totalNum)
    saltpepper = np.reshape(saltpepper,img.shape)
    img = np.add(img,saltpepper)
    del saltpepper # conserve memory|
    
    intensity[i] = img
    
print('--synthetic data creation is completed--')

#%% save to TIF and pkl
setName = 'simulation-001'

this_file_dir = os.path.join(os.path.dirname(os.path.abspath("./")),
                            'Dropbox (ASU)','Research')
intensityFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Simulation-data', 'TIF-files', setName)

fname_save_tiff = intensityFolder + '.tif'
img_to_save = tifffile.transpose_axes(intensity, "TZYX", asaxes="TZCYXS")
tifffile.imwrite(fname_save_tiff, img_to_save, imagej=True)

# save to PKL
data = {"pxum": pxum,
        "vol_exp": vol_exp,
        "Dpar": Dpar,
        "Dperp": Dperp,
        "Dpitch": Dpitch,
        "Droll": Droll,
        "Dyaw": Dyaw,
        "cm": cm_px,
        "origin": origin,
        "EuAng": EuAng,
        "localAxes": localAxes}

fname_save_pkl = intensityFolder + '.pkl'
with open(fname_save_pkl, "wb") as f:
     pickle.dump(data, f)
     
#%% View image, threshold, and fit together
viewer = napari.Viewer(ndisplay=3)      
viewer.add_image(intensity, contrast_limits=[0,300],\
                 scale=[0.115,.115,.115], blending='additive',\
                 multiscale=False,colormap='gray',opacity=1)
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
napari.run()


#%% Reload movies to check
fname_save_mov = intensityFolder + '.mov'
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(intensity, contrast_limits=[0,300],\
                 scale=[0.115,.115,.115], blending='additive',\
                 multiscale=False,colormap='gray',opacity=1)
# viewer.add_image(blobBin[frame_start:frame_end], contrast_limits=[0,1],\
#                   scale=[0.115,.115,.115], blending='additive',\
#                   multiscale=False,colormap='green',opacity=0.15)
# viewer.add_image(fitImage[frame_start:frame_end], contrast_limits=[0,1],\
#                   scale=[0.115,.115,.115], blending='additive',\
#                   rendering='iso', iso_threshold = 0.05,\
#                   multiscale=False,colormap='bop orange',opacity=.4)
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
movie = Movie(myviewer=viewer)
movie.create_state_dict_from_script('./moviecommands/mcRotate-v2.txt')
movie.make_movie(fname_save_mov,fps=10)
