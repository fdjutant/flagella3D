#%% Import all necessary libraries
import sys
sys.path.insert(0, './modules')
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})
from sklearn.decomposition import PCA
import napari
import msd
from skimage import measure
from scipy.optimize import least_squares
from naparimovie import Movie
from pathlib import Path
from scipy import optimize
import os.path
import pickle
import time
import tifffile
import sys
sys.stdout.write('\a')
sys.stdout.flush()

# time settings in the light sheet
pxum = 0.115
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
exp3D_sec = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 
pxum = 0.115

this_file_dir = os.path.join(os.path.dirname(os.path.abspath("./")),
                            'Dropbox (ASU)','Research')
thresholdFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Flagella-data', 'threshold-labKit')
intensityFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Flagella-data', 'TIF-files')

thresholdFiles = list(Path(thresholdFolder).glob("*-LabKit-*.tif"))
intensityFiles = list(Path(intensityFolder).glob("*.tif"))

# chose which file to analyze
whichFiles = 76
imgs_thresh = tifffile.imread(thresholdFiles[whichFiles])
imgs = tifffile.imread(intensityFiles[whichFiles])
print(intensityFiles[whichFiles].name)
print(thresholdFiles[whichFiles].name)

#%% Compute CM then generate n1, n2, n3
nt = len(imgs_thresh)
# nt = 130

frame_start = 0
frame_end = nt

blobBin = []
xb = []
xp = []
cm = np.zeros((nt,3))
n1s = np.zeros((nt, 3))
n2s = np.zeros((nt, 3))
n3s = np.zeros((nt, 3))
m2s = np.zeros((nt, 3))
m3s = np.zeros((nt, 3))
r_coms = np.zeros((nt, 3))
flagella_len = np.zeros(nt)
radial_dist_pt = np.zeros(nt)
blob_size = np.zeros(nt)
fitImage = np.zeros(imgs.shape,dtype='uint8')
params = np.zeros([nt,3])

tstart = time.perf_counter()

# for frame in range(nt):
for frame in range(frame_start,frame_end):
    
    print('frame: %d, time (sec): %.2f' %(frame, time.perf_counter()-tstart) )
       
    # grab current image
    img = np.array(imgs_thresh[frame]).astype('bool')
    
    # erosion: thinning the threshold
    # kernel = np.ones((4,4), np.uint8)
    # img = cv2.erode(img, kernel, iterations=1)

    # label and measure every clusters
    blobs = measure.label(img, background=0)
    labels = np.arange(1, blobs.max() + 1, dtype=int)
    sizes = np.array([np.sum(blobs == l) for l in labels])
    
    # keep only the largest cluster  
    max_ind = np.argmax(sizes)
    blob_size[frame] = sizes[max_ind]
    
    # mask showing which pixels ae in largest cluster
    blob = blobs == labels[max_ind]
    
    # store threshold/binarized image
    blobBin.append(blob)
    
    # ######################################
    # extract coordinates and center of mass
    # ######################################
    # extract coordinates
    X0 = np.argwhere(blob).astype('float')  # coordinates 
    xb.append(X0)                           # store coordinates
    
    # compute center of mass
    CM1 = np.array([sum(X0[:,j]) for j in range(X0.shape[1])])/X0.shape[0]
    cm[frame,:] = CM1 # store center of mass
    
    # ##############################################################
    # determine axis n1 from PCA and consistency with previous point
    # ##############################################################
    coords = X0 - CM1 # shift all the coordinates into origin
    pca = PCA(n_components=3)
    pca.fit(coords)
    n1s[frame] = pca.components_[0]
    m2s[frame] = pca.components_[1]
    m3s[frame] = pca.components_[2]
    
    # choose the sign of current n1 so it is as close as possible to n1 at the previous timestep
    if frame > frame_start and np.linalg.norm(n1s[frame] - n1s[frame -1]) > np.linalg.norm(n1s[frame] + n1s[frame - 1]):
        n1s[frame] = -n1s[frame]
        m2s[frame] = -m2s[frame]
        m3s[frame] = -m3s[frame]
        
    # ensure n1 x m2 = m3. Want to avoid possibility that n1 x m2 = -m3
    if np.linalg.norm(np.cross(n1s[frame], m2s[frame]) - m3s[frame]) > 1e-12:
        m3s[frame] = -m3s[frame]
        # check it worked
        assert np.linalg.norm(np.cross(n1s[frame], m2s[frame]) - m3s[frame]) < 1e-12
    
    # #####################################
    # rotate flagella on the principal axes
    # #####################################
    pts_on_n1 = n1s[frame, 0] * coords[:, 0] +\
                n1s[frame, 1] * coords[:, 1] +\
                n1s[frame, 2] * coords[:, 2]
    pts_on_m2 = m2s[frame, 0] * coords[:, 0] +\
                m2s[frame, 1] * coords[:, 1] +\
                m2s[frame, 2] * coords[:, 2]
    pts_on_m3 = m3s[frame, 0] * coords[:, 0] +\
                m3s[frame, 1] * coords[:, 1] +\
                m3s[frame, 2] * coords[:, 2]
    coord_on_principal = np.stack([pts_on_n1,
                                   pts_on_m2,
                                   pts_on_m3],axis=1)
    xp.append(coord_on_principal)

    # ##########################
    # Curve fit helix projection
    # ##########################
    # Fix amplitude and wave number, and set initial guess for phase
    amplitude = 1.65
    wave_number = 0.28
    phase = np.pi/10
    
    def cosine_fn(pts_on_n1,a): # model for "y" projection (on xz plane)
        return a[0] * np.cos(a[1] * pts_on_n1 + a[2])

    def sine_fn(pts_on_n1,a): # model for "z" projection (on xy plane)
        return a[0] * np.sin(a[1] * pts_on_n1 + a[2])
    
    def cost_fn(a):
        cost = np.concatenate((cosine_fn(pts_on_n1, a) -
                   pts_on_m2, sine_fn(pts_on_n1,a)-pts_on_m3)) / pts_on_n1.size
        return cost
    
    def jacobian_fn(a): #Cost gradient

        dy = cosine_fn(pts_on_n1,a) - pts_on_m2
        dz = sine_fn(pts_on_n1,a) - pts_on_m3
        
        g0 = dy  * np.cos(a[1] * pts_on_n1 + a[2]) +\
             dz  * np.sin(a[1] * pts_on_n1 + a[2])
        g2 = -dy * np.sin(a[1] * pts_on_n1 + a[2]) +\
              dz  * np.cos(a[1] * pts_on_n1 + a[2])
        g1 = pts_on_n1 * g2
        
        return np.array([g0.sum(),g1.sum(),g2.sum()])*2 / len(pts_on_n1)
    
    init_params = np.array([amplitude, wave_number, phase])
    lower_bounds = [1.5, 0.25, -np.inf]
    upper_bounds = [2.5, 0.3, np.inf]
    
    # fix a parameter: radius and wave number
    # results_fit = least_squares(lambda p: cost_fn([amplitude, wave_number, p[0]]),
    #                             init_params[2],
    #                             bounds=(lower_bounds[2], upper_bounds[2]))
    # phase = results_fit["x"][0]

    # fix a parameter: only radius    
    # results_fit = least_squares(lambda p: cost_fn([amplitude, p[0], p[1]]),
    #                             init_params[1:3],
    #                             bounds=(lower_bounds[1:3], upper_bounds[1:3]))
    # wave_number = results_fit["x"][0]
    # phase = results_fit["x"][1]

    # fix none    
    results_fit = least_squares(lambda p: cost_fn([p[0], p[1], p[2]]),
                                init_params[0:3],
                                bounds=(lower_bounds[0:3], upper_bounds[0:3]))
    amplitude = results_fit["x"][0]
    wave_number = results_fit["x"][1]
    phase = results_fit["x"][2]

    # Save fit parameters
    fit_params = np.array([amplitude, wave_number, phase])
    params[frame,:] = fit_params
    
    # #################################################
    # Construct 3D matrix for the fit for visualization    
    # #################################################
    # construct helix with some padding
    x = np.linspace(min(pts_on_n1),max(pts_on_n1),5000)
    ym = cosine_fn(x,fit_params)                          # mid
    yt = cosine_fn(x,fit_params) + 0.5*fit_params[1]      # top
    yb = cosine_fn(x,fit_params) - 0.5*fit_params[1]      # bottom
    zm = sine_fn(x,fit_params)                            # mid
    zt = sine_fn(x,fit_params)   + 0.5*fit_params[1]      # top
    zb = sine_fn(x,fit_params)   - 0.5*fit_params[1]      # bottom
    
    # stack the coordinates
    fit_P = np.array([x,yb,zb]).T
    fit_P = np.append(fit_P, np.array([x,yb,zm]).T,axis=0)
    fit_P = np.append(fit_P, np.array([x,yb,zt]).T,axis=0)
    fit_P = np.append(fit_P, np.array([x,ym,zb]).T,axis=0)
    fit_P = np.append(fit_P, np.array([x,ym,zm]).T,axis=0)
    fit_P = np.append(fit_P, np.array([x,ym,zt]).T,axis=0)
    fit_P = np.append(fit_P, np.array([x,yt,zb]).T,axis=0)
    fit_P = np.append(fit_P, np.array([x,yt,zm]).T,axis=0)
    fit_P = np.append(fit_P, np.array([x,yt,zt]).T,axis=0)
    
    # matrix rotation
    mrot = np.linalg.inv(np.vstack([n1s[frame],m2s[frame],m3s[frame]]))
    
    # inverse transform
    # fit_X = pca.inverse_transform(fit_P)+ CM1
    fit_X = np.matmul(mrot,fit_P.T).T + CM1
    fit_X = fit_X.astype('int')  
    fit_X = np.unique(fit_X,axis=0)
    
    # prepare our model image
    fit_img = np.zeros(img.shape)
    for idx in fit_X:
        i,j,k = idx
        if i < img.shape[0] and j < img.shape[1] and k < img.shape[2]:
            fit_img[i,j,k] = 1  # value of 1 for the fit
    fitImage[frame] = fit_img

    # ###################################
    # re-measure center of mass after fit
    # ###################################
    # extract coordinates
    X0_post_fit = np.argwhere(fit_img).astype('float')  # coordinates 
    
    # compute center of mass
    CM1 = np.array([sum(X0_post_fit[:,j]) for j in range(X0_post_fit.shape[1])])/X0_post_fit.shape[0]
    cm[frame,:] = CM1 # store center of mass

    # ##########################################
    # determine the flagella length along the n1
    # ##########################################
    flagella_len[frame] = np.max(pts_on_n1) - np.min(pts_on_n1)
    
    # ########################################
    # compute n2 and n3 from phase information
    # ######################################## 
    
    n2s[frame] = m2s[frame] * np.cos(phase) - m3s[frame] * np.sin(phase)
    n2s[frame] = n2s[frame] / np.linalg.norm(n2s[frame])
    # n3s[frame] = m2s[frame] * np.sin(phase) + m3s[frame] * np.cos(phase)
    
    assert n1s[frame].dot(n2s[frame]) < 1e-12

    # generate n3 such that coordinate system is right-handed
    n3s[frame] = np.cross(n1s[frame], n2s[frame])
    n3s[frame] = n3s[frame] / np.linalg.norm(n3s[frame])
    
    assert n1s[frame].dot(n3s[frame]) < 1e-12
    
    # negative control: not tracking any points
    # n2s[frame] = m2s[frame]
    # n3s[frame] = m3s[frame]

blobBin = np.array(blobBin)
print(thresholdFiles[whichFiles].name)

#%% View image, threshold, and fit together
viewer = napari.Viewer(ndisplay=3)      
viewer.add_image(imgs[frame_start:frame_end], contrast_limits=[110,250],\
                 scale=[0.115,.115,.115], blending='additive',\
                 multiscale=False,colormap='gray',opacity=1)
# viewer.add_image(blobBin[frame_start:frame_end], contrast_limits=[0,1],\
#                   scale=[0.115,.115,.115], blending='additive',\
#                   multiscale=False,colormap='green',opacity=0.4)
viewer.add_image(fitImage[frame_start:frame_end], contrast_limits=[0,1],\
                  scale=[0.115,.115,.115], blending='additive',\
                  multiscale=False,colormap='bop orange',opacity=1)
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
napari.run()

#%% Reload movies to check
# fname_save_mov = os.path.join(os.path.dirname(intensityFolder), 'MOV-files',
#                  intensityFiles[whichFiles].name[:-4] + '.mov')
# viewer = napari.Viewer(ndisplay=3)
# viewer.add_image(imgs[frame_start:frame_end], contrast_limits=[120,250],\
#                  scale=[0.115,.115,.115], blending='additive',\
#                  multiscale=False,colormap='gray',opacity=1)
# viewer.scale_bar.visible=True
# viewer.scale_bar.unit='um'
# viewer.scale_bar.position='top_right'
# viewer.axes.visible = True
# movie = Movie(myviewer=viewer)
# movie.create_state_dict_from_script('./moviecommands/mcRotate.txt')
# movie.make_movie(fname_save_mov,fps=10)

fname_save_mov = os.path.join(os.path.dirname(intensityFolder), 'MOV-files',
                 intensityFiles[whichFiles].name[:-4] + '-fit.mov')
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(imgs[frame_start:frame_end], contrast_limits=[110,250],\
                 scale=[0.115,.115,.115], blending='additive',\
                 multiscale=False,colormap='gray',opacity=1)
viewer.add_image(fitImage[frame_start:frame_end], contrast_limits=[0,1],\
                  scale=[0.115,.115,.115], blending='additive',\
                  multiscale=False,colormap='bop orange',opacity=1)
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
movie = Movie(myviewer=viewer)
movie.create_state_dict_from_script('./moviecommands/mcRotate.txt')
movie.make_movie(fname_save_mov,fps=10)
print('\a')