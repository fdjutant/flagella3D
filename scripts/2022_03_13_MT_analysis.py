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
from naparimovie import Movie
from pathlib import Path
import scipy.signal
from scipy import optimize
import cv2 
import os.path
import pickle

# time settings in the light sheet
pxum = 0.115
camExposure_ms = 3
sweep_um = 25
stepsize_nm = 400
exp3D_ms = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 
pxum = 0.115

# load raw images
# search_dir = Path(r"\\10.206.26.21\opm2\franky-sample-images")

setName = 'suc90_25um_3ms'
# dataName = 'timelapse_2022_03_06-01_52_30' # A and B
# dataName = 'timelapse_2022_03_06-02_16_09' # A and B and C
# dataName = 'timelapse_2022_03_06-01_57_50' # A 
# dataName = 'timelapse_2022_03_06-02_10_48' # A 
dataName = 'timelapse_2022_03_06-02_22_46' # A and B

this_file_dir = os.path.join(os.path.dirname(os.path.abspath("./")),
                            'Dropbox (ASU)','Research')
savingFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          '20220305_' + setName, dataName)

intensityFiles = list(Path(savingFolder).glob("*.npy"))
# thresholdFiles = list(Path(savingFolder).glob("*threshold.npy"))
whichFiles = intensityFiles[1]

imgs = da.from_npy_stack(whichFiles)
# imgs_thresh = da.from_npy_stack(thresholdFiles[0])

#%% binarization, extract coordinates, and compute CM
blobBin = []
xb = []
xp = []
nt = len(imgs)
cm = np.zeros((nt,3))
n1s = np.zeros((nt, 3))
n2s = np.zeros((nt, 3))
n3s = np.zeros((nt, 3))
r_coms = np.zeros((nt, 3))
flagella_len = np.zeros(nt)
radial_dist_pt = np.zeros(nt)


for frame in range(nt):
# for frame in range(0,1):
    
    print('frame:', frame)
    
    # ###############
    # threshold image
    # ###############
    
    # grab current image
    img_now = np.array(imgs[frame])

    # median filter image
    # img_now_med = scipy.signal.medfilt(img_now, kernel_size=(3,3,3))
    # img_now_med = img_now

    # threshold image
    img_thresh_med = img_now > 0.2 * np.max(img_now)
    
    # #############################################
    # pick cluster closest to the previous position
    # #############################################
    
    # label each pixels for each clusters
    blobs = np.uint8(measure.label(img_thresh_med, background=0))
    
    # close holes
    # kernel = np.ones((6,6), np.uint8); 
    # blobs = cv2.morphologyEx(blobs, cv2.MORPH_CLOSE, kernel, iterations=1)

    # pick the largest for the first one
    if frame == 0:   
        labels = np.unique(blobs.ravel())[1:] # spit out label numbers/tags
        sizes = np.array([np.argwhere(blobs==l).shape[0] for l in labels])
        keep = labels[np.argwhere((sizes == max(sizes)))[0]]
        blob = blobs == keep
        
        # constraint for the next frame with CM location
        X_f0 = np.argwhere(blob).astype('float')
        CM_f0 = np.array([sum(X_f0[:,j]) for j in range(X_f0.shape[1])])/X_f0.shape[0]
    # for the subsequent frame, pick the cluster closest to the previous frame    
    else: 
        labels = np.unique(blobs.ravel())[1:]
        CM_fN_diff = np.zeros(len(labels))
        CM_fN = np.zeros([len(labels), 3])
        for l in range(len(labels)): # goes through every clusters
            X_fN = np.argwhere(blobs==l+1)
            CM_fN[l] = np.array([sum(X_fN[:,j]) for j in range(X_fN.shape[1])])/X_fN.shape[0]
            CM_fN_diff[l] = np.linalg.norm(CM_f0-CM_fN[l])
        closest_idx = np.where(CM_fN_diff == min(CM_fN_diff))[0][0]
        CM_f0 = CM_fN[closest_idx] # update the constraint value
        keep = labels[closest_idx]
        blob = blobs == keep
    
    # store threshold/binarized image
    blobBin.append(blob)
    
    # ######################################
    # extract coordinates and center of mass
    # ######################################
    
    # extract coordinates
    X0 = np.argwhere(blob).astype('float') # coordinates 
    xb.append(X0) # store coordinates
    
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
    n2s[frame] = pca.components_[1]
    n3s[frame] = pca.components_[2]

    # choose the sign of current n1 so it is as close as possible to n1 at the previous timestep
    if frame > 0 and np.linalg.norm(n1s[frame] - n1s[frame -1]) > np.linalg.norm(n1s[frame] + n1s[frame - 1]):
        n1s[frame] = -n1s[frame]
        n2s[frame] = -n2s[frame]
        n3s[frame] = -n3s[frame]
        
    # #####################################
    # rotate flagella on the principal axes
    # #####################################
    dist_projected_along_n1 = n1s[frame, 0] * coords[:, 0] +\
                              n1s[frame, 1] * coords[:, 1] +\
                              n1s[frame, 2] * coords[:, 2]
    dist_projected_along_n2 = n2s[frame, 0] * coords[:, 0] +\
                              n2s[frame, 1] * coords[:, 1] +\
                              n2s[frame, 2] * coords[:, 2]
    dist_projected_along_n3 = n3s[frame, 0] * coords[:, 0] +\
                              n3s[frame, 1] * coords[:, 1] +\
                              n3s[frame, 2] * coords[:, 2]
    coord_on_principal = np.stack([dist_projected_along_n1,
                                   dist_projected_along_n2,
                                   dist_projected_along_n3],axis=1)
    xp.append(coord_on_principal)

    # ##########################################
    # determine the flagella length along the n1
    # ##########################################
    flagella_len[frame] = np.max(dist_projected_along_n1) - np.min(dist_projected_along_n1)

    # ##########################################
    # find the furthest point along the flagella
    # and the positive n1 direction
    # and use this to determine n2
    # ##########################################
    ind_pt = np.argmax(dist_projected_along_n1)
    coord_pt = coords[ind_pt]

    # project out n1
    coord_pt_proj = coord_pt - (coord_pt.dot(n1s[frame])) * n1s[frame]

    # check the radial distance of this point from the center
    radial_dist_pt[frame] = np.linalg.norm(coord_pt_proj)

    # generate n2 from this
    # n2s[frame] = coord_pt_proj / np.linalg.norm(coord_pt_proj)

    # assert n1s[frame].dot(n2s[frame]) < 1e-12

    # generate n3 such that coordinate system is right-handed
    # n3s[frame] = np.cross(n1s[frame], n2s[frame])
        
# convert to dask array
blobBin = np.array(blobBin)
xp = np.array(xp, dtype=object)

# compute rotation displacement
nt = len(n1s)
dpitch = np.zeros(nt)
droll = np.zeros(nt)
dyaw = np.zeros(nt)
for frame in range(nt-1):
    dpitch[frame] = np.dot(n2s[frame], n1s[frame+1] - n1s[frame])
    droll[frame] = np.dot(n3s[frame], n2s[frame+1] - n2s[frame])
    dyaw[frame] = np.dot(n1s[frame], n3s[frame+1] - n3s[frame])

EuAng = np.zeros([nt,3])
for frame in range(nt):
    EuAng[frame,0] = np.sum(dpitch[0:frame+1])
    EuAng[frame,1] = np.sum(droll[0:frame+1])
    EuAng[frame,2] = np.sum(dyaw[0:frame+1])
    
disp_pitch = np.diff(EuAng[:,0])
disp_roll = np.diff(EuAng[:,1])
disp_yaw = np.diff(EuAng[:,2])

disp_Ang = np.stack([disp_pitch,disp_roll,disp_yaw],axis=1)
firstone = np.array([[0,0,0]])
disp_Ang = np.vstack([firstone, disp_Ang])

# compute translation displacement
disp_n1 = []; disp_n2 = []; disp_n3 =[];
for i in range(nt-1):
    
    # displacement in Cartesian coordinates
    deltaX = ( cm[i+1,0] - cm[i,0] ) * 0.115
    deltaY = ( cm[i+1,1] - cm[i,1] ) * 0.115
    deltaZ = ( cm[i+1,2] - cm[i,2] ) * 0.115
    deltaXYZ = np.array([deltaX, deltaY, deltaZ])
    
    # displcament in local axes
    disp_n1.append(n1s[i,0]*deltaXYZ[0] + 
                   n1s[i,1]*deltaXYZ[1] +
                   n1s[i,2]*deltaXYZ[2]) # parallel
    disp_n2.append(n2s[i,0]*deltaXYZ[0] + 
                   n2s[i,1]*deltaXYZ[1] +
                   n2s[i,2]*deltaXYZ[2]) # perp1
    disp_n3.append(n3s[i,0]*deltaXYZ[0] + 
                   n3s[i,1]*deltaXYZ[1] +
                   n3s[i,2]*deltaXYZ[2]) # perp2
disp_n1 = np.array(disp_n1)
disp_n2 = np.array(disp_n2)
disp_n3 = np.array(disp_n3)

disp = np.stack([disp_n1,disp_n2,disp_n3],axis=1)
firstone = np.array([[0,0,0]])
disp= np.vstack([firstone,disp])

print('Length [um] = %.2f with std = %.2f' %(np.mean(flagella_len)*0.115,
                                             np.std(flagella_len)*0.115))
print(nt)

#%% Check thresholding
viewer = napari.Viewer(ndisplay=3)      
viewer.add_image(imgs, contrast_limits=[100,400],\
                    scale=[0.115,.115,.115],\
                    multiscale=False,colormap='gray',opacity=1)
viewer.add_image(blobBin, contrast_limits=[0,1],\
                    scale=[0.115,.115,.115],\
                    multiscale=False,colormap='green',opacity=0.5)
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.axes.visible = True
napari.run()

#%% Perform vector analysis & MSD
# initialize msd
msd_N = []; msd_S1 = []; msd_S2 = []; msd_NR = []
msd_P = []; msd_R = []; msd_Y = []; msd_CM = []
nInterval = 50

# center-of-mass tracking
nt = len(cm)
dstCM = np.zeros(nt)
for i in range(len(cm)): dstCM[i] = np.linalg.norm(cm[i])

# MSD: mean square displacement

MSD_N, MSD_S1, MSD_S2, MSD_NR = msd.trans_MSD_Namba(nt,
                                          cm, EuAng[:,1],
                                          n1s, n2s, n3s,
                                          exp3D_ms, nInterval)
MSD_P = msd.regMSD(nt, EuAng[:,0], exp3D_ms, nInterval)
MSD_R = msd.regMSD(nt, EuAng[:,1], exp3D_ms, nInterval)
MSD_Y = msd.regMSD(nt, EuAng[:,2], exp3D_ms, nInterval)
MSD_CM = msd.regMSD(nt, dstCM, exp3D_ms, nInterval)

msd_N.append(MSD_N); msd_S1.append(MSD_S1); msd_S2.append(MSD_S2)
msd_NR.append(MSD_NR)
msd_P.append(MSD_P); msd_R.append(MSD_R); msd_Y.append(MSD_Y)
msd_CM.append(MSD_CM)

# Fit MSD with y = Const + B*x for N, S, NR, PY, R
Nfit = 10
xtime = np.linspace(1,Nfit,Nfit)
def MSDfit(x, a, b): return b + a * x  
fit_N,fitN_const  = optimize.curve_fit(MSDfit, xtime, MSD_N[0:Nfit])[0]
fit_S,fitS_const  = optimize.curve_fit(MSDfit, xtime,
                        np.mean([MSD_S1[0:Nfit],MSD_S2[0:Nfit]],axis=0))[0]
fit_NR,fitNR_const = optimize.curve_fit(MSDfit, xtime, MSD_NR[0:Nfit])[0]
fit_PY,fitPY_const = optimize.curve_fit(MSDfit, xtime,
                          np.mean([MSD_P[0:Nfit],MSD_Y[0:Nfit]],axis=0))[0]
fit_R,fitR_const   = optimize.curve_fit(MSDfit, xtime, MSD_R[0:Nfit])[0]
fit_CM,fitCM_const = optimize.curve_fit(MSDfit, xtime, MSD_CM[0:Nfit])[0]

# Additional fit
fit_S1,fitS1_const  = optimize.curve_fit(MSDfit, xtime, MSD_S1[0:Nfit])[0]
fit_S2,fitS2_const  = optimize.curve_fit(MSDfit, xtime, MSD_S2[0:Nfit])[0]
fit_P, fitP_const  = optimize.curve_fit(MSDfit, xtime, MSD_P[0:Nfit])[0]
fit_Y, fitY_const  = optimize.curve_fit(MSDfit, xtime, MSD_Y[0:Nfit])[0]

# plot MSD and fit
xaxis = np.arange(1,nInterval+1)
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 12})
fig0,ax0 = plt.subplots(dpi=150, figsize=(6,5))
fig0.suptitle('data: %s\n' %os.path.basename(whichFiles) +
              '$N_{trajectory}$ = %i' %nt + ', '
              '$L$ = %.3f $\pm$ %.3f $\mu$m' %(np.round(np.mean(flagella_len)*pxum,3),
                                                  np.round(np.std(flagella_len)*pxum,3))
              )
ax0.plot(xaxis*exp3D_ms,MSD_N,c='C0',marker="^",mfc='none',
          ms=5,ls='None',alpha=1)   
# ax0.plot(xaxis*exp3D_ms,np.mean([MSD_S1,MSD_S2],axis=0),
ax0.plot(xaxis*exp3D_ms,MSD_S2,c='C1',marker="s",mfc='none',
          ms=5,ls='None',alpha=1)
ax0.plot(xaxis*exp3D_ms,fitN_const + fit_N*xaxis,
         c='C0',alpha=1,label='_nolegend_')
ax0.plot(xaxis*exp3D_ms,fitS2_const + fit_S2*xaxis,
         c='C1',alpha=1,label='_nolegend_')
ax0.set_xlabel(r'Lag time [sec]');
ax0.set_ylabel(r'MSD [$\mu m^2$]')
# ax0.set_ylim([0, 2]);
ax0.set_xlim([0, nInterval*exp3D_ms])
ax0.legend(['$D_\parallel=$ %.3f $\mu$m/sec$^2$' %np.round(fit_N/(2*exp3D_ms),3),
            # '$D_{\perp}=$ %.3f $\mu$m/sec$^2$' %np.round(fit_S2/(2*exp3D_ms),3) ])
            "$D_{\perp}=$ %.3f $\mu$m/sec$^2$, %.3f $\mu$m/sec$^2$" %(np.round(fit_S1/(2*exp3D_ms),3),
                                      np.round(fit_S2/(2*exp3D_ms),3)) ])

# Store diffusion coefficient and length as *pkl
savingPKL = os.path.dirname(whichFiles) + os.path.basename(whichFiles)[-6:-4] 
data = {"lengthMean": np.mean(flagella_len)*pxum,
        "lengthSTD": np.std(flagella_len)*pxum,
        "Dpar": fit_N/(2*exp3D_ms),
        "Dperp": fit_S2/(2*exp3D_ms),
        "Drot": fit_P/(2*exp3D_ms),
        "MSD_par": MSD_N,
        "MSD_perp": MSD_S2,
        # "MSD_perp": (MSD_S1 + MSD_S2)*0.5,
        # "MSD_rot": 0.5*(MSD_P + MSD_Y),
        "MSD_rot": MSD_P,
        "exp3D_ms": exp3D_ms}
with open(savingPKL + '-result.pkl', "wb") as f:
      pickle.dump(data, f)

#%% save movie  
savingPKL = os.path.dirname(whichFiles) + os.path.basename(whichFiles)[-6:-4] 
savingMovie_Time = savingPKL + '-Time-Threshold.mov'
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(imgs, contrast_limits=[0,300],\
                    scale=[0.115,.115,.115],\
                    multiscale=False,colormap='gray',opacity=1)
viewer.add_image(blobBin, contrast_limits=[0,1],\
                    scale=[0.115,.115,.115],\
                    multiscale=False,colormap='green',opacity=0.5)
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
movie = Movie(myviewer=viewer)
movie.create_state_dict_from_script('./moviecommands/mcTime.txt')
movie.make_movie(savingMovie_Time,fps=10)

savingMovie_Rotate = savingPKL + '-Rotate-Threshold.mov'
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(imgs, contrast_limits=[0,300],\
                    scale=[0.115,.115,.115],\
                    multiscale=False,colormap='gray',opacity=1)
viewer.add_image(blobBin, contrast_limits=[0,1],\
                    scale=[0.115,.115,.115],\
                    multiscale=False,colormap='green',opacity=0.5)
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
movie = Movie(myviewer=viewer)
movie.create_state_dict_from_script('./moviecommands/mcRotate.txt')
movie.make_movie(savingMovie_Rotate,fps=10)
