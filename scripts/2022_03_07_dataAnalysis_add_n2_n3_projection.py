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

# time settings in the light sheet
pxum = 0.115
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
exp3D_ms = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 
pxum = 0.115

#%% load raw images
search_dir = Path(r"\\10.206.26.21\opm2\franky-sample-images")
intensityFiles = list(Path(search_dir).glob("*A.npy"))
thresholdFiles = list(Path(search_dir).glob("*threshold.npy"))

imgs = da.from_npy_stack(intensityFiles[0])
imgs_thresh = da.from_npy_stack(thresholdFiles[0])

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
    img_now_med = scipy.signal.medfilt(img_now, kernel_size=(3, 3, 3))

    # threshold image
    img_thresh_med = img_now_med > 0.8 * np.max(img_now_med)
    
    # #############################################
    # pick cluster closest to the previous position
    # #############################################
    
    # label each pixels for each clusters
    blobs = np.uint8(measure.label(img_thresh_med, background=0))
    
    # close holes
    kernel = np.ones((6,6), np.uint8); 
    blobs = cv2.morphologyEx(blobs, cv2.MORPH_CLOSE, kernel, iterations=1)

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
    # pca = PCA(n_components=3)
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
    n2s[frame] = coord_pt_proj / np.linalg.norm(coord_pt_proj)

    assert n1s[frame].dot(n2s[frame]) < 1e-12

    # generate n3 such that coordinate system is right-handed
    n3s[frame] = np.cross(n1s[frame], n2s[frame])
        
# convert to dask array
blobBin = da.from_array(blobBin)
xp = np.array(xp)

#%% compute rotation displacement
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

print('Length [um] = %.2f with std = %.2f' %(np.mean(flagella_len)*0.115,np.std(flagella_len)*0.115))
print(nt)

#%% Plot the coordinate points
for frame in range(len(blobBin)):
# for frame in range(0,1):
        
    xb0 = xb[frame] - cm[frame]
    xp0 = xp[frame]

    fig = plt.figure(dpi=150, figsize = (10, 6))
    fig.suptitle('data: %s\n' %os.path.basename(thresholdFiles[0]) +
                  'frame-num = ' + str(frame).zfill(3) + ', '
                  'length = %.3f $\mu$m' %np.round(flagella_len[frame]*pxum,3) + ','
                  'radius = %.3f $\mu$m\n' %np.round(radial_dist_pt[frame]*pxum,3) +
                   '$\Delta_\parallel$ = %.3f $\mu$m, ' %np.round(disp[frame,0],3) +
                   '$\Delta_{\perp 1}$ = %.3f $\mu$m, ' %np.round(disp[frame,1],3) +
                   '$\Delta_{\perp 2}$ = %.3f $\mu$m\n' %np.round(disp[frame,2],3) +
                  '$\Delta_\psi$ = %.3f rad, ' %np.round(disp_Ang[frame,1],3) +
                  '$\Delta_\gamma$ = %.3f rad, ' %np.round(disp_Ang[frame,0],3) +
                  '$\Delta_\phi$ = %.3f rad\n' %np.round(disp_Ang[frame,2],3)
                  )
    ax0 = fig.add_subplot(231,projection='3d')
    ax2 = fig.add_subplot(232,projection='3d')
    ax3 = fig.add_subplot(235,projection='3d')
    ax4 = fig.add_subplot(234,projection='3d')
    ax5 = fig.add_subplot(233,projection='3d')
    ax6 = fig.add_subplot(236,projection='3d')
    pxum = 0.115

    ## plot 1
    x, y, z = np.array([[-40,0,0],[0,-40,0],[0,0,-40]])*pxum
    u, v, w = np.array([[60,0,0],[0,60,0],[0,0,60]])*pxum
    ax0.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
    edgePoint = 40
    ax0.set_ylim(-edgePoint*pxum,edgePoint*pxum)
    ax0.set_xlim(-edgePoint*pxum,edgePoint*pxum)
    ax0.set_zlim(-edgePoint*pxum,edgePoint*pxum)
    ax0.view_init(elev=30, azim=30)
    ax0.set_xlabel(r'x [$\mu m$]'); ax0.set_ylabel(r'y [$\mu m$]')
    ax0.set_zlabel(r'z [$\mu m$]')
    ax0.scatter(xb0[:,0]*pxum, xb0[:,1]*pxum,\
                xb0[:,2]*pxum, c = 'k',alpha=0.1, s=10)
    # ax0.scatter(endpt[frame,0]*pxum,\
    #             endpt[frame,1]*pxum,\
    #             endpt[frame,2]*pxum, c = 'r', alpha=0.5, s=50) 
    origin = [0,0,0]
    X, Y, Z = zip(origin)
    Un1, Vn1, Wn1 = zip(list(5*n1s[frame])) 
    Un2, Vn2, Wn2 = zip(list(5*n2s[frame])) 
    Un3, Vn3, Wn3 = zip(list(5*n3s[frame]))
    ax0.quiver(X,Y,Z,Un1,Vn1,Wn1,color='r')
    ax0.quiver(X,Y,Z,Un2,Vn2,Wn2,color='g')
    ax0.quiver(X,Y,Z,Un3,Vn3,Wn3,color='b')

    ## plot 2
    x, y, z = np.array([[-40,0,0],[0,-40,0],[0,0,-40]])*pxum
    u, v, w = np.array([[60,0,0],[0,60,0],[0,0,60]])*pxum
    ax2.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
    edgePoint = 40
    ax2.set_ylim(-edgePoint*pxum,edgePoint*pxum)
    ax2.set_xlim(-edgePoint*pxum,edgePoint*pxum)
    ax2.set_zlim(-edgePoint*pxum,edgePoint*pxum)
    ax2.view_init(elev=0, azim=90)
    ax2.set_xlabel(r'x [$\mu m$]')
    ax2.set_yticks([])
    ax2.set_zlabel(r'z [$\mu m$]')
    ax2.scatter(xb0[:,0]*pxum, xb0[:,1]*pxum,\
                xb0[:,2]*pxum, c = 'k',alpha=0.1, s=10)
    # ax2.scatter(endpt[frame,0]*pxum,\
    #             endpt[frame,1]*pxum,\
    #             endpt[frame,2]*pxum, c = 'r', alpha=0.5, s=50) 
    origin = [0,0,0]
    X, Y, Z = zip(origin)
    Un1, Vn1, Wn1 = zip(list(5*n1s[frame])) 
    Un2, Vn2, Wn2 = zip(list(5*n2s[frame])) 
    Un3, Vn3, Wn3 = zip(list(5*n3s[frame]))
    ax2.quiver(X,Y,Z,Un1,Vn1,Wn1,color='r')
    ax2.quiver(X,Y,Z,Un2,Vn2,Wn2,color='g')
    ax2.quiver(X,Y,Z,Un3,Vn3,Wn3,color='b')

    ## plot 3
    x, y, z = np.array([[-40,0,0],[0,-40,0],[0,0,-40]])*pxum
    u, v, w = np.array([[60,0,0],[0,60,0],[0,0,60]])*pxum
    ax3.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
    edgePoint = 40
    ax3.set_ylim(-edgePoint*pxum,edgePoint*pxum)
    ax3.set_xlim(-edgePoint*pxum,edgePoint*pxum)
    ax3.set_zlim(-edgePoint*pxum,edgePoint*pxum)
    ax3.view_init(elev=0, azim=0)
    ax3.set_xticks([])
    ax3.set_ylabel(r'y [$\mu m$]')
    ax3.set_zlabel(r'z [$\mu m$]')
    ax3.scatter(xb0[:,0]*pxum, xb0[:,1]*pxum,\
                xb0[:,2]*pxum, c = 'k',alpha=0.1, s=10)
    # ax3.scatter(endpt[frame,0]*pxum,\
    #            endpt[frame,1]*pxum,\
    #            endpt[frame,2]*pxum, c = 'r', alpha=0.5, s=50) 
    origin = [0,0,0]
    X, Y, Z = zip(origin)
    Un1, Vn1, Wn1 = zip(list(5*n1s[frame])) 
    Un2, Vn2, Wn2 = zip(list(5*n2s[frame])) 
    Un3, Vn3, Wn3 = zip(list(5*n3s[frame]))
    ax3.quiver(X,Y,Z,Un1,Vn1,Wn1,color='r')
    ax3.quiver(X,Y,Z,Un2,Vn2,Wn2,color='g')
    ax3.quiver(X,Y,Z,Un3,Vn3,Wn3,color='b')

    ## plot 4
    x, y, z = np.array([[-40,0,0],[0,-40,0],[0,0,-40]])*pxum
    u, v, w = np.array([[60,0,0],[0,60,0],[0,0,60]])*pxum
    ax4.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
    edgePoint = 40
    ax4.set_ylim(-edgePoint*pxum,edgePoint*pxum)
    ax4.set_xlim(-edgePoint*pxum,edgePoint*pxum)
    ax4.set_zlim(-edgePoint*pxum,edgePoint*pxum)
    ax4.view_init(elev=90, azim=0)
    ax4.set_xlabel(r'x [$\mu m$]')
    ax4.set_ylabel(r'y [$\mu m$]')
    ax4.set_zticks([])
    ax4.scatter(xb0[:,0]*pxum, xb0[:,1]*pxum,\
                xb0[:,2]*pxum, c = 'k',alpha=0.1, s=10)
    # ax4.scatter(endpt[frame,0]*pxum,\
    #            endpt[frame,1]*pxum,\
    #            endpt[frame,2]*pxum, c = 'r', alpha=0.5, s=50) 
    origin = [0,0,0]
    X, Y, Z = zip(origin)
    Un1, Vn1, Wn1 = zip(list(5*n1s[frame])) 
    Un2, Vn2, Wn2 = zip(list(5*n2s[frame])) 
    Un3, Vn3, Wn3 = zip(list(5*n3s[frame]))
    ax4.quiver(X,Y,Z,Un1,Vn1,Wn1,color='r')
    ax4.quiver(X,Y,Z,Un2,Vn2,Wn2,color='g')
    ax4.quiver(X,Y,Z,Un3,Vn3,Wn3,color='b')
    
    ## plot 5
    x, y, z = np.array([[-40,0,0],[0,-40,0],[0,0,-40]])*pxum
    u, v, w = np.array([[60,0,0],[0,60,0],[0,0,60]])*pxum
    ax5.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
    edgePoint = 40
    ax5.set_ylim(-edgePoint*pxum,edgePoint*pxum)
    ax5.set_xlim(-edgePoint*pxum,edgePoint*pxum)
    ax5.set_zlim(-edgePoint*pxum,edgePoint*pxum)
    ax5.view_init(elev=0, azim=90)
    ax5.set_xlabel(r'x [$\mu m$]')
    ax5.set_yticks([])
    ax5.set_zlabel(r'z [$\mu m$]')
    ax5.scatter(xp0[:,0]*pxum, xp0[:,1]*pxum,\
                xp0[:,2]*pxum, c = 'k',alpha=0.1, s=10)
    #ax5.figure.savefig(os.path.join(savingFolder, ThName + '-' + 
    #                                str(frame).zfill(3) + '.png'))
    
    ## plot 6
    x, y, z = np.array([[-40,0,0],[0,-40,0],[0,0,-40]])*pxum
    u, v, w = np.array([[60,0,0],[0,60,0],[0,0,60]])*pxum
    ax6.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
    edgePoint = 40
    ax6.set_ylim(-edgePoint*pxum,edgePoint*pxum)
    ax6.set_xlim(-edgePoint*pxum,edgePoint*pxum)
    ax6.set_zlim(-edgePoint*pxum,edgePoint*pxum)
    ax6.view_init(elev=90, azim=0)
    ax6.set_xlabel(r'x [$\mu m$]')
    ax6.set_ylabel(r'y [$\mu m$]')
    ax6.set_zticks([])
    ax6.scatter(xp0[:,0]*pxum, xp0[:,1]*pxum,\
                xp0[:,2]*pxum, c = 'k',alpha=0.1, s=10)
    
    
    # save to folder
    savingSnapshots = os.path.join(thresholdFiles[0], 'snapshots')
    if os.path.isdir(savingSnapshots) != True:
        os.mkdir(savingSnapshots) # create path if non-existent
    ax6.figure.savefig(os.path.join(savingSnapshots, 
                                    str(frame).zfill(3) + '.png'))

#%% Perform vector analysis & MSD
# initialize msd
msd_N = []; msd_S1 = []; msd_S2 = []; msd_NR = []
msd_P = []; msd_R = []; msd_Y = []; msd_CM = []
nInterval = 50

# compute 3D exposure time
pxum = 0.115; 
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
exp3D_ms = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 

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
fit_S1,fitP_const  = optimize.curve_fit(MSDfit, xtime, MSD_S1[0:Nfit])[0]
fit_S2,fitY_const  = optimize.curve_fit(MSDfit, xtime, MSD_S2[0:Nfit])[0]
fit_P, fitP_const  = optimize.curve_fit(MSDfit, xtime, MSD_P[0:Nfit])[0]
fit_Y, fitY_const  = optimize.curve_fit(MSDfit, xtime, MSD_Y[0:Nfit])[0]

# plot MSD and fit
xaxis = np.arange(1,nInterval+1)
fig0,ax0 = plt.subplots(dpi=75, figsize=(6,5))
ax0.plot(xaxis*exp3D_ms,MSD_N,c='k',marker="^",mfc='none',
          ms=5,ls='None',alpha=0.5)   
ax0.plot(xaxis*exp3D_ms,np.mean([MSD_S1,MSD_S2],axis=0),
         c='k',marker="s",mfc='none',
          ms=5,ls='None',alpha=0.5)
ax0.plot(xaxis*exp3D_ms,fitN_const + fit_N*xaxis,
         c='k',alpha=0.5,label='_nolegend_')
ax0.plot(xaxis*exp3D_ms,fitS_const + fit_S*xaxis,
         c='k',alpha=0.5,label='_nolegend_')
ax0.set_xlabel(r'Lag time [sec]');
ax0.set_ylabel(r'MSD [$\mu m^2$]')
# ax0.set_ylim([0, 2]);
ax0.set_xlim([0, nInterval*exp3D_ms])
ax0.legend(["lengthwise","sidewise"])

print(fit_N/(2*exp3D_ms))
print(fit_S/(2*exp3D_ms))
print(fit_N/fit_S)

#%% Check binarization
viewer = napari.Viewer(ndisplay=3)      
viewer.add_image(imgs, contrast_limits=[100,140],\
                    scale=[0.115,.115,.115],\
                    multiscale=False,colormap='gray',opacity=0.5)
viewer.add_image(blobBin, contrast_limits=[0,1],\
                    scale=[0.115,.115,.115],\
                    multiscale=False,colormap='green',opacity=0.5)
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
napari.run()

#%% save movie  
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(imgs, contrast_limits=[100,140],\
                    scale=[0.115,.115,.115],\
                    multiscale=False,colormap='gray',opacity=0.5)
viewer.add_image(blobBin, contrast_limits=[0,1],\
                    scale=[0.115,.115,.115],\
                    multiscale=False,colormap='green',opacity=0.5)
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
movie = Movie(myviewer=viewer)
movie.create_state_dict_from_script('../moviecommands/mcTime.txt')
movie.make_movie(pathMovie,fps=10)