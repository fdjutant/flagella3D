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
from scipy.optimize import minimize, least_squares
from naparimovie import Movie
from pathlib import Path
import scipy.signal
from scipy import optimize
import os.path
import pickle
import time

# time settings in the light sheet
pxum = 0.115
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
exp3D_ms = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 
pxum = 0.115

# load raw images
# search_dir = Path(r"\\10.206.26.21\opm2\franky-sample-images")

this_file_dir = os.path.join(os.path.dirname(os.path.abspath("./")),
                            'Dropbox (ASU)','Research')
thresholdFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Flagella-data', 'threshold-iterative')
intensityFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Flagella-data', 'suc-40')

thresholdFiles = list(Path(thresholdFolder).glob("*threshold*.npy"))
intensityFiles = list(Path(intensityFolder).glob("*.npy"))

whichFiles = 0
imgs_thresh = np.load(thresholdFiles[whichFiles])
imgs = da.from_npy_stack(intensityFiles[whichFiles])
print(thresholdFiles[whichFiles])

#%% Compute CM then generate n1, n2, n3
blobBin = []
xb = []
xp = []
nt = len(imgs_thresh)
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
fitImage = np.zeros(imgs.shape)
params = np.zeros([nt,3])
sin_params = np.zeros([nt,3])
cos_params = np.zeros([nt,3])
regcoeff = np.zeros(nt)
NiterRec = np.zeros(nt)

tstart = time.perf_counter()

frame_start = 0
frame_end = 300

# for frame in range(nt):
for frame in range(frame_start,frame_end):
    
    print('frame: %d, time: %.2f' %(frame, time.perf_counter()-tstart) )
       
    # import image
    img = imgs[frame]    
    
    # compute threshold pixel number
    blob = imgs_thresh[frame]
    blob_size[frame] = np.sum(blob == True)
    
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
    if frame > 0 and np.linalg.norm(n1s[frame] - n1s[frame -1]) > np.linalg.norm(n1s[frame] + n1s[frame - 1]):
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
    dist_projected_along_n1 = n1s[frame, 0] * coords[:, 0] +\
                              n1s[frame, 1] * coords[:, 1] +\
                              n1s[frame, 2] * coords[:, 2]
    dist_projected_along_m2 = m2s[frame, 0] * coords[:, 0] +\
                              m2s[frame, 1] * coords[:, 1] +\
                              m2s[frame, 2] * coords[:, 2]
    dist_projected_along_m3 = m3s[frame, 0] * coords[:, 0] +\
                              m3s[frame, 1] * coords[:, 1] +\
                              m3s[frame, 2] * coords[:, 2]
    coord_on_principal = np.stack([dist_projected_along_n1,
                                   dist_projected_along_m2,
                                   dist_projected_along_m3],axis=1)
    xp.append(coord_on_principal)

    # ##########################
    # Curve fit helix projection
    # ##########################
    xx = dist_projected_along_n1
    yy = dist_projected_along_m2
    zz = dist_projected_along_m3
    
    # Minimize both projection together
    amplitude = 1.65
    period = 25; wave_number = 0.28
    phase = np.pi/10
    
    def cosine_fn(xx,a): # model for "y" projection (on xz plane)
        return a[0] * np.cos(a[1] * xx + a[2])

    def sine_fn(xx,a): # model for "z" projection (on xy plane)
        return a[0] * np.sin(a[1] * xx + a[2])
    
    def cost_fn(a):
        cost = np.concatenate((cosine_fn(xx, a) - yy, sine_fn(xx,a)-zz)) / xx.size
        return cost
    
    def jacobian_fn(a): #Cost gradient

        dy = cosine_fn(xx,a) - yy
        dz = sine_fn(xx,a) - zz
        
        g0 = dy  * np.cos(a[1] * xx + a[2]) + dz  * np.sin(a[1] * xx + a[2])
        g2 = -dy * np.sin(a[1] * xx + a[2]) + dz  * np.cos(a[1] * xx + a[2])
        g1 = xx * g2
        
        return np.array([g0.sum(),g1.sum(),g2.sum()])*2 / len(xx)
    
    init_params = np.array([amplitude, wave_number, phase])
    lower_bounds = [0, 0, -np.inf]
    upper_bounds = [np.inf, np.inf, np.inf]
    
    # results_fit = least_squares(cost_fn, init_params,
    #                             bounds=(lower_bounds, upper_bounds))
    # fit_params = results_fit["x"]
    
    # fix a parameter: radius and wave number
    results_fit = least_squares(lambda p: cost_fn([amplitude, wave_number, p[0]]),
                                init_params[2],
                                bounds=(lower_bounds[2], upper_bounds[2]))
    fit_params = np.array([amplitude, wave_number, results_fit["x"][0]])

    print(fit_params)
    
    # Save fit parameters
    params[frame,:] = fit_params
    
    # #################################################
    # Construct 3D matrix for the fit for visualization    
    # #################################################
    # construct helix with some padding
    x = np.linspace(min(xx),max(xx),5000)
    ym = cosine_fn(x,fit_params)                        # mid
    yt = cosine_fn(x,fit_params)+0.5*fit_params[1]      # top
    yb = cosine_fn(x,fit_params)-0.5*fit_params[1]      # bot
    zm = sine_fn(x,fit_params)                        # mid
    zt = sine_fn(x,fit_params)+0.5*fit_params[1]      # top
    zb = sine_fn(x,fit_params)-0.5*fit_params[1]      # bot
    
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
        fit_img[i,j,k] = 1  # value of 1 for the fit
    fitImage[frame] = fit_img

    # ##########################################
    # determine the flagella length along the n1
    # ##########################################
    flagella_len[frame] = np.max(dist_projected_along_n1) -\
                          np.min(dist_projected_along_n1)

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

#%% View image and threshold together
viewer = napari.Viewer(ndisplay=3)      
viewer.add_image(imgs[frame_start:frame_end], contrast_limits=[100,400],\
                 scale=[0.115,.115,.115], blending='additive',\
                 multiscale=False,colormap='gray',opacity=1)
viewer.add_image(fitImage[frame_start:frame_end], contrast_limits=[0,1],\
                 scale=[0.115,.115,.115], blending='additive',\
                 multiscale=False,colormap='red',opacity=0.5)
viewer.add_image(imgs_thresh[frame_start:frame_end], contrast_limits=[0,1],\
                 scale=[0.115,.115,.115], blending='additive',\
                 multiscale=False,colormap='green',opacity=0.5)
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
napari.run()

#%% Plot 2D projections (XY, XZ, YZ)
for frame in range(frame_start,frame_end):
    
    xb0 = xb[frame] - cm[frame]
    xp0 = xp[frame]
    
    # x-y plot
    fig = plt.figure(dpi=150, figsize = (15, 6))
    ax0 = fig.add_subplot(231)
    ax1 = fig.add_subplot(232)
    ax2 = fig.add_subplot(233)
    ax3 = fig.add_subplot(234)
    ax4 = fig.add_subplot(235)
    ax5 = fig.add_subplot(236)
    
    ax0.axis('equal')
    ax0.scatter(xb0[:,0],xb0[:,1],c='k',alpha=0.3)
    ax0.set_xlabel(r'x [px]'); ax0.set_ylabel(r'y [px]')
    # fig0.savefig('filename.pdf')
    
    ax1.axis('equal')
    ax1.scatter(xb0[:,0],xb0[:,2],c='k',alpha=0.3)
    ax1.set_xlabel(r'x [px]'); ax1.set_ylabel(r'z [px]')
    
    # y-z plot
    ax2.axis('equal')
    ax2.scatter(xb0[:,1],xb0[:,2],c='k',alpha=0.3)
    ax2.set_xlabel(r'y [px]'); ax2.set_ylabel(r'z [px]')
    
    ax3.axis('equal')
    ax3.scatter(xp0[:,0],xp0[:,1],c='r',alpha=0.3)
    ax3.set_xlabel(r'x [px]'); ax3.set_ylabel(r'y [px]')
    
    # x-z plot
    ax4.axis('equal')
    ax4.scatter(xp0[:,0],xp0[:,2],c='r',alpha=0.3)
    ax4.set_xlabel(r'x [px]'); ax4.set_ylabel(r'z [px]')
    
    ax5.axis('equal')
    ax5.scatter(xp0[:,1],xp0[:,2],c='r',alpha=0.3)
    ax5.set_xlabel(r'y [px]'); ax5.set_ylabel(r'z [px]')

#%% Compute translation displacements and angles
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

print('Length [um] = %.2f with std = %.2f'
      %(np.mean(flagella_len)*0.115,np.std(flagella_len)*0.115))
print(nt)

plt.plot(dist_projected_along_n1,dist_projected_along_m2,'x')
plt.plot(yy,fit_along_N3,'o')

#%% Print to PNG: snapshot of threshold from 6 angles
for frame in range(nt):
# for frame in range(0,1):
        
    xb0 = xb[frame] - cm[frame]
    xp0 = xp[frame]

    fig = plt.figure(dpi=150, figsize = (10, 6))
    # fig.suptitle('data: %s\n' %os.path.basename(thresholdFiles[whichFiles]) +
    #               'frame-num = ' + str(frame).zfill(3) + ', '
    #               'length = %.3f $\mu$m' %np.round(flagella_len[frame]*pxum,3) + ','
    #               'radius = %.3f $\mu$m\n' %np.round(radial_dist_pt[frame]*pxum,3) +
    #                '$\Delta_\parallel$ = %.3f $\mu$m, ' %np.round(disp[frame,0],3) +
    #                '$\Delta_{\perp 1}$ = %.3f $\mu$m, ' %np.round(disp[frame,1],3) +
    #                '$\Delta_{\perp 2}$ = %.3f $\mu$m\n' %np.round(disp[frame,2],3) +
    #               '$\Delta_\psi$ = %.3f rad, ' %np.round(disp_Ang[frame,1],3) +
    #               '$\Delta_\gamma$ = %.3f rad, ' %np.round(disp_Ang[frame,0],3) +
    #               '$\Delta_\phi$ = %.3f rad\n' %np.round(disp_Ang[frame,2],3)
    #               )
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
    ax0.quiver(X,Y,Z,Un1,Vn1,Wn1,color='C0')
    ax0.quiver(X,Y,Z,Un2,Vn2,Wn2,color='C1')
    ax0.quiver(X,Y,Z,Un3,Vn3,Wn3,color='C2')

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
    ax2.quiver(X,Y,Z,Un1,Vn1,Wn1,color='C0')
    ax2.quiver(X,Y,Z,Un2,Vn2,Wn2,color='C1')
    ax2.quiver(X,Y,Z,Un3,Vn3,Wn3,color='C2')

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
    ax4.quiver(X,Y,Z,Un1,Vn1,Wn1,color='C0')
    ax4.quiver(X,Y,Z,Un2,Vn2,Wn2,color='C1')
    ax4.quiver(X,Y,Z,Un3,Vn3,Wn3,color='C2')
    
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
    # snapshotFolder = os.path.join(
    #                  os.path.dirname(thresholdFiles[whichFiles]),
    #                  os.path.basename(thresholdFiles[whichFiles])[:-4]
    #                  + '-snapshots')
    # if os.path.isdir(snapshotFolder) != True:
    #     os.mkdir(snapshotFolder) # create path if non-existent
    # ax6.figure.savefig(os.path.join(snapshotFolder, 
    #                                 str(frame).zfill(3) + '.png'))

#%% Plot length (and) radius (and) threshold pixel total vs time 
fig = plt.figure(dpi=150, figsize = (40, 15))
plt.rcParams.update({'font.size': 30})
fig.suptitle('data: %s (' %os.path.basename(thresholdFiles[whichFiles]) +
              'Nt = %d' %nt + ', '
              'L = %.3f $\pm$ %.3f $\mu$m'
              %(np.mean(flagella_len)*pxum,
                np.std(flagella_len)*pxum) + ', '
              'R = %.3f $\pm$ %.3f $\mu$m)'
              %(np.mean(radial_dist_pt)*pxum,
                np.std(radial_dist_pt)*pxum) )
ax0 = fig.add_subplot(231)
ax1 = fig.add_subplot(232)
ax2 = fig.add_subplot(233)
ax3 = fig.add_subplot(234)
ax4 = fig.add_subplot(235)
ax5 = fig.add_subplot(236)
pxum = 0.115

ax0.plot(np.arange(0,nt),flagella_len*pxum,'k')
ax0.set_xlabel(r'frame-num')
ax0.set_ylabel(r'length [$\mu m$]')

ax1.plot(np.arange(0,nt),radial_dist_pt*pxum,'k')
ax1.set_xlabel(r'frame-num')
ax1.set_ylabel(r'radius [$\mu m$]')

ax2.plot(np.arange(0,nt),blob_size,'k')
ax2.set_xlabel(r'frame-num')
ax2.set_ylabel(r'pixel total number')

ax3.plot(np.arange(1,nt),disp_n1*pxum,'C0')
ax3.plot(np.arange(1,nt),disp_n2*pxum,'C1')
ax3.plot(np.arange(1,nt),disp_n3*pxum,'C2')
ax3.set_xlabel(r'frame-num')
ax3.set_ylabel(r'displacement along local axes [$\mu m$]')
ax3.legend(['$n_1$','$n_2$','$n_3$'], loc="upper right", ncol = 3)

ax4.plot(np.arange(1,nt),disp_roll,'C0')
ax4.set_xlabel(r'frame-num')
ax4.set_ylabel(r'$\psi$ [rad]')
ax4.set_ylim(min(disp_roll)-0.1,max(disp_roll)+0.1)

ax5.plot(np.arange(1,nt),disp_pitch,'C1')
ax5.plot(np.arange(1,nt),disp_yaw,'C2')
ax5.set_xlabel(r'frame-num')
ax5.set_ylabel(r'$\beta$ or $\gamma$ [rad]')
ax5.set_ylim(min(disp_roll)-0.1,max(disp_roll)+0.1)
ax5.legend(['pitch','yaw'],loc="upper right", ncol = 2)

ax5.figure.savefig(os.path.join(snapshotFolder, 'summary.png'))

print('one dataset takes [min]:', (time.perf_counter()-tstart)/60)
