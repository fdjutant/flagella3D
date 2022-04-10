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
whichFiles = 91
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
fitImage = np.zeros(imgs.shape)
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
viewer.add_image(imgs[frame_start:frame_end], contrast_limits=[100,400],\
                 scale=[0.115,.115,.115], blending='additive',\
                 multiscale=False,colormap='gray',opacity=1)
viewer.add_image(blobBin[frame_start:frame_end], contrast_limits=[0,1],\
                 scale=[0.115,.115,.115], blending='additive',\
                 multiscale=False,colormap='green',opacity=0.2)
viewer.add_image(fitImage[frame_start:frame_end], contrast_limits=[0,1],\
                 scale=[0.115,.115,.115], blending='additive',\
                 multiscale=False,colormap='red',opacity=0.2)
viewer.scale_bar.visible=True
viewer.scale_bar.unit='um'
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
napari.run()

#%% Compute translation displacements and angles
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

fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
plt.plot(disp_n1,'purple',alpha=0.75)
plt.plot(disp_n2,'C1',alpha=0.75)
plt.plot(disp_n3,'C2',alpha=0.75)
ax0.set_xlabel(r'Time [sec]')
ax0.set_ylabel(r'Displacement [$\mu m^2$]')
# ax0.set_ylim([0, 0.5])
# ax0.set_xlim([0, 3.2])
ax0.legend(['$n_1$', '$n_2$', '$n_3$'], ncol=3)
ax0.set_title(thresholdFiles[whichFiles].name)

print('Filename: %s' %thresholdFiles[whichFiles].name)
print('Number of frames = %d, length (std) [um] = %.2f (%.2f)'
      %(nt, np.mean(flagella_len)*0.115,np.std(flagella_len)*0.115))

# Compute MSD and curve fit
# initialize msd
msd_n1 = []; msd_n2 = []; msd_n3 = []; co_msd = []
msad_roll = []; msad_pitch = []; msad_yaw = []; msd_CM = []
nInterval = 50

# center-of-mass tracking
nt = len(cm)
dstCM = np.zeros(nt)
for i in range(len(cm)): dstCM[i] = np.linalg.norm(cm[i])

# MSD: mean square displacement
MSD_n1, MSD_n2, MSD_n3, CO_MSD = msd.trans_MSD_Namba(nt,
                                          cm, EuAng[:,1],
                                          n1s, n2s, n3s,
                                          exp3D_sec, nInterval)
MSAD_P = msd.regMSD_Namba(nt, EuAng[:,0], exp3D_sec, nInterval)
MSAD_R = msd.regMSD_Namba(nt, EuAng[:,1], exp3D_sec, nInterval)
MSAD_Y = msd.regMSD_Namba(nt, EuAng[:,2], exp3D_sec, nInterval)
MSD_CM = msd.regMSD_Namba(nt, dstCM, exp3D_sec, nInterval)

# Fit MSD with y = Const + B*x for N, S, NR, PY, R
Nfit = 10
xtime = np.linspace(1,Nfit,Nfit)
def MSDfit(x, a, b): return b + a * x  

# fit MSD and MSAD
fit_n1, fit_n1_const  = optimize.curve_fit(MSDfit, xtime, MSD_n1[0:Nfit])[0]
fit_n2n3, fit_n2n3_const  = optimize.curve_fit(MSDfit, xtime,
                        np.mean([MSD_n2[0:Nfit],MSD_n3[0:Nfit]],axis=0))[0]
fit_CO, fit_CO_const = optimize.curve_fit(MSDfit, xtime, CO_MSD[0:Nfit])[0]
fit_PY, fit_PY_const = optimize.curve_fit(MSDfit, xtime,
                          np.mean([MSAD_P[0:Nfit],MSAD_Y[0:Nfit]],axis=0))[0]
fit_R,fit_R_const   = optimize.curve_fit(MSDfit, xtime, MSAD_R[0:Nfit])[0]
fit_CM,fit_CM_const = optimize.curve_fit(MSDfit, xtime, MSD_CM[0:Nfit])[0]

# Additional fit
fit_n2,fit_n2_const  = optimize.curve_fit(MSDfit, xtime, MSD_n2[0:Nfit])[0]
fit_n3,fit_n3_const  = optimize.curve_fit(MSDfit, xtime, MSD_n3[0:Nfit])[0]
fit_P, fit_P_const  = optimize.curve_fit(MSDfit, xtime, MSAD_P[0:Nfit])[0]
fit_Y, fit_Y_const  = optimize.curve_fit(MSDfit, xtime, MSAD_Y[0:Nfit])[0]

# Compute diffusion coefficients
D_trans = np.stack([fit_n1, fit_n2, fit_n3]) / (2*exp3D_sec)
D_rot = np.stack([fit_R, fit_P, fit_Y]) / (2*exp3D_sec)
D_CO = fit_CO / (2*exp3D_sec)

print('translational diffusion coefficients [um2/sec]:\n', D_trans)
print('rotational diffusion coefficients [rad2/sec]:\n', D_rot)
print('co-diffusion coefficients [um x rad]:\n', D_CO)

# plot
xaxis = np.arange(1,nInterval+1)
plt.rcParams.update({'font.size': 18})
fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
ax0.plot(xaxis*exp3D_sec, MSD_n1,
         c='purple',marker="s",mfc='none',
         ms=5,ls='None',alpha=1)   
ax0.plot(xaxis*exp3D_sec, MSD_n2,
         c='C1',marker="s",mfc='none',
         ms=5,ls='None',alpha=1)
ax0.plot(xaxis*exp3D_sec, MSD_n3,
         c='C2',marker="s",mfc='none',
         ms=5,ls='None',alpha=1)
ax0.plot(xaxis*exp3D_sec, fit_n1_const + fit_n1*xaxis,
         c='purple',alpha=1,label='_nolegend_')
ax0.plot(xaxis*exp3D_sec,fit_n2_const + fit_n2*xaxis,
         c='C1',alpha=1,label='_nolegend_')
ax0.plot(xaxis*exp3D_sec,fit_n3_const + fit_n3*xaxis,
         c='C2',alpha=1,label='_nolegend_')
ax0.set_xlabel(r'Lag time [sec]')
ax0.set_ylabel(r'MSD [$\mu m^2$]')
# ax0.set_ylim([0, 0.5])
# ax0.set_xlim([0, 3.2])
ax0.legend(['$n_1$', '$n_2$', '$n_3$'], ncol=3)
ax0.set_title(thresholdFiles[whichFiles].name)

# plot CO-MSD
xaxis = np.arange(1,nInterval+1)
plt.rcParams.update({'font.size': 18})
fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
ax0.plot(xaxis*exp3D_sec, CO_MSD,
         c='k',marker="s",mfc='none',
         ms=5,ls='None',alpha=1)   
ax0.plot(xaxis*exp3D_sec, fit_CO_const + fit_CO*xaxis,
         c='k',alpha=1,label='_nolegend_')
ax0.set_xlabel(r'Lag time [sec]')
ax0.set_ylabel(r'CO-MSD [$\mu m\times rad$]')
ax0.set_title(thresholdFiles[whichFiles].name)

# Compute A, B, D
kB = 1.380649e-23  # J / K
T = 273 + 25       # K

# from bead measurement
vis70 = 2.84e-3
vis50 = 1.99e-3
vis40 = 1.77e-3

D_n1 = D_trans[0] * 1e-12
D_n1_psi = D_CO * 1e-6
D_psi = D_rot[0]

A_per_vis = D_psi * kB * T / (D_n1 * D_psi - D_n1_psi**2) / vis50
B_per_vis = -D_n1_psi * kB * T / (D_n1 * D_psi - D_n1_psi**2) / vis50
D_per_vis = D_n1 * kB * T / (D_n1 * D_psi - D_n1_psi**2) / vis50

print('propulsion matrix\n A/vis, B/vis, D/vis = %.2E %.2E %.2E'
      %(A_per_vis, B_per_vis, D_per_vis))

#%% Store tracking information to PKL
savingPKL = os.path.join(thresholdFiles[whichFiles].parent,
                         thresholdFiles[whichFiles].with_suffix('.pkl'))
data = {
        "data_name": intensityFiles[whichFiles].name[:-4],
        "flagella_length": flagella_len,
        "exp3D_sec": exp3D_sec,
        "pxum": pxum,
        "cm": cm,
        "disp": disp,
        "disp_Ang": disp_Ang,
        "MSD": np.stack([MSD_n1, MSD_n2, MSD_n3],axis=1),
        "MSAD": np.stack([MSAD_R, MSAD_P, MSAD_Y],axis=1),
        "CO_MSD": CO_MSD,
        "D_trans": D_trans,
        "D_rot": D_rot,
        "D_co": D_CO,
        "A_per_vis": A_per_vis,
        "B_per_vis": B_per_vis,
        "D_per_vis": D_per_vis
        }
with open(savingPKL, "wb") as f:
      pickle.dump(data, f)
print('%s is saved' %(os.path.basename(savingPKL)))

#%% MSAD: roll
plt.rcParams.update({'font.size': 18})
fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
ax0.plot(xaxis*exp3D_sec, MSAD_R,
         c='purple',marker="o",mfc='none',
         ms=5,ls='None',alpha=1)   
ax0.plot(xaxis*exp3D_sec, fit_R_const + fit_R*xaxis,
         c='purple',alpha=1,label='_nolegend_')
ax0.set_xlabel(r'Lag time [sec]')
ax0.set_ylabel(r'MSAD [rad$^2$]')
ax0.set_ylim([0, 3])
ax0.set_xlim([0, 3.2])
ax0.set_yticks([0,1,2,3])
ax0.legend(['$R$'])
# ax0.figure.savefig(pdfFolder + '/fig3-MSAD-R.pdf')

# MSAD: pitch/yaw
fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
ax0.plot(xaxis*exp3D_sec, MSAD_P,
         c='C1',marker="o",mfc='none',
         ms=5,ls='None',alpha=1)
ax0.plot(xaxis*exp3D_sec, MSAD_Y,
         c='C2',marker="o",mfc='none',
         ms=5,ls='None',alpha=1)
ax0.plot(xaxis*exp3D_sec,fit_P_const + fit_P*xaxis,
         c='C1',alpha=1,label='_nolegend_')
ax0.plot(xaxis*exp3D_sec,fit_Y_const + fit_Y*xaxis,
         c='C2',alpha=1,label='_nolegend_')
ax0.set_xlabel(r'Lag time [sec]')
ax0.set_ylabel(r'MSAD [rad$^2$]')
ax0.set_ylim([0, 0.08])
ax0.set_yticks([0,0.02,0.04,0.06,0.08])
ax0.set_xlim([0, 3.2])
ax0.legend(['$P$', '$Y$'], ncol=2)