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

setName = 'suc-40'

this_file_dir = os.path.join(os.path.dirname(os.path.abspath("./")),
                            'Dropbox (ASU)','Research')
loadFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Flagella-data', setName)

intensityFiles = list(Path(loadFolder).glob("*.npy"))
# thresholdFiles = list(Path(savingFolder).glob("*threshold.npy"))
whichFiles = intensityFiles[0]

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
thresh_size = np.zeros(nt)
threshold = np.zeros(nt)
num_iterations = np.zeros(nt)
# starting threshold guess as fraction of maximum pixel value
max_thresh_iterations = 10
min_thresh_size_diff = 100
thresh_value = 0.8
thresh_max_start = 0.9
thresh_min_start = 0.7

tstart = time.perf_counter()
for frame in range(nt):
    
    #print('frame:', frame)
    
    # grab current image
    img_now = np.array(imgs[frame])

    # median filter image
    img_now_med = scipy.signal.medfilt(img_now, kernel_size=(3,3,3))
        
    # iteratively threshold using binary search
    thresh_max = thresh_max_start
    thresh_min = thresh_min_start
    thresh_value_temp = thresh_value
    for ii in range(max_thresh_iterations):
        if ii > 0:
            thresh_constraint = thresh_size[0] - thresh_size_temp
            
            # todo: investigate case where bounces back and forth between
            # two values the whole time... probably related to thresholding
            # integer data
            
            # if constrain was met, break out of loop
            if np.abs(thresh_constraint) < min_thresh_size_diff:
                break
            
            if thresh_constraint > 0:
                
                # if not enough thresholded points, decrease threshold
                # since thres_value_temp is too high a threshold, update thresh_max
                thresh_max = thresh_value_temp
                
                # new threshold value guess
                thresh_value_temp = 0.5 * (thresh_value_temp + thresh_min)
                
            else:
                
                # if too many thresholded points, increase threshold
                # since thresh_value_temp is too low, update thresh_min
                thresh_min = thresh_value_temp
                thresh_value_temp = 0.5 * (thresh_value_temp + thresh_max)
        
        # threshold intensity image with the default value
        img_thresh_med = img_now_med > thresh_value_temp * np.max(img_now_med)
        
        # label and measure every clusters
        blobs = measure.label(img_thresh_med, background=0)
        labels = np.arange(1, blobs.max() + 1, dtype=int)
        sizes = np.array([np.sum(blobs == l) for l in labels])
        
        # keep only the largest cluster  
        max_ind = np.argmax(sizes)
        thresh_size_temp = sizes[max_ind]
        
        # mask showing which pixels ae in largest cluster
        blob = blobs == labels[max_ind]
    
        # print iteration number 
        print('frame = %d, iteration = %d, # thresholded pixels = %d, threshold = %.6f, elapsed time = %0.2fs'
              % (frame, ii, thresh_size_temp, thresh_value_temp, time.perf_counter() - tstart) )
        
        # if first frame, only do one iteration
        if frame == 0:
            break

    # track number of iterations run
    num_iterations[frame] = ii
    # size of thresholded region
    thresh_size[frame] = sizes[max_ind]
    # threshold value
    threshold[frame] = thresh_value_temp
    # store threshold/binarized image
    blobBin.append(blob)

blobBin = np.array(blobBin)
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
viewer.scale_bar.position='top_right'
viewer.axes.visible = True
napari.run()

#%% plot threshold size
plt.figure(dpi=600, figsize=(10,7))
plt.plot(thresh_size)
plt.xlabel('frame-num')
plt.ylabel('threshold array size')
plt.ylim([0,4100])

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
        "Drot": fit_PY/(2*exp3D_ms),
        "MSD_par": MSD_N,
        "MSD_perp": MSD_S2,
        # "MSD_perp": (MSD_S1 + MSD_S2)*0.5,
        "MSD_rot": 0.5*(MSD_P + MSD_Y),
        "exp3D_ms": exp3D_ms}
# with open(savingPKL + '-result.pkl', "wb") as f:
#       pickle.dump(data, f)

#%% save movie  
savingMovie_Time = Path(str(whichFiles)[:-6] + '-Time-Threshold-4.mov')
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

savingMovie_Rotate = Path(str(whichFiles)[:-6] + '-Rotate-Threshold.mov')
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
