import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})
from sklearn.decomposition import PCA
from scipy import stats, optimize
from scipy.ndimage import gaussian_filter
import pandas as pd
import time
# import napari
from matmatrix import *
import helixFun
import imProcess
import msd
from msd import regMSD
import movingHx  
import glob
from natsort import natsorted, ns
from pathlib import Path

path = r"C:\Users\labuser\Dropbox (ASU)\Research\DNA-Rotary-Motor\Helical-nanotubes\Light-sheet-OPM\Result-data"
# path = r"D:\Dropbox (ASU)\Research\DNA-Rotary-Motor\Helical-nanotubes\Light-sheet-OPM\Result-data"
# path = r"/mnt/opm2/20211022_franky/"

fName = path + "/20211022a_suc40_h15um"
# fName = path + "/20211022b_suc40_h30um"
# fName = path + "/20211018a_suc50_h15um"
# fName = path + "/20211018b_suc50_h30um"
# fName = path + "/20211022c_suc70_h15um"
# fName = path + "/20211022d_suc70_h30um" 

images = glob.glob(fName + '/*.npy')

vis70 = 673 # 70% sucrose, unit: mPa.s (Quintas et al. 2005)
vis50 = 15.04 # 50% sucrose, unit: mPa.s (Telis et al. 2005)
vis40 = 6.20 # 40% sucrose, unit: mPa.s (Telis et al. 2005)
suc_per = str(40)
vis = vis40

# Parameters
pxum = 0.115; 
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
# vol_exp = 1/ (sweep_um/stepsize_nm * camExposure_ms)
vol_exp = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm)  # in sec
thresvalue_in = 0.8

start = time.perf_counter()
#%% Go through every folder
for j in range(len(images)):

    fileIm = natsorted(glob.glob(images[j] + '/*.npy'))      
    intensity0 = np.concatenate([np.load(f) for f in fileIm])
    # intensity0 = da.from_npy_stack(images[j]) # with dask-array
    intensity = intensity0[:,:,:,:] 
    Nframes = intensity.shape[0]
    
    # Image analysis and curve fitting
    xb = []; xp = []; xp0 = []
    blobSkel = []; blobBin =[]; blobSize = []
    eigenvec = []; blobRaw = []; xb0 = []
    
    cm = np.zeros([Nframes,3]); lenfla = np.zeros(Nframes);
    eigenvec = np.zeros([Nframes,3,3]);
    coord = []; sizeAll = np.zeros([Nframes])
    localAxes = np.zeros([Nframes,3,3]);
    endpt = np.zeros([Nframes]).astype('int')
        
    for frame in range(Nframes):
    
        tstart_thresh = time.perf_counter()
        # Image processing
        thresvalue = thresvalue_in; sizes = 0;   
        fromImgPro = imProcess.ImPro(intensity[frame],\
                                     thresvalue)
        img = fromImgPro.thresVol()                   # binary image
        sizes = max(fromImgPro.selectLargest()[0])    # largest body only
        
        att = 0;
        if sizes < 900:
            att = 1
            while sizes < 900 and att < 20: 
                thresvalue = thresvalue - 0.025
                fromImgPro = imProcess.ImPro(intensity[frame],thresvalue)
                img = fromImgPro.thresVol()                 
                sizes = max(fromImgPro.selectLargest()[0])
                att = att + 1
        print("thresholding took %0.2fs with %d attemps"\
              % (time.perf_counter() - tstart_thresh, att))

        blob = fromImgPro.BlobAndSkel()
                   
        X0 = fromImgPro.extCoord()          # extract coordinates
        CM1 = fromImgPro.computeCM()        # compute center of mass

        # Store binary image, coordinates and center of mass
        blobBin.append(blob); blobSize.append(sizes);
        blobRaw.append(img);
        xb.append(X0); cm[frame,:] = CM1
           
        # Compute the flagella length    
        lenfla[frame] = flaLength(X0)*pxum
        
        # Use PCA to find the rotation matrix
        X = X0 - CM1 # shift all the coordinates into origin
        pca = PCA(n_components=3)
        pca.fit(X)
        axes = pca.components_
        
        # Make the PCA consistent
        if frame == 0:
            axes_ref = axes
        else:
            axes, axes_ref = consistentPCA(axes, axes_ref)
        
        # Find the second vector orthogonal to the major axis using skeletonize
        if frame == 0:
            ep_ref = 0
            ep, Coord = endPoints(X0, CM1, axes)
        else:
            ep, Coord = endPoints(X0, CM1, axes)
            ep_ref = ep
        endpt[frame] = ep.astype('int')
        xb0.append(Coord)
        
        # Use Gram-Schmidt to find n2, then find n3 with the cross
        n1 = axes[0] / np.linalg.norm(axes[0])
        n2 = Coord[ep] - np.array([0,0,0])
        n2 -= n2.dot(n1) * n1 / np.linalg.norm(n1)**2
        n2 /= np.linalg.norm(n2)
        n3 = np.cross(n1,n2)
        n3 /= np.linalg.norm(n3)
        localAxes[frame,0] = n1; localAxes[frame,1] = n2; localAxes[frame,2] = n3;
        
        # Rotate to the principal axes
        P0 = np.matmul(axes,X.T).T
        xp0.append(P0)  
        eigenvec[frame] = axes
    
        # Print each volume frame has finished
        end = time.perf_counter()
        print('Folder:',images[j][-40:],"(",j+1,"out of",len(images),")",
              '\n frame#:',frame,'out of',len(intensity),
              '\n length (um):', np.round(lenfla[frame],2),
              '\n elapsed (min):',np.round((end-start)/60,2))
    
    # Save threshold coordinates to external file for further review 
    fileName = images[j]
    blobBin = da.from_array(blobBin)
    da.to_npy_stack(fileName[:len(fileName)-4] + '-threshold.npy',blobBin)      

    # Compute pitch, roll, and yaw 
    n1 = localAxes[:,0]; n2 = localAxes[:,1]; n3 = localAxes[:,2]
    dpitch = np.zeros(Nframes); droll = np.zeros(Nframes); dyaw = np.zeros(Nframes)
    for frame in range(Nframes-1):
        dpitch[frame] = np.dot(n2[frame], n1[frame+1] - n1[frame])
        droll[frame] = np.dot(n3[frame], n2[frame+1] - n2[frame])
        dyaw[frame] = np.dot(n1[frame], n3[frame+1] - n3[frame])
        
    # can you get EuAng diretly from n1, n2, n3 without integral?
    EuAng = np.zeros([Nframes,3]);
    for frame in range(Nframes):
        EuAng[frame,0] = np.sum(dpitch[0:frame+1])
        # is this faster?
        #EuAng[frame, 0] = EuAng[frame - 1, 0] + dpitch[0, frame + 1]
        EuAng[frame,1] = np.sum(droll[0:frame+1])
        EuAng[frame,2] = np.sum(dyaw[0:frame+1])
    
    # Compute absolute angle relative to x, y,and z axis
    n1 = localAxes[:,0]
    dirAng = np.zeros([Nframes,3]); 
    # dirAng[:, 0] = np.arccos(n1[:, 0]) #this will be faster and etc for 1,2
    for frame in range(Nframes):
        dirAng[frame,0] = np.arccos(n1[frame,0])
        dirAng[frame,1] = np.arccos(n1[frame,1])
        dirAng[frame,2] = np.arccos(n1[frame,2])
    
    # Store all the tracking information
    ate = np.asarray([EuAng, dirAng, cm, localAxes])
    np.save(fileName[:len(fileName)-4] + "-results",ate)
    
    # All the MSD of interest
    fromMSD = msd.theMSD(0.8, Nframes, cm, dirAng,\
                         EuAng[:,1], localAxes, vol_exp)
    time_x, MSD_N, MSD_S, MSD_combo = fromMSD.trans_combo_MSD()
    # time_x, MSD_combo = fromMSD.combo_MSD()
    time_x, MSD_pitch = regMSD(0.8, Nframes, EuAng[:,0], vol_exp)
    time_x, MSD_roll = regMSD(0.8, Nframes, EuAng[:,1], vol_exp)
    time_x, MSD_yaw = regMSD(0.8, Nframes, EuAng[:,2], vol_exp)

    # Fit the MSDs curve
    nData = np.int32(0.1*Nframes) # number of data fitted
    def MSDfit(x, a):
        return a * x   
    fitN = optimize.curve_fit(MSDfit, time_x[0:nData], MSD_N[0:nData],p0=0.1)
    fitS = optimize.curve_fit(MSDfit, time_x[0:nData], MSD_S[0:nData],p0=0.1)
    fitPitch = optimize.curve_fit(MSDfit, time_x[0:nData], MSD_pitch[0:nData],p0=0.1)
    fitRoll = optimize.curve_fit(MSDfit, time_x[0:nData], MSD_roll[0:nData],p0=0.1)
    fitYaw = optimize.curve_fit(MSDfit, time_x[0:nData], MSD_yaw[0:nData],p0=0.1)
    fitCombo = optimize.curve_fit(MSDfit, time_x[0:nData], MSD_combo[0:nData],p0=0.1)

    # Compute all parameters
    radfla = np.zeros(Nframes)
    pitfla = np.zeros(Nframes)
    for i in range(Nframes):
        radfla[i] =  (max(xp0[i][:,2])-min(xp0[i][:,2]) ) *pxum
        pitfla[i] = lenfla[i]/2.5
    
    print('SUMMARY for ' + fName + ' with total',Nframes,'frames')
    print("flagella radius [um] = ", np.mean(radfla),\
          " with std = ", np.std(radfla))
    print("flagella length [um] = ", np.mean(lenfla),\
          " with std = ", np.std(lenfla))
    print("flagella pitch-length [um] = ", np.mean(pitfla),\
          " with std = ", np.std(pitfla))
    print("Fit for parallel, perpen-1, perpen-2:",fitN[0], fitS[0])
    print("Fit for pitch, roll, yaw:",fitPitch[0], fitRoll[0], fitYaw[0])
    print("Fit for combo:",fitCombo[0])
    print("Matrix A, B, D for " + fName)
    A, B, D = BernieMatrix(fitN[0]*1e-12,fitRoll[0],fitCombo[0]*1e-6)
    A2, B2, D2 = BernieMatrix(fitN[0]*1e-12*(vis),fitRoll[0]*(vis),\
                              fitCombo[0]*1e-6*(vis)) 
    print('A, B, D:', A, B, D)
    print("A, B, D (adjusted '+ suc_per + '\% sucrose):", A2, B2, D2)
    
    # print to excel
    data = [['number of frames', Nframes],\
            ['radius [um]', np.mean(radfla),np.std(radfla)],\
            ['length [um]', np.mean(lenfla),np.std(lenfla)],\
            ['pitch [um]', np.mean(pitfla),np.std(pitfla)],\
            ['trans-fit [um^2/sec^2]',fitN[0][0], fitS[0][0]],\
            ['rotation-fit [rad^2/sec^2]',fitPitch[0][0], fitRoll[0][0], fitYaw[0][0]],\
            ['combo-fit [um.rad/sec^2]',fitCombo[0][0]],\
            ['A, B, D', A[0], B[0], D[0]],\
            ['A, B, D (adjusted '+ suc_per + '\% sucrose)', A2[0], B2[0], D2[0]]\
                ]
    df = pd.DataFrame(data)
    df.to_excel(fileName[:len(fileName)-4] + '.xlsx', index = False, header = False)  
