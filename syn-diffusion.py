# Import all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})
from sklearn.decomposition import PCA
from scipy import stats, optimize
from scipy.ndimage import gaussian_filter
import pandas as pd
import time
import napari
from matmatrix import *
import helixFun
import imProcess
import msd
from msd import regMSD, trans_stepSize,\
                rot_stepSize, trans_stepSize_all

import movingHx

# time settings in the light sheet
pxum = 0.115; 
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
vol_exp = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) / 10

# input helix geometry
length = 8; radius = 0.3; pitchHx = 2.5 # all three in um
chirality = 1;  # left-handed: 1, right-handed: -1
resol = 100     # number of points/resolutions

# input helix motion
Nframes = 300; spin = 0; drift = 0;
Dperp = 0.3; Dpar = 2*Dperp;            # in um^2/sec
Dpitch = 0.1; Droll = 5; Dyaw = 0.1;    # in rad^2/sec

# input helix-and-noise level
hxInt = 200; hxVar = 0;
noiseInt = 0; noiseVar = 0;

# Image analysis and curve fitting
xb = []; xb0 = []; xp = []; xp0 = []
blobSkel = []; blobBin =[]; blobSize = []
eigenvec = []; blobRaw = []; xb0 = []
    
lenfla = np.zeros(Nframes)
cmOutput = np.zeros([Nframes,3]); 
localAxesOutput = np.zeros([Nframes,3,3]);
eigenvec = np.zeros([Nframes,3,3]);
coord = [];
endpt = np.zeros([Nframes]).astype('int')
ep_ref = np.zeros([Nframes,3])

start = time.perf_counter()

nTMax = 10; nSepTotal = 3;
nPoints, nDivider, errorTh = (np.zeros([nSepTotal]) for _ in range(3));
sigma2_par, sigma2_perp1, sigma2_perp2 = (np.zeros([nTMax,nSepTotal]) for _ in range(3));
sigma2_P, sigma2_R, sigma2_Y = (np.zeros([nTMax,nSepTotal]) for _ in range(3));
sigma2_NR = np.zeros([nTMax,nSepTotal]);
fitN_1per, fitS_1per, fitS2_1per, fitNR_1per = (np.zeros([nTMax,nSepTotal]) for _ in range(4));
fitP_1per, fitR_1per, fitY_1per = (np.zeros([nTMax,nSepTotal]) for _ in range(3));
fitN_5per, fitS_5per, fitS2_5per, fitNR_5per = (np.zeros([nTMax,nSepTotal]) for _ in range(4));
fitP_5per, fitR_5per, fitY_5per = (np.zeros([nTMax,nSepTotal]) for _ in range(3));
fitN_10per, fitS_10per, fitS2_10per, fitNR_10per = (np.zeros([nTMax,nSepTotal]) for _ in range(4));
fitP_10per, fitR_10per, fitY_10per = (np.zeros([nTMax,nSepTotal]) for _ in range(3));

cm_traj = []; EuAng_traj = []; localAxes_traj = [];
transSS_traj = []; rotSS_traj = []; NRSS_traj = [];
time_x_traj = []; MSD_N_traj = [];
MSD_S_traj = []; MSD_S2_traj = [];
MSD_NR_traj = [];
MSD_P_traj = []; MSD_R_traj = [];
MSD_Y_traj = [];

print('Initialization time (sec)',\
      np.round(time.perf_counter()-start,1))

# COMPUTE
nT =  1;
while nT <= nTMax:
    # Generate diffusion
    fromHx = movingHx.createMovHx(length, radius, pitchHx,\
                                  chirality, resol, Nframes,\
                                  Dpar, Dperp, Dpitch, Droll, Dyaw,\
                                  spin, drift, hxInt, hxVar, noiseInt,\
                                  noiseVar, vol_exp)
    cm, EuAng, localAxes = fromHx.movHx()
    print('------------------------------------------')
    print('Generate trajectory #:', nT, 'time (sec):',\
          np.round(time.perf_counter()-start,1))
    
    # generate the step size
    cm_sep = []; EuAng_sep = []; localAxes_sep = [];
    transSS_sep = []; rotSS_sep = []; NRSS_sep = [];
    time_x_sep = [];
    MSD_N_sep = []; MSD_S_sep = []; MSD_S2_sep = []; MSD_NR_sep = [];
    MSD_P_sep = []; MSD_R_sep = []; MSD_Y_sep = []; 
    for k in np.linspace(1,nSepTotal,nSepTotal)[::1].astype('int'):
    
        # Number divider & points
        nPoints[k-1] = len(cm[::k])
        nDivider[k-1] = k
        errorTh[k-1] = np.sqrt(2/(len(cm[::k])-1))
    
        # Save cm, EuAng, localAxes
        cm_sep.append(cm[::k]);
        EuAng_sep.append(EuAng[::k]);
        localAxes_sep.append(localAxes[::k]);
    
        # Step size technique
        transSS = trans_stepSize(cm[::k], localAxes[::k])
        transSS_sep.append(transSS)
        amp_par, mean_par, sigmaPar = fitCDF(transSS[:,0])
        amp_perp1, mean_perp1, sigmaPerp1 = fitCDF(transSS[:,1])
        amp_perp2, mean_perp2, sigmaPerp2 = fitCDF(transSS[:,2])
        sigma2_par[nT-1,k-1] = sigmaPar**2
        sigma2_perp1[nT-1,k-1] = sigmaPerp1**2
        sigma2_perp2[nT-1,k-1] = sigmaPerp2**2

        rotSS = rot_stepSize(EuAng[::k])
        rotSS_sep.append(rotSS)
        amp_P, mean_P, sigmaP = fitCDF(rotSS[:,0])
        amp_R, mean_R, sigmaR = fitCDF(rotSS[:,1])
        amp_Y, mean_Y, sigmaY = fitCDF(rotSS[:,2])
        sigma2_P[nT-1,k-1] = sigmaP**2
        sigma2_R[nT-1,k-1] = sigmaR**2
        sigma2_Y[nT-1,k-1] = sigmaY**2

        NRSS = transSS[:,0] * rotSS[:,1]
        NRSS_sep.append(NRSS)
        amp_NR, mean_NR, sigmaNR = fitCDF(NRSS)
        sigma2_NR[nT-1,k-1] = sigmaNR**2
        print('traj-#:', nT, 'SS-iteration-#:', k, 'time (sec):',\
              np.round(time.perf_counter()-start,1))
    
        # MSD technique
        fromMSD = msd.theMSD(len(cm[::k]), cm[::k], EuAng[:,1],
                              localAxes[::k], vol_exp*k)
        time_x, MSD_N, MSD_S, MSD_S2, MSD_NR = fromMSD.trans_combo_MSD()
        time_xP, MSD_P = regMSD(len(cm[::k]), EuAng[:,0], vol_exp)
        time_xR, MSD_R = regMSD(len(cm[::k]), EuAng[:,1], vol_exp)
        time_xY, MSD_Y = regMSD(len(cm[::k]), EuAng[:,2], vol_exp)
        print('traj-#:', nT, 'MSD-iteration-#:', k, 'time (sec):',\
              np.round(time.perf_counter()-start,1))
        time_x_sep.append(time_x); MSD_N_sep.append(MSD_N);
        MSD_S_sep.append(MSD_S); MSD_S2_sep.append(MSD_S2);
        MSD_NR_sep.append(MSD_NR);
        MSD_P_sep.append(MSD_P); MSD_R_sep.append(MSD_R);
        MSD_Y_sep.append(MSD_Y);
        
        # Fit 1% of the data
        rFit = 0.01;
        Ndata = np.round(rFit*len(cm[::k])).astype('int');
        fitN_1per[nT-1,k-1] = optimize.curve_fit(MSDfit, time_x[0:Ndata], MSD_N[0:Ndata],p0=0.1)[0]
        fitS_1per[nT-1,k-1] = optimize.curve_fit(MSDfit, time_x[0:Ndata], MSD_S[0:Ndata],p0=0.1)[0]
        fitS2_1per[nT-1,k-1] = optimize.curve_fit(MSDfit, time_x[0:Ndata], MSD_S2[0:Ndata],p0=0.1)[0]
        fitNR_1per[nT-1,k-1] = optimize.curve_fit(MSDfit, time_x[0:Ndata], MSD_NR[0:Ndata],p0=0.1)[0]
        fitP_1per[nT-1,k-1] = optimize.curve_fit(MSDfit, time_x[0:Ndata], MSD_P[0:Ndata],p0=0.1)[0]
        fitR_1per[nT-1,k-1] = optimize.curve_fit(MSDfit, time_x[0:Ndata], MSD_R[0:Ndata],p0=0.1)[0]
        fitY_1per[nT-1,k-1] = optimize.curve_fit(MSDfit, time_x[0:Ndata], MSD_Y[0:Ndata],p0=0.1)[0]
        
        # Fit 5% of the data
        rFit = 0.05;
        Ndata = np.round(rFit*len(cm[::k])).astype('int');
        fitN_5per[nT-1,k-1] = optimize.curve_fit(MSDfit, time_x[0:Ndata], MSD_N[0:Ndata],p0=0.1)[0]
        fitS_5per[nT-1,k-1] = optimize.curve_fit(MSDfit, time_x[0:Ndata], MSD_S[0:Ndata],p0=0.1)[0]
        fitS2_5per[nT-1,k-1] = optimize.curve_fit(MSDfit, time_x[0:Ndata], MSD_S2[0:Ndata],p0=0.1)[0]    
        fitNR_5per[nT-1,k-1] = optimize.curve_fit(MSDfit, time_x[0:Ndata], MSD_NR[0:Ndata],p0=0.1)[0]
        fitP_5per[nT-1,k-1] = optimize.curve_fit(MSDfit, time_x[0:Ndata], MSD_P[0:Ndata],p0=0.1)[0]
        fitR_5per[nT-1,k-1] = optimize.curve_fit(MSDfit, time_x[0:Ndata], MSD_R[0:Ndata],p0=0.1)[0]
        fitY_5per[nT-1,k-1] = optimize.curve_fit(MSDfit, time_x[0:Ndata], MSD_Y[0:Ndata],p0=0.1)[0]

        print('traj-#:', nT, 'FitMSD iteration-#:', k, 'time (sec):',\
              np.round(time.perf_counter()-start,0))
    
    cm_traj.append(cm_sep)
    EuAng_traj.append(EuAng_sep)
    localAxes_traj.append(localAxes_sep)
    transSS_traj.append(transSS_sep)
    rotSS_traj.append(rotSS_sep)
    NRSS_traj.append(NRSS_sep)
    time_x_traj.append(time_x_sep)
    MSD_N_traj.append(MSD_N_sep)
    MSD_S_traj.append(MSD_S_sep)
    MSD_S2_traj.append(MSD_S2_sep)
    MSD_NR_traj.append(MSD_NR_sep)
    MSD_P_traj.append(MSD_P_sep)
    MSD_R_traj.append(MSD_R_sep)
    MSD_Y_traj.append(MSD_Y_sep)
    
    del cm, EuAng, localAxes 
    nT += 1        

cm_traj = np.array(cm_traj,dtype=object)
EuAng_traj = np.array(EuAng_traj,dtype=object)
localAxes_traj = np.array(localAxes_traj,dtype=object)
transSS_traj = np.array(transSS_traj,dtype=object)
rotSS_traj = np.array(rotSS_traj,dtype=object)
NRSS_traj = np.array(NRSS_traj,dtype=object)
time_x_traj = np.array(time_x_traj,dtype=object)
MSD_N_traj = np.array(MSD_N_traj,dtype=object)
MSD_S_traj = np.array(MSD_S_traj,dtype=object)
MSD_S2_traj = np.array(MSD_S2_traj,dtype=object)
MSD_NR_traj = np.array(MSD_NR_traj,dtype=object)
MSD_P_traj = np.array(MSD_P_traj,dtype=object)
MSD_R_traj = np.array(MSD_R_traj,dtype=object)
MSD_Y_traj = np.array(MSD_Y_traj,dtype=object)

# save to numpy
folderName = 'syn-result/nT50-nF3000-nSep200/';
np.save(folderName+'Dperp.npy', Dperp)
np.save(folderName+'vol_exp.npy', vol_exp)
np.save(folderName+'nPoints.npy', nPoints)
np.save(folderName+'nDivider.npy', nDivider)
np.save(folderName+'errorTh.npy', errorTh)
np.save(folderName+'cm_traj.npy', cm_traj)
np.save(folderName+'EuAng_traj.npy', EuAng_traj)
np.save(folderName+'localAxes_traj.npy', localAxes_traj)
np.save(folderName+'transSS_traj.npy', transSS_traj)
np.save(folderName+'rotSS_traj.npy', rotSS_traj)
np.save(folderName+'NRSS_traj.npy', NRSS_traj)
np.save(folderName+'sigma2_par.npy', sigma2_par)
np.save(folderName+'sigma2_perp1.npy', sigma2_perp1)
np.save(folderName+'sigma2_perp2.npy', sigma2_perp2)
np.save(folderName+'sigma2_NR.npy', sigma2_NR)
np.save(folderName+'sigma2_P.npy', sigma2_P)
np.save(folderName+'sigma2_R.npy', sigma2_R)
np.save(folderName+'sigma2_Y.npy', sigma2_Y)
np.save(folderName+'time_x_traj.npy', time_x_traj)
np.save(folderName+'MSD_N_traj.npy', MSD_N_traj)
np.save(folderName+'MSD_S_traj.npy', MSD_S_traj)
np.save(folderName+'MSD_S2_traj.npy', MSD_S2_traj)
np.save(folderName+'MSD_NR_traj.npy', MSD_NR_traj)
np.save(folderName+'MSD_P_traj.npy', MSD_P_traj)
np.save(folderName+'MSD_R_traj.npy', MSD_R_traj)
np.save(folderName+'MSD_Y_traj.npy', MSD_Y_traj)
np.save(folderName+'fitN_1per.npy', fitN_1per)
np.save(folderName+'fitS_1per.npy', fitS_1per)
np.save(folderName+'fitS2_1per.npy', fitS2_1per)
np.save(folderName+'fitNR_1per.npy', fitNR_1per)
np.save(folderName+'fitN_5per.npy', fitN_5per)
np.save(folderName+'fitS_5per.npy', fitS_5per)
np.save(folderName+'fitS2_5per.npy', fitS2_5per)
np.save(folderName+'fitNR_5per.npy', fitNR_5per)
np.save(folderName+'fitP_1per.npy', fitP_1per)
np.save(folderName+'fitR_1per.npy', fitR_1per)
np.save(folderName+'fitY_1per.npy', fitY_1per)
np.save(folderName+'fitP_5per.npy', fitP_5per)
np.save(folderName+'fitR_5per.npy', fitR_5per)
np.save(folderName+'fitY_5per.npy', fitY_5per)