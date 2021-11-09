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
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})

path = r"C:\Users\labuser\Dropbox (ASU)\Research\DNA-Rotary-Motor\Helical-nanotubes\Light-sheet-OPM\Result-data"
runNum = 'run-02'

xls70_h15 = glob.glob(path + '/20211022c_suc70_h15um/' + runNum + '/*.xlsx')
xls70_h30 = glob.glob(path + '/20211022d_suc70_h30um/' + runNum + '/*.xlsx')
npy70_h15 = glob.glob(path + '/20211022c_suc70_h15um/' + runNum + '/*-results.npy')
npy70_h30 = glob.glob(path + '/20211022d_suc70_h30um/' + runNum + '/*-results.npy')

xls50_h15 = glob.glob(path + '/20211018a_suc50_h15um/' + runNum + '/*.xlsx')
xls50_h30 = glob.glob(path + '/20211018b_suc50_h30um/' + runNum + '/*.xlsx')
npy50_h15 = glob.glob(path + '/20211018a_suc50_h15um/' + runNum + '/*-results.npy')
npy50_h30 = glob.glob(path + '/20211018b_suc50_h30um/' + runNum + '/*-results.npy')

xls40_h15 = glob.glob(path + '/20211022a_suc40_h15um/' + runNum + '/*.xlsx')
xls40_h30 = glob.glob(path + '/20211022b_suc40_h30um/' + runNum + '/*.xlsx')
npy40_h15 = glob.glob(path + '/20211022a_suc40_h15um/' + runNum + '/*-results.npy')
npy40_h30 = glob.glob(path + '/20211022b_suc40_h30um/' + runNum + '/*-results.npy')

vis70 = 673 # 70% sucrose, unit: mPa.s (Quintas et al. 2005)
vis50 = 15.04 # 50% sucrose, unit: mPa.s (Telis et al. 2005)
vis40 = 6.20 # 40% sucrose, unit: mPa.s (Telis et al. 2005)


theXLS = xls70_h30
theNPY = npy70_h30
vis = vis70
sur_per = str(70)
rData = 0.05

#%% Recompute diffusion coefficient from tracking

# Compute exposure time
pxum = 0.115; 
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
vol_exp = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 
j = 0;

# Go through every data sets
# for j in range(1):
for j in range(len(theNPY)):
    print(theNPY[j])
    fName = theNPY[j]
    EuAng, dirAng, cm, localAxes = np.load(theNPY[j])  
    Nframes = len(cm)
    dataOld = pd.read_excel(theXLS[j], index_col=None).to_numpy()
    geo_mean = dataOld[0:3,1]    # geo: radius, length, pitch
    geo_std = dataOld[0:3,2]     
        
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
    
    A, B, D = BernieMatrix(fitN[0]*1e-12,fitRoll[0],fitCombo[0]*1e-6)
    A2, B2, D2 = BernieMatrix(fitN[0]*1e-12*(vis),fitRoll[0]*(vis),\
                              fitCombo[0]*1e-6*(vis)) 
    # print to excel
    data = [['number of frames', Nframes],\
            ['radius [um]', geo_mean[0], geo_std[0]],\
            ['length [um]', geo_mean[1], geo_std[1]],\
            ['pitch [um]', geo_mean[2], geo_std[2]],\
            ['trans-fit [um^2/sec^2]',fitN[0][0], fitS[0][0]],\
            ['rotation-fit [rad^2/sec^2]',fitPitch[0][0], fitRoll[0][0], fitYaw[0][0]],\
            ['combo-fit [um.rad/sec^2]',fitCombo[0][0]],\
            ['A, B, D', A[0], B[0], D[0]],\
            ['A, B, D (adjusted '+ sur_per + '\% sucrose)', A2[0], B2[0], D2[0]]\
                ]
    df = pd.DataFrame(data)
    df.to_excel(fName[:-5] + '-' + str(int(rData*100)) +'per.xlsx',\
                index = False, header = False)  