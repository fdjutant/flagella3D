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
from lmfit import Model
from scipy.special import erf
from matmatrix import *
import helixFun
import imProcess
import msd
from msd import regMSD, trans_stepSize, rot_stepSize
import movingHx  
import glob
from natsort import natsorted, ns
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})

# path = r"C:\Users\labuser\Dropbox (ASU)\Research\DNA-Rotary-Motor\Helical-nanotubes\Light-sheet-OPM\Result-data"
path = r"D:\Dropbox (ASU)\Research\DNA-Rotary-Motor\Helical-nanotubes\Light-sheet-OPM\Result-data"
runNum = 'run-03'

xls70_h15 = glob.glob(path + '/20211022c_suc70_h15um/' + runNum + '/*.xlsx')
npy70_h15 = glob.glob(path + '/20211022c_suc70_h15um/' + runNum + '/*-angleCM.npy')
vec70_h15 = glob.glob(path + '/20211022c_suc70_h15um/' + runNum + '/*-vectorN.npy')

xls70_h30 = glob.glob(path + '/20211022d_suc70_h30um/' + runNum + '/*.xlsx')
npy70_h30 = glob.glob(path + '/20211022d_suc70_h30um/' + runNum + '/*-angleCM.npy')
vec70_h30 = glob.glob(path + '/20211022d_suc70_h30um/' + runNum + '/*-vectorN.npy')

xls50_h15 = glob.glob(path + '/20211018a_suc50_h15um/' + runNum + '/*.xlsx')
npy50_h15 = glob.glob(path + '/20211018a_suc50_h15um/' + runNum + '/*-angleCM.npy')
vec50_h15 = glob.glob(path + '/20211018a_suc50_h15um/' + runNum + '/*-vectorN.npy')

xls50_h30 = glob.glob(path + '/20211018b_suc50_h30um/' + runNum + '/*.xlsx')
npy50_h30 = glob.glob(path + '/20211018b_suc50_h30um/' + runNum + '/*-angleCM.npy')
vec50_h30 = glob.glob(path + '/20211018b_suc50_h30um/' + runNum + '/*-vectorN.npy')

xls40_h15 = glob.glob(path + '/20211022a_suc40_h15um/' + runNum + '/*.xlsx')
npy40_h15 = glob.glob(path + '/20211022a_suc40_h15um/' + runNum + '/*-angleCM.npy')
vec40_h15 = glob.glob(path + '/20211022a_suc40_h15um/' + runNum + '/*-vectorN.npy')

xls40_h30 = glob.glob(path + '/20211022b_suc40_h30um/' + runNum + '/*.xlsx')
npy40_h30 = glob.glob(path + '/20211022b_suc40_h30um/' + runNum + '/*-angleCM.npy')
vec40_h30 = glob.glob(path + '/20211022b_suc40_h30um/' + runNum + '/*-vectorN.npy')

vis70 = 673     # 70%(w/w) sucrose [mPa.s] (Quintas et al. 2005)
vis50 = 15.04   # 50%(w/w) sucrose [mPa.s] (Telis et al. 2005)
vis40 = 6.20    # 40%(w/w) sucrose [mPa.s] (Telis et al. 2005)

# Compute exposure time
pxum = 0.115; 
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
vol_exp = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 

#%% Recompute diffusion coefficient from tracking
# Go through every data sets
k = 0;
for k in range (6):
    if k == 0: 
        theXLS = xls40_h15; theNPY = npy40_h15; theVEC = vec40_h15;\
        vis = vis40; sur_per = str(40)
    elif k == 1: 
        theXLS = xls40_h30; theNPY = npy40_h30; theVEC = vec40_h30;\
        vis = vis40; sur_per = str(40)
    elif k == 2:
        theXLS = xls50_h15; theNPY = npy50_h15; theVEC = vec50_h15;\
        vis = vis50; sur_per = str(50)
    elif k == 3:
        theXLS = xls50_h30; theNPY = npy50_h30; theVEC = vec50_h30;\
        vis = vis50; sur_per = str(50)
    elif k == 4:
        theXLS = xls70_h15; theNPY = npy70_h15; theVEC = vec70_h15;\
        vis = vis70; sur_per = str(70)
    else:
        theXLS = xls70_h30; theNPY = npy70_h30; theVEC = vec70_h30;\
            vis = vis70; sur_per = str(70)  
            
    # Go through every data sets
    for j in range(len(theNPY)):
        print(theXLS[j])
        fName = theXLS[j]
        EuAng, dirAng, cm = np.load(theNPY[j])  
        localAxes = np.load(theVEC[j])
        Nframes = len(cm)
        dataOld = pd.read_excel(theXLS[j], index_col=None).to_numpy()
        geo_mean = dataOld[0:3,1]    # geo: radius, length, pitch
        geo_std = dataOld[0:3,2]     
            
        # generate the step size
        transSS = trans_stepSize(cm, localAxes)
        rotSS =  rot_stepSize(EuAng)
    
        def gauss_cdf(x, mu, sigma):
            Fx = 1/2.0 * (1. + erf( (x-mu)/ (sigma*np.sqrt(2.))))
            return Fx
        
        def gauss_pdf(x, mu, sigma):
            Fx = np.exp(-0.5*(x-mu)**2/sigma**2) / (sigma * np.sqrt(2*np.pi))
            return Fx
    
        def fitCDF(x):
            model = Model(gauss_cdf, prefix='g1_')
            params = model.make_params(g1_mu = 0, g1_sigma = 0.5)
            yaxis = np.linspace(0,1,len(x), endpoint=False)
            xaxis = np.sort(x)
            result = model.fit(yaxis,params,x=xaxis)
            mean = result.params['g1_mu'].value
            sigma = result.params['g1_sigma'].value
            return mean, sigma
    
        # Translation
        mean_par, sigma_par = fitCDF(transSS[:,0])
        mean_perp1, sigma_perp1 = fitCDF(transSS[:,1])
        mean_perp2, sigma_perp2 = fitCDF(transSS[:,2])
        Dt_par = sigma_par**2 / (2 * vol_exp)
        Dt_perp1 = sigma_perp1**2 / (2 * vol_exp)
        Dt_perp2 = sigma_perp2**2 / (2 * vol_exp)
        
        # Rotation
        mean_pitch, sigma_pitch = fitCDF(rotSS[:,0])
        mean_roll, sigma_roll = fitCDF(rotSS[:,1])
        mean_yaw, sigma_yaw = fitCDF(rotSS[:,2])
        D_pitch = sigma_pitch**2 / (2 * vol_exp)
        D_roll = sigma_roll**2 / (2 * vol_exp)
        D_yaw = sigma_yaw**2 / (2 * vol_exp)
        
        D_combo = np.sqrt(Dt_par*D_roll)
        
        A, B, D = BernieMatrix(Dt_par*1e-12,D_roll,D_combo*1e-6)
        A2, B2, D2 = BernieMatrix(Dt_par*1e-12*(vis),D_roll*(vis),\
                                  D_combo*1e-6*(vis)) 
        # print to excel
        data = [['number of frames', Nframes],\
                ['radius [um]', geo_mean[0], geo_std[0]],\
                ['length [um]', geo_mean[1], geo_std[1]],\
                ['pitch [um]', geo_mean[2], geo_std[2]],\
                ['trans-fit [um^2/sec^2]',Dt_par, Dt_perp1, Dt_perp2],\
                ['rotation-fit [rad^2/sec^2]',D_pitch, D_roll, D_yaw],\
                ['combo-fit [um.rad/sec^2]',D_combo],\
                ['A, B, D', A, B, D],\
                ['A, B, D (adjusted '+ sur_per + '\% sucrose)',\
                 A2, B2, D2]\
                    ]
        df = pd.DataFrame(data)
        # df.to_excel(fName[:-5] + '-StSz.xlsx',\
        #             index = False, header = False)  
        df.to_excel(fName[:-20] + '/SS/' + fName[-19:-5] + '-StSz.xlsx',\
                    index = False, header = False)  