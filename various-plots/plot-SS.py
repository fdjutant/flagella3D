import sys
sys.path.insert(0, '../modules')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matmatrix import fitCDF, fit2CDF, gauss_cdf, gauss_two_cdf
import msd
import glob

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})

#%% Input files
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

rData = 0.02

# Recompute diffusion coefficient from tracking
theXLS = xls70_h30; theNPY = npy70_h30; theVEC = vec70_h30;\
vis = vis70; sur_per = str(70)    

# save to array
nInterval = 20;
nDivider = np.zeros([nInterval]);
amp_par, amp_par_1, amp_par_2 = (np.zeros([nInterval]) for _ in range(3));
mean_par, mean_par_1, mean_par_2 = (np.zeros([nInterval]) for _ in range(3));
sigma_par, sigma_par_1, sigma_par_2 = (np.zeros([nInterval]) for _ in range(3));
amp_perp1, amp_perp1_1, amp_perp1_2 = (np.zeros([nInterval]) for _ in range(3));
mean_perp1, mean_perp1_1, mean_perp1_2 = (np.zeros([nInterval]) for _ in range(3));
sigma_perp1, sigma_perp1_1, sigma_perp1_2 = (np.zeros([nInterval]) for _ in range(3));
amp_perp2, amp_perp2_1, amp_perp2_2 = (np.zeros([nInterval]) for _ in range(3));
mean_perp2, mean_perp2_1, mean_perp2_2 = (np.zeros([nInterval]) for _ in range(3));
sigma_perp2, sigma_perp2_1, sigma_perp2_2 = (np.zeros([nInterval]) for _ in range(3));

#%% Go through every data sets
# for j in range(3,3):
for j in range(1):
    
    print(theXLS[j])
    fName = theXLS[j]
    EuAng, dirAng, cm = np.load(theNPY[j])  
    localAxes = np.load(theVEC[j])
    Nframes = len(cm)
    dataOld = pd.read_excel(theXLS[j], index_col=None).to_numpy()
    geo_mean = dataOld[0:3,1]    # geo: radius, length, pitch
    geo_std = dataOld[0:3,2]     


    # generate the step size
    fromSS = msd.theSS(len(cm), cm, EuAng[:,1],
                          localAxes, vol_exp, nInterval)
    SS_N, SS_S, SS_S2 = fromSS.trans_SS()
    
    for k in range(1,nInterval+1):    
        
        nDivider[k-1] = k
        
        # generate the step size
        transSS = np.array([SS_N[k-1], SS_S[k-1], SS_S2[k-1]]).T
        
        # SINGLE
        amp_par[k-1], mean_par[k-1], sigma_par[k-1] = fitCDF(transSS[:,0])
        amp_perp1[k-1], mean_perp1[k-1], sigma_perp1[k-1] = fitCDF(transSS[:,1])
        amp_perp2[k-1], mean_perp2[k-1], sigma_perp2[k-1] = fitCDF(transSS[:,2]) 
    
        xplot = np.linspace(-1.5,1.5,1000, endpoint=False)
        y_par = gauss_cdf(xplot, amp_par[k-1], mean_par[k-1], sigma_par[k-1])
        y_perp1 = gauss_cdf(xplot, amp_perp1[k-1], mean_perp1[k-1], sigma_perp1[k-1])
        y_perp2 = gauss_cdf(xplot, amp_perp2[k-1], mean_perp2[k-1], sigma_perp2[k-1])
        
        # DOUBLE
        amp_par_1[k-1], mean_par_1[k-1], mean_par_2[k-1],\
            sigma_par_1[k-1], sigma_par_2[k-1] = fit2CDF(transSS[:,0])
        amp_perp1_1[k-1], mean_perp1_1[k-1], mean_perp1_2[k-1],\
            sigma_perp1_1[k-1], sigma_perp1_2[k-1] = fit2CDF(transSS[:,1])
        amp_perp2_1[k-1], mean_perp2_1[k-1], mean_perp2_2[k-1],\
            sigma_perp2_1[k-1], sigma_perp2_2[k-1] = fit2CDF(transSS[:,2]) 
        amp_par_2[k-1] = 1. - amp_par_1[k-1];
        amp_perp1_2[k-1] = 1. - amp_perp1_1[k-1];
        amp_perp2_2[k-1] = 1. - amp_perp2_1[k-1];
        
        y_par_2cdf = gauss_two_cdf(xplot, amp_par_1[k-1],\
                                   mean_par_1[k-1], mean_par_2[k-1],\
                                   sigma_par_1[k-1], sigma_par_2[k-1])
        y_perp1_2cdf = gauss_two_cdf(xplot, amp_perp1_1[k-1],\
                                     mean_perp1_1[k-1], mean_perp1_2[k-1],\
                                     sigma_perp1_1[k-1], sigma_perp1_2[k-1])
        y_perp2_2cdf = gauss_two_cdf(xplot, amp_perp2_1[k-1],\
                                     mean_perp2_1[k-1], mean_perp2_2[k-1],\
                                     sigma_perp2_1[k-1], sigma_perp2_2[k-1])
        
        # Translation
        plt.rcParams.update({'font.size': 15})
        fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))
        ax1.plot(xplot, y_par,'C0', alpha=0.5)
        ax1.plot(xplot, y_perp1,'C1', alpha=0.5)
        ax1.plot(xplot, y_perp2,'C2', alpha=0.5)
        ax1.plot(xplot, y_par_2cdf,'k', alpha=0.5)
        # ax1.plot(xplot, y_perp1_2cdf,'k')
        # ax1.plot(xplot, y_perp2_2cdf,'k')
        ax1.plot(np.sort(transSS[:,0]),
                  np.linspace(0,1,len(transSS[:,0]),endpoint=False),\
                  'C0o',ms=3, alpha=0.5)
        ax1.plot(np.sort(transSS[:,1]),
                  np.linspace(0,1,len(transSS[:,1]),endpoint=False),\
                  'C1o',ms=3, alpha=0.5)
        ax1.plot(np.sort(transSS[:,2]),
                  np.linspace(0,1,len(transSS[:,2]),endpoint=False),\
                  'C2o',ms=3, alpha=0.5)
        ax1.set_title(theXLS[j][-19:-5] + ' (interval = ' + str(k) + ')')
        ax1.set_xlabel(r'Step size [$\mu$m]');
        ax1.set_ylabel(r'Cumulative Probability')
        ax1.set_ylim([-0.05, 1.1]); ax1.set_xlim([-1.5, 1.5]);
        ax1.legend(['parallel','perpendicular-1',\
                    'perpendicular-2','parallel-2CDF'])
        
#%% Plot all amplitudes
plt.rcParams.update({'font.size': 15})
fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))
ax1.plot(nDivider, amp_par_1,'C0o-', alpha=0.5)
ax1.plot(nDivider, amp_par_2,'C1o-', alpha=0.5)
ax1.plot(nDivider, amp_par,'C2o-', alpha=0.5)
ax1.set_title(theXLS[j][-19:-5])
ax1.set_xlabel(r'Number of separation');
ax1.set_ylabel(r'Amplitude')
ax1.set_ylim([-0.1, 1.1]);
ax1.set_xlim([0.5, nInterval+1]);
ax1.legend(['amplitude 1','amplitude 2','amplitude (single CDF)'])

#%% Plot all sigma2
plt.rcParams.update({'font.size': 15})
fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))
ax1.plot(nDivider, sigma_par_1**2,'C0o-', alpha=0.5)
ax1.plot(nDivider, sigma_par**2,'C2o-', alpha=0.5)
ax1.set_title(theXLS[j][-19:-5])
ax1.set_xlabel(r'Number of separation');
ax1.set_ylabel(r'Variance $[\mu m^2]$')
ax1.set_ylim([0, 0.75]);
ax1.set_xlim([0.95, nInterval]);
ax1.legend([r'$\sigma^2_1$ (double CDF)',r'$\sigma^2$ (single CDF)'])

#%% Plot all sigma2
plt.rcParams.update({'font.size': 15})
fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))
ax1.plot(nDivider, sigma_par_2**2,'C1o-', alpha=0.5)
ax1.set_title(theXLS[j][-19:-5])
ax1.set_xlabel(r'Number of separation');
ax1.set_ylabel(r'Variance $[\mu m^2]$')
# ax1.set_ylim([0, 0.75]);
ax1.set_xlim([0.95, nInterval]);
ax1.legend([r'$\sigma^2_2$ (double CDF)'])
            
