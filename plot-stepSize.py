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
from msd import regMSD, trans_stepSize, rot_stepSize, trans_stepSize_all
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

rData = 0.02

# Recompute diffusion coefficient from tracking
theXLS = xls70_h30; theNPY = npy70_h30; theVEC = vec70_h30;\
vis = vis70; sur_per = str(70)    

# Go through every data sets
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
    transSS = trans_stepSize(cm, localAxes)
    # rotSS =  rot_stepSize(EuAng)
    
    Dt_par = []; Dt_perp1 = []; Dt_perp2 = [];
        
    # SINGLE
    amp_par, mean_par, sigma_par = fitCDF(transSS[:,0])
    amp_perp1, mean_perp1, sigma_perp1 = fitCDF(transSS[:,1])
    amp_perp2, mean_perp2, sigma_perp2 = fitCDF(transSS[:,2])    
    Dt_par = sigma_par**2 / (2 * 3 * vol_exp)
    Dt_perp1 = sigma_perp1**2 / (2 * 3 * vol_exp)
    Dt_perp2 = sigma_perp2**2 / (2 * 3 * vol_exp)
    # print(Dt_par, Dt_perp1, Dt_perp2)
    
    # Plot
    xplot = np.linspace(-1.5,1.5,1000, endpoint=False)
    y_par = gauss_cdf(xplot, amp_par, mean_par, sigma_par)
    y_perp1 = gauss_cdf(xplot, amp_perp1, mean_perp1, sigma_perp1)
    y_perp2 = gauss_cdf(xplot, amp_perp2, mean_perp2, sigma_perp2)
    ypdf_par = gauss_pdf(xplot, amp_par, mean_par, sigma_par)
    ypdf_perp1 = gauss_pdf(xplot, amp_perp1, mean_perp1, sigma_perp1)
    ypdf_perp2 = gauss_pdf(xplot, amp_perp2, mean_perp2, sigma_perp2)
    
    # DOUBLE
    amp_par_1, amp_par_2, mean_par_1, mean_par_2,\
        sigma_par_1, sigma_par_2 = fit2CDF(transSS[:,0])
    amp_perp1_1, amp_perp1_2, mean_perp1_1, mean_perp1_2,\
        sigma_perp1_1, sigma_perp1_2 = fit2CDF(transSS[:,1])
    amp_perp2_1, amp_perp2_2, mean_perp2_1, mean_perp2_2,\
        sigma_perp2_1, sigma_perp2_2 = fit2CDF(transSS[:,2])    
    # Dt_par = sigma_par**2 / (2 * 3 * vol_exp)
    # Dt_perp1 = sigma_perp1**2 / (2 * 3 * vol_exp)
    # Dt_perp2 = sigma_perp2**2 / (2 * 3 * vol_exp)
    
    # Plot
    y_par_2cdf = gauss_two_cdf(xplot, amp_par_1, amp_par_2,
                               mean_par_1, mean_par_2,
                               sigma_par_1, sigma_par_2)
    y_perp1_2cdf = gauss_two_cdf(xplot, amp_perp1_1, amp_perp1_2,
                                 mean_perp1_1, mean_perp1_2,\
                                 sigma_perp1_1, sigma_perp1_2)
    y_perp2_2cdf = gauss_two_cdf(xplot, amp_perp2_1, amp_perp2_2,
                                 mean_perp2_1, mean_perp2_2,\
                                 sigma_perp2_1, sigma_perp2_2)
    ypdf_par_2cdf = gauss_two_pdf(xplot, amp_par_1, amp_par_2,
                               mean_par_1, mean_par_2,
                               sigma_par_1, sigma_par_2)
    ypdf_perp1_2cdf = gauss_two_pdf(xplot, amp_perp1_1, amp_perp1_2,
                                 mean_perp1_1, mean_perp1_2,\
                                 sigma_perp1_1, sigma_perp1_2)
    ypdf_perp2_2cdf = gauss_two_pdf(xplot, amp_perp2_1, amp_perp2_2,
                                 mean_perp2_1, mean_perp2_2,\
                                 sigma_perp2_1, sigma_perp2_2)
    
    # Plot CDF
    plt.rcParams.update({'font.size': 15})
    fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))
    ax1.plot(xplot, y_par,'C0', alpha=0.5)
    ax1.plot(xplot, y_perp1,'C1', alpha=0.5)
    ax1.plot(xplot, y_perp2,'C2', alpha=0.5)
    ax1.plot(xplot, y_par_2cdf,'k')
    ax1.plot(xplot, y_perp1_2cdf,'k')
    ax1.plot(xplot, y_perp2_2cdf,'k')
    ax1.plot(np.sort(transSS[:,0]),
              np.linspace(0,1,len(transSS[:,0]),endpoint=False),\
              'C0o',ms=3, alpha=0.5)
    ax1.plot(np.sort(transSS[:,1]),
              np.linspace(0,1,len(transSS[:,1]),endpoint=False),\
              'C1o',ms=3, alpha=0.5)
    ax1.plot(np.sort(transSS[:,2]),
              np.linspace(0,1,len(transSS[:,2]),endpoint=False),\
              'C2o',ms=3, alpha=0.5)
    ax1.set_title(theXLS[j][-19:-5] + '.npy')
    ax1.set_xlabel(r'Step size [$\mu$m]');
    ax1.set_ylabel(r'Cumulative Probability')
    ax1.set_ylim([-0.05, 1.1]); #ax1.set_xlim([0, r_xaxis]);
    ax1.legend(['parallel','perpendicular-1','perpendicular-2'])
    # ax1.figure.savefig(path + '/PDF/trans-suc40-CDF.pdf')

    # plot PDF
    plt.rcParams.update({'font.size': 15})
    fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))
    ax1.plot(xplot, ypdf_par,'C0', alpha=0.8)
    ax1.plot(xplot, ypdf_perp1,'C1', alpha=0.8)
    ax1.plot(xplot, ypdf_perp2,'C2', alpha=0.8)
    ax1.plot(xplot, ypdf_par_2cdf,'k')
    ax1.plot(xplot, ypdf_perp1_2cdf,'k')
    ax1.plot(xplot, ypdf_perp2_2cdf,'k')
    ax1.hist(transSS[:,0], bins='fd', density=True, color='C0', alpha=0.3)
    ax1.hist(transSS[:,1], bins='fd', density=True, color='C1', alpha=0.3)
    ax1.hist(transSS[:,2], bins='fd', density=True, color='C2', alpha=0.3)
    ax1.set_title(theXLS[j][-19:-5] + '.npy')
    ax1.set_xlabel(r'Step size [$\mu$m]');
    ax1.set_ylabel(r'Probability density')
    ax1.set_xlim([-1, 1]);
    ax1.legend(['parallel','perpendicular-1','perpendicular-2'])
    # ax1.figure.savefig(path + '/PDF/trans-suc40-PDF.pdf')
        