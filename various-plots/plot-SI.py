#%% Import modules and files
import sys
sys.path.insert(0, './modules')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matmatrix import MSDfit
from scipy import optimize
import msd
import glob

# Input files
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
exp3D_ms = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 

# Recompute diffusion coefficient from tracking
theXLS = xls40_h15 + xls40_h30;
theNPY = npy40_h15 + npy40_h30;
theVEC = vec40_h15 + vec40_h30;
vis = vis40; sucPer = str(40)    

# save to array
nInterval = 50; xaxis = np.arange(1,nInterval+1)
Nfit = 10; # number of fitting points
amp_par, mean_par, sigma_par = (np.zeros([nInterval]) for _ in range(3));
amp_perp1, mean_perp1, sigma_perp1 = (np.zeros([nInterval]) for _ in range(3));
amp_perp2, mean_perp2, sigma_perp2 = (np.zeros([nInterval]) for _ in range(3));
msd_N_all = []; msd_S_all = []; msd_S2_all = [];

plotTrans = True; plotRot = True; plotCombo = True;

#%% Go through every data sets
for j in range(len(theXLS)):
# for j in range(1):
    
    print(theXLS[j])
    fName = theXLS[j]
    EuAng, dirAng, cm = np.load(theNPY[j])  
    localAxes = np.load(theVEC[j])
    Nframes = len(cm)
    dataOld = pd.read_excel(theXLS[j], index_col=None).to_numpy()
    geo_mean = dataOld[0:3,1]    # geo: radius, length, pitch
    geo_std = dataOld[0:3,2]     

    # MSD: mean square displacement
    fromMSD = msd.theMSD(len(cm), cm, EuAng[:,1],localAxes, exp3D_ms, nInterval)
    MSD_N, MSD_S, MSD_S2, MSD_NR = fromMSD.trans_combo_MSD()
    MSD_P = msd.regMSD(len(cm), EuAng[:,0], exp3D_ms, nInterval)
    MSD_R = msd.regMSD(len(cm), EuAng[:,1], exp3D_ms, nInterval)
    MSD_Y = msd.regMSD(len(cm), EuAng[:,2], exp3D_ms, nInterval)
    msd_N_all.append(MSD_N); msd_S_all.append(MSD_N); msd_S2_all.append(MSD_S2);
    
    xtime = np.linspace(1,Nfit,Nfit)
    fitN,fitN_const   = optimize.curve_fit(MSDfit, xtime, MSD_N[0:Nfit])[0]
    fitS,fitS_const   = optimize.curve_fit(MSDfit, xtime,\
                        np.mean([MSD_S[0:Nfit],MSD_S[0:Nfit]],axis=0))[0]
    fitNR,fitNR_const = optimize.curve_fit(MSDfit, xtime, MSD_NR[0:Nfit])[0]
    fitP,fitP_const  = optimize.curve_fit(MSDfit, xtime, MSD_P[0:Nfit])[0]
    fitR,fitR_const  = optimize.curve_fit(MSDfit, xtime, MSD_R[0:Nfit])[0]
    fitY,fitY_const  = optimize.curve_fit(MSDfit, xtime, MSD_Y[0:Nfit])[0]
        
    plt.rcParams.update({'font.size': 15})
    if plotTrans:     
        fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
        ax0.plot(xaxis*exp3D_ms,MSD_N,c='k',marker="^",mfc='none',
                  ms=5,ls='None',alpha=0.5)   
        ax0.plot(xaxis*exp3D_ms,np.mean([MSD_S,MSD_S],axis=0),
                 c='k',marker="s",mfc='none',
                  ms=5,ls='None',alpha=0.5)
        ax0.plot(xaxis*exp3D_ms,fitN_const + fitN*xaxis,
                 c='C1',alpha=1,label='_nolegend_')
        ax0.plot(xaxis*exp3D_ms,fitS_const + fitS*xaxis,
                 c='C1',alpha=1,label='_nolegend_')
        ax0.set_title(theXLS[j][-19:-5] + " (" + str(Nfit) +
                      " fitting points)")
        # ax0.set_xscale('log'); ax0.set_yscale('log'); 
        ax0.set_xlabel(r'Lag time [sec]');
        ax0.set_ylabel(r'MSD [$\mu m^2$]')
        # ax0.set_ylim([0, 0.7]);
        ax0.set_xlim([0, nInterval*exp3D_ms]);
        ax0.legend(["lengthwise","sidewise"])
        ax0.figure.savefig(path + '/all-data/suc'+ str(sucPer) +
                           '-MSD/Trans-' + str(j) + '.png')
        
    if plotRot:
        fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
        ax0.plot(xaxis*exp3D_ms,MSD_P,c='k',marker="^",mfc='none',
                  ms=5,ls='None',alpha=0.5)   
        ax0.plot(xaxis*exp3D_ms,MSD_Y,c='k',marker="s",mfc='none',
                  ms=5,ls='None',alpha=0.5)
        ax0.plot(xaxis*exp3D_ms,fitP_const + fitP*xaxis,c='C1',
                 alpha=1,label='_nolegend_')
        ax0.plot(xaxis*exp3D_ms,fitY_const + fitY*xaxis,c='C1',
                 alpha=1,label='_nolegend_')
        ax0.set_title(theXLS[j][-19:-5] + " (" + str(Nfit) +
                      " fitting points)")
        # ax0.set_xscale('log'); ax0.set_yscale('log'); 
        ax0.set_xlabel(r'Lag time [sec]')
        ax0.set_ylabel(r'MSD [rad$^2$]')
        # ax0.set_ylim([0, 0.7]);        
        ax0.set_xlim([0, nInterval*exp3D_ms])
        ax0.legend(["pitch","yaw"])
        ax0.figure.savefig(path + '/all-data/suc'+ str(sucPer) +
                           '-MSD/PitchYaw-' + str(j) + '.png')
        
        fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
        ax0.plot(xaxis*exp3D_ms,MSD_R,c='k',marker="o",mfc='none',
                  ms=5,ls='None',alpha=0.5)   
        ax0.plot(xaxis*exp3D_ms,fitR_const + fitR*xaxis,c='C1',
                 alpha=1,label='_nolegend_')
        ax0.set_title(theXLS[j][-19:-5] + " (" + str(Nfit) +
                      " fitting points)")
        # ax0.set_xscale('log'); ax0.set_yscale('log'); 
        ax0.set_xlabel(r'Lag time [sec]')
        ax0.set_ylabel(r'MSD [rad$^2$]')
        # ax0.set_ylim([0, 0.7]);        
        ax0.set_xlim([0, nInterval*exp3D_ms])
        ax0.legend(["roll"])
        ax0.figure.savefig(path + '/all-data/suc'+ str(sucPer) +
                           '-MSD/Roll-' + str(j) + '.png')

    if plotCombo:        
        fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
        ax0.plot(xaxis*exp3D_ms,MSD_NR,c='k',marker="o",mfc='none',\
                  ms=5,ls='None',alpha=0.5)   
        ax0.plot(xaxis*exp3D_ms,fitNR_const + fitNR*xaxis,c='C1',
                 alpha=1,label='_nolegend_')
        ax0.set_title(theXLS[j][-19:-5] + " (" + str(Nfit) +
                      " fitting points)")
        # ax0.set_xscale('log'); ax0.set_yscale('log'); 
        ax0.set_xlabel(r'Lag time [sec]');
        ax0.set_ylabel(r'MSD [rad$\cdot\mu$m]')
        # ax0.set_ylim([0, 0.7]);        
        ax0.set_xlim([0, nInterval*exp3D_ms])
        ax0.legend(["lengthwise x roll"])
        ax0.figure.savefig(path + '/all-data/suc'+ str(sucPer) +
                           '-MSD/Combo-' + str(j) + '.png')