#%% Import modules and files
import sys
sys.path.insert(0, './modules')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matmatrix import fitCDF, gauss_cdf, MSDfit
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
vol_exp = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 

# Recompute diffusion coefficient from tracking
theXLS = xls70_h15 + xls70_h30;
theNPY = npy70_h15 + npy70_h30;
theVEC = vec70_h15 + vec70_h30;
vis = vis70; sur_per = str(70)    

# save to array
nInterval = 50; xaxis = np.arange(1,nInterval+1)  
Nfit = 10; # number of fitting points
nDivider = np.zeros([nInterval]);
amp_par, mean_par, sigma_par = (np.zeros([nInterval]) for _ in range(3));
amp_perp1, mean_perp1, sigma_perp1 = (np.zeros([nInterval]) for _ in range(3));
amp_perp2, mean_perp2, sigma_perp2 = (np.zeros([nInterval]) for _ in range(3));
msd_N_all = []; msd_S_all = []; msd_S2_all = [];

plotCDF = False; plotMSD = False; plotBoth = False;

#%% Go through every data sets
for j in range(len(theXLS)):
# for j in range(2,3):
    
    print(theXLS[j])
    fName = theXLS[j]
    EuAng, dirAng, cm = np.load(theNPY[j])  
    localAxes = np.load(theVEC[j])
    Nframes = len(cm)
    dataOld = pd.read_excel(theXLS[j], index_col=None).to_numpy()
    geo_mean = dataOld[0:3,1]    # geo: radius, length, pitch
    geo_std = dataOld[0:3,2]     

    # MSD: mean square displacement
    fromMSD = msd.theMSD(len(cm), cm, EuAng[:,1],localAxes, vol_exp, nInterval)
    time_x, MSD_N, MSD_S, MSD_S2, MSD_NR = fromMSD.trans_combo_MSD()
    time_xP, MSD_P = msd.regMSD(len(cm), EuAng[:,0], vol_exp, nInterval)
    time_xR, MSD_R = msd.regMSD(len(cm), EuAng[:,1], vol_exp, nInterval)
    time_xY, MSD_Y = msd.regMSD(len(cm), EuAng[:,2], vol_exp, nInterval)
    msd_N_all.append(MSD_N); msd_S_all.append(MSD_N); msd_S2_all.append(MSD_S2);
    
    xtime = np.linspace(1,Nfit,Nfit)
    fitN,fitN_const   = optimize.curve_fit(MSDfit, xtime, MSD_N[0:Nfit])[0]
    fitS,fitS_const   = optimize.curve_fit(MSDfit, xtime, MSD_S[0:Nfit])[0]
    fitS2,fitS2_const = optimize.curve_fit(MSDfit, xtime, MSD_S2[0:Nfit])[0]
    fitNR,_ = optimize.curve_fit(MSDfit, xtime, MSD_NR[0:Nfit])[0]
    fitP,_  = optimize.curve_fit(MSDfit, xtime, MSD_P[0:Nfit])[0]
    fitR,_  = optimize.curve_fit(MSDfit, xtime, MSD_R[0:Nfit])[0]
    fitY,_  = optimize.curve_fit(MSDfit, xtime, MSD_Y[0:Nfit])[0]
    
    # CDF from step size
    fromSS = msd.theSS(len(cm), cm, EuAng[:,1],
                          localAxes, vol_exp, nInterval)
    SS_N, SS_S, SS_S2 = fromSS.trans_SS()
    
    for k in range(1,nInterval+1):    
        nDivider[k-1] = k
        
        # generate the step size
        transSS = np.array([SS_N[k-1], SS_S[k-1], SS_S2[k-1]]).T
        
        # compute sigma^2 from CDF
        amp_par[k-1], mean_par[k-1], sigma_par[k-1] = fitCDF(transSS[:,0])
        amp_perp1[k-1], mean_perp1[k-1], sigma_perp1[k-1] = fitCDF(transSS[:,1])
        amp_perp2[k-1], mean_perp2[k-1], sigma_perp2[k-1] = fitCDF(transSS[:,2]) 

        # plot the CDF
        if plotCDF:
            xplot = np.linspace(-1.5,1.5,1000, endpoint=False)
            y_par = gauss_cdf(xplot, amp_par[k-1], mean_par[k-1], sigma_par[k-1])
            y_perp1 = gauss_cdf(xplot, amp_perp1[k-1], mean_perp1[k-1], sigma_perp1[k-1])
            y_perp2 = gauss_cdf(xplot, amp_perp2[k-1], mean_perp2[k-1], sigma_perp2[k-1])        
            plt.rcParams.update({'font.size': 15})
            fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))
            ax1.plot(xplot, y_par,'C0', alpha=0.5)
            ax1.plot(xplot, y_perp1,'C1', alpha=0.5)
            ax1.plot(xplot, y_perp2,'C2', alpha=0.5)
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
                        'perpendicular-2'])
    
    # Plot sigma2 from CDF and MSD
    if plotBoth:
        plt.rcParams.update({'font.size': 15})
        fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))
        ax1.plot(nDivider, sigma_par**2,'C0^-', alpha=0.5)
        ax1.plot(nDivider, MSD_N,'C1^-', alpha=0.5)
        ax1.plot(nDivider, sigma_perp1**2,'C0s-', alpha=0.5)
        ax1.plot(nDivider, MSD_S,'C1s-', alpha=0.5)
        ax1.plot(nDivider, sigma_perp2**2,'C0o-', alpha=0.5)
        ax1.plot(nDivider, MSD_S2,'C1o-', alpha=0.5)
        ax1.set_title(theXLS[j][-19:-5] + " (lengthwise)")
        ax1.set_xlabel(r'Number of separation');
        ax1.set_ylabel(r'Variance $[\mu m^2]$')
        # ax1.set_ylim([0, 0.7]);
        ax1.set_xlim([0.95, nInterval]);
        ax1.legend([r'from CDF',r'direct or MSD'])
    
    if plotMSD:     
        fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
        xAxis = time_x;
        ax0.plot(nDivider,MSD_N,c='k',marker="^",mfc='none',\
                  ms=5,ls='None',alpha=0.5)   
        ax0.plot(nDivider,MSD_S,c='k',marker="s",mfc='none',\
                  ms=5,ls='None',alpha=0.5)
        ax0.plot(nDivider,MSD_S2,c='k',marker="o",mfc='none',\
                  ms=5,ls='None',alpha=0.5)
        ax0.plot(nDivider,fitN_const + fitN*nDivider,c='C1',alpha=1,label='_nolegend_')
        ax0.plot(nDivider,fitS_const + fitS*nDivider,c='C1',alpha=1,label='_nolegend_')
        ax0.plot(nDivider,fitS2_const + fitS2*nDivider,c='C1',alpha=1,label='_nolegend_')
        ax0.set_title(theXLS[j][-19:-5] + " (" + str(Nfit) + " fitting points)")
        # ax0.set_xscale('log'); ax0.set_yscale('log'); 
        ax0.set_xlabel(r'Number of separation');
        ax0.set_ylabel(r'MSD [$\mu m^2$/sec]')
        # ax0.set_ylim([0, 0.7]);
        ax0.set_xlim([0.95, nInterval]);
        ax0.legend(["lengthwise","sidewise-1", "sidewise-2"])
    
#%% Plot all MSD 
msd_N_all = np.array(msd_N_all);
msd_S_all = np.array(msd_S_all);
msd_S2_all = np.array(msd_S2_all);

fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
for i in range(len(msd_N_all)):
    ax0.plot(nDivider,msd_N_all[i],alpha=0.5)   
ax0.set_title("Lengthwise ("+ sur_per + "% sucrose)")
# ax0.set_xscale('log'); ax0.set_yscale('log'); 
ax0.set_xlabel(r'Number of separation');
ax0.set_ylabel(r'MSD [$\mu m^2$/sec]')
ax0.set_ylim([0, 2.5]);
ax0.set_xlim([0.95, nInterval]);

fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
for i in range(len(msd_S_all)):
    ax0.plot(nDivider,0.5*(msd_S_all[i]+msd_S2_all[i]),alpha=0.5)   
ax0.set_title("Sidewise ("+ sur_per + "% sucrose)")
# ax0.set_xscale('log'); ax0.set_yscale('log'); 
ax0.set_xlabel(r'Number of separation');
ax0.set_ylabel(r'MSD [$\mu m^2$/sec]')
ax0.set_ylim([0, 2.5]);
ax0.set_xlim([0.95, nInterval]);

fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
for i in range(len(msd_S_all)):
    ax0.plot(nDivider,msd_N_all[i],'--',alpha=0.5)   
    ax0.plot(nDivider,0.5*(msd_S_all[i]+msd_S2_all[i]),alpha=1)   
ax0.set_title("Sidewise ("+ sur_per + "% sucrose)")
# ax0.set_xscale('log'); ax0.set_yscale('log'); 
ax0.set_xlabel(r'Number of separation');
ax0.set_ylabel(r'MSD [$\mu m^2$/sec]')
ax0.set_ylim([0, 2.5]);
ax0.set_xlim([0.95, nInterval]);