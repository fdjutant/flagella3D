# Import all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})
from scipy import stats, optimize
from matmatrix import *
import msd
from msd import regMSD, trans_stepSize,\
                rot_stepSize, trans_stepSize_all

folderName = 'syn-result/nT50-nF3000-nSep200/';
Dperp = np.load(folderName+'Dperp.npy', allow_pickle=True)
Dpar = 2*Dperp; 
vol_exp = np.load(folderName+'vol_exp.npy', allow_pickle=True)
nPoints = np.load(folderName+'nPoints.npy', allow_pickle=True)
nDivider = np.load(folderName+'nDivider.npy', allow_pickle=True)
errorTh = np.load(folderName+'errorTh.npy', allow_pickle=True)
cm_traj = np.load(folderName+'cm_traj.npy',allow_pickle=True)
EuAng_traj = np.load(folderName+'EuAng_traj.npy',allow_pickle=True)
localAxes_traj = np.load(folderName+'localAxes_traj.npy',allow_pickle=True)
transSS_traj = np.load(folderName+'transSS_traj.npy',allow_pickle=True)
rotSS_traj = np.load(folderName+'rotSS_traj.npy',allow_pickle=True)
NRSS_traj = np.load(folderName+'NRSS_traj.npy',allow_pickle=True)
time_x_traj = np.load(folderName+'time_x_traj.npy',allow_pickle=True)
sigma2_par = np.load(folderName+'sigma2_par.npy',allow_pickle=True)
sigma2_perp1 = np.load(folderName+'sigma2_perp1.npy',allow_pickle=True)
sigma2_perp2 = np.load(folderName+'sigma2_perp2.npy',allow_pickle=True)
sigma2_NR = np.load(folderName+'sigma2_NR.npy',allow_pickle=True)
sigma2_P = np.load(folderName+'sigma2_P.npy',allow_pickle=True)
sigma2_R = np.load(folderName+'sigma2_R.npy',allow_pickle=True)
sigma2_Y = np.load(folderName+'sigma2_Y.npy',allow_pickle=True)
MSD_N_traj = np.load(folderName+'MSD_N_traj.npy',allow_pickle=True)
MSD_S_traj = np.load(folderName+'MSD_S_traj.npy',allow_pickle=True)
MSD_S2_traj = np.load(folderName+'MSD_S2_traj.npy',allow_pickle=True)
MSD_NR_traj = np.load(folderName+'MSD_NR_traj.npy',allow_pickle=True)
MSD_P_traj = np.load(folderName+'MSD_P_traj.npy',allow_pickle=True)
MSD_R_traj = np.load(folderName+'MSD_R_traj.npy',allow_pickle=True)
MSD_Y_traj = np.load(folderName+'MSD_Y_traj.npy',allow_pickle=True)
fitN_1per = np.load(folderName+'fitN_1per.npy',allow_pickle=True)
fitS_1per = np.load(folderName+'fitS_1per.npy',allow_pickle=True)
fitS2_1per = np.load(folderName+'fitS2_1per.npy',allow_pickle=True)
fitNR_1per = np.load(folderName+'fitNR_1per.npy',allow_pickle=True)
fitN_5per = np.load(folderName+'fitN_5per.npy',allow_pickle=True)
fitS_5per = np.load(folderName+'fitS_5per.npy',allow_pickle=True)
fitS2_5per = np.load(folderName+'fitS2_5per.npy',allow_pickle=True)
fitNR_5per = np.load(folderName+'fitNR_5per.npy',allow_pickle=True)
fitP_1per = np.load(folderName+'fitP_1per.npy',allow_pickle=True)
fitR_1per = np.load(folderName+'fitR_1per.npy',allow_pickle=True)
fitY_1per = np.load(folderName+'fitY_1per.npy',allow_pickle=True)
fitP_5per = np.load(folderName+'fitP_5per.npy',allow_pickle=True)
fitR_5per = np.load(folderName+'fitR_5per.npy',allow_pickle=True)
fitY_5per = np.load(folderName+'fitY_5per.npy',allow_pickle=True)
nSepTotal = cm_traj.shape[1]

#%% Plot diffusion constants (trans, rot, parallel x roll)
xaxis = np.arange(1,nSepTotal+1)      
plt.rcParams.update({'font.size': 15})

fig,ax = plt.subplots(dpi=300, figsize=(14,5))
ax2 = ax.twiny()
ax.errorbar(xaxis, np.mean(sigma2_par/(6*vol_exp*xaxis),axis=0),
              yerr=np.std(sigma2_par/(6*vol_exp*xaxis),axis=0),
              color='k', marker="^",alpha=0.5,
              capsize=2, elinewidth = 0.5)
ax.errorbar(xaxis, np.mean(sigma2_perp1/(6*vol_exp*xaxis),axis=0),
              yerr=np.std(sigma2_perp1/(6*vol_exp*xaxis),axis=0),
              color='k', marker="s",alpha=0.5,
              capsize=2, elinewidth = 0.5, label='_nolegend_')
ax.errorbar(xaxis, np.mean(sigma2_perp2/(6*vol_exp*xaxis),axis=0),
              yerr=np.std(sigma2_perp2/(6*vol_exp*xaxis),axis=0),
              color='k', marker="o",alpha=0.5,
              capsize=2, elinewidth = 0.5, label='_nolegend_')
ax.plot(xaxis, np.ones(len(xaxis))*Dpar, 'r')
ax.plot(xaxis, np.ones(len(xaxis))*Dperp, 'r')
ax.fill_between(xaxis, np.ones(len(xaxis))*Dpar-errorTh,\
                np.ones(len(xaxis))*Dpar+errorTh, facecolor='r',\
                alpha=0.1)
ax.fill_between(xaxis, np.ones(len(xaxis))*Dperp-errorTh,\
                np.ones(len(xaxis))*Dperp+errorTh, facecolor='r',\
                alpha=0.1)
ax.set_xticks(nDivider);
ax2.set_xticks(nDivider);
ax.set_xticklabels(nPoints.astype('int'))
ax.legend(["Step-size", "Ground-Truth"])
ax2.set_xlabel(r'Sampling rate ratio (full/reduced)');
ax.set_xlabel(r'Number of frames');
ax.set_ylabel(r'$D [\mu m^2/sec]$') 
ax2.set_xlim(0.5,20.5)
ax.set_xlim(0.5,20.5)

fig,ax = plt.subplots(dpi=300, figsize=(14,5))
ax2 = ax.twiny()
ax.errorbar(xaxis, np.mean(fitN_1per/(6),axis=0),
              yerr=np.std(fitN_1per/(6),axis=0),
              color='g', marker="^",alpha=0.5,
              capsize=2, elinewidth = 0.5)
ax.errorbar(xaxis, np.mean(fitS_1per/(6),axis=0),
              yerr=np.std(fitS_1per/(6),axis=0),
              color='g', marker="s",alpha=0.5,
              capsize=2, elinewidth = 0.5, label='_nolegend_')
ax.errorbar(xaxis, np.mean(fitS2_1per/(6),axis=0),
              yerr=np.std(fitS2_1per/(6),axis=0),
              color='g', marker="o",alpha=0.5,
              capsize=2, elinewidth = 0.5, label='_nolegend_')
ax.plot(xaxis, np.ones(len(xaxis))*Dpar, 'r')
ax.plot(xaxis, np.ones(len(xaxis))*Dperp, 'r')
ax.fill_between(xaxis, np.ones(len(xaxis))*Dpar-errorTh,\
                np.ones(len(xaxis))*Dpar+errorTh, facecolor='r',\
                alpha=0.1)
ax.fill_between(xaxis, np.ones(len(xaxis))*Dperp-errorTh,\
                np.ones(len(xaxis))*Dperp+errorTh, facecolor='r',\
                alpha=0.1)
ax.set_xticks(nDivider);
ax2.set_xticks(nDivider);
ax.set_xticklabels(nPoints.astype('int'))
ax.legend(["MSD","Ground-Truth"])
ax2.set_xlabel(r'Sampling rate ratio (full/reduced)');
ax.set_xlabel(r'Number of frames');
ax.set_ylabel(r'$D [\mu m^2/sec]$') 
ax2.set_xlim(0.5,20.5)
ax.set_xlim(0.5,20.5)

fig,ax = plt.subplots(dpi=300, figsize=(14,5))
ax2 = ax.twiny()
ax.errorbar(xaxis, np.mean(fitNR_1per/(6),axis=0),
              yerr=np.std(fitNR_1per/(6),axis=0),
              color='g', marker="o",alpha=0.5,
              capsize=2, elinewidth = 0.5)
ax.errorbar(xaxis, np.mean(fitNR_5per/(6),axis=0),
              yerr=np.std(fitNR_5per/(6),axis=0),
              color='b', marker="o",alpha=0.5,
              capsize=2, elinewidth = 0.5)
ax.errorbar(xaxis, np.mean(sigma2_NR/(6*vol_exp*xaxis),axis=0),
              yerr=np.std(sigma2_NR/(6*vol_exp*xaxis),axis=0),
              color='k', marker="o",alpha=0.5,
              capsize=2, elinewidth = 0.5, label='_nolegend_')
ax.set_xticks(nDivider);
ax2.set_xticks(nDivider);
ax.set_xticklabels(nPoints.astype('int'))
ax.legend(["1\% fit","5\% fit"])
ax2.set_xlabel(r'Sampling rate ratio (full/reduced)');
ax.set_xlabel(r'Number of frames');
ax.set_ylabel(r'$D_{\parallel R} [\mu m^2/sec]$') 
ax2.set_xlim(0.5,20.5)
ax.set_xlim(0.5,20.5)

#%% Plot fluctuation
singleTraj = True;
if singleTraj: 
    
    # Choose which trajectory and number of separation
    nT = 1; nSep = 2; 
    x_axis = np.linspace(0,len(cm_traj[nT,nSep][:,0])-1,\
                         num=len(cm_traj[nT,nSep][:,0]))*vol_exp   
    
    # plot x, y, z
    plt.rcParams.update({'font.size': 12})
    fig01,ax01 = plt.subplots(dpi=300, figsize=(6,2))
    ax01.plot(x_axis,cm_traj[nT,nSep][:,0]*0.115,\
              c='k',marker="^",mfc='none',ms=4,alpha=0.2)
    ax01.plot(x_axis,cm_traj[nT,nSep][:,1]*0.115,\
              c='k',marker="s",mfc='none',ms=4,alpha=0.2)
    ax01.plot(x_axis,cm_traj[nT,nSep][:,2]*0.115,\
              c='k',marker="o",mfc='none',ms=4,alpha=0.2)
    ax01.legend(["x","y","z"])
    ax01.set_xlabel(r'$\Delta t$ [sec]');
    ax01.set_ylabel(r'$r [\mu m]$') 
    
    # plot n1_x, n1_y, n1_z
    fig01,ax01 = plt.subplots(dpi=300, figsize=(6,2))
    ax01.plot(x_axis,localAxes_traj[nT,nSep][:,0,0],\
              c='C0',alpha=0.4)
    ax01.plot(x_axis,localAxes_traj[nT,nSep][:,0,1],\
              c='C1',alpha=0.4)
    ax01.plot(x_axis,localAxes_traj[nT,nSep][:,0,2],\
              c='C2',alpha=0.4)
    ax01.legend([r"$n_{1x}$",r"$n_{1y}$",r"$n_{1z}$"])
    ax01.set_xlabel(r'$\Delta t$ [sec]');
    ax01.set_ylabel(r'$n_1$') 
    
    # plot pitch, roll, yaw
    plt.rcParams.update({'font.size': 12})
    fig01,ax01 = plt.subplots(dpi=300, figsize=(6,2))
    ax01.plot(x_axis,EuAng_traj[nT,nSep][:,0],\
              c='k',marker="^",mfc='none',ms=4,alpha=0.2)
    ax01.plot(x_axis,EuAng_traj[nT,nSep][:,1],\
              c='k',marker="s",mfc='none',ms=4,alpha=0.2)
    ax01.plot(x_axis,EuAng_traj[nT,nSep][:,2],\
              c='k',marker="o",mfc='none',ms=4,alpha=0.2)
    ax01.legend(["pitch","roll","yaw"])
    ax01.set_xlabel(r'$\Delta t$ [sec]');
    ax01.set_ylabel('Angle [rad]') 
    

#%% Plot StepSize
singleStep = True;
if singleStep: 
    
    # Choose which trajectory and number of separation
    nT = 1; nSep = 0; 
    
    # Combo (parallel x roll) - CDF & PDF
    xplot = np.linspace(-2,2,1000, endpoint=False)
    y_NR = gauss_cdf(xplot, 1, 0, np.sqrt(sigma2_NR[nT,nSep]))
    ypdf_NR = gauss_pdf(xplot, 1, 0, np.sqrt(sigma2_NR[nT,nSep]))
    
    plt.rcParams.update({'font.size': 15})
    fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))
    ax1.plot(xplot, y_NR,'C0', alpha=0.5)
    ax1.plot(np.sort(NRSS_traj[nT,nSep]),
              np.linspace(0,1,len(transSS_traj[nT,nSep][:,0]),
                          endpoint=False),'k',ms=3, alpha=0.5)
    ax1.set_xlabel(r'Step size [$\mu$m$\times$rad]');
    ax1.set_ylabel(r'Cumulative Probability')
    ax1.set_ylim([-0.05, 1.1]); 

    plt.rcParams.update({'font.size': 15})
    fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))
    ax1.plot(xplot, ypdf_NR,'k', alpha=0.8)
    ax1.hist(NRSS_traj[nT,nSep], bins='fd',
             density=True, color='k', alpha=0.3)
    ax1.set_xlabel(r'Step size [$\mu$m$\times$rad]');
    ax1.set_ylabel(r'Probability density')
    
    # Rotation - CDF & PDF
    xplot = np.linspace(-2,2,1000, endpoint=False)
    y_P = gauss_cdf(xplot, 1, 0, np.sqrt(sigma2_P[nT,nSep]))
    y_R = gauss_cdf(xplot, 1, 0, np.sqrt(sigma2_R[nT,nSep]))
    y_Y = gauss_cdf(xplot, 1, 0, np.sqrt(sigma2_Y[nT,nSep]))      
    ypdf_P = gauss_pdf(xplot, 1, 0, np.sqrt(sigma2_P[nT,nSep]))
    ypdf_R = gauss_pdf(xplot, 1, 0, np.sqrt(sigma2_R[nT,nSep]))
    ypdf_Y = gauss_pdf(xplot, 1, 0, np.sqrt(sigma2_Y[nT,nSep]))
    
    plt.rcParams.update({'font.size': 15})
    fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))
    ax1.plot(xplot, y_P,'C0', alpha=0.5)
    ax1.plot(xplot, y_R,'C1', alpha=0.5)
    ax1.plot(xplot, y_Y,'C2', alpha=0.5)
    ax1.plot(np.sort(rotSS_traj[nT,nSep][:,0]),
              np.linspace(0,1,len(transSS_traj[nT,nSep][:,0]),
                          endpoint=False),'C0o',ms=3, alpha=0.5)
    ax1.plot(np.sort(rotSS_traj[nT,nSep][:,1]),
              np.linspace(0,1,len(transSS_traj[nT,nSep][:,1]),
                          endpoint=False),'C1o',ms=3, alpha=0.5)
    ax1.plot(np.sort(rotSS_traj[nT,nSep][:,2]),
              np.linspace(0,1,len(transSS_traj[nT,nSep][:,2]),
                          endpoint=False),'C2o',ms=3, alpha=0.5)
    ax1.set_xlabel(r'Step size [$\mu$m]');
    ax1.set_ylabel(r'Cumulative Probability')
    ax1.set_ylim([-0.05, 1.1]); #ax1.set_xlim([0, r_xaxis]);
    ax1.legend(['pitch','roll','yaw'])

    plt.rcParams.update({'font.size': 15})
    fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))
    ax1.plot(xplot, ypdf_P,'C0', alpha=0.8)
    ax1.plot(xplot, ypdf_R,'C1', alpha=0.8)
    ax1.plot(xplot, ypdf_Y,'C2', alpha=0.8)
    ax1.hist(rotSS_traj[nT,nSep][:,0], bins='fd',
             density=True, color='C0', alpha=0.3)
    ax1.hist(rotSS_traj[nT,nSep][:,1], bins='fd',
             density=True, color='C1', alpha=0.3)
    ax1.hist(rotSS_traj[nT,nSep][:,2], bins='fd',
             density=True, color='C2', alpha=0.3)
    ax1.set_xlabel(r'Step size [$\mu$m]');
    ax1.set_ylabel(r'Probability density')
    # ax1.set_xlim([-2.5, 2.5]);
    ax1.legend(['pitch','roll','yaw'])
    
    # Translation - CDF & PDF
    xplot = np.linspace(-1,1,1000, endpoint=False)
    y_par = gauss_cdf(xplot, 1, 0, np.sqrt(sigma2_par[nT,nSep]))
    y_perp1 = gauss_cdf(xplot, 1, 0, np.sqrt(sigma2_perp1[nT,nSep]))
    y_perp2 = gauss_cdf(xplot, 1, 0, np.sqrt(sigma2_perp2[nT,nSep]))      
    ypdf_par = gauss_pdf(xplot, 1, 0, np.sqrt(sigma2_par[nT,nSep]))
    ypdf_perp1 = gauss_pdf(xplot, 1, 0, np.sqrt(sigma2_perp1[nT,nSep]))
    ypdf_perp2 = gauss_pdf(xplot, 1, 0, np.sqrt(sigma2_perp2[nT,nSep]))
    
    plt.rcParams.update({'font.size': 15})
    fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))
    ax1.plot(xplot, y_par,'C0', alpha=0.5)
    ax1.plot(xplot, y_perp1,'C1', alpha=0.5)
    ax1.plot(xplot, y_perp2,'C2', alpha=0.5)
    ax1.plot(np.sort(transSS_traj[nT,nSep][:,0]),
              np.linspace(0,1,len(transSS_traj[nT,nSep][:,0]),endpoint=False),\
              'C0o',ms=3, alpha=0.5)
    ax1.plot(np.sort(transSS_traj[nT,nSep][:,1]),
              np.linspace(0,1,len(transSS_traj[nT,nSep][:,1]),endpoint=False),\
              'C1o',ms=3, alpha=0.5)
    ax1.plot(np.sort(transSS_traj[nT,nSep][:,2]),
              np.linspace(0,1,len(transSS_traj[nT,nSep][:,2]),endpoint=False),\
              'C2o',ms=3, alpha=0.5)
    ax1.set_xlabel(r'Step size [$\mu$m]');
    ax1.set_ylabel(r'Cumulative Probability')
    ax1.set_ylim([-0.05, 1.1]); #ax1.set_xlim([0, r_xaxis]);
    ax1.legend(['parallel','perpendicular-1','perpendicular-2'])

    plt.rcParams.update({'font.size': 15})
    fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))
    ax1.plot(xplot, ypdf_par,'C0', alpha=0.8)
    ax1.plot(xplot, ypdf_perp1,'C1', alpha=0.8)
    ax1.plot(xplot, ypdf_perp2,'C2', alpha=0.8)
    ax1.hist(transSS_traj[nT,nSep][:,0], bins='fd', density=True, color='C0', alpha=0.3)
    ax1.hist(transSS_traj[nT,nSep][:,1], bins='fd', density=True, color='C1', alpha=0.3)
    ax1.hist(transSS_traj[nT,nSep][:,2], bins='fd', density=True, color='C2', alpha=0.3)
    ax1.set_xlabel(r'Step size [$\mu$m]');
    ax1.set_ylabel(r'Probability density')
    # ax1.set_xlim([-2.5, 2.5]);
    ax1.legend(['parallel','perpendicular-1','perpendicular-2'])

#%% Plot the MSD
singleMSD = True
if singleMSD: 

    # Choose which trajectory and number of separation
    nT = 1; nSep = 0; 
    
    # Plot specific MSD      
    fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
    xAxis = time_x_traj[nT,nSep];
    ax0.plot(xAxis,MSD_N_traj[nT,nSep],c='k',marker="^",mfc='none',ms=9,ls='None',alpha=0.5,label='_nolegend_')   
    ax0.plot(xAxis,MSD_S_traj[nT,nSep],c='k',marker="s",mfc='none',ms=9,ls='None',alpha=0.5,label='_nolegend_')
    ax0.plot(xAxis,MSD_S2_traj[nT,nSep],c='k',marker="o",mfc='none',ms=9,ls='None',alpha=0.5,label='_nolegend_')
    ax0.plot(xAxis,fitN_1per[nT,nSep]*xAxis,c='g',alpha=0.2)
    ax0.plot(xAxis,fitS_1per[nT,nSep]*xAxis,c='g',alpha=0.2,label='_nolegend_')
    ax0.plot(xAxis,fitS2_1per[nT,nSep]*xAxis,c='g',alpha=0.2,label='_nolegend_')
    ax0.plot(xAxis,fitN_5per[nT,nSep]*xAxis,c='b',alpha=0.2)
    ax0.plot(xAxis,fitS_5per[nT,nSep]*xAxis,c='b',alpha=0.2,label='_nolegend_')
    ax0.plot(xAxis,fitS2_5per[nT,nSep]*xAxis,c='b',alpha=0.2,label='_nolegend_')
    ax0.legend(["Fitting 1\%","Fitting 5\%"])
    ax0.set_xscale('log'); ax0.set_yscale('log'); 
    ax0.set_title('MSD translation')
    ax0.set_xlabel(r'Log($\tau$) [sec]');
    ax0.set_ylabel(r'Log(MSD) [$\mu m^2$/sec]')
    # ax0.set_ylim([np.exp(-0.5*10e-1),np.exp(10^4)])
    # ax0.legend(["parallel","perpendicular", "perpendicular-2"])

    fig3,ax3 = plt.subplots(dpi=300, figsize=(6,5))
    ax3.plot(xAxis,MSD_P_traj[nT,nSep],c='k',marker="^",mfc='none',ms=9,ls='None',alpha=0.5,label='_nolegend_')
    ax3.plot(xAxis,MSD_R_traj[nT,nSep],c='k',marker="s",mfc='none',ms=9,ls='None',alpha=0.5,label='_nolegend_')   
    ax3.plot(xAxis,MSD_Y_traj[nT,nSep],c='k',marker="o",mfc='none',ms=9,ls='None',alpha=0.5,label='_nolegend_')   
    ax3.plot(xAxis,fitP_1per[nT,nSep]*xAxis,c='g',alpha=0.2)
    ax3.plot(xAxis,fitR_1per[nT,nSep]*xAxis,c='g',alpha=0.2,label='_nolegend_')
    ax3.plot(xAxis,fitY_1per[nT,nSep]*xAxis,c='g',alpha=0.2,label='_nolegend_')
    ax3.plot(xAxis,fitP_5per[nT,nSep]*xAxis,c='b',alpha=0.2)
    ax3.plot(xAxis,fitR_5per[nT,nSep]*xAxis,c='b',alpha=0.2,label='_nolegend_')
    ax3.plot(xAxis,fitY_5per[nT,nSep]*xAxis,c='b',alpha=0.2,label='_nolegend_')
    ax3.legend(["Fitting 1\%","Fitting 5\%"])
    ax3.set_xscale('log'); ax3.set_yscale('log'); 
    ax3.set_title('MSAD for pitch, yaw, and roll')
    ax3.set_xlabel(r'Log($\tau$) [sec]');
    ax3.set_ylabel(r'Log(MSAD) [rad$^2$/sec]')
    # ax3.legend(["pitch","roll","yaw"])   
    
    fig4,ax4 = plt.subplots(dpi=300, figsize=(6,5))
    ax4.plot(xAxis,MSD_NR_traj[nT,nSep],c='k',alpha=0.5,label='_nolegend_')
    ax4.plot(xAxis,fitNR_1per[nT,nSep]*xAxis,c='g',alpha=0.2)
    ax4.plot(xAxis,fitNR_5per[nT,nSep]*xAxis,c='b',alpha=0.2)
    ax4.legend(["Fitting 1\%","Fitting 5\%"])
    ax4.set_title('MSD combo')
    ax4.set_xlabel(r'Log($\tau$) [sec]');
    ax4.set_ylabel(r'$\langle \Delta Y \Delta\psi\rangle [\mu m\cdot rad/sec]$') 