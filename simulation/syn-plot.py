#%% Load data
import sys
sys.path.insert(0, './modules')
import numpy as np
import matplotlib.pyplot as plt
from matmatrix import gauss_cdf, gauss_pdf
import pickle

fName = "nT10-nSep20-nFrame3000" + ".pkl"
folderName = r"D:/Dropbox (ASU)/Research/DNA-Rotary-Motor/Helical-nanotubes/Light-sheet-OPM/Result-data/synthetic-data/"
with open(folderName+fName, "rb") as f:
      data_loaded = pickle.load(f)

Dperp = data_loaded["Dperp"]
Dpar = 2*Dperp; 
vol_exp = data_loaded["vol_exp"]
nPoints = data_loaded["nPoints"]
nDivider = data_loaded["nDivider"]
errorTh = data_loaded["errorTh"]
cm_traj = data_loaded["cm_traj"]
EuAng_traj = data_loaded["EuAng_traj"]
localAxes_traj = data_loaded["localAxes_traj"]
transSS_traj = data_loaded["transSS_traj"]
rotSS_traj = data_loaded["rotSS_traj"]
NRSS_traj = data_loaded["NRSS_traj"]
time_x_traj = data_loaded["time_x_traj"]
sigma2_N = data_loaded["sigma2_N"]
sigma2_S = data_loaded["sigma2_S"]
sigma2_S2 = data_loaded["sigma2_S2"]
sigma2_NR = data_loaded["sigma2_NR"]
sigma2_N_MSD = data_loaded["sigma2_N_MSD"]
sigma2_S_MSD = data_loaded["sigma2_S_MSD"]
sigma2_S2_MSD = data_loaded["sigma2_S2_MSD"]
sigma2_NR_MSD = data_loaded["sigma2_NR_MSD"]
sigma2_P = data_loaded["sigma2_P"]
sigma2_R = data_loaded["sigma2_R"]
sigma2_Y = data_loaded["sigma2_Y"]
MSD_N_traj = data_loaded["MSD_N_traj"]
MSD_S_traj = data_loaded["MSD_S_traj"]
MSD_S2_traj = data_loaded["MSD_S2_traj"]
MSD_NR_traj = data_loaded["MSD_NR_traj"]
MSD_P_traj = data_loaded["MSD_P_traj"]
MSD_R_traj = data_loaded["MSD_R_traj"]
MSD_Y_traj = data_loaded["MSD_Y_traj"]
Ndata_a = data_loaded["Ndata_a"]
Ndata_b = data_loaded["Ndata_b"]
fitN_a = data_loaded["fitN_a"]
fitS_a = data_loaded["fitS_a"]
fitS2_a = data_loaded["fitS2_a"]
fitNR_a = data_loaded["fitNR_a"]
fitN_b = data_loaded["fitN_b"]
fitS_b = data_loaded["fitS_b"]
fitS2_b = data_loaded["fitS2_b"]
fitNR_b = data_loaded["fitNR_b"]
fitP_a = data_loaded["fitP_a"]
fitR_a = data_loaded["fitR_a"]
fitY_a = data_loaded["fitY_a"]
fitP_b = data_loaded["fitP_b"]
fitR_b = data_loaded["fitR_b"]
fitY_b = data_loaded["fitY_b"]
nSepTotal = cm_traj.shape[1]

#%% Plot diffusion coefficients (translation)
xaxis = np.arange(1,nSepTotal+1)      
plt.rcParams.update({'font.size': 15})

fig,ax = plt.subplots(dpi=300, figsize=(14,5))
ax2 = ax.twiny()
ax.errorbar(xaxis, np.mean(sigma2_N/(6*vol_exp*xaxis),axis=0),
              yerr=np.std(sigma2_N/(6*vol_exp*xaxis),axis=0),
              color='k', marker="^",alpha=0.5,
              capsize=2, elinewidth = 0.5)
ax.errorbar(xaxis, np.mean(sigma2_S/(6*vol_exp*xaxis),axis=0),
              yerr=np.std(sigma2_S/(6*vol_exp*xaxis),axis=0),
              color='k', marker="s",alpha=0.5,
              capsize=2, elinewidth = 0.5, label='_nolegend_')
ax.errorbar(xaxis, np.mean(sigma2_S2/(6*vol_exp*xaxis),axis=0),
              yerr=np.std(sigma2_S2/(6*vol_exp*xaxis),axis=0),
              color='k', marker="o",alpha=0.5,
              capsize=2, elinewidth = 0.5, label='_nolegend_')
ax.plot(xaxis, np.ones(len(xaxis))*Dpar, 'r')
ax.plot(xaxis, np.ones(len(xaxis))*Dperp, 'r', label='_nolegend_')
ax.fill_between(xaxis, Dpar * (np.ones(len(xaxis))-errorTh),\
                Dpar * (np.ones(len(xaxis))+errorTh), facecolor='r',\
                alpha=0.1, label='_nolegend_')
ax.fill_between(xaxis, Dperp * (np.ones(len(xaxis))-errorTh),\
                Dperp * (np.ones(len(xaxis))+errorTh), facecolor='r',\
                alpha=0.1, label='_nolegend_')
ax.set_xticks(nDivider);
ax2.set_xticks(nDivider);
ax.set_xticklabels(nPoints.astype('int'))
ax.legend(["Ground-Truth","Step-size"])
ax2.set_xlabel(r'Sampling rate ratio (full/reduced)');
ax.set_xlabel(r'Number of frames');
ax.set_ylabel(r'$D [\mu m^2/sec]$') 
ax2.set_xlim(0.5,20.5)
ax.set_xlim(0.5,20.5)

fig,ax = plt.subplots(dpi=300, figsize=(14,5))
ax2 = ax.twiny()
ax.errorbar(xaxis, np.mean(sigma2_N_MSD/(6*vol_exp*xaxis),axis=0),
              yerr=np.std(sigma2_N_MSD/(6*vol_exp*xaxis),axis=0),
              color='C0', marker="^",alpha=0.5,
              capsize=2, elinewidth = 0.5)
ax.errorbar(xaxis, np.mean(sigma2_S_MSD/(6*vol_exp*xaxis),axis=0),
              yerr=np.std(sigma2_S_MSD/(6*vol_exp*xaxis),axis=0),
              color='C0', marker="s",alpha=0.5,
              capsize=2, elinewidth = 0.5, label='_nolegend_')
ax.errorbar(xaxis, np.mean(sigma2_S2_MSD/(6*vol_exp*xaxis),axis=0),
              yerr=np.std(sigma2_S2_MSD/(6*vol_exp*xaxis),axis=0),
              color='C0', marker="o",alpha=0.5,
              capsize=2, elinewidth = 0.5, label='_nolegend_')
ax.plot(xaxis, np.ones(len(xaxis))*Dpar, 'r')
ax.plot(xaxis, np.ones(len(xaxis))*Dperp, 'r', label='_nolegend_')
ax.fill_between(xaxis, Dpar * (np.ones(len(xaxis))-errorTh),\
                Dpar * (np.ones(len(xaxis))+errorTh), facecolor='r',\
                alpha=0.1, label='_nolegend_')
ax.fill_between(xaxis, Dperp * (np.ones(len(xaxis))-errorTh),\
                Dperp * (np.ones(len(xaxis))+errorTh), facecolor='r',\
                alpha=0.1, label='_nolegend_')
ax.set_xticks(nDivider);
ax2.set_xticks(nDivider);
ax.set_xticklabels(nPoints.astype('int'))
ax.legend(["Ground-Truth","1st-point MSD"])
ax2.set_xlabel(r'Sampling rate ratio (full/reduced)');
ax.set_xlabel(r'Number of frames');
ax.set_ylabel(r'$D [\mu m^2/sec]$') 
ax2.set_xlim(0.5,20.5)
ax.set_xlim(0.5,20.5)

fig,ax = plt.subplots(dpi=300, figsize=(14,5))
ax2 = ax.twiny()
ax.errorbar(xaxis, np.mean(fitN_a/(6*vol_exp*xaxis),axis=0),
              yerr=np.std(fitN_a/(6*vol_exp*xaxis),axis=0),
              color='b', marker="^",alpha=0.5,
              capsize=2, elinewidth = 0.5)
ax.errorbar(xaxis, np.mean(fitS_a/(6*vol_exp*xaxis),axis=0),
              yerr=np.std(fitS_a/(6*vol_exp*xaxis),axis=0),
              color='b', marker="s",alpha=0.5,
              capsize=2, elinewidth = 0.5, label='_nolegend_')
ax.errorbar(xaxis, np.mean(fitS2_a/(6*vol_exp*xaxis),axis=0),
              yerr=np.std(fitS2_a/(6*vol_exp*xaxis),axis=0),
              color='b', marker="o",alpha=0.5,
              capsize=2, elinewidth = 0.5, label='_nolegend_')
ax.plot(xaxis, np.ones(len(xaxis))*Dpar, 'r')
ax.plot(xaxis, np.ones(len(xaxis))*Dperp, 'r', label='_nolegend_')
ax.fill_between(xaxis, Dpar * (np.ones(len(xaxis))-errorTh),\
                Dpar * (np.ones(len(xaxis))+errorTh), facecolor='r',\
                alpha=0.1, label='_nolegend_')
ax.fill_between(xaxis, Dperp * (np.ones(len(xaxis))-errorTh),\
                Dperp * (np.ones(len(xaxis))+errorTh), facecolor='r',\
                alpha=0.1, label='_nolegend_')
ax.set_xticks(nDivider);
ax2.set_xticks(nDivider);
ax.set_xticklabels(nPoints.astype('int'))
ax.legend(["Ground-Truth",
           "MSD ("+ str(Ndata_a-1) +" fitting points)"])
ax2.set_xlabel(r'Sampling rate ratio (full/reduced)');
ax.set_xlabel(r'Number of frames');
ax.set_ylabel(r'$D [\mu m^2/sec]$') 
ax2.set_xlim(0.5,20.5)
ax.set_xlim(0.5,20.5)

fig,ax = plt.subplots(dpi=300, figsize=(14,5))
ax2 = ax.twiny()
ax.errorbar(xaxis, np.mean(fitN_b/(6*vol_exp*xaxis),axis=0),
              yerr=np.std(fitN_b/(6*vol_exp*xaxis),axis=0),
              color='g', marker="^",alpha=0.5,
              capsize=2, elinewidth = 0.5)
ax.errorbar(xaxis, np.mean(fitS_b/(6*vol_exp*xaxis),axis=0),
              yerr=np.std(fitS_b/(6*vol_exp*xaxis),axis=0),
              color='g', marker="s",alpha=0.5,
              capsize=2, elinewidth = 0.5, label='_nolegend_')
ax.errorbar(xaxis, np.mean(fitS2_b/(6*vol_exp*xaxis),axis=0),
              yerr=np.std(fitS2_b/(6*vol_exp*xaxis),axis=0),
              color='g', marker="o",alpha=0.5,
              capsize=2, elinewidth = 0.5, label='_nolegend_')
ax.plot(xaxis, np.ones(len(xaxis))*Dpar, 'r')
ax.plot(xaxis, np.ones(len(xaxis))*Dperp, 'r', label='_nolegend_')
ax.fill_between(xaxis, Dpar * (np.ones(len(xaxis))-errorTh),\
                Dpar * (np.ones(len(xaxis))+errorTh), facecolor='r',\
                alpha=0.1, label='_nolegend_')
ax.fill_between(xaxis, Dperp * (np.ones(len(xaxis))-errorTh),\
                Dperp * (np.ones(len(xaxis))+errorTh), facecolor='r',\
                alpha=0.1, label='_nolegend_')
ax.set_xticks(nDivider);
ax2.set_xticks(nDivider);
ax.set_xticklabels(nPoints.astype('int'))
ax.legend(["Ground-Truth",
           "MSD ("+ str(Ndata_b-1) +" fitting points)"])
ax2.set_xlabel(r'Sampling rate ratio (full/reduced)');
ax.set_xlabel(r'Number of frames');
ax.set_ylabel(r'$D [\mu m^2/sec]$') 
ax2.set_xlim(0.5,20.5)
ax.set_xlim(0.5,20.5)

#%% Plot diffusion coefficients (rotation)
xaxis = np.arange(1,nSepTotal+1)  
errorTh1 = np.zeros([nSepTotal])    
for k in range(len(xaxis)):
    errorTh1[k-1] = np.sqrt(2/(xaxis.T[k]))

Dpitch = 0.1
plt.rcParams.update({'font.size': 15})
fig,ax = plt.subplots(dpi=300, figsize=(14,5))
ax2 = ax.twiny()
ax.errorbar(xaxis, np.mean(sigma2_P/(6*vol_exp*xaxis),axis=0),
              yerr=np.std(sigma2_P/(6*vol_exp*xaxis),axis=0),
              color='C0', marker="^",alpha=0.5,
              capsize=2, elinewidth = 0.5)
ax.errorbar(xaxis, np.mean(sigma2_Y/(6*vol_exp*xaxis),axis=0),
              yerr=np.std(sigma2_Y/(6*vol_exp*xaxis),axis=0),
              color='C1', marker="s",alpha=0.5,
              capsize=2, elinewidth = 0.5)
ax.plot(xaxis, np.ones(len(xaxis))*0.1, 'r')
ax.fill_between(xaxis, Dpitch * (np.ones(len(xaxis))-errorTh),\
                Dpitch * (np.ones(len(xaxis))+errorTh), facecolor='r',\
                alpha=0.1, label='_nolegend_')
ax.set_xticks(nDivider);
ax2.set_xticks(nDivider);
ax.set_xticklabels(nPoints.astype('int'))
ax.legend(["Ground truth","Pitch","Yaw"])
ax2.set_xlabel(r'Sampling rate ratio (full/reduced)');
ax.set_xlabel(r'Number of frames');
ax.set_ylabel(r'$D [rad^2/sec]$') 
ax2.set_xlim(0.5,20.5)
ax.set_xlim(0.5,20.5)

Droll = 5;
fig,ax = plt.subplots(dpi=300, figsize=(14,5))
ax2 = ax.twiny()
ax.errorbar(xaxis, np.mean(sigma2_R/(6*vol_exp*xaxis),axis=0),
              yerr=np.std(sigma2_R/(6*vol_exp*xaxis),axis=0),
              color='C2', marker="^",alpha=0.5,
              capsize=2, elinewidth = 0.5)
ax.plot(xaxis, np.ones(len(xaxis))*Droll, 'r')
ax.fill_between(xaxis, Droll * (np.ones(len(xaxis))-errorTh),\
                Droll * (np.ones(len(xaxis))+errorTh), facecolor='r',\
                alpha=0.1, label='_nolegend_')
ax.set_xticks(nDivider);
ax2.set_xticks(nDivider);
ax.set_xticklabels(nPoints.astype('int'))
ax.legend(["Ground truth","Roll"])
ax2.set_xlabel(r'Sampling rate ratio (full/reduced)');
ax.set_xlabel(r'Number of frames');
ax.set_ylabel(r'$D [rad^2/sec]$') 
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
    nT = 1; nSep = 9; 
    
    # Combo (parallel x roll) - CDF & PDF
    xplot = np.linspace(-2.5,2.5,1000, endpoint=False)
    # y_NR = gauss_cdf(xplot, 1, 0, np.sqrt(sigma2_NR[nT,nSep]))
    # ypdf_NR = gauss_pdf(xplot, 1, 0, np.sqrt(sigma2_NR[nT,nSep]))
    
    plt.rcParams.update({'font.size': 15})
    fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))
    # ax1.plot(xplot, y_NR,'C0', alpha=0.5)
    ax1.plot(np.sort(NRSS_traj[nT,nSep]),
              np.linspace(0,1,len(transSS_traj[nT,nSep][:,0]),
                          endpoint=False),'k',ms=3, alpha=0.5)
    ax1.set_xlabel(r'Step size [$\mu$m$\times$rad]');
    ax1.set_ylabel(r'Cumulative Probability')
    ax1.set_ylim([-0.05, 1.1]); ax1.set_xlim([-2.5, 2.5]); 

    plt.rcParams.update({'font.size': 15})
    fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))
    # ax1.plot(xplot, ypdf_NR,'k', alpha=0.8)
    ax1.hist(NRSS_traj[nT,nSep], bins='fd',
             density=True, color='k', alpha=0.3)
    ax1.set_xlabel(r'Step size [$\mu$m$\times$rad]');
    ax1.set_xlim([-2.5, 2.5]);
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
    ax1.set_ylim([-0.05, 1.1]); ax1.set_xlim([-2, 2]);
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
    ax1.set_xlim([-2, 2]);
    ax1.legend(['pitch','roll','yaw'])
    
    # Translation - CDF & PDF
    xplot = np.linspace(-2,2,1000, endpoint=False)
    y_par = gauss_cdf(xplot, 1, 0, np.sqrt(sigma2_N[nT,nSep]))
    y_perp1 = gauss_cdf(xplot, 1, 0, np.sqrt(sigma2_S[nT,nSep]))
    y_perp2 = gauss_cdf(xplot, 1, 0, np.sqrt(sigma2_S2[nT,nSep]))      
    ypdf_par = gauss_pdf(xplot, 1, 0, np.sqrt(sigma2_N[nT,nSep]))
    ypdf_perp1 = gauss_pdf(xplot, 1, 0, np.sqrt(sigma2_S[nT,nSep]))
    ypdf_perp2 = gauss_pdf(xplot, 1, 0, np.sqrt(sigma2_S2[nT,nSep]))
    
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
    ax1.set_ylim([-0.05, 1.1]); ax1.set_xlim([-4, 4]);
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
    ax1.set_xlim([-2.5, 2.5]);
    ax1.legend(['parallel','perpendicular-1','perpendicular-2'])

#%% Plot the MSD
singleMSD = True
if singleMSD: 

    # Choose which trajectory and number of separation
    nT = 1; nSep = 0; 
    
    # Plot specific MSD      
    fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
    xAxis = time_x_traj[nT,nSep];
    ax0.plot(xAxis,MSD_N_traj[nT,nSep],c='k',marker="^",mfc='none',ms=2,ls='None',alpha=0.5,label='_nolegend_')   
    ax0.plot(xAxis,MSD_S_traj[nT,nSep],c='k',marker="s",mfc='none',ms=2,ls='None',alpha=0.5,label='_nolegend_')
    ax0.plot(xAxis,MSD_S2_traj[nT,nSep],c='k',marker="o",mfc='none',ms=2,ls='None',alpha=0.5,label='_nolegend_')
    ax0.plot(xAxis,fitN_a[nT,nSep]*xAxis,c='g',alpha=1)
    ax0.plot(xAxis,fitS_a[nT,nSep]*xAxis,c='g',alpha=1,label='_nolegend_')
    ax0.plot(xAxis,fitS2_a[nT,nSep]*xAxis,c='g',alpha=1,label='_nolegend_')
    ax0.plot(xAxis,fitN_b[nT,nSep]*xAxis,c='b',alpha=1)
    ax0.plot(xAxis,fitS_b[nT,nSep]*xAxis,c='b',alpha=1,label='_nolegend_')
    ax0.plot(xAxis,fitS2_b[nT,nSep]*xAxis,c='b',alpha=1,label='_nolegend_')
    ax0.legend(["("+ str(Ndata_a) +" fitting points)",
                "("+ str(Ndata_b) +" fitting points)"])
    # ax0.set_xscale('log'); ax0.set_yscale('log'); 
    ax0.set_title('MSD translation')
    ax0.set_xlabel(r'Log($\tau$) [sec]');
    ax0.set_ylabel(r'Log(MSD) [$\mu m^2$/sec]')
    # ax0.set_ylim([np.exp(-0.5*10e-1),np.exp(10^4)])
    # ax0.legend(["parallel","perpendicular", "perpendicular-2"])

    fig3,ax3 = plt.subplots(dpi=300, figsize=(6,5))
    ax3.plot(xAxis,MSD_P_traj[nT,nSep],c='k',marker="^",mfc='none',ms=9,ls='None',alpha=0.5,label='_nolegend_')
    ax3.plot(xAxis,MSD_R_traj[nT,nSep],c='k',marker="s",mfc='none',ms=9,ls='None',alpha=0.5,label='_nolegend_')   
    ax3.plot(xAxis,MSD_Y_traj[nT,nSep],c='k',marker="o",mfc='none',ms=9,ls='None',alpha=0.5,label='_nolegend_')   
    ax3.plot(xAxis,fitP_a[nT,nSep]*xAxis,c='g',alpha=0.2)
    ax3.plot(xAxis,fitR_a[nT,nSep]*xAxis,c='g',alpha=0.2,label='_nolegend_')
    ax3.plot(xAxis,fitY_a[nT,nSep]*xAxis,c='g',alpha=0.2,label='_nolegend_')
    ax3.plot(xAxis,fitP_b[nT,nSep]*xAxis,c='b',alpha=0.2)
    ax3.plot(xAxis,fitR_b[nT,nSep]*xAxis,c='b',alpha=0.2,label='_nolegend_')
    ax3.plot(xAxis,fitY_b[nT,nSep]*xAxis,c='b',alpha=0.2,label='_nolegend_')
    ax3.legend(["("+ str(Ndata_a) +" fitting points)",
                "("+ str(Ndata_b) +" fitting points)"])
    # ax3.set_xscale('log'); ax3.set_yscale('log'); 
    ax3.set_title('MSAD for pitch, yaw, and roll')
    ax3.set_xlabel(r'Log($\tau$) [sec]');
    ax3.set_ylabel(r'Log(MSAD) [rad$^2$/sec]')
    # ax3.legend(["pitch","roll","yaw"])   
    
    fig4,ax4 = plt.subplots(dpi=300, figsize=(6,5))
    ax4.plot(xAxis,MSD_NR_traj[nT,nSep],c='k',alpha=0.5,label='_nolegend_')
    ax4.plot(xAxis,fitNR_a[nT,nSep]*xAxis,c='g',alpha=0.2)
    ax4.plot(xAxis,fitNR_b[nT,nSep]*xAxis,c='b',alpha=0.2)
    ax4.legend(["("+ str(Ndata_a) +" fitting points)",
                "("+ str(Ndata_b) +" fitting points)"])
    ax4.set_title('MSD combo')
    ax4.set_xlabel(r'Log($\tau$) [sec]');
    ax4.set_ylabel(r'$\langle \Delta Y \Delta\psi\rangle [\mu m\cdot rad/sec]$') 