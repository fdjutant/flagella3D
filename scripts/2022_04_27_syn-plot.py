#%% Load data
import sys
sys.path.insert(0, './modules')
import numpy as np
import matplotlib.pyplot as plt
# plt.show()
from matmatrix import gauss_cdf, gauss_pdf
import pickle
from scipy import integrate
import os.path
from pathlib import Path
import datetime

fName = "nT100-nSep11-nFrame3000000" + ".pkl"

tstamp = datetime.datetime.now().strftime('%Y_%m_%d_%H;%M;%S')


this_file_dir = os.path.join(os.path.dirname(os.path.abspath("./")),
                            'Dropbox (ASU)','Research')
path = os.path.join(this_file_dir,
                'DNA-Rotary-Motor', 'Helical-nanotubes',
                'Light-sheet-OPM', 'Result-data',
                'synthetic-data', fName)
result_dir = os.path.join(os.path.dirname(os.path.dirname(path)),'PDF')

path = Path()

with open(path, "rb") as f:
      data_loaded = pickle.load(f)
loadTraj = False

Dpar = data_loaded["Dpar"]
Dperp = data_loaded["Dperp"]
Dpitch = data_loaded["Dpitch"]
Droll = data_loaded["Droll"]
Dyaw = data_loaded["Dyaw"]
interval = data_loaded["interval"]
vol_exp = data_loaded["vol_exp"]
nPoints = data_loaded["nPoints"]
nDivider = data_loaded["nDivider"]
errorTh = data_loaded["errorTh"]
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
Ndata_a = data_loaded["Ndata_a"]
fitN_a = data_loaded["fitN_a"]
fitS_a = data_loaded["fitS_a"]
fitS2_a = data_loaded["fitS2_a"]
fitNR_a = data_loaded["fitNR_a"]
fitP_a = data_loaded["fitP_a"]
fitR_a = data_loaded["fitR_a"]
fitY_a = data_loaded["fitY_a"]

if loadTraj:
    cm_traj = data_loaded["cm_traj"]
    EuAng_traj = data_loaded["EuAng_traj"]
    localAxes_traj = data_loaded["localAxes_traj"]
    transSS_traj = data_loaded["transSS_traj"]
    rotSS_traj = data_loaded["rotSS_traj"]
    NRSS_traj = data_loaded["NRSS_traj"]
    MSD_N_traj = data_loaded["MSD_N_traj"]
    MSD_S_traj = data_loaded["MSD_S_traj"]
    MSD_S2_traj = data_loaded["MSD_S2_traj"]
    MSD_NR_traj = data_loaded["MSD_NR_traj"]
    MSD_P_traj = data_loaded["MSD_P_traj"]
    MSD_R_traj = data_loaded["MSD_R_traj"]
    MSD_Y_traj = data_loaded["MSD_Y_traj"]
    
#%% print parameters
print('vol_exp_sec = %.4f, Dpar = %.2f, Dperp = %.2f, Droll = %.2f, Dyaw = %.2f, Dpitch = %.2f'
      %(vol_exp, Dpar, Dperp, Droll, Dyaw, Dpitch))

#%% Plot diffusion coefficients (translation)
plt.rcParams.update({'font.size': 15})

# diffusion constant from step size
fig,ax = plt.subplots(dpi=300, figsize=(14,5))
dperp_measure = sigma2_S/(2*vol_exp*interval)
dperp2_measure = sigma2_S2/(2*vol_exp*interval)
dpar_measure = sigma2_N/(2*vol_exp*interval)

ax.errorbar(interval, np.mean(dpar_measure, axis=0),
              yerr=np.std(dpar_measure,axis=0),
              color='k', marker="^",alpha=0.5,
              capsize=2, elinewidth = 0.5)
ax.errorbar(interval, np.mean(dperp_measure,axis=0),
              yerr=np.std(dperp_measure,axis=0),
              color='k', marker="s",alpha=0.5,
              capsize=2, elinewidth = 0.5, label='_nolegend_')
ax.errorbar(interval, np.mean(dperp2_measure, axis=0),
              yerr=np.std(dperp2_measure,axis=0),
              color='k', marker="o",alpha=0.5,
              capsize=2, elinewidth = 0.5, label='_nolegend_')
ax.plot(interval, np.ones(len(interval))*Dpar, 'r')
ax.plot(interval, np.ones(len(interval))*Dperp, 'r', label='_nolegend_')
ax.fill_between(interval, Dpar * (np.ones(len(interval))-errorTh),\
                Dpar * (np.ones(len(interval))+errorTh), facecolor='r',\
                alpha=0.1, label='_nolegend_')
ax.fill_between(interval, Dperp * (np.ones(len(interval))-errorTh),\
                Dperp * (np.ones(len(interval))+errorTh), facecolor='r',\
                alpha=0.1, label='_nolegend_')
# ax.legend(["Ground-Truth ($\parallel$ & $\perp$)","Step-size"])
ax.set_xlabel(r'Fitting every ith frame');
ax.set_ylabel(r'$D [\mu m^2/sec]$') 
ax.set_title('Diffusion coefficients from step size for ' +
             '$\Delta t$ = {}ms ({} trajectories)'
                 .format(vol_exp * 1e3, len(sigma2_N)))
ymax = np.max([np.max(np.mean(dpar_measure, axis=0)),
               np.max(np.mean(dperp_measure, axis=0)),
               np.max(np.mean(dperp2_measure, axis=0)),
               np.max(Dpar * (np.ones(len(interval))+errorTh))]) * 1.1
ymin = np.min([np.min(np.mean(dpar_measure, axis=0)),
               np.min(np.mean(dperp_measure, axis=0)),
               np.min(np.mean(dperp2_measure, axis=0)),
               np.min(Dperp * (np.ones(len(interval))-errorTh))]) * 0.9 
ax.set_ylim([ymin, ymax])
ax.set_xscale('log')


# diffusion constant from MSD[0]
fig,ax = plt.subplots(dpi=300, figsize=(14,5))
dpar_measure = sigma2_N_MSD/(2*vol_exp*interval)
dperp_measure = sigma2_S_MSD/(2*vol_exp*interval)
dperp2_measure = sigma2_S2_MSD/(2*vol_exp*interval)

ax.errorbar(interval, np.mean(dpar_measure, axis=0),
              yerr=np.std(dpar_measure,axis=0),
              color='C0', marker="^",alpha=0.5,
              capsize=2, elinewidth = 0.5)
ax.errorbar(interval, np.mean(dperp_measure,axis=0),
              yerr=np.std(dperp_measure,axis=0),
              color='C0', marker="s",alpha=0.5,
              capsize=2, elinewidth = 0.5, label='_nolegend_')
ax.errorbar(interval, np.mean(dperp2_measure, axis=0),
              yerr=np.std(dperp2_measure,axis=0),
              color='C0', marker="o",alpha=0.5,
              capsize=2, elinewidth = 0.5, label='_nolegend_')
ax.plot(interval, np.ones(len(interval))*Dpar, 'r')
ax.plot(interval, np.ones(len(interval))*Dperp, 'r', label='_nolegend_')
ax.fill_between(interval, Dpar * (np.ones(len(interval))-errorTh),\
                Dpar * (np.ones(len(interval))+errorTh), facecolor='r',\
                alpha=0.1, label='_nolegend_')
ax.fill_between(interval, Dperp * (np.ones(len(interval))-errorTh),\
                Dperp * (np.ones(len(interval))+errorTh), facecolor='r',\
                alpha=0.1, label='_nolegend_')
# ax.legend(["Ground-Truth ($\parallel$ & $\perp$)","MSD (1^{st} points)"])
ax.set_xlabel(r'Fitting every ith frame');
ax.set_ylabel(r'$D [\mu m^2/sec]$') 
ax.set_title('Diffusion coefficients from MSD[0] for ' +
             '$\Delta t$ = {}ms ({} trajectories)'
                 .format(vol_exp * 1e3, len(sigma2_N)))
ymax = np.max([np.max(np.mean(dpar_measure, axis=0)),
               np.max(np.mean(dperp_measure, axis=0)),
               np.max(np.mean(dperp2_measure, axis=0)),
               np.max(Dpar * (np.ones(len(interval))+errorTh))]) * 1.1
ymin = np.min([np.min(np.mean(dpar_measure, axis=0)),
               np.min(np.mean(dperp_measure, axis=0)),
               np.min(np.mean(dperp2_measure, axis=0)),
               np.min(Dperp * (np.ones(len(interval))-errorTh))]) * 0.9 
ax.set_ylim([ymin, ymax])
ax.set_xscale('log')

# diffusion constant from MSD (3 fitting points)
fig,ax = plt.subplots(dpi=300, figsize=(14,5))
dpar_measure = fitN_a/(2*vol_exp*interval)
dperp_measure = fitS_a/(2*vol_exp*interval)
dperp2_measure = fitS2_a/(2*vol_exp*interval)

ax.errorbar(interval, np.mean(dpar_measure, axis=0),
              yerr=np.std(dpar_measure,axis=0),
              color='k', marker="^",alpha=0.5,
              capsize=2, elinewidth = 0.5)
ax.errorbar(interval, np.mean(dperp_measure,axis=0),
              yerr=np.std(dperp_measure,axis=0),
              color='k', marker="s",alpha=0.5,
              capsize=2, elinewidth = 0.5, label='_nolegend_')
ax.errorbar(interval, np.mean(dperp2_measure, axis=0),
              yerr=np.std(dperp2_measure,axis=0),
              color='k', marker="o",alpha=0.5,
              capsize=2, elinewidth = 0.5, label='_nolegend_')
ax.plot(interval, np.ones(len(interval))*Dpar, 'r')
ax.plot(interval, np.ones(len(interval))*Dperp, 'r', label='_nolegend_')
ax.fill_between(interval, Dpar * (np.ones(len(interval))-errorTh),\
                Dpar * (np.ones(len(interval))+errorTh), facecolor='r',\
                alpha=0.1, label='_nolegend_')
ax.fill_between(interval, Dperp * (np.ones(len(interval))-errorTh),\
                Dperp * (np.ones(len(interval))+errorTh), facecolor='r',\
                alpha=0.1, label='_nolegend_')
# ax.legend(["Ground-Truth ($\parallel$ & $\perp$)","MSD (2 fitting points)"])
ax.set_xlabel(r'Fitting every ith frame');
ax.set_ylabel(r'$D [\mu m^2/sec]$') 
ax.set_title('Diffusion coefficients from MSD (2 fitting points) for ' +
             '$\Delta t$ = {}ms ({} trajectories)'
                 .format(vol_exp * 1e3, len(sigma2_N)))
ymax = np.max([np.max(np.mean(dpar_measure, axis=0)),
               np.max(np.mean(dperp_measure, axis=0)),
               np.max(np.mean(dperp2_measure, axis=0)),
               np.max(Dpar * (np.ones(len(interval))+errorTh))]) * 1.1
ymin = np.min([np.min(np.mean(dpar_measure, axis=0)),
               np.min(np.mean(dperp_measure, axis=0)),
               np.min(np.mean(dperp2_measure, axis=0)),
               np.min(Dperp * (np.ones(len(interval))-errorTh))]) * 0.9 
ax.set_ylim([0.05, ymax])
ax.set_xscale('log')
ax.figure.savefig(result_dir + '/syn-D-trans.pdf')

#%% Plot diffusion coefficients (ROLL)
plt.rcParams.update({'font.size': 15})

# diffusion constant from step size
fig,ax = plt.subplots(dpi=300, figsize=(14,5))
droll_measure = sigma2_R/(2*vol_exp*interval)

ax.errorbar(interval, np.mean(droll_measure,axis=0),
              yerr=np.std(droll_measure,axis=0),
              color='k', marker="s",alpha=0.5,
              capsize=2, elinewidth = 0.5)
ax.plot(interval, np.ones(len(interval))*Droll, 'r')
ax.fill_between(interval, Droll * (np.ones(len(interval))-errorTh),\
                Droll * (np.ones(len(interval))+errorTh),
                facecolor='r',\
                alpha=0.1, label='_nolegend_')
ax.legend(["Ground-Truth (Roll)","Step-size"])
ax.set_xlabel(r'Fitting every ith frame');
ax.set_ylabel(r'$D [rad^2/sec]$') 
ax.set_title('Diffusion coefficients from step size for ' +
             '$\Delta t$ = {}ms ({} trajectories)'
                 .format(vol_exp * 1e3, len(sigma2_N)))
ymax = np.max(Droll * (np.ones(len(interval))+errorTh)) * 1.1
ymin = np.min(Droll * (np.ones(len(interval))-errorTh)) * 0.9 
ax.set_ylim([ymin, ymax])
ax.set_xscale('log')

# diffusion constant from MSD (3 fitting points)
fig,ax = plt.subplots(dpi=300, figsize=(14,5))
droll_measure = fitR_a/(2*vol_exp*interval)

ax.errorbar(interval, np.mean(droll_measure,axis=0),
              yerr=np.std(droll_measure,axis=0),
              color='k', marker="s",alpha=0.5,
              capsize=2, elinewidth = 0.5)
ax.plot(interval, np.ones(len(interval))*Droll, 'r')
ax.fill_between(interval, Droll * (np.ones(len(interval))-errorTh),\
                Droll * (np.ones(len(interval))+errorTh),
                facecolor='r',\
                alpha=0.1, label='_nolegend_')
ax.legend(["Ground-Truth (Roll)","MSD (3 fitting points)"])
ax.set_xlabel(r'Fitting every ith frame');
ax.set_ylabel(r'$D [rad^2/sec]$') 
ax.set_title('Diffusion coefficients from MSD (3 fitting points) for ' +
             '$\Delta t$ = {}ms ({} trajectories)'
                 .format(vol_exp * 1e3, len(sigma2_N)))
ymax = np.max(Droll * (np.ones(len(interval))+errorTh)) * 1.1
ymin = np.min(Droll * (np.ones(len(interval))-errorTh)) * 0.9 
ax.set_ylim([0.9, ymax])
ax.set_xscale('log')
ax.figure.savefig(result_dir + '/syn-D-rot-longitudinal.pdf')

#%% Plot diffusion coefficients (Pitch & Yaw)
plt.rcParams.update({'font.size': 15})

# diffusion constant from step size
fig,ax = plt.subplots(dpi=300, figsize=(14,5))
dpitch_measure = sigma2_P/(2*vol_exp*interval)
dyaw_measure = sigma2_Y/(2*vol_exp*interval)

ax.errorbar(interval, np.mean(dpitch_measure, axis=0),
              yerr=np.std(dpitch_measure,axis=0),
              color='k', marker="^",alpha=0.5,
              capsize=2, elinewidth = 0.5)
ax.errorbar(interval, np.mean(dyaw_measure,axis=0),
              yerr=np.std(dyaw_measure,axis=0),
              color='k', marker="s",alpha=0.5,
              capsize=2, elinewidth = 0.5, label='_nolegend_')
ax.plot(interval, np.ones(len(interval))*Dpitch, 'r')
ax.plot(interval, np.ones(len(interval))*Dyaw, 'r', label='_nolegend_')
ax.fill_between(interval, Dpitch * (np.ones(len(interval))-errorTh),\
                Dpitch * (np.ones(len(interval))+errorTh), facecolor='r',\
                alpha=0.1, label='_nolegend_')
ax.fill_between(interval, Dyaw * (np.ones(len(interval))-errorTh),\
                Dyaw * (np.ones(len(interval))+errorTh), facecolor='r',\
                alpha=0.1, label='_nolegend_')
ax.legend(["Ground-Truth (pitch & yaw)","Step-size"])
ax.set_xlabel(r'Fitting every ith frame');
ax.set_ylabel(r'$D [rad^2/sec]$') 
ax.set_title('Diffusion coefficients from step size for ' +
             '$\Delta t$ = {}ms ({} trajectories)'
                 .format(vol_exp * 1e3, len(sigma2_N)))
ymax = np.max(Dpitch * (np.ones(len(interval))+errorTh)) * 1.1
ymin = np.min(Dyaw * (np.ones(len(interval))-errorTh)) * 0.9 
ax.set_ylim([ymin, ymax])
ax.set_xscale('log')

# diffusion constant from MSD[0]
fig,ax = plt.subplots(dpi=300, figsize=(14,5))
dpitch_measure = fitP_a/(2*vol_exp*interval)
dyaw_measure = fitY_a/(2*vol_exp*interval)

ax.errorbar(interval, np.mean(dpitch_measure, axis=0),
              yerr=np.std(dpitch_measure,axis=0),
              color='k', marker="^",alpha=0.5,
              capsize=2, elinewidth = 0.5)
ax.errorbar(interval, np.mean(dyaw_measure,axis=0),
              yerr=np.std(dyaw_measure,axis=0),
              color='k', marker="s",alpha=0.5,
              capsize=2, elinewidth = 0.5, label='_nolegend_')
ax.plot(interval, np.ones(len(interval))*Dpitch, 'r')
ax.plot(interval, np.ones(len(interval))*Dyaw, 'r', label='_nolegend_')
ax.fill_between(interval, Dpitch * (np.ones(len(interval))-errorTh),\
                Dpitch * (np.ones(len(interval))+errorTh), facecolor='r',\
                alpha=0.1, label='_nolegend_')
ax.fill_between(interval, Dyaw * (np.ones(len(interval))-errorTh),\
                Dyaw * (np.ones(len(interval))+errorTh), facecolor='r',\
                alpha=0.1, label='_nolegend_')
ax.legend(["Ground-Truth (pitch & yaw)","MSD (2 fitting points)"])
ax.set_xlabel(r'Fitting every ith frame');
ax.set_ylabel(r'$D [rad^2/sec]$')  
ax.set_title('Diffusion coefficients from MSD (2 fitting points) for ' +
             '$\Delta t$ = {}ms ({} trajectories)'
                 .format(vol_exp * 1e3, len(sigma2_N)))
ymax = np.max(Dpitch * (np.ones(len(interval))+errorTh)) * 1.1
ymin = np.min(Dyaw * (np.ones(len(interval))-errorTh)) * 0.9 
ax.set_ylim([ymin, ymax])
ax.set_xscale('log')
ax.figure.savefig(result_dir + '/syn-D-rot-transversal.pdf')

#%% Plot fluctuation
singleTraj = False;
if singleTraj: 
    
    # Choose which trajectory and number of separation
    nT = 1; nSep = 2; 
    x_axis = np.linspace(0,len(sigma2_N[nT,nSep][:,0])-1,\
                         num=len(sigma2_N[nT,nSep][:,0]))*vol_exp
    
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
    
    
    # Combo (parallel x roll) - CDF & PDF
    xplot = np.linspace(-2.5,2.5,1000, endpoint=False)
    
    plt.rcParams.update({'font.size': 15})
    fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))
    ax1.plot(np.sort(NRSS_traj[nT,nSep]),
              np.linspace(0,1,len(transSS_traj[nT,nSep][:,0]),
                          endpoint=False),'k',ms=3, alpha=0.5)
    ax1.set_xlabel(r'Step size [$\mu$m$\times$rad]');
    ax1.set_ylabel(r'Cumulative Probability')
    ax1.set_ylim([-0.05, 1.1]); ax1.set_xlim([-2.5, 2.5]); 

    sx2 = sigma2_N[nT,nSep]; sy2 = sigma2_R[nT,nSep];
    zs = np.linspace(0, 2.5, 300)
    def fn(x, z):
        return np.exp(-(x-0)**2 / 2/sx2) *\
               np.exp(-(z-0)**2 / 2/sy2 / x**2)
    
    rs = np.zeros(len(zs))
    for ii in range(len(zs)):
        rs[ii], unc = integrate.quad(lambda x: fn(x, zs[ii]), -np.inf, np.inf)
    zs = np.concatenate((-np.flip(zs[1:]), zs))
    rs = np.concatenate((np.flip(rs[1:]), rs))
    rs /= np.trapz(rs, zs) # normalize
    
    # figh = plt.figure()
    # plt.title("PDF of product of uncorrelated gaussians
    # with $\sigma_x$=%0.2f, $\sigma_y$=%0.2f" % (sx, sy))
    # plt.plot(zs, rs)

    plt.rcParams.update({'font.size': 15})
    fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))
    ax1.plot(zs, rs)
    ax1.hist(NRSS_traj[nT,nSep], bins='fd',
             density=True, color='k', alpha=0.3)
    ax1.set_xlabel(r'Step size [$\mu$m$\times$rad]');
    ax1.set_xlim([-2.5, 2.5]);
    ax1.set_ylabel(r'Probability density')
    
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