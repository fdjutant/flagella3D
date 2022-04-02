#%% Import all necessary libraries
import sys
sys.path.insert(0, '../modules')
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})
from sklearn.decomposition import PCA
import napari
import msd
from skimage import measure
from naparimovie import Movie
from pathlib import Path
import scipy.signal
from scipy import optimize, stats
import cv2 
import os.path
from os.path import dirname as up
import pickle
import time

# time settings in the light sheet
pxum = 0.115
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
exp3D_ms = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 
pxum = 0.115

# load raw images
# search_dir = Path(r"\\10.206.26.21\opm2\franky-sample-images")

setName = 'suc-40'

this_file_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("./"))),
                            'Dropbox (ASU)','Research')
thresholdFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Flagella-data', 'threshold-iterative')

thresholdFiles = list(Path(thresholdFolder).glob("*-threshold.npy"))

for whichFiles in range(len(thresholdFiles)):
# for whichFiles in range(0,5):
    
    # load threshold
    imgs_thresh = np.load(thresholdFiles[whichFiles])
    print(thresholdFiles[whichFiles])
    
    # Compute CM then generate n1, n2, n3
    blobBin = []
    xb = []
    xp = []
    nt = len(imgs_thresh)
    cm = np.zeros((nt,3))
    n1s = np.zeros((nt, 3))
    n2s = np.zeros((nt, 3))
    n3s = np.zeros((nt, 3))
    r_coms = np.zeros((nt, 3))
    flagella_len = np.zeros(nt)
    radial_dist_pt = np.zeros(nt)
    blob_size = np.zeros(nt)
    
    tstart = time.perf_counter()
    
    for frame in range(nt):
        
        # compute threshold pixel number
        blob = imgs_thresh[frame]
        blob_size[frame] = np.sum(blob == True)
        
        # ######################################
        # extract coordinates and center of mass
        # ######################################
        
        # extract coordinates
        X0 = np.argwhere(blob).astype('float') # coordinates 
        xb.append(X0) # store coordinates
        
        # compute center of mass
        CM1 = np.array([sum(X0[:,j]) for j in range(X0.shape[1])])/X0.shape[0]
        cm[frame,:] = CM1 # store center of mass
        
        # ##############################################################
        # determine axis n1 from PCA and consistency with previous point
        # ##############################################################
        coords = X0 - CM1 # shift all the coordinates into origin
        pca = PCA(n_components=3)
        pca.fit(coords)
        n1s[frame] = pca.components_[0]
        n2s[frame] = pca.components_[1]
        n3s[frame] = pca.components_[2]
    
        # choose the sign of current n1 so it is as close as possible to n1 at the previous timestep
        if frame > 0 and np.linalg.norm(n1s[frame] - n1s[frame -1]) > np.linalg.norm(n1s[frame] + n1s[frame - 1]):
            n1s[frame] = -n1s[frame]
            n2s[frame] = -n2s[frame]
            n3s[frame] = -n3s[frame]
            
        # #####################################
        # rotate flagella on the principal axes
        # #####################################
        dist_projected_along_n1 = n1s[frame, 0] * coords[:, 0] +\
                                  n1s[frame, 1] * coords[:, 1] +\
                                  n1s[frame, 2] * coords[:, 2]
        dist_projected_along_n2 = n2s[frame, 0] * coords[:, 0] +\
                                  n2s[frame, 1] * coords[:, 1] +\
                                  n2s[frame, 2] * coords[:, 2]
        dist_projected_along_n3 = n3s[frame, 0] * coords[:, 0] +\
                                  n3s[frame, 1] * coords[:, 1] +\
                                  n3s[frame, 2] * coords[:, 2]
        coord_on_principal = np.stack([dist_projected_along_n1,
                                       dist_projected_along_n2,
                                       dist_projected_along_n3],axis=1)
        xp.append(coord_on_principal)
    
        # ##########################################
        # determine the flagella length along the n1
        # ##########################################
        flagella_len[frame] = np.max(dist_projected_along_n1) - np.min(dist_projected_along_n1)
    
        # ##########################################
        # find the furthest point along the flagella
        # and the positive n1 direction
        # and use this to determine n2
        # ##########################################
        ind_pt = np.argmax(dist_projected_along_n1)
        coord_pt = coords[ind_pt]
    
        # project out n1
        coord_pt_proj = coord_pt - (coord_pt.dot(n1s[frame])) * n1s[frame]
    
        # check the radial distance of this point from the center
        radial_dist_pt[frame] = np.linalg.norm(coord_pt_proj)
    
        # generate n2 from this
        n2s[frame] = coord_pt_proj / np.linalg.norm(coord_pt_proj)
    
        assert n1s[frame].dot(n2s[frame]) < 1e-12
    
        # generate n3 such that coordinate system is right-handed
        n3s[frame] = np.cross(n1s[frame], n2s[frame])
    
    #%% Compute translation displacements and angles
    nt = len(n1s)
    dpitch = np.zeros(nt)
    droll = np.zeros(nt)
    dyaw = np.zeros(nt)
    for frame in range(nt-1):
        dpitch[frame] = np.dot(n2s[frame], n1s[frame+1] - n1s[frame])
        droll[frame] = np.dot(n3s[frame], n2s[frame+1] - n2s[frame])
        dyaw[frame] = np.dot(n1s[frame], n3s[frame+1] - n3s[frame])
    
    EuAng = np.zeros([nt,3])
    for frame in range(nt):
        EuAng[frame,0] = np.sum(dpitch[0:frame+1])
        EuAng[frame,1] = np.sum(droll[0:frame+1])
        EuAng[frame,2] = np.sum(dyaw[0:frame+1])
        
    disp_pitch = np.diff(EuAng[:,0])
    disp_roll = np.diff(EuAng[:,1])
    disp_yaw = np.diff(EuAng[:,2])
    
    disp_Ang = np.stack([disp_pitch,disp_roll,disp_yaw],axis=1)
    firstone = np.array([[0,0,0]])
    disp_Ang = np.vstack([firstone, disp_Ang])
    
    # compute translation displacement
    disp_n1 = []; disp_n2 = []; disp_n3 =[];
    for i in range(nt-1):
        
        # displacement in Cartesian coordinates
        deltaX = ( cm[i+1,0] - cm[i,0] ) * 0.115
        deltaY = ( cm[i+1,1] - cm[i,1] ) * 0.115
        deltaZ = ( cm[i+1,2] - cm[i,2] ) * 0.115
        deltaXYZ = np.array([deltaX, deltaY, deltaZ])
        
        # displcament in local axes
        disp_n1.append(n1s[i,0]*deltaXYZ[0] + 
                       n1s[i,1]*deltaXYZ[1] +
                       n1s[i,2]*deltaXYZ[2]) # parallel
        disp_n2.append(n2s[i,0]*deltaXYZ[0] + 
                       n2s[i,1]*deltaXYZ[1] +
                       n2s[i,2]*deltaXYZ[2]) # perp1
        disp_n3.append(n3s[i,0]*deltaXYZ[0] + 
                       n3s[i,1]*deltaXYZ[1] +
                       n3s[i,2]*deltaXYZ[2]) # perp2
    disp_n1 = np.array(disp_n1)
    disp_n2 = np.array(disp_n2)
    disp_n3 = np.array(disp_n3)
    
    disp = np.stack([disp_n1,disp_n2,disp_n3],axis=1)
    firstone = np.array([[0,0,0]])
    disp= np.vstack([firstone,disp])
    
    #%% Compute MSD and curve fit
    # initialize msd
    msd_n1 = []; msd_n2 = []; msd_n3 = []; co_msd = []
    msad_roll = []; msad_pitch = []; msad_yaw = []; msd_CM = []
    nInterval = 50

    # center-of-mass tracking
    nt = len(cm)
    dstCM = np.zeros(nt)
    for i in range(len(cm)): dstCM[i] = np.linalg.norm(cm[i])

    # MSD: mean square displacement
    MSD_n1, MSD_n2, MSD_n3, CO_MSD = msd.trans_MSD_Namba(nt,
                                              cm, EuAng[:,1],
                                              n1s, n2s, n3s,
                                              exp3D_ms, nInterval)
    MSAD_P = msd.regMSD_Namba(nt, EuAng[:,0], exp3D_ms, nInterval)
    MSAD_R = msd.regMSD_Namba(nt, EuAng[:,1], exp3D_ms, nInterval)
    MSAD_Y = msd.regMSD_Namba(nt, EuAng[:,2], exp3D_ms, nInterval)
    MSD_CM = msd.regMSD_Namba(nt, dstCM, exp3D_ms, nInterval)

    # Fit MSD with y = Const + B*x for N, S, NR, PY, R
    Nfit = 5
    xtime = np.linspace(1,Nfit,Nfit)
    def MSDfit(x, a, b): return b + a * x  
    
    # fit MSD and MSAD
    fit_n1, fit_n1_const  = optimize.curve_fit(MSDfit, xtime, MSD_n1[0:Nfit])[0]
    fit_n2n3, fit_n2n3_const  = optimize.curve_fit(MSDfit, xtime,
                            np.mean([MSD_n2[0:Nfit],MSD_n3[0:Nfit]],axis=0))[0]
    fit_CO, fit_CO_const = optimize.curve_fit(MSDfit, xtime, CO_MSD[0:Nfit])[0]
    fit_PY, fit_PY_const = optimize.curve_fit(MSDfit, xtime,
                              np.mean([MSAD_P[0:Nfit],MSAD_Y[0:Nfit]],axis=0))[0]
    fit_R,fit_R_const   = optimize.curve_fit(MSDfit, xtime, MSAD_R[0:Nfit])[0]
    fit_CM,fit_CM_const = optimize.curve_fit(MSDfit, xtime, MSD_CM[0:Nfit])[0]

    # Additional fit
    fit_n2,fit_n2_const  = optimize.curve_fit(MSDfit, xtime, MSD_n2[0:Nfit])[0]
    fit_n3,fit_n3_const  = optimize.curve_fit(MSDfit, xtime, MSD_n3[0:Nfit])[0]
    fit_P, fit_P_const  = optimize.curve_fit(MSDfit, xtime, MSAD_P[0:Nfit])[0]
    fit_Y, fit_Y_const  = optimize.curve_fit(MSDfit, xtime, MSAD_Y[0:Nfit])[0]

    #%% Plot length (and) radius (and) threshold pixel total vs time 
    fig = plt.figure(dpi=150, figsize = (40, 35))
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.rcParams.update({'font.size': 30})
    fig.suptitle('data: %s (' %os.path.basename(thresholdFiles[whichFiles]) +
                  'Nt = %d' %nt + ', '
                  'L = %.3f $\pm$ %.3f $\mu$m'
                  %(np.mean(flagella_len)*pxum,
                    np.std(flagella_len)*pxum) + ', '
                  'R = %.3f $\pm$ %.3f $\mu$m)'
                  %(np.mean(radial_dist_pt)*pxum,
                    np.std(radial_dist_pt)*pxum) )
    ax0 = fig.add_subplot(431)
    ax1 = fig.add_subplot(432)
    ax2 = fig.add_subplot(433)
    ax3 = fig.add_subplot(434)
    ax4 = fig.add_subplot(435)
    ax5 = fig.add_subplot(436)
    ax6 = fig.add_subplot(437)
    ax7 = fig.add_subplot(438)
    ax8 = fig.add_subplot(439)
    ax9 = fig.add_subplot(4,3,10)
    ax10 = fig.add_subplot(4,3,11)
    ax11 = fig.add_subplot(4,3,12)
    pxum = 0.115
    
    ax0.plot(np.arange(0,nt),flagella_len*pxum,'k')
    ax0.set_xlabel(r'frame-num')
    ax0.set_ylabel(r'length [$\mu m$]')
    
    ax1.plot(np.arange(0,nt),radial_dist_pt*pxum,'k')
    ax1.set_xlabel(r'frame-num')
    ax1.set_ylabel(r'radius [$\mu m$]')
    
    ax2.plot(np.arange(0,nt),blob_size,'k')
    ax2.set_xlabel(r'frame-num')
    ax2.set_ylabel(r'pixel total number')
    
    ax3.plot(np.arange(1,nt),disp_n1*pxum,'C0')
    ax3.plot(np.arange(1,nt),disp_n2*pxum,'C1')
    ax3.plot(np.arange(1,nt),disp_n3*pxum,'C2')
    ax3.set_xlabel(r'frame-num')
    ax3.set_ylabel(r'displacement along local axes [$\mu m$]')
    ax3.legend(['$n_1$','$n_2$','$n_3$'], loc="upper right", ncol = 3)
    
    ax4.plot(np.arange(1,nt),disp_roll,'C0')
    ax4.set_xlabel(r'frame-num')
    ax4.set_ylabel(r'$\Delta\psi$ [rad]')
    ax4.set_ylim(min(disp_roll)-0.1,max(disp_roll)+0.1)
    
    ax5.plot(np.arange(1,nt),disp_pitch,'C1')
    ax5.plot(np.arange(1,nt),disp_yaw,'C2')
    ax5.set_xlabel(r'frame-num')
    ax5.set_ylabel(r'$\Delta\beta$ or $\Delta\gamma$ [rad]')
    ax5.set_ylim(min(disp_roll)-0.1,max(disp_roll)+0.1)
    ax5.legend(['pitch','yaw'],loc="upper right", ncol = 2)
    
    # translation
    xaxis = np.arange(1,nInterval+1)
    ax6.plot(xaxis*exp3D_ms, MSD_n1,c='C0',marker="^",mfc='none',
              ms=8,ls='None',alpha=1)   
    ax6.plot(xaxis*exp3D_ms,np.mean([MSD_n2,MSD_n3],axis=0),
              c='k',marker="^",mfc='none',
              ms=8,ls='None',alpha=1)
    ax6.plot(xaxis*exp3D_ms, MSD_n2,c='C1',marker="s",mfc='none',
              ms=8,ls='None',alpha=.3)
    ax6.plot(xaxis*exp3D_ms, MSD_n3,c='C2',marker="s",mfc='none',
              ms=8,ls='None',alpha=.3)
    ax6.plot(xaxis*exp3D_ms, fit_n1_const + fit_n1*xaxis,
             c='C0',alpha=1,label='_nolegend_')
    ax6.plot(xaxis*exp3D_ms, fit_n2n3_const + fit_n2n3*xaxis,
             c='k',alpha=1,label='_nolegend_')
    ax6.plot(xaxis*exp3D_ms, fit_n2_const + fit_n2*xaxis,
             c='C1',alpha=.3,label='_nolegend_')
    ax6.plot(xaxis*exp3D_ms, fit_n3_const + fit_n3*xaxis,
             c='C2',alpha=.3,label='_nolegend_')
    ax6.set_xlabel(r'Lag time [sec]');
    ax6.set_ylabel(r'MSD [$\mu m^2$]')
    # ax0.set_ylim([0, 2]);
    ax6.set_xlim([0, nInterval*exp3D_ms])
    ax6.legend(['$D_\parallel=$ %.3f $\mu$m$^2$/sec' %np.round(fit_n1/(2*exp3D_ms),3),
                # '$D_{\perp}=$ %.3f $\mu$m/sec$^2$' %np.round(fit_n3/(2*exp3D_ms),3) ])
                "$D_{\perp}=$ %.3f, %.3f $\mu$m$^2$/sec" 
                %(np.round(fit_n2/(2*exp3D_ms),3),
                  np.round(fit_n3/(2*exp3D_ms),3)) ])
    
    # roll
    xaxis = np.arange(1,nInterval+1)
    ax7.plot(xaxis*exp3D_ms, MSAD_R,c='C0',marker="s",mfc='none',
              ms=8,ls='None',alpha=1)
    ax7.plot(xaxis*exp3D_ms, fit_R_const + fit_R*xaxis,
             c='C0',alpha=1,label='_nolegend_')
    ax7.set_xlabel(r'Lag time [sec]');
    ax7.set_ylabel(r'MSAD [rad$^2$]')
    # ax0.set_ylim([0, 2]);
    ax7.set_xlim([0, nInterval*exp3D_ms])
    ax7.legend(['$D_{R}=$ %.3f rad$^2$/sec' %np.round(fit_R/(2*exp3D_ms),3)])
    
    # pitch and yaw
    xaxis = np.arange(1,nInterval+1)
    ax8.plot(xaxis*exp3D_ms, MSAD_P,c='C1',marker="s",mfc='none',
              ms=8,ls='None',alpha=.3)
    ax8.plot(xaxis*exp3D_ms, MSAD_Y,c='C2',marker="s",mfc='none',
              ms=8,ls='None',alpha=.3)
    ax8.plot(xaxis*exp3D_ms,np.mean([MSAD_P,MSAD_Y],axis=0),
              c='k',marker="^",mfc='none',
              ms=8,ls='None',alpha=1)
    ax8.plot(xaxis*exp3D_ms, fit_PY_const + fit_PY*xaxis,
             c='k',alpha=1,label='_nolegend_')
    ax8.plot(xaxis*exp3D_ms, fit_P_const + fit_P*xaxis,
             c='C1',alpha=.3,label='_nolegend_')
    ax8.plot(xaxis*exp3D_ms, fit_Y_const + fit_Y*xaxis,
             c='C2',alpha=.3,label='_nolegend_')
    ax8.set_xlabel(r'Lag time [sec]');
    ax8.set_ylabel(r'MSAD [rad$^2$]')
    # ax0.set_ylim([0, 2]);
    ax8.set_xlim([0, nInterval*exp3D_ms])
    ax8.legend(['$D_{P}=$ %.3f rad$^2$/sec' %np.round(fit_P/(2*exp3D_ms),3),
                "$D_{Y}=$ %.3f rad$^2$/sec" %np.round(fit_Y/(2*exp3D_ms),3)])
    
    # co-diffusion: correlation
    xaxis = np.arange(1,nInterval+1)
    ax9.plot(xaxis*exp3D_ms, CO_MSD,c='k',marker="s",mfc='none',
              ms=8,ls='None',alpha=1)
    ax9.plot(xaxis*exp3D_ms, fit_CO_const + fit_CO*xaxis,
             c='k',alpha=1,label='_nolegend_')
    ax9.set_xlabel(r'Lag time [sec]');
    ax9.set_ylabel(r'CO-MSD [$\mu$m x rad]')
    ax9.set_xlim([0, nInterval*exp3D_ms])
    ax9.legend(['$D_{\parallel R}=$ %.3f $\mu$m x rad/sec'
                %np.round(fit_CO/(2*exp3D_ms),3)])
    
    # translation vs rotation (longitudinal axis)
    lsq_res = stats.linregress(disp_n1, disp_roll)
    ax10.plot(disp_n1, disp_roll, 'ko', ms=15, mew=1.5, mfc='none',
              ls='None',alpha=0.5)
    ax10.plot(disp_n1, lsq_res[1] + lsq_res[0] * disp_n1, 'k')
    ax10.set_xlabel(r'$\Delta_\parallel$ [$\mu m$]')
    ax10.set_ylabel(r'$\Delta\psi$ [rad]')
    ax10.grid(True, which='both')
    ax10.legend(['$\Delta_\psi / \Delta_\parallel=$ %.3f rad/$\mu$m'
                %np.round(lsq_res[0],3)])

    # CM: x, y, z 
    ax11.plot(np.arange(0,nt),cm[:,0]-cm[0,0],'r')
    ax11.plot(np.arange(0,nt),cm[:,1]-cm[0,1],'g')
    ax11.plot(np.arange(0,nt),cm[:,2]-cm[0,2],'b')
    ax11.set_xlabel(r'frame-num')
    ax11.set_ylabel(r'CM [$\mu m$]')
    ax11.legend(['x','y','z'],loc="upper right", ncol = 3)

    # save to single-folder    
    summaryFolder = os.path.join(up(up(up(thresholdFiles[whichFiles]))),
                                 'Flagella-data','summary')
    if os.path.isdir(summaryFolder) != True:
        os.mkdir(summaryFolder) # create path if non-existent
    ax11.figure.savefig(os.path.join(summaryFolder,
                       os.path.basename(thresholdFiles[whichFiles])[:-14] +
                       '-summary.png' ))
    plt.close(fig)

    #%% Store tracking information to PKL
    pklFolder = os.path.join(up(up(up(thresholdFiles[whichFiles]))),
                             'Flagella-data','summary-PKL-5fitPoints')
    savingPKL = os.path.join(pklFolder,
                       os.path.basename(thresholdFiles[whichFiles])[:-14] +
                       '-summary.pkl')
    data = {"flagella_length": flagella_len,
            "flagella_radius": radial_dist_pt,
            "exp3D_ms": exp3D_ms,
            "pxum": pxum,
            "cm": cm,
            "disp": disp,
            "disp_Ang": disp_Ang,
            "MSD": np.stack([MSD_n1, MSD_n2, MSD_n3],axis=1),
            "MSAD": np.stack([MSAD_R, MSAD_P, MSAD_Y],axis=1),
            "CO_MSD": CO_MSD,
            "D_trans": np.stack([fit_n1, fit_n2, fit_n3]) / (2*exp3D_ms),
            "D_rot": np.stack([fit_R, fit_P, fit_Y]) / (2*exp3D_ms)}
    with open(savingPKL, "wb") as f:
          pickle.dump(data, f)

    print('one dataset takes [sec]: %.2f' %(time.perf_counter()-tstart))