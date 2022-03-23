#%% Import all necessary libraries
import sys
sys.path.insert(0, './modules')
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})
from pathlib import Path
from scipy import optimize, stats
import os.path
from os.path import dirname as up
import pickle
from matplotlib.transforms import Affine2D
import seaborn as sns

# time settings in the light sheet
pxum = 0.115
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
exp3D_ms = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 
pxum = 0.115

#%% Loading pickle files
this_file_dir = os.path.join(os.path.dirname(os.path.abspath("./")),
                            'Dropbox (ASU)','Research')
pklFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Flagella-data', 'summary-PKL')
pdfFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Flagella-data', 'various-plots')
pklFiles = list(Path(pklFolder).glob("suc70*.pkl"))
pklFiles50 = list(Path(pklFolder).glob("suc50*.pkl"))
pklFiles40 = list(Path(pklFolder).glob("suc40*.pkl"))

# sucrose 70 (w/v)
D_trans = np.zeros([len(pklFiles),3])
D_rot = np.zeros([len(pklFiles),3])
flagella_length_mean = np.zeros(len(pklFiles))
flagella_length_std = np.zeros(len(pklFiles))
cm = []
disp = []
disp_Ang = []
MSD = []
MSAD = []
CO_MSD = []
for j in range(len(pklFiles)):
    with open(pklFiles[j], "rb") as f:
          data_loaded = pickle.load(f)
    exp3D_ms = data_loaded["exp3D_ms"]
    pxum = data_loaded["pxum"]
    flagella_length_mean[j] = np.mean(data_loaded["flagella_length"])*pxum
    flagella_length_std[j] = np.std(data_loaded["flagella_length"])*pxum
    cm.append(data_loaded["cm"])
    disp.append(data_loaded["disp"])
    disp_Ang.append(data_loaded["disp_Ang"])
    D_trans[j] = data_loaded["D_trans"]
    D_rot[j] = data_loaded["D_rot"]
    MSD.append(data_loaded["MSD"])
    MSAD.append(data_loaded["MSAD"])
    CO_MSD.append(data_loaded["CO_MSD"])
MSD = np.array(MSD, dtype=object)
MSAD = np.array(MSAD, dtype=object)
CO_MSD = np.array(CO_MSD,dtype=object)
cm = np.array(cm, dtype=object)
disp = np.array(disp, dtype=object)
disp_Ang = np.array(disp_Ang, dtype=object)
whichFiles = 0

# sucrose 50 (w/v)
D_trans_50 = np.zeros([len(pklFiles50),3])
D_rot_50 = np.zeros([len(pklFiles50),3])
flagella_length_mean_50 = np.zeros(len(pklFiles50))
flagella_length_std_50 = np.zeros(len(pklFiles50))
cm_50 = []
disp_50 = []
disp_Ang_50 = []
MSD_50 = []
MSAD_50 = []
CO_MSD_50 = []
for j in range(len(pklFiles50)):
    with open(pklFiles50[j], "rb") as f:
          data_loaded = pickle.load(f)
    exp3D_ms = data_loaded["exp3D_ms"]
    pxum = data_loaded["pxum"]
    flagella_length_mean_50[j] = np.mean(data_loaded["flagella_length"])*pxum
    flagella_length_std_50[j] = np.std(data_loaded["flagella_length"])*pxum
    cm_50.append(data_loaded["cm"])
    disp_50.append(data_loaded["disp"])
    disp_Ang_50.append(data_loaded["disp_Ang"])
    D_trans_50[j] = data_loaded["D_trans"]
    D_rot_50[j] = data_loaded["D_rot"]
    MSD_50.append(data_loaded["MSD"])
    MSAD_50.append(data_loaded["MSAD"])
    CO_MSD_50.append(data_loaded["CO_MSD"])
MSD_50 = np.array(MSD_50, dtype=object)
MSAD_50 = np.array(MSAD_50, dtype=object)
CO_MSD_50 = np.array(CO_MSD_50,dtype=object)
cm_50 = np.array(cm_50, dtype=object)
disp_50 = np.array(disp_50, dtype=object)
disp_Ang_50 = np.array(disp_Ang_50, dtype=object)

# sucrose 40 (w/v)
D_trans_40 = np.zeros([len(pklFiles40),3])
D_rot_40 = np.zeros([len(pklFiles40),3])
flagella_length_mean_40 = np.zeros(len(pklFiles40))
flagella_length_std_40 = np.zeros(len(pklFiles40))
cm_40 = []
disp_40 = []
disp_Ang_40 = []
MSD_40 = []
MSAD_40 = []
CO_MSD_40 = []
for j in range(len(pklFiles40)):
    with open(pklFiles40[j], "rb") as f:
          data_loaded = pickle.load(f)
    exp3D_ms = data_loaded["exp3D_ms"]
    pxum = data_loaded["pxum"]
    flagella_length_mean_40[j] = np.mean(data_loaded["flagella_length"])*pxum
    flagella_length_std_40[j] = np.std(data_loaded["flagella_length"])*pxum
    cm_40.append(data_loaded["cm"])
    disp_40.append(data_loaded["disp"])
    disp_Ang_40.append(data_loaded["disp_Ang"])
    D_trans_40[j] = data_loaded["D_trans"]
    D_rot_40[j] = data_loaded["D_rot"]
    MSD_40.append(data_loaded["MSD"])
    MSAD_40.append(data_loaded["MSAD"])
    CO_MSD_40.append(data_loaded["CO_MSD"])
MSD_40 = np.array(MSD_40, dtype=object)
MSAD_40 = np.array(MSAD_40, dtype=object)
CO_MSD_40 = np.array(CO_MSD_40,dtype=object)
cm_40 = np.array(cm_40, dtype=object)
disp_40 = np.array(disp_40, dtype=object)
disp_Ang_40 = np.array(disp_Ang_40, dtype=object)

#%% CM: x, y, z
cm_specific = cm[whichFiles]*pxum

fig0,ax0 = plt.subplots(dpi=300, figsize=(10,5))
plt.rcParams.update({'font.size': 18})
xtrack = np.arange(0,len(cm[whichFiles][:,0]))
ax0.plot(xtrack*exp3D_ms,cm_specific[:,0]-cm_specific[0,0],alpha=1,c='r')   
ax0.plot(xtrack*exp3D_ms,cm_specific[:,1]-cm_specific[0,1],alpha=1,c='g')   
ax0.plot(xtrack*exp3D_ms,cm_specific[:,2]-cm_specific[0,2],alpha=1,c='b')   
ax0.set_xlabel(r'Time [sec]')
ax0.set_ylabel(r'Displacement [$\mu$m]')
ax0.legend([r"$\Delta x$","$\Delta y$","$\Delta z$"], ncol=3)
ax0.set_ylim([-1.5, 1.5])
ax0.set_xlim([0, 20])
ax0.figure.savefig(pdfFolder + '/fig3-tracking-CM.pdf')  

#%% Translation displacement
disp_specific = disp[whichFiles]

fig0,ax0 = plt.subplots(dpi=300, figsize=(10,5))
plt.rcParams.update({'font.size': 18})
xtrack = np.arange(0,len(cm[whichFiles][:,0]))
ax0.plot(xtrack*exp3D_ms,disp_specific[:,0],alpha=.75,c='purple',lw=1.2)   
ax0.plot(xtrack*exp3D_ms,disp_specific[:,1],alpha=.75,c='C1',lw=1.2)   
ax0.plot(xtrack*exp3D_ms,disp_specific[:,2],alpha=.75,c='C2',lw=1.2)   
ax0.set_xlabel(r'Time [sec]')
ax0.set_ylabel(r'Displacement on local axes [$\mu$m]')
ax0.legend([r"$\Delta_{n_1}$","$\Delta_{n_2}$","$\Delta_{n_3}$"], ncol=3)
ax0.set_ylim([-0.6, 0.6])
ax0.set_yticks([-0.5,0,0.5])
ax0.set_xlim([0, 20])
ax0.set_xticks([0,5,10,15,20])
ax0.figure.savefig(pdfFolder + '/fig3-tracking-disp.pdf')  

trans0 = disp_specific[:,0][:-1] + disp_specific[:,0][1:]
trans1 = disp_specific[:,1][:-1] + disp_specific[:,1][1:]
trans2 = disp_specific[:,2][:-1] + disp_specific[:,2][1:]

fig0,ax0 = plt.subplots(dpi=300, figsize=(10,5))
plt.rcParams.update({'font.size': 18})
xtrack = np.arange(1,len(cm[whichFiles][:,0]))
ax0.plot(xtrack*exp3D_ms,trans0,alpha=.75,c='purple',lw=1.2)   
ax0.plot(xtrack*exp3D_ms,trans1,alpha=.75,c='C1',lw=1.2)   
ax0.plot(xtrack*exp3D_ms,trans2,alpha=.75,c='C2',lw=1.2)   
ax0.set_xlabel(r'Time [sec]')
ax0.set_ylabel(r'Translational position [$\mu$m]')
ax0.legend([r"$\Delta_{n_1}$","$\Delta_{n_2}$","$\Delta_{n_3}$"], ncol=3)
ax0.set_ylim([-0.6, 0.6])
ax0.set_yticks([-0.5,0,0.5])
ax0.set_xlim([0, 20])
ax0.set_xticks([0,5,10,15,20])

#%% Rotational displacement
whichFiles = 0
disp_Ang_specific = disp_Ang[whichFiles]

fig0,ax0 = plt.subplots(dpi=300, figsize=(10,5))
plt.rcParams.update({'font.size': 18})
xtrack = np.arange(0,len(cm[whichFiles][:,0]))
ax0.plot(xtrack*exp3D_ms,disp_Ang_specific[:,0],alpha=.75,c='C1',lw=1.2)   
ax0.plot(xtrack*exp3D_ms,disp_Ang_specific[:,1],alpha=.75,c='purple',lw=1.2)   
ax0.plot(xtrack*exp3D_ms,disp_Ang_specific[:,2],alpha=.75,c='C2',lw=1.2)   
ax0.set_xlabel(r'Time [sec]')
ax0.set_ylabel(r'Rotational displacement [rad]')
ax0.legend([r"$\Delta_{n_1}$","$\Delta_{n_2}$","$\Delta_{n_3}$"], ncol=3)
# ax0.set_ylim([-0.6, 0.6])
# ax0.set_yticks([-0.5,0,0.5])
ax0.set_xlim([0, 20])
ax0.set_xticks([0,5,10,15,20])
ax0.figure.savefig(pdfFolder + '/fig3-tracking-disp-ang.pdf')  

EuAng0 = disp_Ang_specific[:,0][:-1] + disp_Ang_specific[:,0][1:]
EuAng1 = disp_Ang_specific[:,1][:-1] + disp_Ang_specific[:,1][1:]
EuAng2 = disp_Ang_specific[:,2][:-1] + disp_Ang_specific[:,2][1:]

fig0,ax0 = plt.subplots(dpi=300, figsize=(10,5))
plt.rcParams.update({'font.size': 18})
xtrack = np.arange(1,len(cm[whichFiles][:,0]))
ax0.plot(xtrack*exp3D_ms,EuAng0,alpha=.75,c='C1',lw=1.2)   
ax0.plot(xtrack*exp3D_ms,EuAng1,alpha=.75,c='purple',lw=1.2)   
ax0.plot(xtrack*exp3D_ms,EuAng2,alpha=.75,c='C2',lw=1.2)   
ax0.set_xlabel(r'Time [sec]')
ax0.set_ylabel(r'Rotational position [rad]')
ax0.legend([r"$\Delta_{n_1}$","$\Delta_{n_2}$","$\Delta_{n_3}$"], ncol=3)
# ax0.set_ylim([-0.6, 0.6])
# ax0.set_yticks([-0.5,0,0.5])
ax0.set_xlim([0, 20])
ax0.set_xticks([0,5,10,15,20])

#%% MSD
MSD_specific = MSD[whichFiles]
MSD_n1 = MSD_specific[:,0]
MSD_n2 = MSD_specific[:,1]
MSD_n3 = MSD_specific[:,2]

nInterval = 50
xaxis = np.arange(1,nInterval+1)

Nfit = 5
xtime = np.linspace(1,Nfit,Nfit)
def MSDfit(x, a, b): return b + a * x  

# fit MSD
fit_n1, fit_n1_const  = optimize.curve_fit(MSDfit, xtime, MSD_n1[0:Nfit])[0]
fit_n2n3, fit_n2n3_const  = optimize.curve_fit(MSDfit, xtime,
                        np.mean([MSD_n2[0:Nfit],MSD_n3[0:Nfit]],axis=0))[0]
fit_n2,fit_n2_const  = optimize.curve_fit(MSDfit, xtime, MSD_n2[0:Nfit])[0]
fit_n3,fit_n3_const  = optimize.curve_fit(MSDfit, xtime, MSD_n3[0:Nfit])[0]

print('D_n1 = %.2f, D_n2 = %.2f, D_n3 = %.2f'
      %(fit_n1/(2*exp3D_ms), fit_n2/(2*exp3D_ms), fit_n3/(2*exp3D_ms)))

plt.rcParams.update({'font.size': 18})
fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
ax0.plot(xaxis*exp3D_ms, MSD_n1,
         c='purple',marker="s",mfc='none',
         ms=5,ls='None',alpha=1)   
ax0.plot(xaxis*exp3D_ms, MSD_n2,
         c='C1',marker="s",mfc='none',
         ms=5,ls='None',alpha=1)
ax0.plot(xaxis*exp3D_ms, MSD_n3,
         c='C2',marker="s",mfc='none',
         ms=5,ls='None',alpha=1)
ax0.plot(xaxis*exp3D_ms, fit_n1_const + fit_n1*xaxis,
         c='purple',alpha=1,label='_nolegend_')
ax0.plot(xaxis*exp3D_ms,fit_n2_const + fit_n2*xaxis,
         c='C1',alpha=1,label='_nolegend_')
ax0.plot(xaxis*exp3D_ms,fit_n3_const + fit_n3*xaxis,
         c='C2',alpha=1,label='_nolegend_')
ax0.set_xlabel(r'Lag time [sec]')
ax0.set_ylabel(r'MSD [$\mu m^2$]')
ax0.set_ylim([0, 0.5])
ax0.set_xlim([0, 3.2])
ax0.legend(['$n_1$', '$n_2$', '$n_3$'], ncol=3)
ax0.figure.savefig(pdfFolder + '/fig3-MSD.pdf')

#%% MSAD
MSAD_specific = MSAD[whichFiles]
MSAD_n1 = MSAD_specific[:,0]
MSAD_n2 = MSAD_specific[:,1]
MSAD_n3 = MSAD_specific[:,2]

nInterval = 50
xaxis = np.arange(1,nInterval+1)

Nfit = 5
xtime = np.linspace(1,Nfit,Nfit)
def MSDfit(x, a, b): return b + a * x  

# fit MSD
fit_R, fit_R_const = optimize.curve_fit(MSDfit, xtime, MSAD_n1[0:Nfit])[0]
fit_PY, fit_PY_const = optimize.curve_fit(MSDfit, xtime,
                        np.mean([MSAD_n2[0:Nfit],MSAD_n3[0:Nfit]],axis=0))[0]
fit_P, fit_P_const  = optimize.curve_fit(MSDfit, xtime, MSAD_n2[0:Nfit])[0]
fit_Y, fit_Y_const  = optimize.curve_fit(MSDfit, xtime, MSAD_n3[0:Nfit])[0]

print('D_R = %.2f, D_P = %.2f, D_Y = %.2f'
      %(fit_R/(2*exp3D_ms), fit_P/(2*exp3D_ms), fit_Y/(2*exp3D_ms)))

plt.rcParams.update({'font.size': 18})
fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
ax0.plot(xaxis*exp3D_ms, MSAD_n1,
         c='purple',marker="o",mfc='none',
         ms=5,ls='None',alpha=1)   
ax0.plot(xaxis*exp3D_ms, fit_R_const + fit_R*xaxis,
         c='purple',alpha=1,label='_nolegend_')
ax0.set_xlabel(r'Lag time [sec]')
ax0.set_ylabel(r'MSAD [rad$^2$]')
ax0.set_ylim([0, 3])
ax0.set_xlim([0, 3.2])
ax0.set_yticks([0,1,2,3])
ax0.legend(['$R$'])
ax0.figure.savefig(pdfFolder + '/fig3-MSAD-R.pdf')

fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
ax0.plot(xaxis*exp3D_ms, MSAD_n2,
         c='C1',marker="o",mfc='none',
         ms=5,ls='None',alpha=1)
ax0.plot(xaxis*exp3D_ms, MSAD_n3,
         c='C2',marker="o",mfc='none',
         ms=5,ls='None',alpha=1)
ax0.plot(xaxis*exp3D_ms,fit_P_const + fit_P*xaxis,
         c='C1',alpha=1,label='_nolegend_')
ax0.plot(xaxis*exp3D_ms,fit_Y_const + fit_Y*xaxis,
         c='C2',alpha=1,label='_nolegend_')
ax0.set_xlabel(r'Lag time [sec]')
ax0.set_ylabel(r'MSAD [rad$^2$]')
ax0.set_ylim([0, 0.08])
ax0.set_yticks([0,0.02,0.04,0.06,0.08])
ax0.set_xlim([0, 3.2])
ax0.legend(['$P$', '$Y$'], ncol=2)
ax0.figure.savefig(pdfFolder + '/fig3-MSAD-PY.pdf')

fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
ax0.plot(xaxis*exp3D_ms, MSAD_n1,
         c='purple',marker="o",mfc='none',
         ms=5,ls='None',alpha=1)   
ax0.plot(xaxis*exp3D_ms, MSAD_n2,
         c='C1',marker="o",mfc='none',
         ms=5,ls='None',alpha=1)
ax0.plot(xaxis*exp3D_ms, MSAD_n3,
         c='C2',marker="o",mfc='none',
         ms=5,ls='None',alpha=1)
ax0.plot(xaxis*exp3D_ms, fit_R_const + fit_R*xaxis,
         c='purple',alpha=1,label='_nolegend_')
ax0.plot(xaxis*exp3D_ms,fit_P_const + fit_P*xaxis,
         c='C1',alpha=1,label='_nolegend_')
ax0.plot(xaxis*exp3D_ms,fit_Y_const + fit_Y*xaxis,
         c='C2',alpha=1,label='_nolegend_')
ax0.set_xlabel(r'Lag time [sec]')
ax0.set_ylabel(r'MSAD [rad$^2$]')
ax0.set_ylim([-0.1, 2.5])
ax0.set_yticks([0,1,2])
ax0.set_xlim([0, 3.2])
ax0.legend(['R','P', 'Y'], ncol=3)
ax0.figure.savefig(pdfFolder + '/fig3-MSAD.pdf')

#%% Translation diffusion vs Length
length_diff = np.stack([flagella_length_mean, flagella_length_std, 
                        D_trans[:,0], D_trans[:,1], D_trans[:,2],
                        D_rot[:,0], D_rot[:,1], D_rot[:,2]]).T
length_diff_sorted = length_diff[length_diff[:,0].argsort()]

length_diff_50 = np.stack([flagella_length_mean_50, flagella_length_std_50, 
                        D_trans_50[:,0], D_trans_50[:,1], D_trans_50[:,2],
                        D_rot_50[:,0], D_rot_50[:,1], D_rot_50[:,2]]).T
length_diff_sorted_50 = length_diff_50[length_diff_50[:,0].argsort()]

length_diff_40 = np.stack([flagella_length_mean_40, flagella_length_std_40, 
                        D_trans_40[:,0], D_trans_40[:,1], D_trans_40[:,2],
                        D_rot_40[:,0], D_rot_40[:,1], D_rot_40[:,2]]).T
length_diff_sorted_40 = length_diff_40[length_diff_40[:,0].argsort()]

# theory
kB = 1.380649e-23  # J / K
T = 273 + 20       # K
diam = 6e-7        # m (flagella helical diameter ~0.6 um)
length_flagellum = np.arange(4,10,0.1)

# 70%(w/v) suc
vis70 = 3e-3     # Pa.sec (suc70(w/v) ~ 3 mPa.sec)
D_n1 = 1e12 * kB * T * (np.log(length_flagellum*1e-6/diam)) /\
           (2 * np.pi * vis70 * length_flagellum*1e-6)
D_n2n3 = 1e12 * kB * T * (np.log(length_flagellum*1e-6/diam) ) /\
           (4 * np.pi * vis70 * length_flagellum*1e-6) 
plt.figure(dpi=300, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
plt.errorbar(length_diff_sorted[:,0], length_diff_sorted[:,2],
              xerr=length_diff_sorted[:,1],
              marker="o", markersize=10, color='purple', linestyle="none",
              capsize=10, capthick=1.5, alpha=1)
plt.errorbar(length_diff_sorted[:,0],
             length_diff_sorted[:,3],
             xerr=length_diff_sorted[:,1], 
             marker="o", markersize=10, color='C1', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.7)
plt.errorbar(length_diff_sorted[:,0],
             length_diff_sorted[:,4],
             xerr=length_diff_sorted[:,1], 
             marker="o", markersize=10, color='C2', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.7)
plt.plot(length_flagellum, D_n1, 'purple', label='_nolegend_')
plt.plot(length_flagellum, D_n2n3, 'C1', label='_nolegend_')
plt.xlabel(r'$L [\mu m]$')
plt.ylabel(r'$D_{\parallel}$ or $D_{\perp}$ [$\mu m^2$/sec]')
plt.legend(['longitudinal', 'transversal'])
plt.ylim([0, 0.1])
plt.xlim([4, 10])
plt.savefig(pdfFolder + '/Length-vs-Dtrans-70.pdf')

# 50%(w/v) suc
vis50 = 2e-3     # Pa.sec (suc70(w/v) ~ 3 mPa.sec)
D_n1_50 = 1e12 * kB * T * (np.log(length_flagellum*1e-6/diam)) /\
           (2 * np.pi * vis50 * length_flagellum*1e-6)
D_n2n3_50 = 1e12 * kB * T * (np.log(length_flagellum*1e-6/diam) ) /\
           (4 * np.pi * vis50 * length_flagellum*1e-6) 
plt.figure(dpi=300, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
plt.errorbar(length_diff_sorted_50[:,0], length_diff_sorted_50[:,2],
              xerr=length_diff_sorted_50[:,1],
              marker="o", markersize=10, color='purple', linestyle="none",
              capsize=10, capthick=1.5, alpha=1)
plt.errorbar(length_diff_sorted_50[:,0],
             length_diff_sorted_50[:,3],
             xerr=length_diff_sorted_50[:,1], 
             marker="o", markersize=10, color='C1', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.7)
plt.errorbar(length_diff_sorted_50[:,0],
             length_diff_sorted_50[:,4],
             xerr=length_diff_sorted_50[:,1], 
             marker="o", markersize=10, color='C2', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.7)
plt.plot(length_flagellum, D_n1_50, 'purple', label='_nolegend_')
plt.plot(length_flagellum, D_n2n3_50, 'C1', label='_nolegend_')
plt.xlabel(r'$L [\mu m]$')
plt.ylabel(r'$D_{\parallel}$ or $D_{\perp}$ [$\mu m^2$/sec]')
plt.legend(['longitudinal', 'transversal'])
plt.ylim([0, 0.2])
plt.xlim([4, 10])
plt.savefig(pdfFolder + '/Length-vs-Dtrans-50.pdf')

# 40%(w/v) suc
vis40 = 1.3e-3     # Pa.sec (suc70(w/v) ~ 3 mPa.sec)
D_n1_40 = 1e12 * kB * T * (np.log(length_flagellum*1e-6/diam)) /\
           (2 * np.pi * vis40 * length_flagellum*1e-6)
D_n2n3_40 = 1e12 * kB * T * (np.log(length_flagellum*1e-6/diam) ) /\
           (4 * np.pi * vis40 * length_flagellum*1e-6) 
plt.figure(dpi=300, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
plt.errorbar(length_diff_sorted_40[:,0], length_diff_sorted_40[:,2],
              xerr=length_diff_sorted_40[:,1],
              marker="o", markersize=10, color='purple', linestyle="none",
              capsize=10, capthick=1.5, alpha=1)
plt.errorbar(length_diff_sorted_40[:,0],
             length_diff_sorted_40[:,3],
             xerr=length_diff_sorted_40[:,1], 
             marker="o", markersize=10, color='C1', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.7)
plt.errorbar(length_diff_sorted_40[:,0],
             length_diff_sorted_40[:,4],
             xerr=length_diff_sorted_40[:,1], 
             marker="o", markersize=10, color='C2', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.7)
plt.plot(length_flagellum, D_n1_40, 'purple', label='_nolegend_')
plt.plot(length_flagellum, D_n2n3_40, 'C1', label='_nolegend_')
plt.xlabel(r'$L [\mu m]$')
plt.ylabel(r'$D_{\parallel}$ or $D_{\perp}$ [$\mu m^2$/sec]')
plt.legend(['longitudinal', 'transversal'])
plt.ylim([0, 0.35])
plt.xlim([4, 10])
plt.savefig(pdfFolder + '/Length-vs-Dtrans-40.pdf')

#%% Rotation diffusion vs Length
# theory
kB = 1.380649e-23  # J / K
T = 273 + 20       # K
diam = 6e-7        # m (flagella helical diameter ~0.6 um)
vis70 = 3.5e-3     # Pa.sec (suc90 ~ 2.84 mPa.sec)
length_flagellum = np.arange(4,10,0.1)
D_PY = 3 * kB * T * (np.log(length_flagellum*1e-6/diam) ) /\
           (np.pi * vis70 * (length_flagellum*1e-6)**3)    
           
# 70%(w/v) suc
vis70 = 3e-3     # Pa.sec (suc70(w/v) ~ 3 mPa.sec)
D_PY = 3 * kB * T * (np.log(length_flagellum*1e-6/diam) ) /\
           (np.pi * vis70 * (length_flagellum*1e-6)**3)  
plt.figure(dpi=300, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
plt.errorbar(length_diff_sorted[:,0],
             length_diff_sorted[:,6],
             xerr=length_diff_sorted[:,1], 
             marker="o", markersize=10, color='C1', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.7)
plt.errorbar(length_diff_sorted[:,0],
             length_diff_sorted[:,7],
             xerr=length_diff_sorted[:,1], 
             marker="o", markersize=10, color='C2', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.7)
plt.plot(length_flagellum, D_PY, 'C1', label='_nolegend_')
plt.xlabel(r'$L [\mu m]$')
plt.ylabel(r'$D_\beta$ and $D_\gamma$ [rad$^2$/sec]')
plt.legend(['$D_P$', '$D_Y$'])
plt.ylim([0, 0.04])
plt.xlim([4, 10])
plt.savefig(pdfFolder + '/Length-vs-Drot-70.pdf')

# 50%(w/v) suc
vis50 = 1.84e-3     # Pa.sec (suc70(w/v) ~ 3 mPa.sec)
D_PY_50 = 3 * kB * T * (np.log(length_flagellum*1e-6/diam) ) /\
           (np.pi * vis50 * (length_flagellum*1e-6)**3)  
plt.figure(dpi=300, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
plt.errorbar(length_diff_sorted_50[:,0],
             length_diff_sorted_50[:,6],
             xerr=length_diff_sorted_50[:,1], 
             marker="o", markersize=10, color='C1', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.7)
plt.errorbar(length_diff_sorted_50[:,0],
             length_diff_sorted_50[:,7],
             xerr=length_diff_sorted_50[:,1], 
             marker="o", markersize=10, color='C2', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.7)
plt.plot(length_flagellum, D_PY_50, 'C1', label='_nolegend_')
plt.xlabel(r'$L [\mu m]$')
plt.ylabel(r'$D_\beta$ and $D_\gamma$ [rad$^2$/sec]')
plt.legend(['$D_P$', '$D_Y$'])
plt.ylim([0, 0.04])
plt.xlim([4, 10])
plt.savefig(pdfFolder + '/Length-vs-Drot-50.pdf')

# 40%(w/v) suc
vis40 = 1.1e-3     # Pa.sec (suc70(w/v) ~ 3 mPa.sec)
D_PY_40 = 3 * kB * T * (np.log(length_flagellum*1e-6/diam) ) /\
           (np.pi * vis40 * (length_flagellum*1e-6)**3)  
plt.figure(dpi=300, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
plt.errorbar(length_diff_sorted_40[:,0],
             length_diff_sorted_40[:,6],
             xerr=length_diff_sorted_40[:,1], 
             marker="o", markersize=10, color='C1', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.7)
plt.errorbar(length_diff_sorted_40[:,0],
             length_diff_sorted_40[:,7],
             xerr=length_diff_sorted_40[:,1], 
             marker="o", markersize=10, color='C2', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.7)
plt.plot(length_flagellum, D_PY_40, 'C1', label='_nolegend_')
plt.xlabel(r'$L [\mu m]$')
plt.ylabel(r'$D_\beta$ and $D_\gamma$ [rad$^2$/sec]')
plt.legend(['$D_P$', '$D_Y$'])
plt.ylim([0, 0.1])
plt.xlim([4, 10])
plt.savefig(pdfFolder + '/Length-vs-Drot-40.pdf')

#%% Jitter plot: translation diffusion
mean_n1 = [np.mean(D_trans_40[:,0]), np.mean(D_trans_50[:,0]),
          np.mean(D_trans[:,0])]
std_n1 = [np.std(D_trans_40[:,0]), np.std(D_trans_50[:,0]),
          np.std(D_trans[:,0])]
mean_n2 = [np.mean(D_trans_40[:,1]),
           np.mean(np.mean([D_trans_50[:,1], D_trans_50[:,1]], axis=0)),
           np.mean(np.mean([D_trans[:,1], D_trans[:,1]], axis=0))]
std_n2 = [np.std(D_trans_40[:,1]),
          np.std(np.mean([D_trans_50[:,1], D_trans_50[:,1]], axis=0)),
          np.std(np.mean([D_trans[:,1], D_trans[:,1]], axis=0))]
xlabel = ["1.77", "1.96", "2.84"] 
# xlabel = ["16.67%", "18.96%", "27.01%"]

plt.rcParams.update({'font.size': 28})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
trans1 = Affine2D().translate(-0.15, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.15, 0.0) + ax.transData
sns.stripplot(data=[D_trans_40[:,0], D_trans_50[:,0], D_trans[:,0]],
              color="purple", alpha=0.7,
              transform=trans1, marker="o", size=8, jitter=0.08)
sns.stripplot(data=[D_trans_40[:,1], D_trans_50[:,1], D_trans[:,1]],
              color="C1", alpha=0.7,
              transform=trans2, marker="o", size=8, jitter=0.08)
sns.stripplot(data=[D_trans_40[:,2], D_trans_50[:,2], D_trans[:,2]],
              color="C2", alpha=0.7,
              transform=trans2, marker="o", size=8, jitter=0.08)
ax.errorbar(xlabel, mean_n1, yerr=std_n1, marker="_", markersize=20,
            color='k', linestyle="none",
            transform=trans1, capsize=10)
ax.errorbar(xlabel, mean_n2, yerr=std_n2, marker="_", markersize=20,
            color='k', linestyle="none",
            transform=trans2, capsize=10, capthick=1.5)
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$D_\parallel$ and $D_\perp$ [$\mu m^2$/sec]')
ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
# ax.set_xlabel(r'%(w/w) sucrose concentration')
ax.set_ylim([0, 0.35])
ax.figure.savefig(pdfFolder + '/Jitter-Dtrans.pdf')

#%% Jitter plot: rotation diffusion
mean_n1 = [np.mean(D_rot_40[:,0]), np.mean(D_rot_50[:,0]),
          np.mean(D_rot[:,0])]
std_n1 = [np.std(D_rot_40[:,0]), np.std(D_rot_50[:,0]),
          np.std(D_rot[:,0])]
mean_n2 = [np.mean(np.mean([D_rot_40[:,1], D_rot_40[:,1]], axis=0)),
           np.mean(np.mean([D_rot_50[:,1], D_rot_50[:,1]], axis=0)),
           np.mean(np.mean([D_rot[:,1], D_rot[:,1]], axis=0))]
std_n2 = [np.std(D_rot_40[:,1]),
          np.std(np.mean([D_rot_50[:,1], D_rot_50[:,1]], axis=0)),
          np.std(np.mean([D_rot[:,1], D_rot[:,1]], axis=0))]
xlabel = ["1.77", "1.96", "2.84"] 
# xlabel = ["16.67%", "18.96%", "27.01%"]

# Roll
plt.rcParams.update({'font.size': 28})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
sns.stripplot(data=[D_rot_40[:,0], D_rot_50[:,0], D_rot[:,0]],
              color="purple", alpha=0.7,
              marker="o", size=8, jitter=0.08)
ax.errorbar(xlabel, mean_n1, yerr=std_n1, marker="_", markersize=20,
            color='k', linestyle="none", capsize=10)
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$D_\psi$ [rad$^2$/sec]')
ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
# ax.set_xlabel(r'%(w/w) sucrose concentration')
ax.set_ylim([0, 2.5])
plt.show()
ax.figure.savefig(pdfFolder + '/Jitter-Drot-R.pdf')

# Pitch and yaw
plt.rcParams.update({'font.size': 28})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
sns.stripplot(data=[D_rot_40[:,1], D_rot_50[:,1], D_rot[:,1]],
              color="C1", alpha=0.7,
              marker="o", size=8, jitter=0.08)
sns.stripplot(data=[D_rot_40[:,2], D_rot_50[:,2], D_rot[:,2]],
              color="C2", alpha=0.7,
              marker="o", size=8, jitter=0.08)
ax.errorbar(xlabel, mean_n2, yerr=std_n2, marker="_", markersize=20,
            color='k', linestyle="none", capsize=10)
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$D_\beta$ and $D_\gamma$ [rad$^2$/sec]')
ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
# ax.set_xlabel(r'%(w/w) sucrose concentration')
ax.set_ylim([0, 0.07])
ax.figure.savefig(pdfFolder + '/Jitter-Drot-PY.pdf')
