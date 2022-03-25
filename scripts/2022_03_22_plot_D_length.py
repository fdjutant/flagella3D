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

#%% Rotation diffusion vs Length: pitch/yaw
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
plt.savefig(pdfFolder + '/Length-vs-Drot-PY-70.pdf')

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
plt.savefig(pdfFolder + '/Length-vs-Drot-PY-50.pdf')

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
plt.savefig(pdfFolder + '/Length-vs-Drot-PY-40.pdf')

#%% Rotation diffusion vs Length: roll
# theory
kB = 1.380649e-23  # J / K
T = 273 + 20       # K
diam = 6e-7        # m (flagella helical diameter ~0.6 um)
vis70 = 3.5e-3     # Pa.sec (suc90 ~ 2.84 mPa.sec)
length_flagellum = np.arange(4,10,0.1)
D_R = 3 * kB * T * (np.log(length_flagellum*1e-6/diam) ) /\
           (np.pi * vis70 * (length_flagellum*1e-6)**3)    
           
# 70%(w/v) suc
vis70 = 3e-3     # Pa.sec (suc70(w/v) ~ 3 mPa.sec)
D_R = 3 * kB * T * (np.log(length_flagellum*1e-6/diam) ) /\
           (np.pi * vis70 * (length_flagellum*1e-6)**3)  
plt.figure(dpi=300, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
plt.errorbar(length_diff_sorted[:,0],
             length_diff_sorted[:,5],
             xerr=length_diff_sorted[:,1], 
             marker="o", markersize=10, color='purple', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.7)
# plt.plot(length_flagellum, D_R, 'purple', label='_nolegend_')
plt.xlabel(r'$L [\mu m]$')
plt.ylabel(r'$D_\psi$ [rad$^2$/sec]')
plt.legend(['$D_R$'])
# plt.ylim([0, 0.04])
plt.xlim([4, 10])
plt.savefig(pdfFolder + '/Length-vs-Drot-R-70.pdf')

# 50%(w/v) suc
vis50 = 1.84e-3     # Pa.sec (suc70(w/v) ~ 3 mPa.sec)
D_R_50 = 3 * kB * T * (np.log(length_flagellum*1e-6/diam) ) /\
           (np.pi * vis50 * (length_flagellum*1e-6)**3)  
plt.figure(dpi=300, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
plt.errorbar(length_diff_sorted_50[:,0],
             length_diff_sorted_50[:,5],
             xerr=length_diff_sorted_50[:,1], 
             marker="o", markersize=10, color='purple', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.7)
# plt.plot(length_flagellum, D_R_50, 'C1', label='_nolegend_')
plt.xlabel(r'$L [\mu m]$')
plt.ylabel(r'$D_\psi$ [rad$^2$/sec]')
plt.legend(['$D_R$'])
# plt.ylim([0, 0.04])
plt.xlim([4, 10])
plt.savefig(pdfFolder + '/Length-vs-Drot-50.pdf')

# 40%(w/v) suc
vis40 = 1.1e-3     # Pa.sec (suc70(w/v) ~ 3 mPa.sec)
D_R_40 = 3 * kB * T * (np.log(length_flagellum*1e-6/diam) ) /\
           (np.pi * vis40 * (length_flagellum*1e-6)**3)  
plt.figure(dpi=300, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
plt.errorbar(length_diff_sorted_40[:,0],
             length_diff_sorted_40[:,5],
             xerr=length_diff_sorted_40[:,1], 
             marker="o", markersize=10, color='purple', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.7)
# plt.plot(length_flagellum, D_R_40, 'C1', label='_nolegend_')
plt.xlabel(r'$L [\mu m]$')
plt.ylabel(r'$D_\psi$ [rad$^2$/sec]')
plt.legend(['$D_R$'])
# plt.ylim([0, 0.1])
plt.xlim([4, 10])
plt.savefig(pdfFolder + '/Length-vs-Drot-R-40.pdf')