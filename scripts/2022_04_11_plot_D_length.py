#%% Import all necessary libraries
import sys
sys.path.insert(0, './modules')
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})
from pathlib import Path
import os.path
import pickle
from matplotlib.transforms import Affine2D
import seaborn as sns
from scipy.stats import sem

# time settings in the light sheet
pxum = 0.115
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
exp3D_sec = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 
pxum = 0.115

# Loading pickle files
this_file_dir = os.path.join(os.path.dirname(os.path.abspath("./")),
                            'Dropbox (ASU)','Research')
pklFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Flagella-data', 'PKL-files')
pdfFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Flagella-data', 'various-plots')
pklFiles70 = list(Path(pklFolder).glob("suc70*.pkl"))
pklFiles50 = list(Path(pklFolder).glob("suc50*.pkl"))
pklFiles40 = list(Path(pklFolder).glob("suc40*.pkl"))

# Compute A, B, D
kB = 1.380649e-23  # J / K
T = 273 + 25       # K

# sucrose 70 (w/v)
D_trans_70 = np.zeros([len(pklFiles70),3])
D_rot_70 = np.zeros([len(pklFiles70),3])
D_CO_70 = np.zeros(len(pklFiles70))
A_per_vis_70 = np.zeros(len(pklFiles70))
B_per_vis_70 = np.zeros(len(pklFiles70))
D_per_vis_70 = np.zeros(len(pklFiles70))
flagella_length_mean_70 = np.zeros(len(pklFiles70))
flagella_length_std_70 = np.zeros(len(pklFiles70))
cm_70 = []
disp_70 = []
disp_Ang_70 = []
MSD_70 = []
MSAD_70 = []
CO_MSD_70 = []
for j in range(len(pklFiles70)):
    with open(pklFiles70[j], "rb") as f:
          data_loaded = pickle.load(f)
    exp3D_sec = data_loaded["exp3D_sec"]
    pxum = data_loaded["pxum"]
    flagella_length_mean_70[j] = np.mean(data_loaded["flagella_length"])*pxum
    flagella_length_std_70[j] = np.std(data_loaded["flagella_length"])*pxum
    cm_70.append(data_loaded["cm"])
    disp_70.append(data_loaded["disp"])
    disp_Ang_70.append(data_loaded["disp_Ang"])
    D_trans_70[j] = data_loaded["D_trans"]
    D_rot_70[j] = data_loaded["D_rot"]
    D_CO_70[j] = data_loaded["D_co"]
    MSD_70.append(data_loaded["MSD"])
    MSAD_70.append(data_loaded["MSAD"])
    CO_MSD_70.append(data_loaded["CO_MSD"])
    
    D_n1 = D_trans_70[j,0] * 1e-12
    D_n1_psi = D_CO_70[j] * 1e-6
    D_psi = D_rot_70[j,0]
    vis70 = 2.84e-3
    A_per_vis_70[j] = D_psi * kB * T / (D_n1 * D_psi - D_n1_psi**2) / vis70
    B_per_vis_70[j] = -D_n1_psi * kB * T / (D_n1 * D_psi - D_n1_psi**2) / vis70
    D_per_vis_70[j] = D_n1 * kB * T / (D_n1 * D_psi - D_n1_psi**2) / vis70
    
MSD_70 = np.array(MSD_70, dtype=object)
MSAD_70 = np.array(MSAD_70, dtype=object)
CO_MSD_70 = np.array(CO_MSD_70,dtype=object)
cm_70 = np.array(cm_70, dtype=object)
disp_70 = np.array(disp_70, dtype=object)
disp_Ang_70 = np.array(disp_Ang_70, dtype=object)

# sucrose 50 (w/v)
D_trans_50 = np.zeros([len(pklFiles50),3])
D_rot_50 = np.zeros([len(pklFiles50),3])
D_CO_50 = np.zeros(len(pklFiles50))
A_per_vis_50 = np.zeros(len(pklFiles50))
B_per_vis_50 = np.zeros(len(pklFiles50))
D_per_vis_50 = np.zeros(len(pklFiles50))
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
    exp3D_sec = data_loaded["exp3D_sec"]
    pxum = data_loaded["pxum"]
    flagella_length_mean_50[j] = np.mean(data_loaded["flagella_length"])*pxum
    flagella_length_std_50[j] = np.std(data_loaded["flagella_length"])*pxum
    cm_50.append(data_loaded["cm"])
    disp_50.append(data_loaded["disp"])
    disp_Ang_50.append(data_loaded["disp_Ang"])
    D_trans_50[j] = data_loaded["D_trans"]
    D_rot_50[j] = data_loaded["D_rot"]
    D_CO_50[j] = data_loaded["D_co"]
    MSD_50.append(data_loaded["MSD"])
    MSAD_50.append(data_loaded["MSAD"])
    CO_MSD_50.append(data_loaded["CO_MSD"])
    
    D_n1 = D_trans_50[j,0] * 1e-12
    D_n1_psi = D_CO_50[j] * 1e-6
    D_psi = D_rot_50[j,0]
    vis50 = 2.84e-3
    A_per_vis_50[j] = D_psi * kB * T / (D_n1 * D_psi - D_n1_psi**2) / vis50
    B_per_vis_50[j] = -D_n1_psi * kB * T / (D_n1 * D_psi - D_n1_psi**2) / vis50
    D_per_vis_50[j] = D_n1 * kB * T / (D_n1 * D_psi - D_n1_psi**2) / vis50
    
MSD_50 = np.array(MSD_50, dtype=object)
MSAD_50 = np.array(MSAD_50, dtype=object)
CO_MSD_50 = np.array(CO_MSD_50,dtype=object)
cm_50 = np.array(cm_50, dtype=object)
disp_50 = np.array(disp_50, dtype=object)
disp_Ang_50 = np.array(disp_Ang_50, dtype=object)

# sucrose 40 (w/v)
D_trans_40 = np.zeros([len(pklFiles40),3])
D_rot_40 = np.zeros([len(pklFiles40),3])
D_CO_40 = np.zeros(len(pklFiles40))
A_per_vis_40 = np.zeros(len(pklFiles40))
B_per_vis_40 = np.zeros(len(pklFiles40))
D_per_vis_40 = np.zeros(len(pklFiles40))
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
    exp3D_sec = data_loaded["exp3D_sec"]
    pxum = data_loaded["pxum"]
    flagella_length_mean_40[j] = np.mean(data_loaded["flagella_length"])*pxum
    flagella_length_std_40[j] = np.std(data_loaded["flagella_length"])*pxum
    cm_40.append(data_loaded["cm"])
    disp_40.append(data_loaded["disp"])
    disp_Ang_40.append(data_loaded["disp_Ang"])
    D_trans_40[j] = data_loaded["D_trans"]
    D_rot_40[j] = data_loaded["D_rot"]
    D_CO_40[j] = data_loaded["D_co"]
    MSD_40.append(data_loaded["MSD"])
    MSAD_40.append(data_loaded["MSAD"])
    CO_MSD_40.append(data_loaded["CO_MSD"])
    
    D_n1 = D_trans_40[j,0] * 1e-12
    D_n1_psi = D_CO_40[j] * 1e-6
    D_psi = D_rot_40[j,0]
    vis40 = 2.84e-3
    A_per_vis_40[j] = D_psi * kB * T / (D_n1 * D_psi - D_n1_psi**2) / vis40
    B_per_vis_40[j] = -D_n1_psi * kB * T / (D_n1 * D_psi - D_n1_psi**2) / vis40
    D_per_vis_40[j] = D_n1 * kB * T / (D_n1 * D_psi - D_n1_psi**2) / vis40
    
MSD_40 = np.array(MSD_40, dtype=object)
MSAD_40 = np.array(MSAD_40, dtype=object)
CO_MSD_40 = np.array(CO_MSD_40,dtype=object)
cm_40 = np.array(cm_40, dtype=object)
disp_40 = np.array(disp_40, dtype=object)
disp_Ang_40 = np.array(disp_Ang_40, dtype=object)

#%% Translation diffusion vs Length
length_diff_70 = np.stack([flagella_length_mean_70, flagella_length_std_70, 
                        D_trans_70[:,0], D_trans_70[:,1], D_trans_70[:,2],
                        D_rot_70[:,0], D_rot_70[:,1], D_rot_70[:,2]]).T
length_diff_sorted_70 = length_diff_70[length_diff_70[:,0].argsort()]

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
T = 273 + 25       # K
length_flagellum = np.arange(3,11,0.1)
diam_filament = 10e-9   # diameter filament = 20 nm
diam_helical = 6e-7     # flagella helical diameter ~0.6 um
arc_length = np.sqrt(length_flagellum**2 + (np.pi*diam_helical*1e6)**2 ) 

# 70%(w/v) suc
vis70 = 2.84e-3     # Pa.sec (suc70(w/v) ~ 3 mPa.sec)
D_n1_70_helical = 1e12 * kB * T * (np.log(length_flagellum*1e-6/diam_helical)
                                   - 0.114) /\
                   (2 * np.pi * vis70 * (length_flagellum*1e-6) )  
D_n1_70_filament = 1e12 * kB * T * (np.log(arc_length*1e-6/diam_filament)
                                    - 0.114) /\
                   (2 * np.pi * vis70 * (arc_length*1e-6) ) 
fig, ax = plt.subplots(dpi=300, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
ax.errorbar(length_diff_sorted_70[:,0], length_diff_sorted_70[:,2],
              xerr=length_diff_sorted_70[:,1],
              marker="o", markersize=10, color='purple', linestyle="none",
              capsize=10, capthick=1.5, alpha=0.5)
ax.plot(length_flagellum, D_n1_70_filament, color='brown', alpha=1)
ax.plot(length_flagellum, D_n1_70_helical, color='blue', alpha=1)
ax.fill_between(length_flagellum, D_n1_70_filament, D_n1_70_helical,
                color='purple', alpha=0.1, label='_nolegend_')
ax.set_xlabel(r'$L [\mu m]$')
ax.set_ylabel(r'$D_{\parallel}$ [$\mu m^2$/sec]')
ax.legend(['longitudinal', 'transversal'])
ax.set_ylim([0, 0.10])
ax.set_xlim([4, 11])
# plt.savefig(pdfFolder + '/Length-vs-Dtrans-n1-70.pdf')

D_n2n3_70_helical = 1e12 * kB * T * (np.log(length_flagellum*1e-6/diam_helical)
                                     + 0.886) /\
                   (4 * np.pi * vis70 * (length_flagellum*1e-6))  
D_n2n3_70_filament = 1e12 * kB * T * (np.log(arc_length*1e-6/diam_filament)
                                      + 0.886) /\
                   (4 * np.pi * vis70 * (arc_length*1e-6))              
fig, ax = plt.subplots(dpi=300, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
ax.errorbar(length_diff_sorted_70[:,0],
             length_diff_sorted_70[:,3],
             xerr=length_diff_sorted_70[:,1], 
             marker="o", markersize=10, color='C1', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.5)
ax.errorbar(length_diff_sorted_70[:,0],
             length_diff_sorted_70[:,4],
             xerr=length_diff_sorted_70[:,1], 
             marker="o", markersize=10, color='C2', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.5)
ax.plot(length_flagellum, D_n2n3_70_filament, color='brown', alpha=1)
ax.plot(length_flagellum, D_n2n3_70_helical, color='blue', alpha=1)
ax.fill_between(length_flagellum, D_n2n3_70_filament, D_n2n3_70_helical,
                color='black', alpha=0.1, label='_nolegend_')
ax.set_xlabel(r'$L [\mu m]$')
ax.set_ylabel(r'$D_{\perp}$ [$\mu m^2$/sec]')
ax.legend(['longitudinal', 'transversal'])
ax.set_ylim([0, 0.10])
ax.set_xlim([4, 11])
# plt.savefig(pdfFolder + '/Length-vs-Dtrans-n2n3-70.pdf')

#%% 50%(w/v) suc
vis50 = 1.99e-3     # Pa.sec (suc70(w/v) ~ 3 mPa.sec)
D_n1_50_helical = 1e12 * kB * T * (np.log(length_flagellum*1e-6/diam_helical)
                                   - 0.114) /\
                   (2 * np.pi * vis50 * (length_flagellum*1e-6))  
D_n1_50_filament = 1e12 * kB * T * (np.log(arc_length*1e-6/diam_filament)
                                    - 0.114) /\
                   (2 * np.pi * vis50 * (arc_length*1e-6)) 
fig, ax = plt.subplots(dpi=300, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
ax.errorbar(length_diff_sorted_50[:,0], length_diff_sorted_50[:,2],
              xerr=length_diff_sorted_50[:,1],
              marker="o", markersize=10, color='purple', linestyle="none",
              capsize=10, capthick=1.5, alpha=0.5)
ax.plot(length_flagellum, D_n1_50_filament, color='brown', alpha=1)
ax.plot(length_flagellum, D_n1_50_helical, color='blue', alpha=1)
ax.fill_between(length_flagellum, D_n1_50_filament, D_n1_50_helical,
                color='purple', alpha=0.1, label='_nolegend_')
ax.set_xlabel(r'$L [\mu m]$')
ax.set_ylabel(r'$D_{\parallel}$ [$\mu m^2$/sec]')
ax.legend(['longitudinal', 'transversal'])
ax.set_ylim([0, 0.25])
ax.set_xlim([4, 11])
# plt.savefig(pdfFolder + '/Length-vs-Dtrans-n1-50.pdf')

D_n2n3_50_helical = 1e12 * kB * T * (np.log(length_flagellum*1e-6/diam_helical)
                                     + 0.886) /\
                   (4 * np.pi * vis50 * (length_flagellum*1e-6))  
D_n2n3_50_filament = 1e12 * kB * T * (np.log(arc_length*1e-6/diam_filament)
                                     + 0.886) /\
                   (4 * np.pi * vis50 * (arc_length*1e-6))              
fig, ax = plt.subplots(dpi=300, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
ax.errorbar(length_diff_sorted_50[:,0],
             length_diff_sorted_50[:,3],
             xerr=length_diff_sorted_50[:,1], 
             marker="o", markersize=10, color='C1', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.5)
ax.errorbar(length_diff_sorted_50[:,0],
             length_diff_sorted_50[:,4],
             xerr=length_diff_sorted_50[:,1], 
             marker="o", markersize=10, color='C2', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.5)
ax.plot(length_flagellum, D_n2n3_50_filament, color='brown', alpha=1)
ax.plot(length_flagellum, D_n2n3_50_helical, color='blue', alpha=1)
ax.fill_between(length_flagellum, D_n2n3_50_filament, D_n2n3_50_helical,
                color='black', alpha=0.1, label='_nolegend_')
ax.set_xlabel(r'$L [\mu m]$')
ax.set_ylabel(r'$D_{\perp}$ [$\mu m^2$/sec]')
ax.legend(['longitudinal', 'transversal'])
ax.set_ylim([0, 0.15])
ax.set_xlim([4, 11])
# plt.savefig(pdfFolder + '/Length-vs-Dtrans-n2n3-50.pdf')

#%% 40%(w/v) suc
vis40 = 1.77e-3     # Pa.sec (suc70(w/v) ~ 3 mPa.sec)
D_n1_40_helical = 1e12 * kB * T * (np.log(length_flagellum*1e-6/diam_helical)
                                   - 0.114) /\
                   (2 * np.pi * vis40 * (length_flagellum*1e-6))  
D_n1_40_filament = 1e12 * kB * T * (np.log(arc_length*1e-6/diam_filament)
                                    - 0.114) /\
                   (2 * np.pi * vis40 * (arc_length*1e-6)) 
fig, ax = plt.subplots(dpi=300, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
ax.errorbar(length_diff_sorted_40[:,0], length_diff_sorted_40[:,2],
              xerr=length_diff_sorted_40[:,1],
              marker="o", markersize=10, color='purple', linestyle="none",
              capsize=10, capthick=1.5, alpha=0.5)
ax.plot(length_flagellum, D_n1_40_filament, color='brown', alpha=1)
ax.plot(length_flagellum, D_n1_40_helical, color='blue', alpha=1)
ax.fill_between(length_flagellum, D_n1_40_filament, D_n1_40_helical,
                color='purple', alpha=0.1, label='_nolegend_')
ax.set_xlabel(r'$L [\mu m]$')
ax.set_ylabel(r'$D_{\parallel}$ [$\mu m^2$/sec]')
ax.legend(['longitudinal', 'transversal'])
ax.set_ylim([0, 0.30])
ax.set_xlim([4, 11])
# plt.savefig(pdfFolder + '/Length-vs-Dtrans-n1-40.pdf')

D_n2n3_40_helical = 1e12 * kB * T * (np.log(length_flagellum*1e-6/diam_helical)
                                     + 0.886) /\
                   (4 * np.pi * vis40 * (length_flagellum*1e-6))  
D_n2n3_40_filament = 1e12 * kB * T * (np.log(arc_length*1e-6/diam_filament)
                                     + 0.886) /\
                   (4 * np.pi * vis40 * (arc_length*1e-6))              
fig, ax = plt.subplots(dpi=300, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
ax.errorbar(length_diff_sorted_40[:,0],
             length_diff_sorted_40[:,3],
             xerr=length_diff_sorted_40[:,1], 
             marker="o", markersize=10, color='C1', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.5)
ax.errorbar(length_diff_sorted_40[:,0],
             length_diff_sorted_40[:,4],
             xerr=length_diff_sorted_40[:,1], 
             marker="o", markersize=10, color='C2', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.5)
ax.plot(length_flagellum, D_n2n3_40_filament, color='brown', alpha=1)
ax.plot(length_flagellum, D_n2n3_40_helical, color='blue', alpha=1)
ax.fill_between(length_flagellum, D_n2n3_40_filament, D_n2n3_40_helical,
                color='black', alpha=0.1, label='_nolegend_')
ax.set_xlabel(r'$L [\mu m]$')
ax.set_ylabel(r'$D_{\perp}$ [$\mu m^2$/sec]')
ax.legend(['longitudinal', 'transversal'])
ax.set_ylim([0, 0.30])
ax.set_xlim([4, 11])
# plt.savefig(pdfFolder + '/Length-vs-Dtrans-n2n3-40.pdf')

#%% Rotation diffusion vs Length
# theory
kB = 1.380649e-23  # J / K
T = 273 + 25       # K
length_flagellum = np.arange(3,10,0.1)  # in um
diam_filament = 10e-9   # diameter filament = 20 nm
diam_helical = 6e-7     # flagella helical diameter ~0.6 um
arc_length = np.sqrt(length_flagellum**2 + (np.pi*diam_helical*1e6)**2 )           

#%% 70%(w/v) suc
vis70 = 2.84e-3     # Pa.sec (suc70(w/v) ~ 3 mPa.sec)
D_PY_70_helical = 3 * kB * T * (np.log(length_flagellum*1e-6/diam_helical) ) /\
                   (np.pi * vis70 * (length_flagellum*1e-6)**3)  
D_PY_70_filament = 3 * kB * T * (np.log(arc_length*1e-6/diam_filament) ) /\
                   (np.pi * vis70 * (arc_length*1e-6)**3)             
fig, ax = plt.subplots(dpi=300, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
ax.errorbar(length_diff_sorted_70[:,0], length_diff_sorted_70[:,6],
             xerr=length_diff_sorted_70[:,1], 
             marker="o", markersize=10, color='C1', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.5)
ax.errorbar(length_diff_sorted_70[:,0], length_diff_sorted_70[:,7],
             xerr=length_diff_sorted_70[:,1], 
             marker="o", markersize=10, color='C2', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.5)
ax.plot(length_flagellum, D_PY_70_filament, color='brown', alpha=1)
ax.plot(length_flagellum, D_PY_70_helical, color='blue', alpha=1)
ax.fill_between(length_flagellum, D_PY_70_filament, D_PY_70_helical,
                color='black', alpha=0.1, label='_nolegend_')
ax.set_xlabel(r'$L [\mu m]$')
ax.set_ylabel(r'$D_\beta$ and $D_\gamma$ [rad$^2$/sec]')
ax.legend([ 'rod (d = 30 nm)', 'rod (d = 0.6 um)','$D_P$', '$D_Y$'])
ax.set_ylim([0, 0.05])
ax.set_xlim([4, 10])
# plt.savefig(pdfFolder + '/Length-vs-Drot-PY-70.pdf')

#%% 50%(w/v) suc
vis50 = 1.99e-3     # Pa.sec (suc70(w/v) ~ 3 mPa.sec)
D_PY_50_helical = 3 * kB * T * (np.log(length_flagellum*1e-6/diam_helical) ) /\
           (np.pi * vis50 * (length_flagellum*1e-6)**3)  
D_PY_50_filament = 3 * kB * T * (np.log(arc_length*1e-6/diam_filament) ) /\
           (np.pi * vis50 * (arc_length*1e-6)**3)             
fig, ax = plt.subplots(dpi=300, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
ax.errorbar(length_diff_sorted_50[:,0],
             length_diff_sorted_50[:,6],
             xerr=length_diff_sorted_50[:,1], 
             marker="o", markersize=10, color='C1', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.5)
ax.errorbar(length_diff_sorted_50[:,0],
             length_diff_sorted_50[:,7],
             xerr=length_diff_sorted_50[:,1], 
             marker="o", markersize=10, color='C2', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.5)
ax.plot(length_flagellum, D_PY_50_filament, color='brown', alpha=1)
ax.plot(length_flagellum, D_PY_50_helical, color='blue', alpha=1)
ax.fill_between(length_flagellum, D_PY_50_filament, D_PY_50_helical,
                color='black', alpha=0.1, label='_nolegend_')
ax.set_xlabel(r'$L [\mu m]$')
ax.set_ylabel(r'$D_\beta$ and $D_\gamma$ [rad$^2$/sec]')
ax.legend([ 'rod (d = 30 nm)', 'rod (d = 0.6 um)','$D_P$', '$D_Y$'])
ax.set_ylim([0, 0.05])
ax.set_xlim([4, 10])
# plt.savefig(pdfFolder + '/Length-vs-Drot-PY-50.pdf')

#%% 40%(w/v) suc
vis40 = 1.77e-3     # Pa.sec (suc70(w/v) ~ 3 mPa.sec)
D_PY_40_helical = 3 * kB * T * (np.log(length_flagellum*1e-6/diam_helical) ) /\
           (np.pi * vis40 * (length_flagellum*1e-6)**3)  
D_PY_40_filament = 3 * kB * T * (np.log(arc_length*1e-6/diam_filament) ) /\
           (np.pi * vis40 * (arc_length*1e-6)**3)             
fig, ax = plt.subplots(dpi=300, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
ax.errorbar(length_diff_sorted_40[:,0], length_diff_sorted_40[:,6],
             xerr=length_diff_sorted_40[:,1], 
             marker="o", markersize=10, color='C1', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.5)
ax.errorbar(length_diff_sorted_40[:,0], length_diff_sorted_40[:,7],
             xerr=length_diff_sorted_40[:,1], 
             marker="o", markersize=12, color='C2', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.5)
ax.plot(length_flagellum, D_PY_40_filament, color='brown', alpha=1)
ax.plot(length_flagellum, D_PY_40_helical, color='blue', alpha=1)
ax.fill_between(length_flagellum, D_PY_40_filament, D_PY_40_helical,
                color='black', alpha=0.1, label='_nolegend_')
ax.set_xlabel(r'$L [\mu m]$')
ax.set_ylabel(r'$D_\beta$ and $D_\gamma$ [rad$^2$/sec]')
ax.legend([ 'rod (d = 30 nm)', 'rod (d = 0.6 um)','$D_P$', '$D_Y$'])
ax.set_ylim([0, 0.05])
ax.set_xlim([4, 10])
# plt.savefig(pdfFolder + '/Length-vs-Drot-PY-40.pdf')

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
vis70 = 2.84e-3     # Pa.sec (suc70(w/v) ~ 3 mPa.sec)
D_R = 3 * kB * T * (np.log(length_flagellum*1e-6/diam) ) /\
           (np.pi * vis70 * (length_flagellum*1e-6)**3)  
plt.figure(dpi=300, figsize=(10,7))
plt.rcParams.update({'font.size': 22})
plt.errorbar(length_diff_sorted_70[:,0],
             length_diff_sorted_70[:,5],
             xerr=length_diff_sorted_70[:,1], 
             marker="o", markersize=10, color='purple', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.7)
# plt.plot(length_flagellum, D_R, 'purple', label='_nolegend_')
plt.xlabel(r'$L [\mu m]$')
plt.ylabel(r'$D_\psi$ [rad$^2$/sec]')
plt.legend(['$D_R$'])
plt.ylim([0, 2.3])
plt.xlim([4, 11])
plt.savefig(pdfFolder + '/Length-vs-Drot-R-70.pdf')

#%% 50%(w/v) suc
vis50 = 1.99e-3     # Pa.sec (suc70(w/v) ~ 3 mPa.sec)
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
plt.ylim([0, 2.3])
plt.xlim([4, 11])
plt.savefig(pdfFolder + '/Length-vs-Drot-R-50.pdf')

#%% 40%(w/v) suc
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
plt.xlabel(r'$L [\mu m]$')
plt.ylabel(r'$D_\psi$ [rad$^2$/sec]')
plt.legend(['$D_R$'])
plt.ylim([0, 2.3])
plt.xlim([4, 11])
plt.savefig(pdfFolder + '/Length-vs-Drot-R-40.pdf')
