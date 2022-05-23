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
from scipy import optimize
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

#%% sort by length
length_diff_70 = np.stack([flagella_length_mean_70, flagella_length_std_70, 
                        D_trans_70[:,0], D_trans_70[:,1], D_trans_70[:,2],
                        D_rot_70[:,0], D_rot_70[:,1], D_rot_70[:,2], 
                        D_CO_70]).T
length_diff_sorted_70 = length_diff_70[length_diff_70[:,0].argsort()]

length_diff_50 = np.stack([flagella_length_mean_50, flagella_length_std_50, 
                        D_trans_50[:,0], D_trans_50[:,1], D_trans_50[:,2],
                        D_rot_50[:,0], D_rot_50[:,1], D_rot_50[:,2],
                        D_CO_50]).T
length_diff_sorted_50 = length_diff_50[length_diff_50[:,0].argsort()]

length_diff_40 = np.stack([flagella_length_mean_40, flagella_length_std_40, 
                        D_trans_40[:,0], D_trans_40[:,1], D_trans_40[:,2],
                        D_rot_40[:,0], D_rot_40[:,1], D_rot_40[:,2],
                        D_CO_40]).T
length_diff_sorted_40 = length_diff_40[length_diff_40[:,0].argsort()]

min_length_um = 6
max_length_um = 10

data_exclusion_40 = length_diff_sorted_40[min_length_um < length_diff_sorted_40[:,0]]
data_exclusion_40 = data_exclusion_40[data_exclusion_40[:,0] < max_length_um]
data_exclusion_50 = length_diff_sorted_50[min_length_um < length_diff_sorted_50[:,0]]
data_exclusion_50 = data_exclusion_50[data_exclusion_50[:,0] < max_length_um]
data_exclusion_70 = length_diff_sorted_70[min_length_um < length_diff_sorted_70[:,0]]
data_exclusion_70 = data_exclusion_70[data_exclusion_70[:,0] < max_length_um]

data_exclusion_all = np.concatenate([data_exclusion_40, data_exclusion_50,
                                      data_exclusion_70])

#%% Fitting the mean with 1/vis (rotation)
mean_n1 = [np.mean(data_exclusion_40[:,5]),
           np.mean(data_exclusion_50[:,5]),
           np.mean(data_exclusion_70[:,5])]
sem_n1 = [sem(data_exclusion_40[:,5]),
          sem(data_exclusion_50[:,5]),
          sem(data_exclusion_70[:,5])]

vis_measured = np.array([1.3,1.96,2.84])
vis_range = np.linspace(1,3,100)
D_theory = np.mean(data_exclusion_70[:,5]) * (vis_measured[2]/vis_range)

plt.rcParams.update({'font.size': 28})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax.errorbar(vis_measured, mean_n1, yerr=sem_n1, marker="_", markersize=50,
            color='k', linestyle="none", capsize=15)
ax.plot(vis_measured, mean_n1,  c='purple', linestyle='None')
ax.plot(vis_range, D_theory, c='k',
        linestyle='-')
ax.set_ylabel(r'$D_\psi$ [rad$^2$/sec]')
ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
ax.set_ylim([0, 2.5])
ax.set_xlim([1, 3])
ax.set_xticks([1.77, 1.96, 2.84])
plt.show()
# ax.figure.savefig(pdfFolder + '/Jitter-Drot-R-fit.pdf')

#%% rotation along n2 and n3
mean_n2 = [np.mean(np.mean([data_exclusion_40[:,6],
                            data_exclusion_40[:,7]], axis=0)),
           np.mean(np.mean([data_exclusion_50[:,6],
                            data_exclusion_50[:,7]], axis=0)),
           np.mean(np.mean([data_exclusion_70[:,6],
                            data_exclusion_70[:,7]], axis=0))]
sem_n2 = [sem(np.mean([data_exclusion_40[:,6],
                       data_exclusion_40[:,7]], axis=0)),
          sem(np.mean([data_exclusion_50[:,6],
                       data_exclusion_50[:,7]], axis=0)),
          sem(np.mean([data_exclusion_70[:,6],
                       data_exclusion_70[:,7]], axis=0))]
vis_measured = np.array([1.3,1.96,2.84])
vis_range = np.linspace(1,3,100)
D_theory = mean_n2[2] * (vis_measured[2]/vis_range)

plt.rcParams.update({'font.size': 28})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax.errorbar(vis_measured, mean_n2, yerr=sem_n2, marker="_", markersize=50,
            color='k', linestyle="none", capsize=15)
ax.plot(vis_measured, mean_n2,  c='purple', linestyle='None')
ax.plot(vis_range, D_theory, c='k', linestyle='-')
ax.set_ylabel(r'$D_\beta$ and $D_\gamma$ [rad$^2$/sec]')
ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
ax.set_ylim([0, 0.05])
ax.set_xlim([1, 3])
ax.set_xticks([1.77, 1.96, 2.84])
plt.show()
ax.figure.savefig(pdfFolder + '/Jitter-Drot-PY-fit.pdf')

#%% Jitter plot: translation diffusion
mean_n1 = [np.mean(data_exclusion_40[:,2]),
           np.mean(data_exclusion_50[:,2]),
           np.mean(data_exclusion_70[:,2])]
sem_n1 =  [sem(data_exclusion_40[:,2]),
           sem(data_exclusion_50[:,2]),
           sem(data_exclusion_70[:,2])]
mean_n2 = [np.mean(np.mean([data_exclusion_40[:,3],
                            data_exclusion_40[:,4]], axis=0)),
           np.mean(np.mean([data_exclusion_50[:,3],
                            data_exclusion_50[:,4]], axis=0)),
           np.mean(np.mean([data_exclusion_70[:,3],
                            data_exclusion_70[:,4]], axis=0))]
sem_n2 = [sem(np.mean([data_exclusion_40[:,3],
                       data_exclusion_40[:,4]], axis=0)),
          sem(np.mean([data_exclusion_50[:,3],
                       data_exclusion_50[:,4]], axis=0)),
          sem(np.mean([data_exclusion_70[:,3],
                       data_exclusion_70[:,4]], axis=0))]
vis_measured = np.array([1.3,1.96,2.84])
vis_range = np.linspace(1,3,100)
D_theory_n1 = mean_n1[2] * (vis_measured[2]/vis_range)
D_theory_n2 = mean_n2[2] * (vis_measured[2]/vis_range)

plt.rcParams.update({'font.size': 28})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax.errorbar(vis_measured, mean_n1, yerr=sem_n1, marker="_", markersize=50,
            color='purple', linestyle="none", capsize=15)
ax.errorbar(vis_measured, mean_n2, yerr=sem_n2, marker="_", markersize=50,
            color='skyblue', linestyle="none", capsize=15)
ax.plot(vis_measured, mean_n2,  c='purple', linestyle='None')
ax.plot(vis_range, D_theory_n1, c='purple', linestyle='-')
ax.plot(vis_range, D_theory_n2, c='skyblue', linestyle='-')
ax.set_ylabel(r'$D_\parallel$ and $D_\perp$ [$\mu m^2$/sec]')
ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
ax.set_ylim([0, 0.25])
ax.set_xlim([1, 3])
ax.set_xticks([1.77, 1.96, 2.84])
plt.show()
ax.figure.savefig(pdfFolder + '/Jitter-Dtrans-fit.pdf')

#%% Co-diffusion
mean_CO = [np.mean(D_CO_40), np.mean(D_CO_50), np.mean(D_CO_70)] 
sem_CO = [sem(D_CO_40), sem(D_CO_50), sem(D_CO_70)]
xlabel = ["1.77", "1.96", "2.84"] 
vis_measured = np.array([1.3,1.96,2.84])
vis_range = np.linspace(1,3,100)
D_theory_CO = mean_CO[2] * (vis_measured[2]/vis_range)

plt.rcParams.update({'font.size': 28})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax.errorbar(vis_measured, mean_CO, yerr=sem_CO, marker="_", markersize=50,
            color='k', linestyle="none", capsize=15)
ax.plot(vis_range, D_theory_CO, c='k', linestyle='-')
ax.set_ylabel(r'$D_{n_1 \psi}$ [$\mu m$ x rad]')
ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
ax.set_ylim([-0.35, 0.22])
ax.set_xlim([1, 3])
ax.set_xticks([1.3, 1.96, 2.84])
ax.axhline(y = 0, color = 'k', label = 'axvline - full height')
plt.show()
ax.figure.savefig(pdfFolder + '/CoDiffusion-Jitter-fit.pdf')
