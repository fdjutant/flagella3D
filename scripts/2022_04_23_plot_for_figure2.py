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

#%% Plot fluctuation
whichFiles = 4
cm = -1*cm_70[whichFiles] * pxum  
xaxis = np.arange(0,len(cm))

plt.rcParams.update({'font.size': 24})
fig0,ax0 = plt.subplots(dpi=300, figsize=(12,5))
ax0.plot(xaxis*exp3D_sec, cm[:,0] - cm[0,0], c='#C3996C',alpha=1)   
ax0.plot(xaxis*exp3D_sec, cm[:,1] - cm[0,1], c='#ED2024',alpha=1)   
ax0.plot(xaxis*exp3D_sec, cm[:,2] - cm[0,2], c='#1D75BC',alpha=1)   
ax0.axhline(y=0, c='k')
ax0.set_xlabel(r'Lag time [sec]')
ax0.set_ylabel(r'$\Delta_{CM}$ [$\mu$m]')
ax0.set_ylim([-2, 2])
ax0.set_xlim([0, 30])
# ax0.legend(['$x$', '$y$', '$z$'])
ax0.figure.savefig(pdfFolder + '/fig2-fluctuation.pdf')


