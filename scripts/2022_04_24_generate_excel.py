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

#%% Write to CSV for Mathematica
result_dir_csv = os.path.join(Path('../6-DOF-Flagella/Flagella').resolve())
nInterval = 50; xaxis = np.arange(1,nInterval+1)

for i in range(len(MSD_70)):
    MSD_70suc = np.array([xaxis*exp3D_sec, MSD_70[i,:,0],
                          0.5*(MSD_70[i,:,1] + MSD_70[i,:,2]),
                          MSAD_70[i,:,0],
                          0.5*(MSAD_70[i,:,1] + MSAD_70[i,:,2]),
                          CO_MSD_70[i]]).T
    fmt = ",".join(["%s"] + ["%10.6e"] * (MSD_70suc.shape[1]-1))
    np.savetxt(result_dir_csv + "/MSD/MSD-70suc-" + str(i).zfill(2) + ".csv",
               MSD_70suc, fmt=fmt,
               header="lag time [sec], MSD-longitudinal [um^2]," +
                      "MSD-transversal [um^2],"+
                      "MSAD-longitudinal [rad^2], MSAD-transversal [rad^2]," +
                      "MLAD [um x rad]",
               comments='')
    
for i in range(len(MSD_50)):
    MSD_50suc = np.array([xaxis*exp3D_sec, MSD_50[i,:,0],
                          0.5*(MSD_50[i,:,1] + MSD_50[i,:,2]),
                          MSAD_50[i,:,0],
                          0.5*(MSAD_50[i,:,1] + MSAD_50[i,:,2]),
                          CO_MSD_50[i]]).T
    fmt = ",".join(["%s"] + ["%10.6e"] * (MSD_50suc.shape[1]-1))
    np.savetxt(result_dir_csv + "/MSD/MSD-50suc-" + str(i).zfill(2) + ".csv",
               MSD_50suc, fmt=fmt,
               header="lag time [sec], MSD-longitudinal [um^2]," +
                      "MSD-transversal [um^2],"+
                      "MSAD-longitudinal [rad^2], MSAD-transversal [rad^2]," +
                      "MLAD [um x rad]",
               comments='')

for i in range(len(MSD_40)):
    MSD_40suc = np.array([xaxis*exp3D_sec, MSD_40[i,:,0],
                          0.5*(MSD_40[i,:,1] + MSD_40[i,:,2]),
                          MSAD_40[i,:,0],
                          0.5*(MSAD_40[i,:,1] + MSAD_40[i,:,2]),
                          CO_MSD_40[i]]).T
    fmt = ",".join(["%s"] + ["%10.6e"] * (MSD_40suc.shape[1]-1))
    np.savetxt(result_dir_csv + "/MSD/MSD-40suc-" + str(i).zfill(2) + ".csv",
               MSD_40suc, fmt=fmt,
               header="lag time [sec], MSD-longitudinal [um^2]," +
                      "MSD-transversal [um^2],"+
                      "MSAD-longitudinal [rad^2], MSAD-transversal [rad^2]," +
                      "MLAD [um x rad]",
               comments='')
    
#%% Diffusion coefficients    
Diff70 = np.hstack([Dt70, np.flip(Dr70,(0,1)),
                    Dnr70.reshape(len(Dnr70),1), eff70.reshape(len(eff70),1)])
fmt = ",".join(["%s"] + ["%10.6e"] * (Diff70.shape[1]-1))
np.savetxt(result_dir_csv + "/Diffusion-coefficients/DiffCoeff-70suc.csv", Diff70, fmt=fmt,
           header="D-trans-longitudinal [um^2/sec]," +
                  "D-trans-transversal [um^2/sec]," +
                  "D-rot-longitudinal [rad^2/sec]," +
                  "D-rot-transversal [rad^2/sec]," +
                  "D-TransRot-longitudinal [um x rad/sec]," +
                  "Efficiency [%]",
           comments='')    

Diff50 = np.hstack([Dt50, np.flip(Dr50,(0,1)),
                    Dnr50.reshape(len(Dnr50),1), eff50.reshape(len(eff50),1)])
fmt = ",".join(["%s"] + ["%10.6e"] * (Diff50.shape[1]-1))
np.savetxt(result_dir_csv + "/Diffusion-coefficients/DiffCoeff-50suc.csv", Diff50, fmt=fmt,
           header="D-trans-longitudinal [um^2/sec]," +
                  "D-trans-transversal [um^2/sec]," +
                  "D-rot-longitudinal [rad^2/sec]," +
                  "D-rot-transversal [rad^2/sec]," +
                  "D-TransRot-longitudinal [um x rad/sec]," +
                  "Efficiency [%]",
           comments='')    

Diff40 = np.hstack([Dt40, np.flip(Dr40,(0,1)),
                    Dnr40.reshape(len(Dnr40),1), eff40.reshape(len(eff40),1)])
fmt = ",".join(["%s"] + ["%10.6e"] * (Diff40.shape[1]-1))
np.savetxt(result_dir_csv + "/Diffusion-coefficients/DiffCoeff-40suc.csv", Diff40, fmt=fmt,
           header="D-trans-longitudinal [um^2/sec]," +
                  "D-trans-transversal [um^2/sec]," +
                  "D-rot-longitudinal [rad^2/sec]," +
                  "D-rot-transversal [rad^2/sec]," +
                  "D-TransRot-longitudinal [um x rad/sec]," +
                  "Efficiency [%]",
           comments='')    