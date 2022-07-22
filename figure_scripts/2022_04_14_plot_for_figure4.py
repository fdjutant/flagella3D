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

#%% Jitter plot: co-diffusion
mean_CO = [np.mean(D_CO_40), np.mean(D_CO_50), np.mean(D_CO_70)] 
std_CO = [sem(D_CO_40), sem(D_CO_50), sem(D_CO_70)]
xlabel = ["1.77", "1.96", "2.84"] 
# xlabel = ["16.67%", "18.96%", "27.01%"]

# Roll
plt.rcParams.update({'font.size': 28})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax.errorbar(xlabel, mean_CO, yerr=std_CO, marker="_", markersize=50,
            color='k', linestyle="none", capsize=10)
sns.stripplot(data=[D_CO_40, D_CO_50, D_CO_70],
              color="k", alpha=0.3,
              marker="o", size=15, jitter=0.08)
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$D_{n_1 \psi}$ [$\mu m$ x rad]')
ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
# ax.set_xlabel(r'%(w/w) sucrose concentration')
ax.set_ylim([-0.35, 0.22])
ax.axhline(y = 0, color = 'k', label = 'axvline - full height')
plt.show()
ax.figure.savefig(pdfFolder + '/CoDiffusion-Jitter.pdf')

# histogram
plt.rcParams.update({'font.size': 28})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax.hist(np.concatenate([D_CO_40, D_CO_50, D_CO_70]),
        bins=12, color = "white", ec="k")
ax.set_xlabel(r'$-D_{n_1\psi}$ [$\mu$m x rad]')
ax.set_ylabel(r'Number of data')
ax.set_xlim([-0.35, 0.22])
ax.set_ylim([0, 32])
ax.figure.savefig(pdfFolder + '/CoDiffusion-Histogram.pdf')

#%% Flagella length
flagella_all_length = np.concatenate([flagella_length_mean_70,
                                      flagella_length_mean_50,
                                      flagella_length_mean_40])
print('flagella_mean = %.2f with sem = %.2f'
      %(np.mean(flagella_all_length), sem(flagella_all_length)) )

plt.rcParams.update({'font.size': 28})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax.hist(flagella_all_length, bins=8, color = "white", ec="k")
ax.set_xlabel(r'Flagella length [$\mu$m]')
ax.set_ylabel(r'Number of data')
ax.set_xlim([4, 13])
ax.set_ylim([0, 30])
ax.figure.savefig(pdfFolder + '/Flagella-length-PDF.pdf')

fig1,ax1 = plt.subplots(dpi=300, figsize=(10,6.5))
ax1.plot(np.sort(flagella_all_length),
         np.linspace(0,1,len(flagella_all_length),
         endpoint=False), 'k',ms=3, alpha=0.5)
ax1.set_xlabel(r'Flagella length [$\mu$m]')
ax1.set_ylabel(r'Cumulative Probability')
ax1.set_ylim([0, 1])
ax1.set_xlim([4, 13])
ax1.figure.savefig(pdfFolder + '/Flagella-length-CDF.pdf')

#%% Compute A, B, and D
kB = 1.380649e-23  # J / K
T = 273 + 25       # K
diameter_flagellum = 0.5e-6   # 0.5 um
length_flagellum = np.mean(flagella_all_length)* 1e-6  

# scale by viscosity and length and diameter
A_per_vis_per_length_40 = A_per_vis_40 / (flagella_length_mean_40 * 1e-6)
A_per_vis_per_length_50 = A_per_vis_50 / (flagella_length_mean_50 * 1e-6)
A_per_vis_per_length_70 = A_per_vis_70 / (flagella_length_mean_70 * 1e-6)

B_per_vis_per_length_40 = B_per_vis_40 / (flagella_length_mean_40 * diameter_flagellum * 1e-6)
B_per_vis_per_length_50 = B_per_vis_50 / (flagella_length_mean_50 * diameter_flagellum * 1e-6)
B_per_vis_per_length_70 = B_per_vis_70 / (flagella_length_mean_70 * diameter_flagellum * 1e-6)

D_per_vis_per_length_40 = D_per_vis_40 / (flagella_length_mean_40 * diameter_flagellum**2 * 1e-6)
D_per_vis_per_length_50 = D_per_vis_50 / (flagella_length_mean_50 * diameter_flagellum**2 * 1e-6)
D_per_vis_per_length_70 = D_per_vis_70 / (flagella_length_mean_70 * diameter_flagellum**2 * 1e-6)

A_per_vis_all = np.concatenate([A_per_vis_per_length_40, 
                                A_per_vis_per_length_50,
                                A_per_vis_per_length_70])
B_per_vis_all = np.concatenate([B_per_vis_per_length_40,
                                B_per_vis_per_length_50,
                                B_per_vis_per_length_70])
# B_per_vis_all = B_per_vis_all[B_per_vis_all < 0]
D_per_vis_all = np.concatenate([D_per_vis_per_length_40,
                                D_per_vis_per_length_50,
                                D_per_vis_per_length_70])
efficiency = np.mean(B_per_vis_all)**2 / (4*np.mean(A_per_vis_all)*np.mean(D_per_vis_all))

Astar_all = A_per_vis_all 
Bstar_all = B_per_vis_all 
Dstar_all = D_per_vis_all

print("Brownian-mean: A*/vis = %.2E, B*/vis = %.2E, D*/vis = %.2E"
      %(np.mean(Astar_all), abs(np.mean(Bstar_all)), np.mean(Dstar_all)) )
print("Brownian-SEM: A*/vis = %.3E, B*/vis = %.3E, D*/vis = %.3E"
      %(sem(Astar_all), sem(Bstar_all), sem(Dstar_all)) )

#%% Optical tweezer data — Chattopadhyay, PNAS(2006)
A_optical_tweezer = 1.48e-8     # [N.s/m]
B_optical_tweezer = 7.90e-16    # [N.s]
D_optical_tweezer = 7.0e-22     # [N.s.m]
visWater = 1e-3                 # Pa.sec
diameter_bundle = 0.5e-6     # 0.5 um
length_bundle = 6.5e-6       # predicted from resistive force theory

Astar_optical_tweezer = A_optical_tweezer / length_bundle
Bstar_optical_tweezer = B_optical_tweezer / (length_bundle * diameter_bundle)
Dstar_optical_tweezer = D_optical_tweezer / (length_bundle * diameter_bundle**2)

print("Optical tweezer:\n A*/vis = %.2E, B*/vis = %.2E, D*/vis = %.2E"
      %(np.mean(Astar_optical_tweezer)/visWater,
        np.mean(Bstar_optical_tweezer)/visWater,
        np.mean(Dstar_optical_tweezer)/visWater) )
print("Optical tweezer:\n A*/vis = %.2E, B*/vis = %.2E, D*/vis = %.2E"
      %(0.04e-8/length_bundle/visWater,
        0.2e-16/(length_bundle * diameter_bundle)/visWater,
        0.1e-22/(length_bundle * diameter_bundle**2)/visWater) )

#%% Purcell paper
A_over_vis = np.array([0.67,0.71,0.74,0.48,0.91]) * 1e-2 * (6*np.pi)       # [m]
B_over_vis = np.array([0.032,0.038,0.018,0.023,0.053]) * 1e-4 * (6*np.pi)  # [m^2]
D_over_vis = np.array([0.076,0.06,0.031,0.053,0.13]) * 1e-6 * (6*np.pi)    # [m^3]
length_wire = np.array([5.2,7.8,9.4,3.1,7.5]) * 1e-2
length_over_wavelength = np.array([5,5,5,3,7])
pitch_angle_radian = np.radians(np.array([55,39,20,55,56]))
diameter_wire = length_wire / (np.pi * np.tan(pitch_angle_radian))

Astar_over_vis = A_over_vis / length_wire
Bstar_over_vis = B_over_vis / (length_wire * diameter_wire)
Dstar_over_vis = D_over_vis / (length_wire * diameter_wire**2)

print("Purcell-mean:\n A*/vis = %.2E, B*/vis = %.2E, D*/vis = %.2E"
      %(np.mean(Astar_over_vis),
        np.mean(Bstar_over_vis),
        np.mean(Dstar_over_vis)) )
print("Purcell-SEM:\n A/vis = %.2E, B = %.2E, D = %.2E"
      %(sem(Astar_over_vis),
        sem(Bstar_over_vis),
        sem(Dstar_over_vis)) )

#%% Rodenborn PNAS (2013)
vis_silicone_oil = 100 # in SI [kg/(m.sec)]
drag_over_velocity = 19              # [N /(sec. m)] (Fig. 7c)   A
thrust_over_rotation = 242.6e-3      # [N m sec]     (Fig. 7a)   B
torque_over_rotation = 16.4e-3       # [N m^-1 sec]  (Fig. 7b)   D
diameter_rodenborn = (2*6.6) * 1e-3             # [m] Figure 7
length_rodenborn = 20 * diameter_rodenborn/2      # [m]

A_rodenborn_mean = drag_over_velocity  / length_rodenborn
B_rodenborn_mean = thrust_over_rotation / (length_rodenborn * diameter_rodenborn)
D_rodenborn_mean = torque_over_rotation / (length_rodenborn * diameter_rodenborn**2)

A_rodenborn_std = 0.5 / length_rodenborn
B_rodenborn_std = 5.5e-3 / (length_rodenborn * diameter_rodenborn)
D_rodenborn_std = 0.2e-3 / (length_rodenborn * diameter_rodenborn**2)

print("Rodenborn-mean: A/vis = %.2E, B = %.2E, D = %.2E"
      %(A_rodenborn_mean/vis_silicone_oil,
        B_rodenborn_mean/vis_silicone_oil,
        D_rodenborn_mean/vis_silicone_oil) )
print("Rodenborn-std: A/vis = %.2E, B = %.2E, D = %.2E"
      %(A_rodenborn_std/vis_silicone_oil,
        B_rodenborn_std/vis_silicone_oil,
        D_rodenborn_std/vis_silicone_oil) )
