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
exp3D_s = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 
pxum = 0.115

# Loading pickle files
this_file_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("./"))),
                            'Dropbox (ASU)','Research')
pklFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Flagella-data', 'summary-PKL-5fitPoints')
pdfFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Flagella-data', 'various-plots')
pklFiles    = list(Path(pklFolder).glob("suc70*.pkl"))
pklFiles50 = list(Path(pklFolder).glob("suc50*.pkl"))
pklFiles40 = list(Path(pklFolder).glob("suc40*.pkl"))

# sucrose 70 (w/v)
D_trans = np.zeros([len(pklFiles),3])
D_rot = np.zeros([len(pklFiles),3])
D_CO = np.zeros(len(pklFiles))
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
    exp3D_s = data_loaded["exp3D_sec"]
    pxum = data_loaded["pxum"]
    flagella_length_mean[j] = np.mean(data_loaded["flagella_length"])*pxum
    flagella_length_std[j] = np.std(data_loaded["flagella_length"])*pxum
    cm.append(data_loaded["cm"])
    disp.append(data_loaded["disp"])
    disp_Ang.append(data_loaded["disp_Ang"])
    D_trans[j] = data_loaded["D_trans"]
    D_rot[j] = data_loaded["D_rot"]
    D_CO[j] = data_loaded["D_co"]
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
D_CO_50 = np.zeros(len(pklFiles50))
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
    exp3D_s = data_loaded["exp3D_sec"]
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
    exp3D_s = data_loaded["exp3D_sec"]
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
MSD_40 = np.array(MSD_40, dtype=object)
MSAD_40 = np.array(MSAD_40, dtype=object)
CO_MSD_40 = np.array(CO_MSD_40,dtype=object)
cm_40 = np.array(cm_40, dtype=object)
disp_40 = np.array(disp_40, dtype=object)
disp_Ang_40 = np.array(disp_Ang_40, dtype=object)

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
ax.errorbar(xlabel, mean_n1, yerr=std_n1, marker="_", markersize=50,
            color='k', linestyle="none",
            transform=trans1, capsize=10)
ax.errorbar(xlabel, mean_n2, yerr=std_n2, marker="_", markersize=50,
            color='k', linestyle="none",
            transform=trans2, capsize=10, capthick=1.5)
sns.stripplot(data=[D_trans_40[:,0], D_trans_50[:,0], D_trans[:,0]],
              color="purple", alpha=0.5,
              transform=trans1, marker="o", size=10, jitter=0.08)
sns.stripplot(data=[D_trans_40[:,1], D_trans_50[:,1], D_trans[:,1]],
              color="C1", alpha=0.5,
              transform=trans2, marker="o", size=10, jitter=0.08)
sns.stripplot(data=[D_trans_40[:,2], D_trans_50[:,2], D_trans[:,2]],
              color="C2", alpha=0.5,
              transform=trans2, marker="o", size=10, jitter=0.08)
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$D_\parallel$ and $D_\perp$ [$\mu m^2$/sec]')
ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
# ax.set_xlabel(r'%(w/w) sucrose concentration')
ax.set_ylim([0, 0.3])
# ax.figure.savefig(pdfFolder + '/Jitter-Dtrans.pdf')

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
ax.errorbar(xlabel, mean_n1, yerr=std_n1, marker="_", markersize=50,
            color='k', linestyle="none", capsize=10)
sns.stripplot(data=[D_rot_40[:,0], D_rot_50[:,0], D_rot[:,0]],
              color="purple", alpha=0.5,
              marker="o", size=10, jitter=0.08)
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$D_\psi$ [rad$^2$/sec]')
ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
# ax.set_xlabel(r'%(w/w) sucrose concentration')
ax.set_ylim([0, 2.5])
plt.show()
# ax.figure.savefig(pdfFolder + '/Jitter-Drot-R.pdf')

# Pitch and yaw
plt.rcParams.update({'font.size': 28})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax.errorbar(xlabel, mean_n2, yerr=std_n2, marker="_", markersize=50,
            color='k', linestyle="none", capsize=10)
sns.stripplot(data=[D_rot_40[:,1], D_rot_50[:,1], D_rot[:,1]],
              color="C1", alpha=0.5,
              marker="o", size=10, jitter=0.08)
sns.stripplot(data=[D_rot_40[:,2], D_rot_50[:,2], D_rot[:,2]],
              color="C2", alpha=0.5,
              marker="o", size=10, jitter=0.08)
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$D_\beta$ and $D_\gamma$ [rad$^2$/sec]')
ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
# ax.set_xlabel(r'%(w/w) sucrose concentration')
ax.set_ylim([0, 0.045])
# ax.figure.savefig(pdfFolder + '/Jitter-Drot-PY.pdf')

#%% Jitter plot: co-diffusion
mean_CO = [np.mean(D_CO_40), np.mean(D_CO_50), np.mean(D_CO)]
std_CO = [np.std(D_CO_40), np.std(D_CO_50), np.std(D_CO)]
xlabel = ["1.77", "1.96", "2.84"] 
# xlabel = ["16.67%", "18.96%", "27.01%"]

# Roll
plt.rcParams.update({'font.size': 28})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax.errorbar(xlabel, mean_CO, yerr=std_CO, marker="_", markersize=50,
            color='k', linestyle="none", capsize=10)
sns.stripplot(data=[D_CO_40, D_CO_50, D_CO],
              color="k", alpha=0.5,
              marker="o", size=10, jitter=0.08)
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$D_{n_1 \psi}$ [rad$^2$/sec]')
ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
# ax.set_xlabel(r'%(w/w) sucrose concentration')
# ax.set_ylim([0, 2.5])
plt.show()

#%% Jitter plot: co-diffusion
mean_CO = [np.mean(D_CO_40), np.mean(D_CO_50), np.mean(D_CO)]
std_CO = [np.std(D_CO_40), np.std(D_CO_50), np.std(D_CO)]
xlabel = ["1.77", "1.96", "2.84"] 
# xlabel = ["16.67%", "18.96%", "27.01%"]

# Roll
plt.rcParams.update({'font.size': 28})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax.errorbar(xlabel, mean_CO, yerr=std_CO, marker="_", markersize=50,
            color='k', linestyle="none", capsize=10)
sns.stripplot(data=[D_CO_40, D_CO_50, D_CO],
              color="k", alpha=0.5,
              marker="o", size=10, jitter=0.08)
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$D_{n_1 \psi}$ [rad$^2$/sec]')
ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
# ax.set_xlabel(r'%(w/w) sucrose concentration')
ax.set_ylim([-0.15, 0.15])
plt.show()
ax.figure.savefig(pdfFolder + '/CoDiffusion-Jitter.pdf')

# histogram
plt.rcParams.update({'font.size': 28})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax.hist(np.concatenate([D_CO_40, D_CO_50, D_CO]),
        bins=8, color = "white", ec="k")
ax.set_xlabel(r'$D_{n_1\psi}$ [$\mu$m x rad]')
ax.set_ylabel(r'Number of data')
ax.set_xlim([-0.15, 0.15])
ax.set_ylim([0, 30])
ax.figure.savefig(pdfFolder + '/CoDiffusion-Histogram.pdf')

#%% Flagella length
flagella_all_length = np.concatenate([flagella_length_mean,
                                      flagella_length_mean_50,
                                      flagella_length_mean_40])
print('flagella_mean = %.2f with std = %.2f'
      %(np.mean(flagella_all_length), np.std(flagella_all_length)) )

plt.rcParams.update({'font.size': 28})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax.hist(flagella_all_length, bins=8, color = "white", ec="k")
ax.set_xlabel(r'Flagella length [$\mu$m]')
ax.set_ylabel(r'Number of data')
ax.set_xlim([4, 10])
ax.set_ylim([0, 25])
# ax.figure.savefig(pdfFolder + '/Flagella-length-PDF.pdf')

fig1,ax1 = plt.subplots(dpi=300, figsize=(10,6.5))
ax1.plot(np.sort(flagella_all_length),
         np.linspace(0,1,len(flagella_all_length),
         endpoint=False), 'k',ms=3, alpha=0.5)
ax1.set_xlabel(r'Flagella length [$\mu$m]')
ax1.set_ylabel(r'Cumulative Probability')
ax1.set_ylim([0, 1])
ax1.set_xlim([4, 10])
# ax1.figure.savefig(pdfFolder + '/Flagella-length-CDF.pdf')

#%% Compute A, B, and D
kB = 1.380649e-23  # J / K
T = 273 + 25       # K
diameter_flagellum = 0.5e-6   # 0.5 um
length_flagellum = np.mean(flagella_all_length)    

D_n1 = D_trans[:,0] * 1e-12
D_n1_psi = D_CO * 1e-6
D_psi = D_rot[:,0]

D_n1_50 = D_trans_50[:,0] * 1e-12
D_n1_psi_50 = D_CO_50 * 1e-6
D_psi_50 = D_rot_50[:,0]

D_n1_40 = D_trans_40[:,0] * 1e-12
D_n1_psi_40 = D_CO_40 * 1e-6
D_psi_40 = D_rot_40[:,0]

A_70 = D_psi * kB * T / (D_n1 * D_psi - D_n1_psi**2)
B_70 = D_n1_psi * kB * T / (D_n1 * D_psi - D_n1_psi**2)
D_70 = D_n1 * kB * T / (D_n1 * D_psi - D_n1_psi**2)

A_50 = D_psi_50 * kB * T / (D_n1_50 * D_psi_50 - D_n1_psi_50**2)
B_50 = D_n1_psi_50 * kB * T / (D_n1_50 * D_psi_50 - D_n1_psi_50**2)
D_50 = D_n1_50 * kB * T / (D_n1_50 * D_psi_50 - D_n1_psi_50**2)

A_40 = D_psi_40 * kB * T / (D_n1_40 * D_psi_40 - D_n1_psi_40**2)
B_40 = D_n1_psi_40 * kB * T / (D_n1_40 * D_psi_40 - D_n1_psi_40**2)
D_40 = D_n1_40 * kB * T / (D_n1_40 * D_psi_40 - D_n1_psi_40**2)

# from bead measurement
vis70 = 2.84e-3
vis50 = 1.99e-3
vis40 = 1.77e-3

A_all_vis = np.concatenate([A_70/vis70, A_50/vis50, A_40/vis40])
B_all_vis = np.concatenate([B_70/vis70, B_50/vis50, B_40/vis40]) 
D_all_vis = np.concatenate([D_70/vis70, D_50/vis50, D_40/vis40])
efficiency = np.mean(B_all_vis)**2 / (4*np.mean(A_all_vis)*np.mean(D_all_vis))

Astar_all = A_all_vis /  length_flagellum
Bstar_all = B_all_vis /  (length_flagellum * diameter_flagellum)
Dstar_all = D_all_vis /  (length_flagellum * diameter_flagellum**2)

print("Brownian-mean: A*/vis = %s, B*/vis = %s, D*/vis = %s"
      %(np.mean(Astar_all), np.mean(Bstar_all), np.mean(Dstar_all)) )
print("Brownian-std: A*/vis = %s, B*/vis = %s, D*/vis = %s"
      %(np.std(Astar_all), np.std(Bstar_all), np.std(Dstar_all)) )

#%% Optical tweezer data — Chattopadhyay, PNAS(2006)
A_optical_tweezer = 1.48e-8     # [N.s/m]
B_optical_tweezer = 7.90e-16    # [N.s]
D_optical_tweezer = 7.0e-22     # [N.s.m]
visWater = 1e-3                 # Pa.sec
print("Optical tweezer: A/vis = %s, B = %s, D = %s"
      %(np.mean(A_optical_tweezer)/visWater,
        np.mean(B_optical_tweezer)/visWater,
        np.mean(D_optical_tweezer)/visWater) )

# Purcell paper
A_over_vis = np.array([0.67,0.71,0.74,0.48,0.91]) * 1e-2 * (6*np.pi)       # [m]
B_over_vis = np.array([0.032,0.038,0.018,0.023,0.053]) * 1e-4 * (6*np.pi)  # [m^2]
D_over_vis = np.array([0.076,0.06,0.031,0.053,0.13]) * 1e-6 * (6*np.pi)    # [m^3]
print("Purcell-mean: A/vis = %.2E, B = %.2E, D = %.2E"
      %(np.mean(A_over_vis),
        np.mean(B_over_vis),
        np.mean(D_over_vis)) )
print("Purcell-std: A/vis = %.2E, B = %.2E, D = %z.2E"
      %(np.std(A_over_vis),
        np.std(B_over_vis),
        np.std(D_over_vis)) )


# Rodenborn PNAS (2013)
vis_silicone_oil = 100 # in SI [kg/(m.sec)]
drag_over_velocity = 19              # [N /(sec. m)] (Fig. 7c)   A
thrust_over_rotation = 242.6e-3      # [N m sec]     (Fig. 7a)   B
torque_over_rotation = 16.4e-3       # [N m^-1 sec]  (Fig. 7b)   D
A_rodenborn_mean = drag_over_velocity 
B_rodenborn_mean = thrust_over_rotation
D_rodenborn_mean = torque_over_rotation

A_rodenborn_std = drag_over_velocity + 0.5 
B_rodenborn_std = thrust_over_rotation + 5.5e-3
D_rodenborn_std = torque_over_rotation + 0.2e-3

print("Rodenborn-mean: A/vis = %.2E, B = %.2E, D = %.2E"
      %(A_rodenborn_mean/vis_silicone_oil,
        B_rodenborn_mean/vis_silicone_oil,
        D_rodenborn_mean/vis_silicone_oil) )
print("Rodenborn-std: A/vis = %.2E, B = %.2E, D = %.2E"
      %(A_rodenborn_std/vis_silicone_oil - A_rodenborn_mean/vis_silicone_oil,
        B_rodenborn_std/vis_silicone_oil - B_rodenborn_mean/vis_silicone_oil,
        D_rodenborn_std/vis_silicone_oil - D_rodenborn_mean/vis_silicone_oil) )

