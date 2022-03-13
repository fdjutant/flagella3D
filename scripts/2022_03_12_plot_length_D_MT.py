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
import seaborn as sns
from matplotlib.transforms import Affine2D

# time settings in the light sheet
pxum = 0.115
camExposure_ms = 3
sweep_um = 25
stepsize_nm = 400
exp3D_ms = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 
pxum = 0.115

# set folder set
setName = 'suc90_25um_3ms'

# load pkl files
this_file_dir = os.path.join(os.path.dirname(os.path.abspath("./")))
loadFolder = os.path.join(this_file_dir,
                          'Dropbox (ASU)','Research',
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          '20220305_' + setName)
saveFolder = os.path.join(this_file_dir,'6-DOF-flagella','microtubules')

intensityFiles = list(Path(loadFolder).glob("*-result.pkl"))

Dpar = np.zeros(len(intensityFiles))
Dperp = np.zeros(len(intensityFiles))
Drot = np.zeros(len(intensityFiles))
flagella_length_mean = np.zeros(len(intensityFiles))
flagella_length_std = np.zeros(len(intensityFiles))
MSD_par = []
MSD_perp = []
MSD_rot = []
for j in range(len(intensityFiles)):

    with open(intensityFiles[j], "rb") as f:
          data_loaded = pickle.load(f)
    
    flagella_length_mean[j] = data_loaded["lengthMean"]
    flagella_length_std[j] = data_loaded["lengthSTD"]
    Dpar[j] = data_loaded["Dpar"]
    Dperp[j] = data_loaded["Dperp"]
    Drot[j] = data_loaded["Drot"]
    MSD_par.append(data_loaded["MSD_par"])
    MSD_perp.append(data_loaded["MSD_perp"])
    MSD_rot.append(data_loaded["MSD_rot"])
    exp3D_ms = data_loaded["exp3D_ms"]
    
MSD_par = np.array(MSD_par)
MSD_perp = np.array(MSD_perp)
MSD_rot = np.array(MSD_rot)
nInterval = 50
xaxis = np.arange(1,nInterval+1)

#%% Write to CSV for Mathematica
# MSD: mean square displacement
for i in range(len(MSD_par)):
    MSD_all = np.array([xaxis*exp3D_ms, MSD_par[i], MSD_perp[i], MSD_rot[i]]).T
    fmt = ",".join(["%s"] + ["%10.6e"] * (MSD_all.shape[1]-1))
    np.savetxt(saveFolder + "/MSD-MT-90suc-" + str(i).zfill(2) + ".csv",
               MSD_all, fmt=fmt,
               header="lag time [sec], MSD-longitudinal [um^2]," +
                      "MSD-transversal [um^2]," + "MSAD-transversal [rad^2]",
                      comments='')

# diffusion coefficients    
Diff_all = np.array([flagella_length_mean, flagella_length_std, 
                      Dpar, Dperp, Drot]).T
fmt = ",".join(["%s"] + ["%10.6e"] * (Diff_all.shape[1]-1))
np.savetxt(saveFolder + "/DiffCoeff-90suc.csv", Diff_all, fmt=fmt,
           header= "length_mean," + "length_std," +
                   "D-trans-longitudinal [um^2/sec]," +
                   "D-trans-transversal [um^2/sec]," +
                   "D-rot-transversal [rad^2/sec],",
           comments='')    

#%% Translation and rotation diffusion coefficients
# translation
mean_N = np.mean(Dpar)
std_N = np.std(Dpar)
mean_S = np.mean(Dperp)
std_S = np.std(Dperp)
xlabel = ["90\% (w/v)"]

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(dpi=300, figsize=(4,6.2))
trans1 = Affine2D().translate(-0.15, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.15, 0.0) + ax.transData
sns.swarmplot(data=Dpar,
              color="C0", alpha=0.7,
              transform=trans1, marker="o", size=12)
sns.swarmplot(data=Dperp,
              color="C1", alpha=0.7,
              transform=trans2, marker="o", size=12)
ax.errorbar(xlabel, mean_N, yerr=std_N, marker="_", markersize=50,
            color='k', linestyle="none",
            transform=trans1, capsize=10)
ax.errorbar(xlabel, mean_S, yerr=std_S, marker="_", markersize=50,
            color='k', linestyle="none",
            transform=trans2, capsize=10, capthick=1.5)
ax.set_xticklabels(xlabel)
# ax.set_title('Translation diffusion')
ax.set_ylabel(r'$D_\parallel$ or $D_\perp$ [$\mu m^2$/sec]')
# ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
ax.set_xlabel(r'%(w/w) sucrose concentration')
ax.set_ylim([0.01, 0.05])
plt.show()
ax.figure.savefig(saveFolder + '/D-trans.pdf')

# rotation
mean_PY = np.mean(Drot)
std_PY = np.std(Drot)
xlabel = ["90\% (w/v)"]

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(dpi=300, figsize=(4,6.2))
sns.swarmplot(data=Drot,
              color="C0", alpha=0.7, marker="o", size=12)
ax.errorbar(xlabel, mean_PY, yerr=std_PY, marker="_", markersize=50,
            color='k', linestyle="none", capsize=10, capthick=1.5)
ax.set_xticklabels(xlabel)
# ax.set_title('Translation diffusion')
ax.set_ylabel(r'$D_\beta$ [$\mu m^2$/sec]')
# ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
ax.set_xlabel(r'%(w/w) sucrose concentration')
ax.set_ylim([0.002, 0.008])
plt.show()
ax.figure.savefig(saveFolder + '/D-rot.pdf')

#%% Plot as a function of length
mt_diff = np.stack([flagella_length_mean, flagella_length_std/2, 
                    Dpar, Dperp, Drot]).T
mt_diff_sorted = mt_diff[mt_diff[:,0].argsort()]

# theory
kB = 1.380649e-23   # J / K
T = 273 + 20        # K
diam = 20e-9        # m (MT diameter = 25 nm)
vis90 = 15e-3       # Pa.sec (suc90 ~ 3 mPa.sec)
length_mt = np.arange(5,9.1,0.1)
diff_trans = 1e12 * kB * T * (3 * np.log(length_mt*1e-6/diam) - 2*0.114 + 0.886) /\
           (8 * np.pi * vis90 * length_mt*1e-6)
diff_par = 1e12 * kB * T * (np.log(length_mt*1e-6/diam)) /\
           (2 * np.pi * vis90 * length_mt*1e-6)
diff_perp = 1e12 * kB * T * (np.log(length_mt*1e-6/diam) ) /\
           (4 * np.pi * vis90 * length_mt*1e-6)           
diff_rot = 3 * kB * T * (np.log(length_mt*1e-6/diam) ) /\
           (np.pi * vis90 * (length_mt*1e-6)**3)                      

plt.figure(dpi=300, figsize=(7,7))
plt.rcParams.update({'font.size': 22})
# plt.errorbar(mt_diff_sorted[:,0], (1/3)*mt_diff_sorted[:,2] +
#              (2/3)*mt_diff_sorted[:,3], xerr=mt_diff[:,1],
#              marker="o", markersize=10, color='k', linestyle="none",
             # capsize=10, capthick=1.5, alpha=0.7)
plt.errorbar(mt_diff_sorted[:,0], mt_diff_sorted[:,2], xerr=mt_diff[:,1],
             marker="o", markersize=10, color='C0', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.7)
plt.errorbar(mt_diff_sorted[:,0], mt_diff_sorted[:,3], xerr=mt_diff[:,1], 
             marker="o", markersize=10, color='C1', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.7)
# plt.plot(length_mt, diff_trans, 'k', label='_nolegend_')
plt.plot(length_mt, diff_par, 'C0', label='_nolegend_')
plt.plot(length_mt, diff_perp, 'C1', label='_nolegend_')
plt.xlabel(r'$L [\mu m]$')
plt.ylabel(r'$D_\parallel$ or $D_\perp$ [$\mu m^2$/sec]')
plt.legend(['longitudinal', 'transversal'])
plt.ylim([0.01, 0.05])
plt.xlim([4.5, 9.5])
plt.savefig(saveFolder + '/D-length-trans.pdf')

plt.figure(dpi=300, figsize=(7,7))
plt.rcParams.update({'font.size': 22})
plt.errorbar(mt_diff_sorted[:,0], mt_diff_sorted[:,4], xerr=mt_diff[:,1], 
             marker="o", markersize=10, color='C0', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.7)
plt.plot(length_mt, diff_rot, 'C0', label='_nolegend_')
plt.xlabel(r'$L [\mu m]$')
plt.ylabel(r'$D_{rot}$ [rad$^2$/sec]')
plt.ylim([0.002, 0.008])
plt.xlim([4.5, 9.5])
plt.savefig(saveFolder + '/D-length-rot.pdf')