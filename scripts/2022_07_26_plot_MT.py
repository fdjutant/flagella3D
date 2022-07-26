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
from scipy.stats import sem

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
this_file_dir = os.path.join(os.path.dirname(os.path.dirname(
                             os.path.abspath("./"))))
loadFolder = os.path.join(this_file_dir,
                          'Dropbox (ASU)','BioMAN_Lab', 'Franky',
                          'Flagella-motor',
                          'Light-sheet-OPM', 'Result-data',
                          'Microtubule-data','PKL-files')
pdfFolder = os.path.join(this_file_dir,
                          'Dropbox (ASU)','BioMAN_Lab', 'Franky',
                          'Flagella-motor',
                          'Light-sheet-OPM', 'Result-data',
                          'Microtubule-data','PKL-files')

intensityFiles = list(Path(loadFolder).glob("*.pkl"))

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
    
    exp3D_sec = data_loaded["exp3D_sec"]
    pxum = data_loaded["pxum"]
    flagella_length_mean[j] = np.mean(data_loaded["flagella_length"]*pxum)
    flagella_length_std[j] = np.std(data_loaded["flagella_length"]*pxum)
    Dpar[j] = data_loaded["D_trans"][0]
    Dperp[j] = data_loaded["D_trans"][1]
    Drot[j] = data_loaded["D_rot"]
    MSD_par.append(data_loaded["MSD"][:,0])
    MSD_perp.append(data_loaded["MSD"][:,1])
    MSD_rot.append(data_loaded["MSAD"][:,1])
    
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
    np.savetxt(pdfFolder + "/MSD-MT-90suc-" + str(i).zfill(2) + ".csv",
               MSD_all, fmt=fmt,
               header="lag time [sec], MSD-longitudinal [um^2]," +
                      "MSD-transversal [um^2]," + "MSAD-transversal [rad^2]",
                      comments='')

# diffusion coefficients    
Diff_all = np.array([flagella_length_mean, flagella_length_std, 
                      Dpar, Dperp, Drot]).T
fmt = ",".join(["%s"] + ["%10.6e"] * (Diff_all.shape[1]-1))
np.savetxt(pdfFolder + "/DiffCoeff-90suc.csv", Diff_all, fmt=fmt,
           header= "length_mean," + "length_std," +
                   "D-trans-longitudinal [um^2/sec]," +
                   "D-trans-transversal [um^2/sec]," +
                   "D-rot-transversal [rad^2/sec],",
           comments='')    

#%% Translation and rotation diffusion coefficients
# translation
mean_N = np.mean(Dpar)
std_N = sem(Dpar)
mean_S = np.mean(Dperp)
std_S = sem(Dperp)
xlabel = ["10"]

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(dpi=300, figsize=(4,6.2))
trans1 = Affine2D().translate(-0.2, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.2, 0.0) + ax.transData
ax.errorbar(xlabel, mean_N, yerr=std_N, marker="_", markersize=50,
            color='k', linestyle="none",
            transform=trans1, capsize=10, capthick=1.5)
ax.errorbar(xlabel, mean_S, yerr=std_S, marker="_", markersize=50,
            color='k', linestyle="none",
            transform=trans2, capsize=10, capthick=1.5)
sns.stripplot(data=Dpar, color="k", alpha=0.5,
              transform=trans1, marker="o", size=15, jitter=0.08)
sns.stripplot(data=Dperp, color="k", alpha=0.5,
              transform=trans2, marker="s", size=15, jitter=0.08)
ax.set_xticklabels(xlabel)
# ax.set_title('Translation diffusion')
ax.set_ylabel(r'$D_\parallel$ or $D_\perp$ [$\mu m^2$/sec]')
ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
# ax.set_xlabel(r'%(w/w) sucrose concentration')
ax.set_ylim([0, 0.05])
plt.show()
ax.figure.savefig(pdfFolder + '\D-trans-MT.pdf')

#%% rotation
mean_PY = np.mean(Drot)
std_PY = sem(Drot)
xlabel = ["90\% (w/v)"]

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(dpi=300, figsize=(4,6.2))
ax.errorbar(xlabel, mean_PY, yerr=std_PY,
            marker="_", markersize=50,
            color='k', linestyle="none",
            capsize=10, capthick=1.5)
sns.stripplot(data=Drot, color="k", alpha=0.5,
              marker="^", size=15, jitter=0.08)
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$D_\beta$ [$\mu m^2$/sec]')
ax.set_xlabel(r'%(w/w) sucrose concentration')
plt.ylim([0, 0.01])
plt.show()
ax.figure.savefig(pdfFolder + '/D-rot.pdf')

#%% Plot as a function of length
mt_diff = np.stack([flagella_length_mean, flagella_length_std, 
                    Dpar, Dperp, Drot]).T
mt_diff_sorted = mt_diff[mt_diff[:,0].argsort()]

# theory
kB = 1.380649e-23   # J / K
T = 273 + 20        # K
diam = 20e-9        # m (MT diameter = 25 nm)
vis90 = 17e-3       # Pa.sec (suc90 ~ 3 mPa.sec)
length_mt = np.arange(5,9.1,0.1)
diff_trans = 1e12 * kB * T * (3 * np.log(length_mt*1e-6/diam) - 2*0.114 + 0.886) /\
           (8 * np.pi * vis90 * length_mt*1e-6)
diff_par = 1e12 * kB * T * (np.log(length_mt*1e-6/diam)) /\
           (2 * np.pi * vis90 * length_mt*1e-6)
diff_perp = 1e12 * kB * T * (np.log(length_mt*1e-6/diam) ) /\
           (4 * np.pi * vis90 * length_mt*1e-6)           
diff_rot = 3 * kB * T * (np.log(length_mt*1e-6/diam) ) /\
           (np.pi * vis90 * (length_mt*1e-6)**3)                      

plt.figure(dpi=300, figsize=(9,7))
plt.rcParams.update({'font.size': 22})
plt.errorbar(mt_diff_sorted[:,0], mt_diff_sorted[:,2], xerr=mt_diff[:,1],
             marker="o", markersize=15, color='k', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.5)
plt.errorbar(mt_diff_sorted[:,0], mt_diff_sorted[:,3], xerr=mt_diff[:,1], 
             marker="s", markersize=15, color='k', linestyle="none",
             capsize=10, capthick=1.5, alpha=0.5)
# plt.plot(length_mt, diff_trans, 'k', label='_nolegend_')
plt.plot(length_mt, diff_par, 'k', label='_nolegend_')
plt.plot(length_mt, diff_perp, 'k', label='_nolegend_')
plt.xlabel(r'$L [\mu m]$')
plt.ylabel(r'$D_\parallel$ or $D_\perp$ [$\mu m^2$/sec]')
plt.legend(['longitudinal', 'transversal'])
plt.ylim([0, 0.05])
plt.xlim([5, 9])
plt.savefig(pdfFolder + '/D-length-trans.pdf')

plt.figure(dpi=300, figsize=(9,7))
plt.rcParams.update({'font.size': 22})
plt.errorbar(mt_diff_sorted[:,0], mt_diff_sorted[:,4], xerr=mt_diff[:,1], 
              marker="^", markersize=15, color='k', linestyle="none",
              capsize=10, capthick=1.5, alpha=0.5)
plt.plot(length_mt, diff_rot, 'k', label='_nolegend_')
plt.xlabel(r'$L [\mu m]$')
plt.ylabel(r'$D_{rot}$ [rad$^2$/sec]')
plt.ylim([0, 0.01])
plt.xlim([5, 9])
plt.savefig(pdfFolder + '/D-length-rot.pdf')