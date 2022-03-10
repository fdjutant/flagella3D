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
saveFolder = os.path.join(this_file_dir,'6-DOF-flagella','MSD')

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
# for MSD plots
result_dir_csv = str(intensityFiles[0])[:-4] + '.csv'
for i in range(len(MSD_par)):
    MSD_all = np.array([xaxis*exp3D_ms, MSD_par[i], MSD_perp[i], MSD_rot[i]]).T
    fmt = ",".join(["%s"] + ["%10.6e"] * (MSD_all.shape[1]-1))
    np.savetxt(saveFolder + "\MSD-MT-suc90-" + str(i).zfill(2) + ".csv",
               MSD_all, fmt=fmt,
               header="lag time [sec], MSD-longitudinal [um^2]," +
                      "MSD-transversal [um^2]," + "MSAD-transversal [rad^2]",
                      comments='')

# for diffusion coefficients

#%% Translation, rotation, and combo diffusion
# Error bar 
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
ax.set_ylim([0, 0.05])
plt.show()
ax.figure.savefig(saveFolder + '/D-trans.pdf')

# # Compute T-test & Kruskal-Wallis
# t40, p40 = stats.ttest_ind(Dt40[:,0],Dt40[:,1]) 
# t50, p50 = stats.ttest_ind(Dt50[:,0],Dt50[:,1]) 
# t70, p70 = stats.ttest_ind(Dt70[:,0],Dt70[:,1]) 
# print('t, p for 40% sucrose: ',np.round(t40, 2), p40)
# print('t, p for 50% sucrose: ',np.round(t50, 2), p50)
# print('t, p for 70% sucrose: ',np.round(t70, 2), p70)
# tKW, pKW = stats.kruskal(Dt40[:,0],Dt40[:,1], Dt50[:,0],Dt50[:,1],
#                          Dt70[:,0],Dt70[:,1]) 
# print('t, p for kruskal-Wallis',tKW, pKW)

