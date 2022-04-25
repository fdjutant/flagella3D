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

#%% translational diffusion
print('for data 40, 50, 70')
print('number of data = %d, %d, %d with a total of %d'
      %(len(data_exclusion_40[:,2]), len(data_exclusion_50[:,2]),
        len(data_exclusion_70[:,2]), 
        len(data_exclusion_all[:,2])) )
print('trans-diffusion at n1 = %.3f(%.3f), %.3f(%.3f), %.3f(%.3f)'
      %(np.mean(data_exclusion_40[:,2]), sem(data_exclusion_40[:,2]),
        np.mean(data_exclusion_50[:,2]), sem(data_exclusion_50[:,2]),
        np.mean(data_exclusion_70[:,2]), sem(data_exclusion_70[:,2])) )
print('trans-diffusion at n2 = %.3f(%.3f), %.3f(%.3f), %.3f(%.3f)'
      %(np.mean(data_exclusion_40[:,3]), sem(data_exclusion_40[:,3]),
        np.mean(data_exclusion_50[:,3]), sem(data_exclusion_50[:,3]),
        np.mean(data_exclusion_70[:,3]), sem(data_exclusion_70[:,3])) )
print('trans-diffusion at n3 = %.3f(%.3f), %.3f(%.3f), %.3f(%.3f)'
      %(np.mean(data_exclusion_40[:,4]), sem(data_exclusion_40[:,4]),
        np.mean(data_exclusion_50[:,4]), sem(data_exclusion_50[:,4]),
        np.mean(data_exclusion_70[:,4]), sem(data_exclusion_70[:,4])) )

print('length [um] = %.3f(%.3f), %.3f(%.3f), %.3f(%.3f), all = %.3f(%.3f)'
      %(np.mean(data_exclusion_40[:,0]), sem(data_exclusion_40[:,0]),
        np.mean(data_exclusion_50[:,0]), sem(data_exclusion_50[:,0]),
        np.mean(data_exclusion_70[:,0]), sem(data_exclusion_70[:,0]),
        np.mean(data_exclusion_all[:,0]), sem(data_exclusion_all[:,0])
        ) )

ratio40 = data_exclusion_40[:,2] /\
          np.mean([data_exclusion_40[:,3],data_exclusion_40[:,4]],axis=0)
ratio50 = data_exclusion_50[:,2] /\
          np.mean([data_exclusion_50[:,3],data_exclusion_50[:,4]],axis=0)
ratio70 = data_exclusion_70[:,2] /\
          np.mean([data_exclusion_70[:,3],data_exclusion_70[:,4]],axis=0)          
print('ratio of translational diffusion = %.2f(%.2f), %.2f(%.2f), %.2f(%.2f)'
      %(np.mean(ratio40), sem(ratio40),
        np.mean(ratio50), sem(ratio50),
        np.mean(ratio70), sem(ratio70) ) )

D_trans40_n1 = data_exclusion_40[:,2]
D_trans50_n1 = data_exclusion_50[:,2]
D_trans70_n1 = data_exclusion_70[:,2]

D_trans40_n2n3 = np.mean([data_exclusion_40[:,3],data_exclusion_40[:,4]],axis=0)
D_trans50_n2n3 = np.mean([data_exclusion_50[:,3],data_exclusion_50[:,4]],axis=0)
D_trans70_n2n3 = np.mean([data_exclusion_70[:,3],data_exclusion_70[:,4]],axis=0)

D_rot40_n1 = data_exclusion_40[:,5]
D_rot50_n1 = data_exclusion_50[:,5]
D_rot70_n1 = data_exclusion_70[:,5]

D_rot40_n2n3 = np.mean([data_exclusion_40[:,6],data_exclusion_40[:,7]],axis=0)
D_rot50_n2n3 = np.mean([data_exclusion_50[:,6],data_exclusion_50[:,7]],axis=0)
D_rot70_n2n3 = np.mean([data_exclusion_70[:,6],data_exclusion_70[:,7]],axis=0)


# average translation diffusion
D_trans_40 = data_exclusion_40[:,2] + data_exclusion_40[:,3] + data_exclusion_40[:,4]
D_trans_50 = data_exclusion_50[:,2] + data_exclusion_50[:,3] + data_exclusion_50[:,4]
D_trans_70 = data_exclusion_70[:,2] + data_exclusion_70[:,3] + data_exclusion_70[:,4]
print('total translation diffusion = %.3f(%.3f), %.3f(%.3f), %.3f(%.3f)'
      %(np.mean(D_trans_40), sem(D_trans_40),
        np.mean(D_trans_50), sem(D_trans_50),
        np.mean(D_trans_70), sem(D_trans_70)))

D_rot_40 = data_exclusion_40[:,5] + data_exclusion_40[:,6] + data_exclusion_40[:,7]
D_rot_50 = data_exclusion_50[:,5] + data_exclusion_50[:,6] + data_exclusion_50[:,7]
D_rot_70 = data_exclusion_70[:,5] + data_exclusion_70[:,6] + data_exclusion_70[:,7]
print('total rotation diffusion = %.3f(%.3f), %.3f(%.3f), %.3f(%.3f)'
      %(np.mean(D_rot_40), sem(D_rot_40),
        np.mean(D_rot_50), sem(D_rot_50),
        np.mean(D_rot_70), sem(D_rot_70)))

#%% rotational diffusion
print('rot-diffusion at n1 = %.3f(%.3f), %.3f(%.3f), %.3f(%.3f)'
      %(np.mean(data_exclusion_40[:,5]), sem(data_exclusion_40[:,5]),
        np.mean(data_exclusion_50[:,5]), sem(data_exclusion_50[:,5]),
        np.mean(data_exclusion_70[:,5]), sem(data_exclusion_70[:,5])) )
print('rot-diffusion at n2 = %.3f(%.3f), %.3f(%.3f), %.3f(%.3f)'
      %(np.mean(data_exclusion_40[:,6]), sem(data_exclusion_40[:,6]),
        np.mean(data_exclusion_50[:,6]), sem(data_exclusion_50[:,6]),
        np.mean(data_exclusion_70[:,6]), sem(data_exclusion_70[:,6])) )
print('rot-diffusion at n3 = %.3f(%.3f), %.3f(%.3f), %.3f(%.3f)'
      %(np.mean(data_exclusion_40[:,7]), sem(data_exclusion_40[:,7]),
        np.mean(data_exclusion_50[:,7]), sem(data_exclusion_50[:,7]),
        np.mean(data_exclusion_70[:,7]), sem(data_exclusion_70[:,7])) )

#%% Jitter plot: rotation diffusion
mean_n1 = [np.mean(data_exclusion_40[:,5]),
           np.mean(data_exclusion_50[:,5]),
           np.mean(data_exclusion_70[:,5])]
std_n1 = [sem(data_exclusion_40[:,5]),
          sem(data_exclusion_50[:,5]),
          sem(data_exclusion_70[:,5])]
mean_n2 = [np.mean(np.mean([data_exclusion_40[:,6],
                            data_exclusion_40[:,7]], axis=0)),
           np.mean(np.mean([data_exclusion_50[:,6],
                            data_exclusion_50[:,7]], axis=0)),
           np.mean(np.mean([data_exclusion_70[:,6],
                            data_exclusion_70[:,7]], axis=0))]
std_n2 = [sem(np.mean([data_exclusion_40[:,6],
                       data_exclusion_40[:,7]], axis=0)),
          sem(np.mean([data_exclusion_50[:,6],
                       data_exclusion_50[:,7]], axis=0)),
          sem(np.mean([data_exclusion_70[:,6],
                       data_exclusion_70[:,7]], axis=0))]
xlabel = ["1.77", "1.96", "2.84"] 
# xlabel = ["16.67%", "18.96%", "27.01%"]

# Pitch and yaw
plt.rcParams.update({'font.size': 28})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax.errorbar(xlabel, mean_n2, yerr=std_n2, marker="_", markersize=50,
            color='k', linestyle="none", capsize=15)
sns.stripplot(data=[data_exclusion_40[:,6],
                    data_exclusion_50[:,6],
                    data_exclusion_70[:,6]],
              color="C1", alpha=0.5,
              marker="o", size=15, jitter=0.08)
sns.stripplot(data=[data_exclusion_40[:,7],
                    data_exclusion_50[:,7],
                    data_exclusion_70[:,7]],
              color="C2", alpha=0.5,
              marker="o", size=15, jitter=0.08)
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$D_\beta$ and $D_\gamma$ [rad$^2$/sec]')
ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
# ax.set_xlabel(r'%(w/w) sucrose concentration')
ax.set_ylim([0, 0.05])
ax.figure.savefig(pdfFolder + '/Jitter-Drot-PY.pdf')


#%% Roll
plt.rcParams.update({'font.size': 28})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax.errorbar(xlabel, mean_n1, yerr=std_n1, marker="_", markersize=50,
            color='k', linestyle="none", capsize=15)
sns.stripplot(data=[D_rot_40[:,0], D_rot_50[:,0], D_rot_70[:,0]],
              color="purple", alpha=0.5,
              marker="o", size=15, jitter=0.08)
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$D_\psi$ [rad$^2$/sec]')
ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
# ax.set_xlabel(r'%(w/w) sucrose concentration')
ax.set_ylim([0, 2.5])
plt.show()
ax.figure.savefig(pdfFolder + '/Jitter-Drot-R.pdf')

#%% Jitter plot: translation diffusion
mean_n1 = [np.mean(data_exclusion_40[:,2]),
           np.mean(data_exclusion_50[:,2]),
           np.mean(data_exclusion_70[:,2])]
std_n1 =  [sem(data_exclusion_40[:,2]),
           sem(data_exclusion_50[:,2]),
           sem(data_exclusion_70[:,2])]
mean_n2 = [np.mean(np.mean([data_exclusion_40[:,3],
                            data_exclusion_40[:,4]], axis=0)),
           np.mean(np.mean([data_exclusion_50[:,3],
                            data_exclusion_50[:,4]], axis=0)),
           np.mean(np.mean([data_exclusion_70[:,3],
                            data_exclusion_70[:,4]], axis=0))]
std_n2 = [sem(np.mean([data_exclusion_40[:,3],
                       data_exclusion_40[:,4]], axis=0)),
          sem(np.mean([data_exclusion_50[:,3],
                       data_exclusion_50[:,4]], axis=0)),
          sem(np.mean([data_exclusion_70[:,3],
                       data_exclusion_70[:,4]], axis=0))]
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
sns.stripplot(data=[data_exclusion_40[:,2],
                    data_exclusion_50[:,2],
                    data_exclusion_70[:,2]],
              color="purple", alpha=0.5,
              transform=trans1, marker="o", size=15, jitter=0.08)
sns.stripplot(data=[data_exclusion_40[:,3],
                    data_exclusion_50[:,3],
                    data_exclusion_70[:,3]],
                color="C1", alpha=0.5,
                transform=trans2, marker="o", size=15, jitter=0.08)
sns.stripplot(data=[data_exclusion_40[:,4],
                    data_exclusion_50[:,4],
                    data_exclusion_70[:,4]],
              color="C2", alpha=0.5,
              transform=trans2, marker="o", size=15, jitter=0.08)
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$D_\parallel$ and $D_\perp$ [$\mu m^2$/sec]')
ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
# ax.set_xlabel(r'%(w/w) sucrose concentration')
ax.set_ylim([0, 0.25])
ax.figure.savefig(pdfFolder + '/Jitter-Dtrans.pdf')


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

#%% Fitting MSD and MSAD
whichFiles = 5
nInterval = 50
xaxis = np.arange(1,nInterval+1)

MSD_n1 = MSD_70[whichFiles,:,0]
MSD_n2 = MSD_70[whichFiles,:,1]
MSD_n3 = MSD_70[whichFiles,:,2]
CO_MSD = CO_MSD_70[whichFiles]


# Fit MSD with y = Const + B*x for N, S, NR, PY, R
Nfit = 10
xtime = np.linspace(1,Nfit,Nfit)
def MSDfit(x, a, b): return b + a * x  

# fit MSD and MSAD
fit_n1, fit_n1_const  = optimize.curve_fit(MSDfit, xtime, MSD_n1[0:Nfit])[0]
fit_n2, fit_n2_const  = optimize.curve_fit(MSDfit, xtime, MSD_n2[0:Nfit])[0]
fit_n3, fit_n3_const  = optimize.curve_fit(MSDfit, xtime, MSD_n3[0:Nfit])[0]

fit_CO, fit_CO_const = optimize.curve_fit(MSDfit, xtime, CO_MSD[0:Nfit])[0]

markerSize = 8

# MSD
plt.rcParams.update({'font.size': 18})
fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
ax0.plot(xaxis*exp3D_sec, MSD_n1,
         c='purple',marker="s",mfc='none',
         ms=markerSize,ls='None',alpha=1)   
ax0.plot(xaxis*exp3D_sec, MSD_n2,
          c='C1',marker="s",mfc='none',
          ms=markerSize,ls='None',alpha=1)
ax0.plot(xaxis*exp3D_sec, MSD_n3,
          c='C2',marker="s",mfc='none',
          ms=markerSize,ls='None',alpha=1)
ax0.plot(xaxis*exp3D_sec,fit_n1_const + fit_n1*xaxis,
          c='purple',alpha=1,label='_nolegend_')
ax0.plot(xaxis*exp3D_sec,fit_n2_const + fit_n2*xaxis,
          c='C1',alpha=1,label='_nolegend_')
ax0.plot(xaxis*exp3D_sec,fit_n3_const + fit_n3*xaxis,
          c='C2',alpha=1,label='_nolegend_')
ax0.set_xlabel(r'Lag time [sec]')
ax0.set_ylabel(r'MSD [$\mu m^2$]')
ax0.set_ylim([0, 0.125])
ax0.set_xlim([0, 1.5])
# ax0.set_yticks([0,1,2,3])
ax0.figure.savefig(pdfFolder + '/fig3-MSD.pdf')

#%% MSAD: Roll
whichFiles = 4
nInterval = 50
xaxis = np.arange(1,nInterval+1)
MSAD_R = MSAD_70[whichFiles,:,0]
MSAD_P = MSAD_70[whichFiles,:,1]
MSAD_Y = MSAD_70[whichFiles,:,2]
fit_R,  fit_R_const   = optimize.curve_fit(MSDfit, xtime, MSAD_R[0:Nfit])[0]
fit_P,  fit_P_const  = optimize.curve_fit(MSDfit, xtime, MSAD_P[0:Nfit])[0]
fit_Y,  fit_Y_const  = optimize.curve_fit(MSDfit, xtime, MSAD_Y[0:Nfit])[0]

plt.rcParams.update({'font.size': 18})
fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
ax0.plot(xaxis*exp3D_sec, MSAD_R,
         c='purple',marker="o",mfc='none',
         ms=markerSize,ls='None',alpha=1)   
ax0.plot(xaxis*exp3D_sec, fit_R_const + fit_R*xaxis,
         c='purple',alpha=1,label='_nolegend_')
ax0.plot(xaxis*exp3D_sec, MSAD_P,
          c='C1',marker="o",mfc='none',
          ms=markerSize,ls='None',alpha=1)
ax0.plot(xaxis*exp3D_sec, MSAD_Y,
          c='C2',marker="o",mfc='none',
          ms=markerSize,ls='None',alpha=1)
ax0.plot(xaxis*exp3D_sec,fit_P_const + fit_P*xaxis,
          c='C1',alpha=1,label='_nolegend_')
ax0.plot(xaxis*exp3D_sec,fit_Y_const + fit_Y*xaxis,
          c='C2',alpha=1,label='_nolegend_')
ax0.set_xlabel(r'Lag time [sec]')
ax0.set_ylabel(r'MSAD [rad$^2$]')
ax0.set_ylim([-0.1, 2])
ax0.set_xlim([0, 1.5])
# ax0.set_yticks([0,1,2,3])
# ax0.legend(['$R$'])
ax0.figure.savefig(pdfFolder + '/fig3-MSAD-R.pdf')

# MSAD: pitch/yaw
fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
ax0.plot(xaxis*exp3D_sec, MSAD_P,
          c='C1',marker="o",mfc='none',
          ms=markerSize,ls='None',alpha=1)
ax0.plot(xaxis*exp3D_sec, MSAD_Y,
          c='C2',marker="o",mfc='none',
          ms=markerSize,ls='None',alpha=1)
ax0.plot(xaxis*exp3D_sec,fit_P_const + fit_P*xaxis,
          c='C1',alpha=1,label='_nolegend_')
ax0.plot(xaxis*exp3D_sec,fit_Y_const + fit_Y*xaxis,
          c='C2',alpha=1,label='_nolegend_')
ax0.set_xlabel(r'Lag time [sec]')
ax0.set_ylabel(r'MSAD [rad$^2$]')
ax0.set_ylim([0, 0.02])
ax0.set_xlim([0, 1.5])
# ax0.legend(['$P$', '$Y$'], ncol=2)
ax0.figure.savefig(pdfFolder + '/fig3-MSAD-PY.pdf')
