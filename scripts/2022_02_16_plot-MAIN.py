#%% Import modules and files
import sys
sys.path.insert(0, './modules')
import numpy as np
import matplotlib.pyplot as plt
from matmatrix import MSDfit
from scipy import optimize, stats
import msd
import glob
import seaborn as sns
from matplotlib.transforms import Affine2D
import os.path
import pickle
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA

# Compute 3D exposure time
pxum = 0.115; 
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
exp3D_ms = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 

# set number of interval computed and fitting points
nInterval = 50; xaxis = np.arange(1,nInterval+1)
Nfit = 10 # number of fitting points

# initialize arrays
EuAng70 = []
EuAng50 = []
EuAng40 = []

disp70_N = []; disp70_S1 = []; disp70_S2 = []; disp70_NR = []
disp50_N = []; disp50_S1 = []; disp50_S2 = []; disp50_NR = []
disp40_N = []; disp40_S1 = []; disp40_S2 = []; disp40_NR = []

msd70_N = []; msd70_S1 = []; msd70_S2 = []; msd70_NR = []
msd50_N = []; msd50_S1 = []; msd50_S2 = []; msd50_NR = []
msd40_N = []; msd40_S1 = []; msd40_S2 = []; msd40_NR = []

msd70_P = []; msd70_R = []; msd70_Y = []; msd70_CM = []
msd50_P = []; msd50_R = []; msd50_Y = []; msd50_CM = []
msd40_P = []; msd40_R = []; msd40_Y = []; msd40_CM = []

this_file_dir = os.path.join(os.path.dirname(os.path.abspath("./")),
                            'Dropbox (ASU)','Research')
path = os.path.join(this_file_dir,
                'DNA-Rotary-Motor', 'Helical-nanotubes',
                'Light-sheet-OPM', 'Result-data',
                'Flagella-all', 'run-05')
files40 = glob.glob(os.path.join(path, "suc40") + '\*.pkl')
files50 = glob.glob(os.path.join(path, "suc50") + '\*.pkl')
files70 = glob.glob(os.path.join(path, "suc70") + '\*.pkl')

fit70_N, fit70_S, fit70_PY, fit70_R, fit70_NR = (np.zeros([len(files70)])
                                                 for _ in range(5))
fit50_N, fit50_S, fit50_PY, fit50_R, fit50_NR = (np.zeros([len(files50)])
                                                 for _ in range(5))
fit40_N, fit40_S, fit40_PY, fit40_R, fit40_NR = (np.zeros([len(files40)])
                                                 for _ in range(5))
fit40_CM = np.zeros(len(files40))
fit50_CM = np.zeros(len(files50))
fit70_CM = np.zeros(len(files70))

fit70_S1, fit70_S2, fit70_P, fit70_Y = (np.zeros([len(files70)])
                                        for _ in range(4))
fit50_S1, fit50_S2, fit50_P, fit50_Y = (np.zeros([len(files50)])
                                        for _ in range(4))
fit40_S1, fit40_S2, fit40_P, fit40_Y = (np.zeros([len(files40)])
                                        for _ in range(4))

result_dir = os.path.join(os.path.dirname(os.path.dirname(path)),'PDF')
result_dir_csv = os.path.join(Path('../6-DOF-Flagella').resolve())
pklName40 = []; pklName50 = []; pklName70 = []


# CORE: process from Pickle
whichSuc = files70
for j in range(len(whichSuc)):
    
    pklName70.append(os.path.basename(whichSuc[j]))

    with open(whichSuc[j], "rb") as f:
          data_loaded = pickle.load(f)
    
    cm = data_loaded["cm"]
    EuAng = data_loaded["EuAng"]
    localAxes = data_loaded["localAxes"]
    LengthMean = data_loaded["length_mean"]
    LengthSTD = data_loaded["length_std"]
    
    Nframes = len(cm)
    n1 = localAxes[:,0]; n2 = localAxes[:,1]; n3 = localAxes[:,2]
    EuAng70.append(EuAng)
    
    # center-of-mass tracking
    dstCM = np.zeros(len(cm))
    for i in range(len(cm)):
        dstCM[i] = np.linalg.norm(cm[i])
    
    # projecting cm-(x,y,z) to longitudinal (n1) & transversal (n2, n3)
    disp_N, disp_S1, disp_S2 = msd.trans_stepSize_Namba(cm, n1, n2, n3)
    disp_N = np.array(disp_N)
    disp_S1 = np.array(disp_S1)
    disp_S2 = np.array(disp_S2)
    disp70_N.append(disp_N); disp70_S1.append(disp_S1); disp70_S2.append(disp_S2); 
    
    # MSD: mean square displacement
    MSD_N, MSD_S1, MSD_S2, MSD_NR = msd.trans_MSD_Namba(Nframes,
                                              cm, EuAng[:,1],
                                              n1, n2, n3,
                                              exp3D_ms, nInterval)
    MSD_P = msd.regMSD(Nframes, EuAng[:,0], exp3D_ms, nInterval)
    MSD_R = msd.regMSD(Nframes, EuAng[:,1], exp3D_ms, nInterval)
    MSD_Y = msd.regMSD(Nframes, EuAng[:,2], exp3D_ms, nInterval)
    MSD_CM = msd.regMSD(Nframes, dstCM, exp3D_ms, nInterval)
    
    msd70_N.append(MSD_N); msd70_S1.append(MSD_S1); msd70_S2.append(MSD_S2)
    msd70_NR.append(MSD_NR)
    msd70_P.append(MSD_P); msd70_R.append(MSD_R); msd70_Y.append(MSD_Y)
    msd70_CM.append(MSD_CM)
    
    # Fit MSD with y = Const + B*x for N, S, NR, PY, R
    xtime = np.linspace(1,Nfit,Nfit)
    fit70_N[j],fitN_const  = optimize.curve_fit(MSDfit, xtime, MSD_N[0:Nfit])[0]
    fit70_S[j],fitS_const  = optimize.curve_fit(MSDfit, xtime,
                            np.mean([MSD_S1[0:Nfit],MSD_S2[0:Nfit]],axis=0))[0]
    fit70_NR[j],fitNR_const = optimize.curve_fit(MSDfit, xtime, MSD_NR[0:Nfit])[0]
    fit70_PY[j],fitPY_const  = optimize.curve_fit(MSDfit, xtime,
                              np.mean([MSD_P[0:Nfit],MSD_Y[0:Nfit]],axis=0))[0]
    fit70_R[j],fitR_const  = optimize.curve_fit(MSDfit, xtime, MSD_R[0:Nfit])[0]
    fit70_CM[j],fitCM_const  = optimize.curve_fit(MSDfit, xtime, MSD_CM[0:Nfit])[0]
    
    # Additional fit
    fit70_S1[j],fitP_const  = optimize.curve_fit(MSDfit, xtime, MSD_S1[0:Nfit])[0]
    fit70_S2[j],fitY_const  = optimize.curve_fit(MSDfit, xtime, MSD_S2[0:Nfit])[0]
    fit70_P[j],fitP_const  = optimize.curve_fit(MSDfit, xtime, MSD_P[0:Nfit])[0]
    fit70_Y[j],fitY_const  = optimize.curve_fit(MSDfit, xtime, MSD_Y[0:Nfit])[0]
    print(os.path.basename(whichSuc[j]))

disp70_N = np.array(disp70_N, dtype=object)
disp70_S1 = np.array(disp70_S1, dtype=object)
disp70_S2 = np.array(disp70_S2, dtype=object)
EuAng70 = np.array(EuAng70, dtype=object)
msd70_N = np.array(msd70_N)
msd70_S1 = np.array(msd70_S1)
msd70_S2 = np.array(msd70_S2)
msd70_NR = np.array(msd70_NR)
msd70_P = np.array(msd70_P)
msd70_R = np.array(msd70_R)
msd70_Y = np.array(msd70_Y)
msd70_CM = np.array(msd70_CM) 

whichSuc = files50
for j in range(len(whichSuc)):
    
    pklName = os.path.basename(whichSuc[j])

    with open(whichSuc[j], "rb") as f:
          data_loaded = pickle.load(f)
    
    cm = data_loaded["cm"]
    EuAng = data_loaded["EuAng"]
    localAxes = data_loaded["localAxes"]
    LengthMean = data_loaded["length_mean"]
    LengthSTD = data_loaded["length_std"]
    
    Nframes = len(cm)
    n1 = localAxes[:,0]; n2 = localAxes[:,1]; n3 = localAxes[:,2]
    EuAng50.append(EuAng)
    
    # center-of-mass tracking
    dstCM = np.zeros(len(cm))
    for i in range(len(cm)):
        dstCM[i] = np.linalg.norm(cm[i])
    
    # projecting cm-(x,y,z) to longitudinal (n1) & transversal (n2, n3)
    disp_N, disp_S1, disp_S2 = msd.trans_stepSize_Namba(cm, n1, n2, n3)
    disp_N = np.array(disp_N)
    disp_S1 = np.array(disp_S1)
    disp_S2 = np.array(disp_S2)
    disp50_N.append(disp_N); disp50_S1.append(disp_S1); disp50_S2.append(disp_S2); 
    
    # MSD: mean square displacement
    MSD_N, MSD_S1, MSD_S2, MSD_NR = msd.trans_MSD_Namba(Nframes,
                                              cm, EuAng[:,1],
                                              n1, n2, n3,
                                              exp3D_ms, nInterval)
    MSD_P = msd.regMSD(Nframes, EuAng[:,0], exp3D_ms, nInterval)
    MSD_R = msd.regMSD(Nframes, EuAng[:,1], exp3D_ms, nInterval)
    MSD_Y = msd.regMSD(Nframes, EuAng[:,2], exp3D_ms, nInterval)
    MSD_CM = msd.regMSD(Nframes, dstCM, exp3D_ms, nInterval)
    
    msd50_N.append(MSD_N); msd50_S1.append(MSD_S1); msd50_S2.append(MSD_S2)
    msd50_NR.append(MSD_NR)
    msd50_P.append(MSD_P); msd50_R.append(MSD_R); msd50_Y.append(MSD_Y)
    msd50_CM.append(MSD_CM)
    
    # Fit MSD with y = Const + B*x for N, S, NR, PY, R
    xtime = np.linspace(1,Nfit,Nfit)
    fit50_N[j],fitN_const  = optimize.curve_fit(MSDfit, xtime, MSD_N[0:Nfit])[0]
    fit50_S[j],fitS_const  = optimize.curve_fit(MSDfit, xtime,
                            np.mean([MSD_S1[0:Nfit],MSD_S2[0:Nfit]],axis=0))[0]
    fit50_NR[j],fitNR_const = optimize.curve_fit(MSDfit, xtime, MSD_NR[0:Nfit])[0]
    fit50_PY[j],fitPY_const  = optimize.curve_fit(MSDfit, xtime,
                              np.mean([MSD_P[0:Nfit],MSD_Y[0:Nfit]],axis=0))[0]
    fit50_R[j],fitR_const  = optimize.curve_fit(MSDfit, xtime, MSD_R[0:Nfit])[0]
    fit50_CM[j],fitCM_const  = optimize.curve_fit(MSDfit, xtime, MSD_CM[0:Nfit])[0]
    
    # Additional fit
    fit50_S1[j],fitP_const  = optimize.curve_fit(MSDfit, xtime, MSD_S1[0:Nfit])[0]
    fit50_S2[j],fitY_const  = optimize.curve_fit(MSDfit, xtime, MSD_S2[0:Nfit])[0]
    fit50_P[j],fitP_const  = optimize.curve_fit(MSDfit, xtime, MSD_P[0:Nfit])[0]
    fit50_Y[j],fitY_const  = optimize.curve_fit(MSDfit, xtime, MSD_Y[0:Nfit])[0]
    print(os.path.basename(whichSuc[j]))
    
disp50_N = np.array(disp50_N, dtype=object)
disp50_S1 = np.array(disp50_S1, dtype=object)
disp50_S2 = np.array(disp50_S2, dtype=object)
EuAng50 = np.array(EuAng50, dtype=object)
msd50_N = np.array(msd50_N)
msd50_S1 = np.array(msd50_S1)
msd50_S2 = np.array(msd50_S2)
msd50_NR = np.array(msd50_NR)
msd50_P = np.array(msd50_P)
msd50_R = np.array(msd50_R)
msd50_Y = np.array(msd50_Y)
msd50_CM = np.array(msd50_CM)

whichSuc = files40
for j in range(len(whichSuc)):
    
    pklName = os.path.basename(whichSuc[j])

    with open(whichSuc[j], "rb") as f:
          data_loaded = pickle.load(f)
    
    cm = data_loaded["cm"]
    EuAng = data_loaded["EuAng"]
    localAxes = data_loaded["localAxes"]
    LengthMean = data_loaded["length_mean"]
    LengthSTD = data_loaded["length_std"]
    
    Nframes = len(cm)
    n1 = localAxes[:,0]; n2 = localAxes[:,1]; n3 = localAxes[:,2]
    EuAng40.append(EuAng)
    
    # center-of-mass tracking
    dstCM = np.zeros(len(cm))
    for i in range(len(cm)):
        dstCM[i] = np.linalg.norm(cm[i])
    
    # projecting cm-(x,y,z) to longitudinal (n1) & transversal (n2, n3)
    disp_N, disp_S1, disp_S2 = msd.trans_stepSize_Namba(cm, n1, n2, n3)
    disp_N = np.array(disp_N)
    disp_S1 = np.array(disp_S1)
    disp_S2 = np.array(disp_S2)
    disp40_N.append(disp_N); disp40_S1.append(disp_S1); disp40_S2.append(disp_S2); 
    
    # MSD: mean square displacement
    MSD_N, MSD_S1, MSD_S2, MSD_NR = msd.trans_MSD_Namba(Nframes,
                                              cm, EuAng[:,1],
                                              n1, n2, n3,
                                              exp3D_ms, nInterval)
    MSD_P = msd.regMSD(Nframes, EuAng[:,0], exp3D_ms, nInterval)
    MSD_R = msd.regMSD(Nframes, EuAng[:,1], exp3D_ms, nInterval)
    MSD_Y = msd.regMSD(Nframes, EuAng[:,2], exp3D_ms, nInterval)
    MSD_CM = msd.regMSD(Nframes, dstCM, exp3D_ms, nInterval)
    
    msd40_N.append(MSD_N); msd40_S1.append(MSD_S1); msd40_S2.append(MSD_S2)
    msd40_NR.append(MSD_NR)
    msd40_P.append(MSD_P); msd40_R.append(MSD_R); msd40_Y.append(MSD_Y)
    msd40_CM.append(MSD_CM)
    
    # Fit MSD with y = Const + B*x for N, S, NR, PY, R
    xtime = np.linspace(1,Nfit,Nfit)
    fit40_N[j],fitN_const  = optimize.curve_fit(MSDfit, xtime, MSD_N[0:Nfit])[0]
    fit40_S[j],fitS_const  = optimize.curve_fit(MSDfit, xtime,
                            np.mean([MSD_S1[0:Nfit],MSD_S2[0:Nfit]],axis=0))[0]
    fit40_NR[j],fitNR_const = optimize.curve_fit(MSDfit, xtime, MSD_NR[0:Nfit])[0]
    fit40_PY[j],fitPY_const  = optimize.curve_fit(MSDfit, xtime,
                              np.mean([MSD_P[0:Nfit],MSD_Y[0:Nfit]],axis=0))[0]
    fit40_R[j],fitR_const  = optimize.curve_fit(MSDfit, xtime, MSD_R[0:Nfit])[0]
    fit40_CM[j],fitCM_const  = optimize.curve_fit(MSDfit, xtime, MSD_CM[0:Nfit])[0]
    
    # Additional fit
    fit40_S1[j],fitP_const  = optimize.curve_fit(MSDfit, xtime, MSD_S1[0:Nfit])[0]
    fit40_S2[j],fitY_const  = optimize.curve_fit(MSDfit, xtime, MSD_S2[0:Nfit])[0]
    fit40_P[j],fitP_const  = optimize.curve_fit(MSDfit, xtime, MSD_P[0:Nfit])[0]
    fit40_Y[j],fitY_const  = optimize.curve_fit(MSDfit, xtime, MSD_Y[0:Nfit])[0]
    print(pklName)

disp40_N = np.array(disp40_N, dtype=object)
disp40_S1 = np.array(disp40_S1, dtype=object)
disp40_S2 = np.array(disp40_S2, dtype=object)
EuAng40 = np.array(EuAng40, dtype=object)
msd40_N = np.array(msd40_N)
msd40_S1 = np.array(msd40_S1)
msd40_S2 = np.array(msd40_S2)
msd40_NR = np.array(msd40_NR)
msd40_P = np.array(msd40_P)
msd40_R = np.array(msd40_R)
msd40_Y = np.array(msd40_Y)
msd40_CM = np.array(msd40_CM)

#%% Correlation: lengtwise translation vs rotation
# import to pandas database
df40 = []
for i in range(len(disp40_N)):
    disp = np.vstack([disp40_N[i], disp40_S1[i], disp40_S2[i]]).T
    df40.append(pd.DataFrame(np.hstack([disp, np.diff(EuAng40[i], axis=0)]),
                        columns=['disp-N','disp-S1','disp-S2',
                                'pitch','roll','yaw']))
df50 = []
for i in range(len(disp50_N)):
    disp = np.vstack([disp50_N[i], disp50_S1[i], disp50_S2[i]]).T
    df50.append(pd.DataFrame(np.hstack([disp, np.diff(EuAng50[i], axis=0)]),
                        columns=['disp-N','disp-S1','disp-S2',
                                'pitch','roll','yaw']))
df70 = []
for i in range(len(disp70_N)):
    disp = np.vstack([disp70_N[i], disp70_S1[i], disp70_S2[i]]).T
    df70.append(pd.DataFrame(np.hstack([disp, np.diff(EuAng70[i], axis=0)]),
                        columns=['disp-N','disp-S1','disp-S2',
                                'pitch','roll','yaw']))

# Separate plots
whichFile = df70
for i in range(len(whichFile)):
# for i in range(1):
    
    # input data
    y = np.array(whichFile[i]["disp-N"])[::1]
    x = np.array(whichFile[i]["roll"])[::1]
    
    # pca analysis
    whData = np.array([x,y])
    pca = PCA()
    pca.fit(whData.T)
    evalue_3sig = 1.8 * np.sqrt(pca.explained_variance_[0]) 
    evalue_3sig_arr = np.array([[-evalue_3sig, evalue_3sig]])
    evector = np.array([pca.components_[0]]).T
    x_comp,y_comp = np.dot(evector, evalue_3sig_arr)
    slope = np.diff(y_comp) / np.diff(x_comp)
    
    # linear least square vs Theil-sen estimator
    res = stats.theilslopes(y, x, 0.90)
    lsq_res = stats.linregress(x, y)

    # plot it
    plt.figure(dpi=300, figsize=(10,6.2))
    plt.rcParams.update({'font.size': 22})
    plt.plot(x, y, 'C0o', mfc='None', ms=8, mew=1.5, label='_nolegend_')
    plt.plot(x_comp,y_comp, 'r')
    plt.plot(x, lsq_res[1] + lsq_res[0] * x, 'b')
    plt.plot(x, res[1] + res[0] * x, 'g')
    plt.legend([str(np.round(slope[0],3)) + ' $\mu$m/rad',
                str(np.round(lsq_res[0],3)) + ' $\mu$m/rad',
                str(np.round(res[0],3)) + ' $\mu$m/rad'], prop={'size': 12})
    plt.ylabel(r'$\Delta_\parallel$ [$\mu$m]')
    plt.xlabel(r'$\Delta\psi$ [rad]')
    plt.grid(True, which='both')
    plt.xlim([-1.5,1.5])
    plt.ylim([-1.5,1.5])
    plt.title(pklName70[i])
plt.show()

#%% Correlation (in a single plot)
## ------- 40% data ------- 
# PCA analysis
allData = []
for i in range(len(df40)): 
    allData.append(np.array([df40[i]["roll"],df40[i]["disp-N"]]).T)
allData = np.concatenate(np.array(allData, dtype=object)).T
pca = PCA()
pca.fit(allData.T)
evalue_3sig = 1.8 * np.sqrt(pca.explained_variance_[0]) 
evalue_3sig_arr = np.array([[-evalue_3sig, evalue_3sig]])
evector = np.array([pca.components_[0]]).T
x_comp,y_comp = np.dot(evector, evalue_3sig_arr)
slope = np.diff(y_comp) / np.diff(x_comp)

# linear least square vs Theil-sen estimator
x = allData[0,:]
y = allData[1,:]
res = stats.theilslopes(y, x, 0.90)
lsq_res = stats.linregress(x, y)

plt.figure(dpi=300, figsize=(10,6.2))
plt.rcParams.update({'font.size': 22})    
for i in range(len(df40)):    
    line = sns.scatterplot(data=df40[i], x="roll", y="disp-N",
                           alpha=0.5, marker="$\circ$", ec="face",
                           label='_nolegend_')
    plt.plot(x_comp,y_comp, 'r')
    plt.plot(x, res[1] + res[0] * x, 'g')
    plt.plot(x, lsq_res[1] + lsq_res[0] * x, 'b')
    plt.legend([str(np.round(slope[0],3)) + ' $\mu$m/rad',
                str(np.round(lsq_res[0],3)) + ' $\mu$m/rad',
                str(np.round(res[0],3)) + ' $\mu$m/rad'], prop={'size': 12})
    plt.xlabel(r'$\Delta_\parallel$ [$\mu$m]')
    plt.ylabel(r'$\Delta\psi$ [rad]')
    plt.grid(True, which='both')
    plt.xlim([-1.5,1.5])
    plt.ylim([-1.1,1.1])
plt.show()

## ------- 50% data ------- 
# PCA analysis
allData = []
for i in range(len(df50)): 
    allData.append(np.array([df50[i]["roll"],df50[i]["disp-N"]]).T)
allData = np.concatenate(np.array(allData, dtype=object)).T
pca = PCA()
pca.fit(allData.T)
evalue_3sig = 1.8 * np.sqrt(pca.explained_variance_[0]) 
evalue_3sig_arr = np.array([[-evalue_3sig, evalue_3sig]])
evector = np.array([pca.components_[0]]).T
x_comp,y_comp = np.dot(evector, evalue_3sig_arr)
slope = np.diff(y_comp) / np.diff(x_comp)

# linear least square vs Theil-sen estimator
x = allData[0,:]
y = allData[1,:]
res = stats.theilslopes(y, x, 0.90)
lsq_res = stats.linregress(x, y)

plt.figure(dpi=300, figsize=(10,6.2))
plt.rcParams.update({'font.size': 22})    
for i in range(len(df50)):    
    line = sns.scatterplot(data=df50[i], x="roll", y="disp-N",
                           alpha=0.5, marker="$\circ$", ec="face",
                           label='_nolegend_')
    plt.plot(x_comp,y_comp, 'r')
    plt.plot(x, res[1] + res[0] * x, 'g')
    plt.plot(x, lsq_res[1] + lsq_res[0] * x, 'b')
    plt.legend([str(np.round(slope[0],3)) + ' $\mu$m/rad',
                str(np.round(lsq_res[0],3)) + ' $\mu$m/rad',
                str(np.round(res[0],3)) + ' $\mu$m/rad'], prop={'size': 12})
    plt.ylabel('')
    plt.xlabel(r'$\Delta\psi$ [rad]')
    plt.grid(True, which='both')
    plt.xlim([-1.5,1.5])
    plt.ylim([-1.1,1.1])
plt.show()

## ------- 70% data ------- 
# PCA analysis
allData = []
for i in range(len(df70)): 
    allData.append(np.array([df70[i]["roll"],df70[i]["disp-N"]]).T)
allData = np.concatenate(np.array(allData, dtype=object)).T
pca = PCA()
pca.fit(allData.T)
evalue_3sig = 1.8 * np.sqrt(pca.explained_variance_[0]) 
evalue_3sig_arr = np.array([[-evalue_3sig, evalue_3sig]])
evector = np.array([pca.components_[0]]).T
x_comp,y_comp = np.dot(evector, evalue_3sig_arr)
slope = np.diff(y_comp) / np.diff(x_comp)

# linear least square vs Theil-sen estimator
x = allData[0,:]
y = allData[1,:]
res = stats.theilslopes(y, x, 0.90)
lsq_res = stats.linregress(x, y)

plt.figure(dpi=300, figsize=(10,6.2))
plt.rcParams.update({'font.size': 22})
for i in range(len(df70)):    
    line = sns.scatterplot(data=df70[i], x="roll", y="disp-N",
                           alpha=0.5, marker="$\circ$", ec="face",
                           label='_nolegend_')
    plt.plot(x_comp,y_comp, 'r')
    plt.plot(x, res[1] + res[0] * x, 'g')
    plt.plot(x, lsq_res[1] + lsq_res[0] * x, 'b')
    plt.legend([str(np.round(slope[0],3)) + ' $\mu$m/rad',
                str(np.round(lsq_res[0],3)) + ' $\mu$m/rad',
                str(np.round(res[0],3)) + ' $\mu$m/rad'], prop={'size': 12})
    plt.ylabel('')
    plt.xlabel(r'$\Delta\psi$ [rad]')
    plt.xlim([-1.5,1.5])
    plt.ylim([-1.1,1.1])
    plt.grid(True, which='both')
plt.show()

#%% Translation, rotation, and combo diffusion
Dt70 = np.zeros([len(fit70_N),2])
Dt70[:,0] = fit70_N / (2*exp3D_ms)
Dt70[:,1] = fit70_S / (2*exp3D_ms)
Dt50 = np.zeros([len(fit50_N),2])
Dt50[:,0] = fit50_N / (2*exp3D_ms)
Dt50[:,1] = fit50_S / (2*exp3D_ms)
Dt40 = np.zeros([len(fit40_N),2])
Dt40[:,0] = fit40_N / (2*exp3D_ms)
Dt40[:,1] = fit40_S / (2*exp3D_ms)

# Error bar 
mean_N = [np.mean(Dt40[:,0]), np.mean(Dt50[:,0]), np.mean(Dt70[:,0])]
std_N = [np.std(Dt40[:,0]), np.std(Dt50[:,0]), np.std(Dt70[:,0])]
mean_S = [np.mean(Dt40[:,1]), np.mean(Dt50[:,1]), np.mean(Dt70[:,1])]
std_S = [np.std(Dt40[:,1]), np.std(Dt50[:,1]), np.std(Dt70[:,1])]
# xlabel = ["1.77 ± 0.75", "1.96 ± 0.83", "2.84 ± 1.13"] 
xlabel = ["16.67%", "18.96%", "27.01%"]
theoUn_N = [np.mean(Dt40[:,0]) * np.sqrt(2/(200-1)),
        np.mean(Dt50[:,0]) * np.sqrt(2/(200-1)),
        np.mean(Dt70[:,0])* np.sqrt(2/(200-1))] 
theoUn_S = [np.mean(Dt40[:,1]) * np.sqrt(2/(200-1)),
        np.mean(Dt50[:,1]) * np.sqrt(2/(200-1)),
        np.mean(Dt70[:,1])* np.sqrt(2/(200-1))] 

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
trans1 = Affine2D().translate(-0.15, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.15, 0.0) + ax.transData
sns.swarmplot(data=[Dt40[:,0], Dt50[:,0], Dt70[:,0]],
              color="C0", alpha=0.7,
              transform=trans1, marker="o", size=12)
sns.swarmplot(data=[Dt40[:,1], Dt50[:,1], Dt70[:,1]],
              color="C1", alpha=0.7,
              transform=trans2, marker="o", size=12)
ax.errorbar(xlabel, mean_N, yerr=std_N, marker="_", markersize=50,
            color='k', linestyle="none",
            transform=trans1, capsize=10)
ax.errorbar(xlabel, mean_S, yerr=std_S, marker="_", markersize=50,
            color='k', linestyle="none",
            transform=trans2, capsize=10, capthick=1.5)
ax.errorbar(xlabel, mean_N, yerr=theoUn_N, marker="_", markersize=50,
            color='r', linestyle="none",
            transform=trans1, capsize=10, capthick=1.5)
ax.errorbar(xlabel, mean_S, yerr=theoUn_S, marker="_", markersize=50,
            color='r', linestyle="none",
            transform=trans2, capsize=10, capthick=1.5)
ax.set_xticklabels(xlabel)
ax.set_title('Translation diffusion')
ax.set_ylabel(r'$D_\parallel$ or $D_\perp$ [$\mu m^2$/sec]')
# ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
ax.set_xlabel(r'%(w/w) sucrose concentration')
ax.set_ylim([0, 0.3])
plt.show()
ax.figure.savefig(result_dir + '/D-trans.pdf')

# Compute T-test & Kruskal-Wallis
t40, p40 = stats.ttest_ind(Dt40[:,0],Dt40[:,1]) 
t50, p50 = stats.ttest_ind(Dt50[:,0],Dt50[:,1]) 
t70, p70 = stats.ttest_ind(Dt70[:,0],Dt70[:,1]) 
print('t, p for 40% sucrose: ',np.round(t40, 2), p40)
print('t, p for 50% sucrose: ',np.round(t50, 2), p50)
print('t, p for 70% sucrose: ',np.round(t70, 2), p70)
tKW, pKW = stats.kruskal(Dt40[:,0],Dt40[:,1], Dt50[:,0],Dt50[:,1],
                         Dt70[:,0],Dt70[:,1]) 
print('t, p for kruskal-Wallis',tKW, pKW)

# Rotation diffusion
Dr70 = np.zeros([len(fit70_P),2])
Dr70[:,0] = fit70_PY / (2*exp3D_ms)
Dr70[:,1] = fit70_R / (2*exp3D_ms)
Dr50 = np.zeros([len(fit50_P),2])
Dr50[:,0] = fit50_PY / (2*exp3D_ms)
Dr50[:,1] = fit50_R / (2*exp3D_ms)
Dr40 = np.zeros([len(fit40_P),2])
Dr40[:,0] = fit40_PY / (2*exp3D_ms)
Dr40[:,1] = fit40_R / (2*exp3D_ms) 

# Error bar 
mean_PY = [np.mean(Dr40[:,0]), np.mean(Dr50[:,0]), np.mean(Dr70[:,0])]
std_PY = [np.std(Dr40[:,0]), np.std(Dr50[:,0]), np.std(Dr70[:,0])]
mean_R = [np.mean(Dr40[:,1]), np.mean(Dr50[:,1]), np.mean(Dr70[:,1])]
std_R = [np.std(Dr40[:,1]), np.std(Dr50[:,1]), np.std(Dr70[:,1])]
# xlabel = ["1.77 ± 0.75", "1.96 ± 0.83", "2.84 ± 1.13"]
xlabel = ["16.67%", "18.96%", "27.01%"]
theoUn_PY = [np.mean(Dr40[:,0]) * np.sqrt(2/(200-1)),
        np.mean(Dr50[:,0]) * np.sqrt(2/(200-1)),
        np.mean(Dr70[:,0])* np.sqrt(2/(200-1))] 
theoUn_R = [np.mean(Dr40[:,1]) * np.sqrt(2/(200-1)),
        np.mean(Dr50[:,1]) * np.sqrt(2/(200-1)),
        np.mean(Dr70[:,1])* np.sqrt(2/(200-1))] 

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
sns.swarmplot(data=[Dr40[:,1], Dr50[:,1], Dr70[:,1]],
              color="C0", alpha=0.7, marker="o", size=12)
ax.errorbar(xlabel, mean_R, yerr=std_R, marker="_", markersize=50,
            color='k', linestyle="none", capsize=10)
ax.errorbar(xlabel, mean_R, yerr=theoUn_R, marker="_", markersize=50,
            color='r', linestyle="none", capsize=10)
ax.set_xticklabels(xlabel)
ax.set_title('Rotation diffusion - longitudinal')
ax.set_ylabel(r'$D_\psi$ [rad$^2$ sec$^{-1}$]')
# ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
ax.set_xlabel(r'%(w/w) sucrose concentration')
ax.set_ylim([0, 2.5])
plt.show()
ax.figure.savefig(result_dir + '/D-rot-longitudinal.pdf')

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
sns.swarmplot(data=[Dr40[:,0], Dr50[:,0], Dr70[:,0]],
              color="C1", alpha=0.7, marker="o", size=12)
ax.errorbar(xlabel, mean_PY, yerr=std_PY, marker="_", markersize=50,
            color='k', linestyle="none", capsize=10)
ax.errorbar(xlabel, mean_PY, yerr=theoUn_PY, marker="_", markersize=50,
            color='r', linestyle="none", capsize=10)
ax.set_xticklabels(xlabel)
ax.set_title('Rotation diffusion - traverse')
ax.set_ylabel(r'$D_\beta$ or $D_\gamma$ [rad$^2$ sec$^{-1}$]')
# ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
ax.set_xlabel(r'%(w/w) sucrose concentration')
ax.set_ylim([0, 0.06])
plt.show()
ax.figure.savefig(result_dir + '/D-rot-transverse.pdf')

# Compute T-test & Kruskal-Wallis
tPY4050, pPY4050 = stats.ttest_ind(Dr40[:,0],Dr50[:,0]) 
tPY5070, pPY5070 = stats.ttest_ind(Dr50[:,0],Dr70[:,0]) 
tR4050, pR4050 = stats.ttest_ind(Dr40[:,1],Dr50[:,1]) 
tR5070, pR5070 = stats.ttest_ind(Dr50[:,1],Dr70[:,1]) 
print('t, p for 40% & 50% transverse: ',np.round(tPY4050, 2), pPY4050)
print('t, p for 50% & 70% transverse: ',np.round(tPY5070, 2), pPY5070)
print('t, p for 40% & 50% longitudinal: ',np.round(tR4050, 2), pR4050)
print('t, p for 40% & 70% longitudinal: ',np.round(tR5070, 2), pR5070)
tPYKW, pPYKW = stats.kruskal(Dr40[:,0],Dr50[:,0],Dr70[:,0]) 
tRKW, pRKW = stats.kruskal(Dr40[:,1],Dr50[:,1],Dr70[:,1]) 
print('t, p for transverse kruskal-Wallis',tPYKW, pPYKW)
print('t, p for longitudinal kruskal-Wallis',tRKW, pRKW)

# Combo diffusion
Dnr40 = fit40_NR / (2*exp3D_ms)
Dnr50 = fit50_NR / (2*exp3D_ms)
Dnr70 = fit70_NR / (2*exp3D_ms)

# Error bar 
mean_NR = [np.mean(Dnr40), np.mean(Dnr50), np.mean(Dnr70)]
std_NR = [np.std(Dnr40), np.std(Dnr50), np.std(Dnr70)]
# xlabel = ["1.77 ± 0.75", "1.96 ± 0.83", "2.84 ± 1.13"]
xlabel = ["16.67%", "18.96%", "27.01%"]
theoUn_NR = [np.mean(Dnr40) * np.sqrt(2/(200-1)),
        np.mean(Dnr50) * np.sqrt(2/(200-1)),
        np.mean(Dnr70)* np.sqrt(2/(200-1))] 

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
sns.swarmplot(data=[Dnr40, Dnr50, Dnr70],
              color="k", alpha=0.5, marker="o", size=12)
ax.errorbar(xlabel, mean_NR, yerr=std_NR, marker="_", markersize=50,
            color='k', linestyle="none", capsize=10)
ax.errorbar(xlabel, mean_NR, yerr=theoUn_NR, marker="_", markersize=50,
            color='r', linestyle="none", capsize=10)
ax.set_xticklabels(xlabel)
ax.set_title('Combinational diffusion')
ax.set_ylabel(r'$D_{Y\psi}$ [$\mu m \times rad$ sec$^{-1}$]')
# ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
ax.set_xlabel(r'%(w/w) sucrose concentration')
ax.set_ylim([0, 0.08])
plt.show()
ax.figure.savefig(result_dir + '/D-combo.pdf')

# Compute T-test & Kruskal-Wallis
tnr4050, pnr4050 = stats.ttest_ind(Dnr40,Dnr50) 
tnr5070, pnr5070 = stats.ttest_ind(Dnr50,Dnr70) 
print('t, p for 40% & 50% combo: ',np.round(tnr4050, 2), pnr4050)
print('t, p for 50% & 70% combo: ',np.round(tnr5070, 2), pnr5070)
tnrKW, pnrKW = stats.kruskal(Dnr40,Dnr50,Dnr70) 
print('t, p for combo kruskal-Wallis',tnrKW, pnrKW)

# Efficiency per helix
eff40 = Dnr40**2 / (4 * Dt40[:,0] * Dr40[:,1]) * 100
eff50 = Dnr50**2 / (4*  Dt50[:,0] * Dr50[:,1]) * 100
eff70 = Dnr70**2 / (4 * Dt70[:,0] * Dr70[:,1]) * 100

mean_eff = [np.mean(eff40), np.mean(eff50), np.mean(eff70)]
std_eff = [np.std(eff40), np.std(eff50), np.std(eff70)]
xlabel = ["1.77 ± 0.75", "1.96 ± 0.83", "2.84 ± 1.13"]

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
sns.swarmplot(data=[eff40, eff50, eff70],
              color="g", alpha=0.5, marker="o", size=12)
ax.errorbar(xlabel, mean_eff, yerr=std_eff, marker="_", markersize=50,
            color='k', linestyle="none", capsize=10)
ax.set_xticklabels(xlabel)
ax.set_title(r'$\varepsilon = D_{Y\psi}^2\ /\ (4 D_Y D_\psi)$')
ax.set_ylabel(r'Efficiency $\varepsilon$ (%)')
ax.set_xlabel(r'viscosity (mPa$\cdot$sec)')
# ax.set_ylim([0, 0.1])
plt.show()
ax.figure.savefig(result_dir + '/D-Efficiency.pdf')

#%% 6-DOF tracking: cm-x,y,z, pitch, roll, yaw
fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
xtrack = np.arange(0,len(cm[:,0]))
ax0.plot(xtrack*exp3D_ms,cm[:,0]-cm[0,0],alpha=0.7,c='k')   
ax0.plot(xtrack*exp3D_ms,cm[:,1]-cm[0,1],alpha=0.5,c='k')   
ax0.plot(xtrack*exp3D_ms,cm[:,2]-cm[0,2],alpha=0.3,c='k')   
ax0.set_title("Sidewise ("+ str(70) + "% sucrose)")
ax0.set_xlabel(r'Time [sec]')
ax0.set_ylabel(r'Displacement [$\mu$m]')
ax0.legend([r"$\Delta x$","$\Delta y$","$\Delta z$"])
ax0.set_ylim([-40, 20])
ax0.set_xlim([0, 30])
ax0.figure.savefig(result_dir + '/6DOF-CM.pdf')  

fig0,ax0 = plt.subplots(dpi=300, figsize=(6,2))
xtrack = np.arange(0,len(cm[:,0]))
ax0.plot(xtrack*exp3D_ms,cm[:,0]-cm[0,0],alpha=0.5,c='k')   
ax0.set_xlabel(r'Time [sec]')
ax0.set_ylabel(r'Displacement [$\mu$m]')
ax0.set_ylim([-40, 20])
ax0.set_xlim([0, 30])
ax0.figure.savefig(result_dir + '/6DOF-CM-x.pdf') 

fig0,ax0 = plt.subplots(dpi=300, figsize=(6,2))
xtrack = np.arange(0,len(cm[:,0]))
ax0.plot(xtrack*exp3D_ms,cm[:,1]-cm[0,1],alpha=0.5,c='k')   
ax0.set_xlabel(r'Time [sec]')
ax0.set_ylabel(r'Displacement [$\mu$m]')
ax0.set_ylim([-20, 40])
ax0.set_xlim([0, 30])
ax0.figure.savefig(result_dir + '/6DOF-CM-y.pdf') 

fig0,ax0 = plt.subplots(dpi=300, figsize=(6,2))
xtrack = np.arange(0,len(cm[:,0]))
ax0.plot(xtrack*exp3D_ms,cm[:,2]-cm[0,2],alpha=0.5,c='k')   
ax0.set_xlabel(r'Time [sec]')
ax0.set_ylabel(r'Displacement [$\mu$m]')
ax0.set_ylim([-40, 20])
ax0.set_xlim([0, 30])
ax0.figure.savefig(result_dir + '/6DOF-CM-z.pdf') 

fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
xtrack = np.arange(0,len(EuAng[:,0]))
ax0.plot(xtrack*exp3D_ms,EuAng[:,0]-EuAng[0,0],alpha=1,c='C1')   
ax0.plot(xtrack*exp3D_ms,EuAng[:,1]-EuAng[0,1],alpha=1,c='C0')   
ax0.plot(xtrack*exp3D_ms,EuAng[:,2]-EuAng[0,2],alpha=1,c='C2')   
ax0.set_xlabel(r'Time [sec]');
ax0.set_ylabel(r'Angular displacement [rad]')
ax0.legend([r"Pitch","Roll","Yaw"])
ax0.set_yticks([-3*np.pi, -2*np.pi, -np.pi, 0, np.pi])
ax0.set_ylim([-10, np.pi])
ax0.set_xlim([0, 30])
ax0.figure.savefig(result_dir + '/6DOF-EuAng.pdf')  

#%% Single MSD only
fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
ax0.plot(xaxis*exp3D_ms,MSD_N,c='k',marker="^",mfc='none',
          ms=5,ls='None',alpha=0.5)   
ax0.plot(xaxis*exp3D_ms,np.mean([MSD_S1,MSD_S2],axis=0),
         c='k',marker="s",mfc='none',
          ms=5,ls='None',alpha=0.5)
ax0.plot(xaxis*exp3D_ms,fitN_const + fit40_N[j]*xaxis,
         c='k',alpha=0.5,label='_nolegend_')
ax0.plot(xaxis*exp3D_ms,fitS_const + fit40_S[j]*xaxis,
         c='k',alpha=0.5,label='_nolegend_')
ax0.set_xlabel(r'Lag time [sec]');
ax0.set_ylabel(r'MSD [$\mu m^2$]')
ax0.set_ylim([0, 2]);
ax0.set_xlim([0, nInterval*exp3D_ms])
ax0.legend(["lengthwise","sidewise"])
ax0.figure.savefig(result_dir + '/single-MSD-trans.pdf')  

fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
ax0.plot(xaxis*exp3D_ms,MSD_R,c='k',marker="^",mfc='none',
          ms=5,ls='None',alpha=0.5)   
ax0.plot(xaxis*exp3D_ms,np.mean([MSD_P,MSD_Y],axis=0),
          c='k',marker="s",mfc='none',
          ms=5,ls='None',alpha=0.5)
ax0.plot(xaxis*exp3D_ms,fitR_const + fit40_R[j]*xaxis,
         c='k',alpha=0.5,label='_nolegend_')
ax0.plot(xaxis*exp3D_ms,fitP_const + fit40_P[j]*xaxis,
          c='k',alpha=0.5,label='_nolegend_')
ax0.set_xlabel(r'Lag time [sec]');
ax0.set_ylabel(r'MSD [rad$^2$]')
ax0.set_ylim([-0.3, 15]);
ax0.set_xlim([0, nInterval*exp3D_ms])
ax0.legend(["longitudinal (roll)","transverse (pitch & yaw)"])
ax0.figure.savefig(result_dir + '/single-MSD-rot.pdf')  

#%% Write to CSV
# MSD
for i in range(len(msd70_N)):
    MSD_70suc = np.array([xaxis*exp3D_ms, msd70_N[i],
                          0.5*(msd70_S1[i]+msd70_S2[i]),
                          msd70_R[i], 0.5*(msd70_P[i]+msd70_Y[i]),
                          msd70_NR[i]]).T
    fmt = ",".join(["%s"] + ["%10.6e"] * (MSD_70suc.shape[1]-1))
    # np.savetxt(result_dir_csv + "/MSD/MSD-70suc-" +str(i).zfill(2) + ".csv",
    np.savetxt(result_dir_csv + "test" + ".csv",
               MSD_70suc, fmt=fmt,
               header="lag time [sec], MSD-longitudinal [um^2]," +
                      "MSD-transversal [um^2],"+
                      "MSAD-longitudinal [rad^2], MSAD-transversal [rad^2]," +
                      "MLAD [um x rad]",
               comments='')
    
for i in range(len(msd50_N)):
    MSD_50suc = np.array([xaxis*exp3D_ms, msd50_N[i],
                          0.5*(msd50_S1[i]+msd50_S2[i]),
                          msd50_R[i], 0.5*(msd50_P[i]+msd50_Y[i]),
                          msd50_NR[i]]).T
    fmt = ",".join(["%s"] + ["%10.6e"] * (MSD_50suc.shape[1]-1))
    np.savetxt(result_dir_csv + "/MSD/MSD-50suc-" + str(i).zfill(2) + ".csv",
               MSD_50suc, fmt=fmt,
               header="lag time [sec], MSD-longitudinal [um^2]," +
                      "MSD-transversal [um^2],"+
                      "MSAD-longitudinal [rad^2], MSAD-transversal [rad^2]," +
                      "MLAD [um x rad]",
               comments='')

for i in range(len(msd40_N)):
    MSD_40suc = np.array([xaxis*exp3D_ms, msd40_N[i],
                          0.5*(msd40_S1[i]+msd40_S2[i]),
                          msd40_R[i], 0.5*(msd40_P[i]+msd40_Y[i]),
                          msd40_NR[i]]).T
    fmt = ",".join(["%s"] + ["%10.6e"] * (MSD_40suc.shape[1]-1))
    np.savetxt(result_dir_csv + "/MSD/MSD-40suc-" + str(i).zfill(2) + ".csv",
               MSD_40suc, fmt=fmt,
               header="lag time [sec], MSD-longitudinal [um^2]," +
                      "MSD-transversal [um^2],"+
                      "MSAD-longitudinal [rad^2], MSAD-transversal [rad^2]," +
                      "MLAD [um x rad]",
               comments='')
    
# Diffusion coefficients    
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

# 6DOF-tracking (figure 1)
# dispXYZ = np.array([xtrack*exp3D_ms, cm[:,0]-cm[0,0],
#                   cm[:,1]-cm[0,1], cm[:,2]-cm[0,2]]).T
# fmt = ",".join(["%s"] + ["%10.6e"] * (dispXYZ.shape[1]-1))
# np.savetxt(result_dir_csv + "Diffusion-coefficients/displacement-xyz.csv", dispXYZ, fmt=fmt,
#            header="time [sec], Delta x [um], Delta y [um], Delta z [um]",
#            comments='')

# dispAng = np.array([xtrack*exp3D_ms, EuAng[:,1]-EuAng[0,1],
#                   EuAng[:,0]-EuAng[0,0], EuAng[:,2]-EuAng[0,2]]).T
# fmt = ",".join(["%s"] + ["%10.6e"] * (dispAng.shape[1]-1))
# np.savetxt(result_dir_csv + "/Diffusion-coefficients/displacement-RollPitchYaw.csv", dispAng, fmt=fmt,
#            header="time [sec], Roll [rad], Pitch [rad], Yaw [rad]",
           # comments='')