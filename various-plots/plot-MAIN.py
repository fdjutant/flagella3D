#%% Import modules and files
import sys
sys.path.insert(0, './modules')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matmatrix import MSDfit
from scipy import optimize
import msd
import glob
import seaborn as sns
from matmatrix import BernieMatrix

# Compute 3D exposure time
pxum = 0.115; 
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
exp3D_ms = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 

# Input files
# path = r"C:\Users\labuser\Dropbox (ASU)\Research\DNA-Rotary-Motor\Helical-nanotubes\Light-sheet-OPM\Result-data"
path = r"D:\Dropbox (ASU)\Research\DNA-Rotary-Motor\Helical-nanotubes\Light-sheet-OPM\Result-data"
runNum = 'run-03'

xls70_h15 = glob.glob(path + '/20211022c_suc70_h15um/' + runNum + '/*.xlsx')
npy70_h15 = glob.glob(path + '/20211022c_suc70_h15um/' + runNum + '/*-angleCM.npy')
vec70_h15 = glob.glob(path + '/20211022c_suc70_h15um/' + runNum + '/*-vectorN.npy')
xls70_h30 = glob.glob(path + '/20211022d_suc70_h30um/' + runNum + '/*.xlsx')
npy70_h30 = glob.glob(path + '/20211022d_suc70_h30um/' + runNum + '/*-angleCM.npy')
vec70_h30 = glob.glob(path + '/20211022d_suc70_h30um/' + runNum + '/*-vectorN.npy')

xls50_h15 = glob.glob(path + '/20211018a_suc50_h15um/' + runNum + '/*.xlsx')
npy50_h15 = glob.glob(path + '/20211018a_suc50_h15um/' + runNum + '/*-angleCM.npy')
vec50_h15 = glob.glob(path + '/20211018a_suc50_h15um/' + runNum + '/*-vectorN.npy')
xls50_h30 = glob.glob(path + '/20211018b_suc50_h30um/' + runNum + '/*.xlsx')
npy50_h30 = glob.glob(path + '/20211018b_suc50_h30um/' + runNum + '/*-angleCM.npy')
vec50_h30 = glob.glob(path + '/20211018b_suc50_h30um/' + runNum + '/*-vectorN.npy')

xls40_h15 = glob.glob(path + '/20211022a_suc40_h15um/' + runNum + '/*.xlsx')
npy40_h15 = glob.glob(path + '/20211022a_suc40_h15um/' + runNum + '/*-angleCM.npy')
vec40_h15 = glob.glob(path + '/20211022a_suc40_h15um/' + runNum + '/*-vectorN.npy')
xls40_h30 = glob.glob(path + '/20211022b_suc40_h30um/' + runNum + '/*.xlsx')
npy40_h30 = glob.glob(path + '/20211022b_suc40_h30um/' + runNum + '/*-angleCM.npy')
vec40_h30 = glob.glob(path + '/20211022b_suc40_h30um/' + runNum + '/*-vectorN.npy')

vis70 = 673     # 70%(w/w) sucrose [mPa.s] (Quintas et al. 2005)
vis50 = 15.04   # 50%(w/w) sucrose [mPa.s] (Telis et al. 2005)
vis40 = 6.20    # 40%(w/w) sucrose [mPa.s] (Telis et al. 2005)

# Recompute diffusion coefficient from tracking
XLS70 = xls70_h15 + xls70_h30
NPY70 = npy70_h15 + npy70_h30
VEC70 = vec70_h15 + vec70_h30

XLS50 = xls50_h15 + xls50_h30
NPY50 = npy50_h15 + npy50_h30
VEC50 = vec50_h15 + vec50_h30

XLS40 = xls40_h15 + xls40_h30
NPY40 = npy40_h15 + npy40_h30
VEC40 = vec40_h15 + vec40_h30

# set number of interval computed and fitting points
nInterval = 50; xaxis = np.arange(1,nInterval+1)
Nfit = 10 # number of fitting points

# initialize arrays
msd70_N = []; msd70_S = []; msd70_S2 = [];
msd50_N = []; msd50_S = []; msd50_S2 = [];
msd40_N = []; msd40_S = []; msd40_S2 = [];
geo70_mean, geo70_std = (np.zeros([len(XLS70),3]) for _ in range(2))
geo50_mean, geo50_std = (np.zeros([len(XLS50),3]) for _ in range(2))
geo40_mean, geo40_std = (np.zeros([len(XLS40),3]) for _ in range(2))
fit70_N, fit70_S, fit70_S2, fit70_P, fit70_R, fit70_Y, fit70_NR = (np.zeros([len(XLS70)]) for _ in range(7))
fit50_N, fit50_S, fit50_S2, fit50_P, fit50_R, fit50_Y, fit50_NR = (np.zeros([len(XLS50)]) for _ in range(7))
fit40_N, fit40_S, fit40_S2, fit40_P, fit40_R, fit40_Y, fit40_NR = (np.zeros([len(XLS40)]) for _ in range(7))

#%% Go through every data sets
for j in range(len(XLS70)): 
    geo = pd.read_excel(XLS70[j], index_col=None).to_numpy()
    geo70_mean[j,:] = geo[0:3,1]    # geo: radius, length, pitch
    geo70_std[j,:] = geo[0:3,2]   
    EuAng, dirAng, cm = np.load(NPY70[j])  
    localAxes = np.load(VEC70[j])

    # MSD: mean square displacement
    fromMSD = msd.theMSD(len(cm), cm, EuAng[:,1],localAxes, exp3D_ms, nInterval)
    MSD_N, MSD_S, MSD_S2, MSD_NR = fromMSD.trans_combo_MSD()
    MSD_P = msd.regMSD(len(cm), EuAng[:,0], exp3D_ms, nInterval)
    MSD_R = msd.regMSD(len(cm), EuAng[:,1], exp3D_ms, nInterval)
    MSD_Y = msd.regMSD(len(cm), EuAng[:,2], exp3D_ms, nInterval)
    msd70_N.append(MSD_N); msd70_S.append(MSD_S); msd70_S2.append(MSD_S2);
    
    # Fit MSD with y = Const + B*x
    xtime = np.linspace(1,Nfit,Nfit)
    fit70_N[j],fitN_const  = optimize.curve_fit(MSDfit, xtime, MSD_N[0:Nfit])[0]
    fit70_S[j],fitS_const  = optimize.curve_fit(MSDfit, xtime,\
                        np.mean([MSD_S[0:Nfit],MSD_S[0:Nfit]],axis=0))[0]
    fit70_NR[j],fitNR_const = optimize.curve_fit(MSDfit, xtime, MSD_NR[0:Nfit])[0]
    fit70_P[j],fitP_const  = optimize.curve_fit(MSDfit, xtime, MSD_P[0:Nfit])[0]
    fit70_R[j],fitR_const  = optimize.curve_fit(MSDfit, xtime, MSD_R[0:Nfit])[0]
    fit70_Y[j],fitY_const  = optimize.curve_fit(MSDfit, xtime, MSD_Y[0:Nfit])[0]
    print(XLS70[j])
    
for j in range(len(XLS50)): 
    geo = pd.read_excel(XLS50[j], index_col=None).to_numpy()
    geo50_mean[j,:] = geo[0:3,1]    # geo: radius, length, pitch
    geo50_std[j,:] = geo[0:3,2]   
    EuAng, dirAng, cm = np.load(NPY50[j])  
    localAxes = np.load(VEC50[j])

    # MSD: mean square displacement
    fromMSD = msd.theMSD(len(cm), cm, EuAng[:,1],localAxes, exp3D_ms, nInterval)
    MSD_N, MSD_S, MSD_S2, MSD_NR = fromMSD.trans_combo_MSD()
    MSD_P = msd.regMSD(len(cm), EuAng[:,0], exp3D_ms, nInterval)
    MSD_R = msd.regMSD(len(cm), EuAng[:,1], exp3D_ms, nInterval)
    MSD_Y = msd.regMSD(len(cm), EuAng[:,2], exp3D_ms, nInterval)
    msd50_N.append(MSD_N); msd50_S.append(MSD_S); msd50_S2.append(MSD_S2);
    
    # Fit MSD with y = Const + B*x
    xtime = np.linspace(1,Nfit,Nfit)
    fit50_N[j],fitN_const  = optimize.curve_fit(MSDfit, xtime, MSD_N[0:Nfit])[0]
    fit50_S[j],fitS_const  = optimize.curve_fit(MSDfit, xtime,\
                        np.mean([MSD_S[0:Nfit],MSD_S[0:Nfit]],axis=0))[0]
    fit50_NR[j],fitNR_const = optimize.curve_fit(MSDfit, xtime, MSD_NR[0:Nfit])[0]
    fit50_P[j],fitP_const  = optimize.curve_fit(MSDfit, xtime, MSD_P[0:Nfit])[0]
    fit50_R[j],fitR_const  = optimize.curve_fit(MSDfit, xtime, MSD_R[0:Nfit])[0]
    fit50_Y[j],fitY_const  = optimize.curve_fit(MSDfit, xtime, MSD_Y[0:Nfit])[0]
    print(XLS50[j])

for j in range(len(XLS40)): 
    geo = pd.read_excel(XLS40[j], index_col=None).to_numpy()
    geo40_mean[j,:] = geo[0:3,1]    # geo: radius, length, pitch
    geo40_std[j,:] = geo[0:3,2]   
    EuAng, dirAng, cm = np.load(NPY40[j])  
    localAxes = np.load(VEC40[j])

    # MSD: mean square displacement
    fromMSD = msd.theMSD(len(cm), cm, EuAng[:,1],localAxes, exp3D_ms, nInterval)
    MSD_N, MSD_S, MSD_S2, MSD_NR = fromMSD.trans_combo_MSD()
    MSD_P = msd.regMSD(len(cm), EuAng[:,0], exp3D_ms, nInterval)
    MSD_R = msd.regMSD(len(cm), EuAng[:,1], exp3D_ms, nInterval)
    MSD_Y = msd.regMSD(len(cm), EuAng[:,2], exp3D_ms, nInterval)
    msd40_N.append(MSD_N); msd40_S.append(MSD_S); msd40_S2.append(MSD_S2);
    
    # Fit MSD with y = Const + B*x
    xtime = np.linspace(1,Nfit,Nfit)
    fit40_N[j],fitN_const  = optimize.curve_fit(MSDfit, xtime, MSD_N[0:Nfit])[0]
    fit40_S[j],fitS_const  = optimize.curve_fit(MSDfit, xtime,\
                        np.mean([MSD_S[0:Nfit],MSD_S[0:Nfit]],axis=0))[0]
    fit40_NR[j],fitNR_const = optimize.curve_fit(MSDfit, xtime, MSD_NR[0:Nfit])[0]
    fit40_P[j],fitP_const  = optimize.curve_fit(MSDfit, xtime, MSD_P[0:Nfit])[0]
    fit40_R[j],fitR_const  = optimize.curve_fit(MSDfit, xtime, MSD_R[0:Nfit])[0]
    fit40_Y[j],fitY_const  = optimize.curve_fit(MSDfit, xtime, MSD_Y[0:Nfit])[0]
    print(XLS40[j])
        
#%% Translation diffusion 
Dt70 = np.zeros([len(fit70_N),2])
Dt50 = np.zeros([len(fit50_N),2])
Dt40 = np.zeros([len(fit40_N),2])
Dt70[:,0] = fit70_N / (2*exp3D_ms); Dt70[:,1] = fit70_S / (2*exp3D_ms);
Dt50[:,0] = fit50_N / (2*exp3D_ms); Dt50[:,1] = fit50_S / (2*exp3D_ms); 
Dt40[:,0] = fit40_N / (2*exp3D_ms); Dt40[:,1] = fit40_S / (2*exp3D_ms); 

xaxis = ['Lengthwise', 'Sidewise']
xpos = np.arange(len(xaxis))
D70_mean = [np.mean(Dt70[:,0]), np.mean(Dt70[:,1])]
D70_std = [np.std(Dt70[:,0]), np.std(Dt70[:,1])]
D50_mean = [np.mean(Dt50[:,0]), np.mean(Dt50[:,1])]
D50_std = [np.std(Dt50[:,0]), np.std(Dt50[:,1])]
D40_mean = [np.mean(Dt40[:,0]), np.mean(Dt40[:,1])]
D40_std = [np.std(Dt40[:,0]), np.std(Dt40[:,1])]

width = 0.3
plt.rcParams.update({'font.size': 22})
fig04a, ax04a = plt.subplots(dpi=300, figsize=(10,6.2))
rects1 = ax04a.bar(xpos - width, D40_mean, width,\
                  label= r'$40\%~(n =\ $'+ str(len(Dt40)) + r'$)$',
                  yerr=D40_std/np.sqrt(len(Dt40)),\
                  align='center', alpha = 0.5, ecolor='black',
                  color='gray', hatch ='o', edgecolor= 'black', capsize=10)
rects2 = ax04a.bar(xpos, D50_mean, width,\
                  label= r'$50\%~(n =\ $'+ str(len(Dt50)) + r'$)$',
                  yerr=D50_std/np.sqrt(len(Dt50)),\
                  align='center', alpha = 0.5, ecolor='black',
                  color='gray', hatch ='x', edgecolor= 'black', capsize=10)
rects3 = ax04a.bar(xpos + width, D70_mean, width,\
                  label= r'$70\%~(n =\ $'+ str(len(Dt70)) + r'$)$',
                  yerr=D70_std/np.sqrt(len(Dt70)),\
                  align='center', alpha = 0.5, ecolor='black',
                  color='gray', hatch ='+', edgecolor= 'black',  capsize=10)
ax04a.set_ylabel(r'$D$ [$\mu m^2$/sec]')
# ax04a.set_title('Translation diffusion')
ax04a.set_xticks(xpos)
ax04a.set_xticklabels(xaxis)
ax04a.legend()
# ax04a.set_ylim([0, 0.6]);
fig04a.tight_layout()
# ax04a.figure.savefig(path + '/PDF/translation-bar.pdf')

# Swarm plot
sns.set_style("whitegrid")
data = [Dt40[:,0], Dt50[:,0], Dt70[:,0]]
xlabel = ["40%", "50%", "70%"]
ylabel = "D [um2/sec]"

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax = sns.swarmplot(data=[Dt40[:,0], Dt50[:,0], Dt70[:,0]], color=".25")
ax = sns.boxplot(data=[Dt40[:,0], Dt50[:,0], Dt70[:,0]])
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$D_\parallel$ [$\mu m^2$/sec]')
ax.set_xlabel(r'sucrose concentration (w/v)')
ax.set_ylim([0, 0.4]);
plt.show()

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax = sns.swarmplot(data=[Dt40[:,1], Dt50[:,1], Dt70[:,1]], color=".25")
ax = sns.boxplot(data=[Dt40[:,1], Dt50[:,1], Dt70[:,1]])
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$D_\perp$ [$\mu m^2$/sec]')
ax.set_xlabel(r'sucrose concentration (w/v)')
ax.set_ylim([0, 0.4]);
plt.show()

#%% Rotation diffusion
Dr70 = np.zeros([len(fit70_P),3])
Dr50 = np.zeros([len(fit50_P),3])
Dr40 = np.zeros([len(fit40_P),3])
Dr70[:,0] = fit70_P / (2*exp3D_ms)
Dr70[:,1] = fit70_R / (2*exp3D_ms)
Dr70[:,2] = fit70_Y / (2*exp3D_ms)
Dr50[:,0] = fit50_P / (2*exp3D_ms)
Dr50[:,1] = fit50_R / (2*exp3D_ms)
Dr50[:,2] = fit50_Y / (2*exp3D_ms) 
Dr40[:,0] = fit40_P / (2*exp3D_ms)
Dr40[:,1] = fit40_R / (2*exp3D_ms) 
Dr40[:,2] = fit40_Y / (2*exp3D_ms)

sns.set_style("whitegrid")
data = [Dt40[:,0], Dr50[:,0], Dt70[:,0]]
xlabel = ["40%", "50%", "70%"]
ylabel = "D [um2/sec]"

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax = sns.swarmplot(data=[Dr40[:,0], Dr50[:,0], Dr70[:,0]], color=".25")
ax = sns.boxplot(data=[Dr40[:,0], Dr50[:,0], Dr70[:,0]])
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$D_{pitch}$ [rad$^2$/sec]')
ax.set_xlabel(r'sucrose concentration (w/v)')
ax.set_ylim([0, 0.12]);
plt.show()

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax = sns.swarmplot(data=[Dr40[:,1], Dr50[:,1], Dr70[:,1]], color=".25")
ax = sns.boxplot(data=[Dr40[:,1], Dr50[:,1], Dr70[:,1]])
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$D_{roll}$ [rad$^2$/sec]')
ax.set_xlabel(r'sucrose concentration (w/v)')
# ax.set_ylim([0, 0.4]);
plt.show()

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax = sns.swarmplot(data=[Dr40[:,2], Dr50[:,2], Dr70[:,2]], color=".25")
ax = sns.boxplot(data=[Dr40[:,2], Dr50[:,2], Dr70[:,2]])
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$D_{yaw}$ [rad$^2$/sec]')
ax.set_xlabel(r'sucrose concentration (w/v)')
ax.set_ylim([0, 0.12]);
plt.show()

#%% Longitudinal displacement x rotation diffusion
Dtr70 = np.zeros([len(fit70_NR)])
Dtr50 = np.zeros([len(fit50_NR)])
Dtr40 = np.zeros([len(fit40_NR)])
Dtr70 = fit70_NR / (2*exp3D_ms)
Dtr50 = fit50_NR / (2*exp3D_ms)
Dtr40 = fit40_NR / (2*exp3D_ms)

sns.set_style("whitegrid")
data = [Dtr40, Dtr50, Dt70]
xlabel = ["40%", "50%", "70%"]
ylabel = "D [um2/sec]"

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax = sns.swarmplot(data=[Dtr40, Dtr50, Dtr70], color=".25")
ax = sns.boxplot(data=[Dtr40, Dtr50, Dtr70])
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$D_{\parallel x roll}$ [rad$^2$/sec]')
ax.set_xlabel(r'sucrose concentration (w/v)')
ax.set_ylim([-0.2, 0.2]);
plt.show()

#%% A, B, D
A70, B70, D70 = BernieMatrix(Dt70[:,0]*1e-12,Dr70[:,1],abs(Dtr70)*1e-6)
A50, B50, D50 = BernieMatrix(Dt50[:,0]*1e-12,Dr50[:,1],abs(Dtr50)*1e-6)
A40, B40, D40 = BernieMatrix(Dt40[:,0]*1e-12,Dr40[:,1],abs(Dtr40)*1e-6)

sns.set_style("whitegrid")
data = [Dtr40, Dtr50, Dt70]
xlabel = ["40%", "50%", "70%"]
ylabel = "D [um2/sec]"

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax = sns.swarmplot(data=[A40, A50, A70], color=".25")
ax = sns.boxplot(data=[A40, A50, A70])
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$A$ [N.s/m]')
ax.set_xlabel(r'sucrose concentration (w/v)')
# ax.set_ylim([-0.2, 0.2]);
plt.show()

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax = sns.swarmplot(data=[-B40, -B50, -B70], color=".25")
ax = sns.boxplot(data=[-B40, -B50, -B70])
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$B$ [N.s]')
ax.set_xlabel(r'sucrose concentration (w/v)')
# ax.set_ylim([-0.2, 0.2]);
plt.show()

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(dpi=300, figsize=(10,6.2))
ax = sns.swarmplot(data=[D40, D50, D70], color=".25")
ax = sns.boxplot(data=[D40, D50, D70])
ax.set_xticklabels(xlabel)
ax.set_ylabel(r'$D$ [N.s.m]')
ax.set_xlabel(r'sucrose concentration (w/v)')
# ax.set_ylim([-0.2, 0.2]);
plt.show()
