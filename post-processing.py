import numpy as np
import glob
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})

path = r"../../Result-data/"
xls70_h15 = glob.glob(path + '20211004f_70suc_h15um/done/*-10per.xlsx')
xls70_h30 = glob.glob(path + '20211004g_70suc_h30um/done/*-10per.xlsx')

xls50_h15 = glob.glob(path + '20211018a_suc50_h15um/done/*-10per.xlsx')
xls50_h30 = glob.glob(path + '20211018b_suc50_h30um/done/*-10per.xlsx')

xls40_h15 = glob.glob(path + '20211022a_suc40_h15um/done/*-10per.xlsx')
xls40_h30 = glob.glob(path + '20211022b_suc40_h30um/done/*-10per.xlsx')

#%% Go through Excel sheets
# 40% sucrose, h = (15, 30) um from wall
data40_h15 = np.ndarray([len(xls40_h15),8,4],dtype=object)
for j in range(len(xls40_h15)):
    data40_h15[j] = pd.read_excel(xls40_h15[j], index_col=None).to_numpy()
geo40_h15_mean = data40_h15[:,0:3,1]    # geo: radius, length, pitch
geo40_h15_std = data40_h15[:,0:3,2]     
Dt40_h15 = data40_h15[:,3,1:4];         # translation diffusion
Dr40_h15 = data40_h15[:,4,1:4];         # rotation diffusion
Dc40_h15 = data40_h15[:,5,1];           # combo diffusion
ABD40_h15 = data40_h15[:,6,1:4];        # propulsion matrix A, B, Ds
ABD40adj_h15 = data40_h15[:,7,1:4];    # propulsion matrix A, B, Ds (adjusted)

data40_h30 = np.ndarray([len(xls40_h30),8,4],dtype=object)
for j in range(len(xls40_h30)):
    data40_h30[j] = pd.read_excel(xls40_h30[j], index_col=None).to_numpy()
geo40_h30_mean = data40_h30[:,0:3,1]    # geo: radius, length, pitch
geo40_h30_std = data40_h30[:,0:3,2]     
Dt40_h30 = data40_h30[:,3,1:4];         # translation diffusion
Dr40_h30 = data40_h30[:,4,1:4];         # rotation diffusion
Dc40_h30 = data40_h30[:,5,1];           # combo diffusion
ABD40_h30 = data40_h30[:,6,1:4];        # propulsion matrix A, B, Ds
ABD40adj_h30 = data40_h30[:,7,1:4];    # propulsion matrix A, B, Ds (adjusted)

# 50% sucrose, h = (15, 30) um from wall
data50_h15 = np.ndarray([len(xls50_h15),8,4],dtype=object)
for j in range(len(xls50_h15)):
    data50_h15[j] = pd.read_excel(xls50_h15[j], index_col=None).to_numpy()
geo50_h15_mean = data50_h15[:,0:3,1]    # geo: radius, length, pitch
geo50_h15_std = data50_h15[:,0:3,2]     
Dt50_h15 = data50_h15[:,3,1:4];         # translation diffusion
Dr50_h15 = data50_h15[:,4,1:4];         # rotation diffusion
Dc50_h15 = data50_h15[:,5,1];           # combo diffusion
ABD50_h15 = data50_h15[:,6,1:4];        # propulsion matrix A, B, Ds
ABD50adj_h15 = data50_h15[:,7,1:4];    # propulsion matrix A, B, Ds (adjusted)

data50_h30 = np.ndarray([len(xls50_h30),8,4],dtype=object)
for j in range(len(xls50_h30)):
    data50_h30[j] = pd.read_excel(xls50_h30[j], index_col=None).to_numpy()
geo50_h30_mean = data50_h30[:,0:3,1]    # geo: radius, length, pitch
geo50_h30_std = data50_h30[:,0:3,2]     
Dt50_h30 = data50_h30[:,3,1:4];         # translation diffusion
Dr50_h30 = data50_h30[:,4,1:4];         # rotation diffusion
Dc50_h30 = data50_h30[:,5,1];           # combo diffusion
ABD50_h30 = data50_h30[:,6,1:4];        # propulsion matrix A, B, Ds
ABD50adj_h30 = data50_h30[:,7,1:4];    # propulsion matrix A, B, Ds (adjusted)

# 70% sucrose, h = (15, 30) um from wall
data70_h15 = np.ndarray([len(xls70_h15),8,4],dtype=object)
for j in range(len(xls70_h15)):
    data70_h15[j] = pd.read_excel(xls70_h15[j], index_col=None).to_numpy()
geo70_h15_mean = data70_h15[:,0:3,1]    # geo: radius, length, pitch
geo70_h15_std = data70_h15[:,0:3,2]     
Dt70_h15 = data70_h15[:,3,1:4];         # translation diffusion
Dr70_h15 = data70_h15[:,4,1:4];         # rotation diffusion
Dc70_h15 = data70_h15[:,5,1];           # combo diffusion
ABD70_h15 = data70_h15[:,6,1:4];        # propulsion matrix A, B, Ds
ABD70adj_h15 = data70_h15[:,7,1:4];    # propulsion matrix A, B, Ds

data70_h30 = np.ndarray([len(xls70_h30),8,4],dtype=object)
for j in range(len(xls70_h30)):
    data70_h30[j] = pd.read_excel(xls70_h30[j], index_col=None).to_numpy()
geo70_h30_mean = data70_h30[:,0:3,1]    # geo: radius, length, pitch
geo70_h30_std = data70_h30[:,0:3,2]     
Dt70_h30 = data70_h30[:,3,1:4];         # translation diffusion
Dr70_h30 = data70_h30[:,4,1:4];         # rotation diffusion
Dc70_h30 = data70_h30[:,5,1];           # combo diffusion
ABD70_h30 = data70_h30[:,6,1:4];        # propulsion matrix A, B, Ds
ABD70adj_h30 = data70_h30[:,7,1:4];    # propulsion matrix A, B, Ds

#%% Translation diffusion - individual
suc_per = str(70)
geo_h15_mean = geo70_h15_mean
geo_h30_mean = geo70_h30_mean
geo_h15_std = geo70_h15_std
geo_h30_std = geo70_h30_std
Dt_h15 = Dt70_h15
Dt_h30 = Dt70_h30

plt.rcParams.update({'font.size': 10})
# diffusion_parallel = f(length)
fig01,ax01 = plt.subplots(dpi=300, figsize=(6,2))
ax01.errorbar(geo_h15_mean[:,1],Dt_h15[:,0],\
              xerr=geo_h15_std[:,1]/np.sqrt(len(geo_h15_std)),
              marker="o",linestyle = 'None',alpha=0.5,capsize=2)
ax01.errorbar(geo_h30_mean[:,1],Dt_h30[:,0],\
              xerr=geo_h30_std[:,1]/np.sqrt(len(geo_h30_std)),\
              marker="o",linestyle = 'None',alpha=0.5,capsize=2)
ax01.set_xlabel(r'length [$\mu m$]');
ax01.set_title('Translation diffusion parallel (' + suc_per + '\% sucrose)')
ax01.set_ylabel(r'$D_\parallel$ [$\mu m^2$/sec]')
ax01.legend(["$h = 15~\mu m$", "$h = 30~\mu m$"])
ax01.set_xlim([4, 12]);

# diffusion_perpendicular1 = f(length)
fig02,ax02 = plt.subplots(dpi=300, figsize=(6,2))
ax02.errorbar(geo_h15_mean[:,1],Dt_h15[:,1],\
              xerr=geo_h15_std[:,1]/np.sqrt(len(geo_h15_std)),\
              marker="o",linestyle = 'None',alpha=0.5,capsize=2)
ax02.errorbar(geo_h30_mean[:,1],Dt_h30[:,1],\
              xerr=geo_h30_std[:,1]/np.sqrt(len(geo_h30_std)),\
              marker="o",linestyle = 'None',alpha=0.5,capsize=2)
ax02.set_xlabel(r'length [$\mu m$]');
ax02.set_title('Translation diffusion perpendicular 1 (' + suc_per + '\% sucrose)')
ax02.set_ylabel(r'$D_\perp$ [$\mu m^2$/sec]')
ax02.legend(["$h = 15~\mu m$", "$h = 30~\mu m$"])
ax02.set_xlim([4, 12]);

# diffusion_perpendicular2 = f(length)
fig03,ax03 = plt.subplots(dpi=300, figsize=(6,2))
ax03.errorbar(geo_h15_mean[:,1],Dt_h15[:,2],\
              xerr=geo_h15_std[:,1]/np.sqrt(len(geo_h15_std)),\
              marker="o",linestyle = 'None',alpha=0.5,capsize=2)
ax03.errorbar(geo_h30_mean[:,1],Dt_h30[:,2],\
              xerr=geo_h30_std[:,1]/np.sqrt(len(geo_h30_std)),\
              marker="o",linestyle = 'None',alpha=0.5,capsize=2)
ax03.set_xlabel(r'length [$\mu m$]');
ax03.set_title('Translation diffusion perpendicular 2 (' + suc_per + '\% sucrose)')
ax03.set_ylabel(r'$D_\perp$ [$\mu m^2$/sec]')
ax03.legend(["$h = 15~\mu m$", "$h = 30~\mu m$"])
ax03.set_xlim([4, 12]);

#%% diffusion parallel VS perpendiculars
# 70% sucrose
xaxis = ['Parallel', 'Perpendicular 1', 'Perpendicular 2']
xpos = np.arange(len(xaxis))
D70_h15_mean = [np.mean(Dt70_h15[:,0]), np.mean(Dt70_h15[:,1]), np.mean(Dt70_h15[:,2])]
D70_h15_std = [np.std(Dt70_h15[:,0]), np.std(Dt70_h15[:,1]), np.std(Dt70_h15[:,2])]
D70_h30_mean = [np.mean(Dt70_h30[:,0]), np.mean(Dt70_h30[:,1]), np.mean(Dt70_h30[:,2])]
D70_h30_std = [np.std(Dt70_h30[:,0]), np.std(Dt70_h30[:,1]), np.std(Dt70_h30[:,2])]

width = 0.35
plt.rcParams.update({'font.size': 22})
fig04a, ax04a = plt.subplots(dpi=300, figsize=(10,6.2))
rects2 = ax04a.bar(xpos + width/2, D70_h15_mean, width,\
                  label=r'$h = 15\pm3~\mu m~(n =\ $'+ str(len(Dt70_h15)) + r'$)$',
                  yerr=D70_h15_std/np.sqrt(len(Dt70_h15)),\
                  align='center', alpha = 0.5, ecolor='black', capsize=10)
rects1 = ax04a.bar(xpos - width/2, D70_h30_mean, width,\
                  label=r'$h = 30\pm3~\mu m~(n =\ $'+ str(len(Dt70_h30)) + r'$)$',
                  yerr=D70_h30_std/np.sqrt(len(Dt70_h30)),\
                  align='center', alpha = 0.5, ecolor='black', capsize=10)
ax04a.set_ylabel(r'$D$ [$\mu m^2$/sec]')
ax04a.set_title('Translation diffusion (' + str(70) + '\% sucrose)')
ax04a.set_xticks(xpos)
ax04a.set_xticklabels(xaxis)
ax04a.legend()
ax04a.set_ylim([0, 3.5]);
fig04a.tight_layout()

# 50% sucrose
xaxis = ['Parallel', 'Perpendicular 1', 'Perpendicular 2']
xpos = np.arange(len(xaxis))
D50_h15_mean = [np.mean(Dt50_h15[:,0]), np.mean(Dt50_h15[:,1]), np.mean(Dt50_h15[:,2])]
D50_h15_std = [np.std(Dt50_h15[:,0]), np.std(Dt50_h15[:,1]), np.std(Dt50_h15[:,2])]
D50_h30_mean = [np.mean(Dt50_h30[:,0]), np.mean(Dt50_h30[:,1]), np.mean(Dt50_h30[:,2])]
D50_h30_std = [np.std(Dt50_h30[:,0]), np.std(Dt50_h30[:,1]), np.std(Dt50_h30[:,2])]

width = 0.35
plt.rcParams.update({'font.size': 22})
fig04b, ax04b = plt.subplots(dpi=300, figsize=(10,6.2))
rects2 = ax04b.bar(xpos + width/2, D50_h15_mean, width,\
                  label=r'$h = 15\pm3~\mu m~(n =\ $'+ str(len(Dt50_h15)) + r'$)$',
                  yerr=D50_h15_std/np.sqrt(len(Dt50_h15)),\
                  align='center', alpha = 0.5, ecolor='black', capsize=10)
rects1 = ax04b.bar(xpos - width/2, D50_h30_mean, width,\
                  label=r'$h = 30\pm3~\mu m~(n =\ $'+ str(len(Dt50_h30)) + r'$)$',
                  yerr=D50_h30_std/np.sqrt(len(Dt50_h30)),\
                  align='center', alpha = 0.5, ecolor='black', capsize=10)
ax04b.set_ylabel(r'$D$ [$\mu m^2$/sec]')
ax04b.set_title('Translation diffusion (' + str(50) + '\% sucrose)')
ax04b.set_xticks(xpos)
ax04b.set_xticklabels(xaxis)
ax04b.legend()
ax04b.set_ylim([0, 3.5]);
fig04b.tight_layout()

# 40% sucrose
xaxis = ['Parallel', 'Perpendicular 1', 'Perpendicular 2']
xpos = np.arange(len(xaxis))
D40_h15_mean = [np.mean(Dt40_h15[:,0]), np.mean(Dt40_h15[:,1]), np.mean(Dt40_h15[:,2])]
D40_h15_std = [np.std(Dt40_h15[:,0]), np.std(Dt40_h15[:,1]), np.std(Dt40_h15[:,2])]
D40_h30_mean = [np.mean(Dt40_h30[:,0]), np.mean(Dt40_h30[:,1]), np.mean(Dt40_h30[:,2])]
D40_h30_std = [np.std(Dt40_h30[:,0]), np.std(Dt40_h30[:,1]), np.std(Dt40_h30[:,2])]

width = 0.35
plt.rcParams.update({'font.size': 22})
fig04c, ax04c = plt.subplots(dpi=300, figsize=(10,6.2))
rects2 = ax04c.bar(xpos + width/2, D40_h15_mean, width,\
                  label=r'$h = 15\pm3~\mu m~(n =\ $'+ str(len(Dt40_h15)) + r'$)$',
                  yerr=D40_h15_std/np.sqrt(len(Dt40_h15)),\
                  align='center', alpha = 0.5, ecolor='black', capsize=10)
rects1 = ax04c.bar(xpos - width/2, D40_h30_mean, width,\
                  label=r'$h = 30\pm3~\mu m~(n =\ $'+ str(len(Dt40_h30)) + r'$)$',
                  yerr=D40_h30_std/np.sqrt(len(Dt40_h30)),\
                  align='center', alpha = 0.5, ecolor='black', capsize=10)
ax04c.set_ylabel(r'$D$ [$\mu m^2$/sec]')
ax04c.set_title('Translation diffusion (' + str(40) + '\% sucrose)')
ax04c.set_xticks(xpos)
ax04c.set_xticklabels(xaxis)
ax04c.legend()
ax04c.set_ylim([0, 3.5]);
fig04c.tight_layout()

#%% Compute p-value
# print(stats.ttest_ind(Dt70_h15[:,0],Dt70_h15[:,1]))
# print(stats.ttest_ind(Dt70_h15[:,0],Dt70_h15[:,2]))
t70_h15, p70_h15 = stats.ttest_ind(Dt70_h15[:,0],np.concatenate((Dt70_h15[:,1],Dt70_h15[:,2]),axis=0)) 
print('t, p for h = 15 um: ',t70_h15, p70_h15/2)

# print(stats.ttest_ind(Dt70_h30[:,0],Dt70_h30[:,1]))
# print(stats.ttest_ind(Dt70_h30[:,0],Dt70_h30[:,2]))
# print(stats.ttest_ind(Dt70_h30[:,0],np.concatenate((Dt70_h30[:,1],Dt70_h30[:,2]),axis=0)) )
t70_h30, p70_h30 = stats.ttest_ind(Dt70_h30[:,0],np.concatenate((Dt70_h30[:,1],Dt70_h30[:,2]),axis=0)) 
print('t, p for h = 30 um: ',t70_h30, p70_h30/2)

#%% Rotational diffusion
suc_per = str(40)
geo_h15_mean = geo40_h15_mean
geo_h30_mean = geo40_h30_mean
geo_h15_std = geo40_h15_std
geo_h30_std = geo40_h30_std
Dr_h15 = Dr40_h15
Dr_h30 = Dr40_h30

# pitch
plt.rcParams.update({'font.size': 10})
fig05a,ax05a = plt.subplots(dpi=300, figsize=(6,2))
ax05a.errorbar(geo_h15_mean[:,1],Dr_h15[:,0],\
              xerr=geo_h15_std[:,1]/np.sqrt(len(geo_h15_std)),
              marker="s",linestyle = 'None',alpha=0.5,capsize=2)
ax05a.errorbar(geo_h30_mean[:,1],Dr_h30[:,0],\
              xerr=geo_h30_std[:,1]/np.sqrt(len(geo_h30_std)),
              marker="s",linestyle = 'None',alpha=0.5,capsize=2)
ax05a.set_xlabel(r'length [$\mu m$]');
ax05a.set_title('Pitch diffusion (' + suc_per + '\% sucrose)')
ax05a.set_ylabel(r'$Dr$ [$rad^2$/sec]')
ax05a.legend(["$h = 15~\mu m$", "$h = 30~\mu m$"])
ax05a.set_xlim([4, 12]);

# roll
plt.rcParams.update({'font.size': 10})
fig05b,ax05b = plt.subplots(dpi=300, figsize=(6,2))
ax05b.errorbar(geo_h15_mean[:,1],Dr_h15[:,1],\
              xerr=geo_h15_std[:,1]/np.sqrt(len(geo_h15_std)),
              marker="s",linestyle = 'None',alpha=0.5,capsize=2)
ax05b.errorbar(geo_h30_mean[:,1],Dr_h30[:,1],\
              xerr=geo_h30_std[:,1]/np.sqrt(len(geo_h30_std)),
              marker="s",linestyle = 'None',alpha=0.5,capsize=2)
ax05b.set_xlabel(r'length [$\mu m$]');
ax05b.set_title('Roll diffusion (' + suc_per + '\% sucrose)')
ax05b.set_ylabel(r'$Dr$ [$rad^2$/sec]')
ax05b.legend(["$h = 15~\mu m$", "$h = 30~\mu m$"])
ax05b.set_xlim([4, 12]);

# yaw
plt.rcParams.update({'font.size': 10})
fig05c,ax05c = plt.subplots(dpi=300, figsize=(6,2))
ax05c.errorbar(geo_h15_mean[:,1],Dr_h15[:,2],\
              xerr=geo_h15_std[:,1]/np.sqrt(len(geo_h15_std)),
              marker="s",linestyle = 'None',alpha=0.5,capsize=2)
ax05c.errorbar(geo_h30_mean[:,1],Dr_h30[:,1],\
              xerr=geo_h30_std[:,1]/np.sqrt(len(geo_h30_std)),
              marker="s",linestyle = 'None',alpha=0.5,capsize=2)
ax05c.set_xlabel(r'length [$\mu m$]');
ax05c.set_title('Yaw diffusion (' + suc_per + '\% sucrose)')
ax05c.set_ylabel(r'$Dr$ [$rad^2$/sec]')
ax05c.legend(["$h = 15~\mu m$", "$h = 30~\mu m$"])
ax05c.set_xlim([4, 12]);

#%% rotational diffusion: pitch, roll, yaw

# 70% sucrose
xaxis = ['Pitch', 'Roll', 'Yaw']
xpos = np.arange(len(xaxis))
Dr70_h15_mean = [np.mean(Dr70_h15[:,0]), np.mean(Dr70_h15[:,1]), np.mean(Dr70_h15[:,2])]
Dr70_h15_std = [np.std(Dr70_h15[:,0]), np.std(Dr70_h15[:,1]), np.std(Dr70_h15[:,2])]
Dr70_h30_mean = [np.mean(Dr70_h30[:,0]), np.mean(Dr70_h30[:,1]), np.mean(Dr70_h30[:,2])]
Dr70_h30_std = [np.std(Dr70_h30[:,0]), np.std(Dr70_h30[:,1]), np.std(Dr70_h30[:,2])]

width = 0.35
plt.rcParams.update({'font.size': 22})
fig06a, ax06a = plt.subplots(dpi=300, figsize=(10,6.2))
rects2 = ax06a.bar(xpos + width/2, Dr70_h15_mean, width,\
                  label=r'$h = 15\pm3~\mu m~(n =\ $'+ str(len(Dr70_h15)) + r'$)$',
                  yerr=Dr70_h15_std/np.sqrt(len(Dr70_h15)),\
                  align='center', alpha = 0.5, ecolor='black', capsize=10)
rects1 = ax06a.bar(xpos - width/2, Dr70_h30_mean, width,\
                  label=r'$h = 15\pm3~\mu m~(n =\ $'+ str(len(Dr70_h30)) + r'$)$',
                  yerr=Dr70_h30_std/np.sqrt(len(Dr70_h30)),\
                  align='center', alpha = 0.5, ecolor='black', capsize=10)
ax06a.set_ylabel(r'$Dr$ [$rad^2$/sec]')
ax06a.set_title('Rotational diffusion (' + str(70) + '\% sucrose)')
ax06a.set_xticks(xpos)
ax06a.set_xticklabels(xaxis)
ax06a.legend()
ax06a.set_ylim([0, 25]);
fig06a.tight_layout()

# 50% sucrose
xaxis = ['Pitch', 'Roll', 'Yaw']
xpos = np.arange(len(xaxis))
Dr50_h15_mean = [np.mean(Dr50_h15[:,0]), np.mean(Dr50_h15[:,1]), np.mean(Dr50_h15[:,2])]
Dr50_h15_std = [np.std(Dr50_h15[:,0]), np.std(Dr50_h15[:,1]), np.std(Dr50_h15[:,2])]
Dr50_h30_mean = [np.mean(Dr50_h30[:,0]), np.mean(Dr50_h30[:,1]), np.mean(Dr50_h30[:,2])]
Dr50_h30_std = [np.std(Dr50_h30[:,0]), np.std(Dr50_h30[:,1]), np.std(Dr50_h30[:,2])]

width = 0.35
plt.rcParams.update({'font.size': 22})
fig06b, ax06b = plt.subplots(dpi=300, figsize=(10,6.2))
rects2 = ax06b.bar(xpos + width/2, Dr50_h15_mean, width,\
                  label=r'$h = 15\pm3~\mu m~(n =\ $'+ str(len(Dr50_h15)) + r'$)$',
                  yerr=Dr50_h15_std/np.sqrt(len(Dr50_h15)),\
                  align='center', alpha = 0.5, ecolor='black', capsize=10)
rects1 = ax06b.bar(xpos - width/2, Dr50_h30_mean, width,\
                  label=r'$h = 15\pm3~\mu m~(n =\ $'+ str(len(Dr50_h30)) + r'$)$',
                  yerr=Dr50_h30_std/np.sqrt(len(Dr50_h30)),\
                  align='center', alpha = 0.5, ecolor='black', capsize=10)
ax06b.set_ylabel(r'$Dr$ [$rad^2$/sec]')
ax06b.set_title('Rotational diffusion (' + str(50) + '\% sucrose)')
ax06b.set_xticks(xpos)
ax06b.set_xticklabels(xaxis)
ax06b.legend()
ax06b.set_ylim([0, 25]);
fig06b.tight_layout()

# 40% sucrose
xaxis = ['Pitch', 'Roll', 'Yaw']
xpos = np.arange(len(xaxis))
Dr40_h15_mean = [np.mean(Dr40_h15[:,0]), np.mean(Dr40_h15[:,1]), np.mean(Dr40_h15[:,2])]
Dr40_h15_std = [np.std(Dr40_h15[:,0]), np.std(Dr40_h15[:,1]), np.std(Dr40_h15[:,2])]
Dr40_h30_mean = [np.mean(Dr40_h30[:,0]), np.mean(Dr40_h30[:,1]), np.mean(Dr40_h30[:,2])]
Dr40_h30_std = [np.std(Dr40_h30[:,0]), np.std(Dr40_h30[:,1]), np.std(Dr40_h30[:,2])]

width = 0.35
plt.rcParams.update({'font.size': 22})
fig06c, ax06c = plt.subplots(dpi=300, figsize=(10,6.2))
rects2 = ax06c.bar(xpos + width/2, Dr40_h15_mean, width,\
                  label=r'$h = 15\pm3~\mu m~(n =\ $'+ str(len(Dr40_h15)) + r'$)$',
                  yerr=Dr40_h15_std/np.sqrt(len(Dr40_h15)),\
                  align='center', alpha = 0.5, ecolor='black', capsize=10)
rects1 = ax06c.bar(xpos - width/2, Dr40_h30_mean, width,\
                  label=r'$h = 30\pm3~\mu m~(n =\ $'+ str(len(Dr40_h30)) + r'$)$',
                  yerr=Dr40_h30_std/np.sqrt(len(Dr40_h30)),\
                  align='center', alpha = 0.5, ecolor='black', capsize=10)
ax06c.set_ylabel(r'$Dr$ [$rad^2$/sec]')
ax06c.set_title('Rotational diffusion (' + str(40) + '\% sucrose)')
ax06c.set_xticks(xpos)
ax06c.set_xticklabels(xaxis)
ax06c.legend()
ax06c.set_ylim([0, 25]);
fig06c.tight_layout()

#%% A, B, D
suc_per = str(40)
geo_h15_mean = geo40_h15_mean
geo_h30_mean = geo40_h30_mean
geo_h15_std = geo40_h15_std
geo_h30_std = geo40_h30_std
ABD_h15 = ABD40_h15
ABD_h30 = ABD40_h30

# A, B, D = f(length)
plt.rcParams.update({'font.size': 10})
fig07a,ax07a = plt.subplots(dpi=300, figsize=(6,2))
ax07a.errorbar(geo_h15_mean[:,1],abs(ABD_h15[:,0]),\
              xerr=geo_h15_std[:,1]/np.sqrt(len(geo_h15_std)),
              marker="s",linestyle = 'None',alpha=0.5,capsize=2)
ax07a.errorbar(geo_h30_mean[:,1],abs(ABD_h30[:,0]),\
              xerr=geo_h30_std[:,1]/np.sqrt(len(geo_h30_std)),
              marker="s",linestyle = 'None',alpha=0.5,capsize=2)
ax07a.set_xlabel(r'length [$\mu m$]');
ax07a.set_title('Resistence matrix A (' + suc_per + '\% sucrose)')
ax07a.set_ylabel(r'$A [N.s/m]$')
ax07a.legend(["$h = 15~\mu m$", "$h = 30~\mu m$"])

plt.rcParams.update({'font.size': 10})
fig07b,ax07b = plt.subplots(dpi=300, figsize=(6,2))
ax07b.errorbar(geo_h15_mean[:,1],abs(ABD_h15[:,1]),\
              xerr=geo_h15_std[:,1]/np.sqrt(len(geo_h15_std)),
              marker="s",linestyle = 'None',alpha=0.5,capsize=2)
ax07b.errorbar(geo_h30_mean[:,1],abs(ABD_h30[:,1]),\
              xerr=geo_h30_std[:,1]/np.sqrt(len(geo_h30_std)),
              marker="s",linestyle = 'None',alpha=0.5,capsize=2)
ax07b.set_xlabel(r'length [$\mu m$]');
ax07b.set_title('Resistence matrix B (' + suc_per + '\% sucrose)')
ax07b.set_ylabel(r'B [N.s]')
ax07b.legend(["$h = 15~\mu m$", "$h = 30~\mu m$"])

plt.rcParams.update({'font.size': 10})
fig07c,ax07c = plt.subplots(dpi=300, figsize=(6,2))
ax07c.errorbar(geo_h15_mean[:,1],ABD_h15[:,2],\
              xerr=geo_h15_std[:,1]/np.sqrt(len(geo_h15_std)),
              marker="s",linestyle = 'None',alpha=0.5,capsize=2)
ax07c.errorbar(geo_h30_mean[:,1],ABD_h30[:,2],\
              xerr=geo_h30_std[:,1]/np.sqrt(len(geo_h30_std)),
              marker="s",linestyle = 'None',alpha=0.5,capsize=2)
ax07c.set_xlabel(r'length [$\mu m$]');
ax07c.set_title('Resistence matrix D (' + suc_per + '\% sucrose)')
ax07c.set_ylabel(r'D [N.s.m]')
ax07c.legend(["$h = 15~\mu m$", "$h = 30~\mu m$"])

#%% ABD = f(distance)

# Propulsion matrix A
xaxis = ['40\%','50\%']
xpos = np.arange(len(xaxis))
A_h15_mean = [np.mean(ABD40_h15[:,0]), np.mean(ABD50_h15[:,0])]
A_h15_std = [np.std(ABD40_h15[:,0])/np.sqrt(len(ABD40_h15[:,0])),\
             np.std(ABD50_h15[:,0])/np.sqrt(len(ABD50_h15[:,0]))]
A_h30_mean = [np.mean(ABD40_h30[:,0]), np.mean(ABD50_h30[:,0])]
A_h30_std = [np.std(ABD40_h30[:,0])/np.sqrt(len(ABD40_h30[:,0])),\
             np.std(ABD50_h30[:,0])/np.sqrt(len(ABD50_h30[:,0]))]

width = 0.35
plt.rcParams.update({'font.size': 22})
fig06a, ax06a = plt.subplots(dpi=300, figsize=(10,6.2))
rects2 = ax06a.bar(xpos + width/2, A_h15_mean, width,\
                  label=r'$h = 15\pm3~\mu m~(n =\ $'+ str(len(ABD70_h15)) + r'$)$',
                  yerr=A_h15_std,\
                  align='center', alpha = 0.5, ecolor='black', capsize=10)
rects1 = ax06a.bar(xpos - width/2, A_h30_mean, width,\
                  label=r'$h = 15\pm3~\mu m~(n =\ $'+ str(len(ABD70_h30)) + r'$)$',
                  yerr=A_h30_std,\
                  align='center', alpha = 0.5, ecolor='black', capsize=10)
ax06a.set_ylabel(r'$A [N.s/m]$')
ax06a.set_title('Propulsion matrix A')
ax06a.set_xticks(xpos)
ax06a.set_xticklabels(xaxis)
ax06a.legend()
# ax06a.set_ylim([0, 25]);
fig06a.tight_layout()

# Propulsion matrix A (adjusted)
xaxis = ['40\%','50\%']
xpos = np.arange(len(xaxis))
Aadj_h15_mean = [np.mean(ABD40adj_h15[:,0]), np.mean(ABD50adj_h15[:,0])]
Aadj_h15_std = [np.std(ABD40adj_h15[:,0])/np.sqrt(len(ABD40adj_h15[:,0])),\
             np.std(ABD50adj_h15[:,0])/np.sqrt(len(ABD50adj_h15[:,0]))]
Aadj_h30_mean = [np.mean(ABD40adj_h30[:,0]), np.mean(ABD50adj_h30[:,0])]
Aadj_h30_std = [np.std(ABD40adj_h30[:,0])/np.sqrt(len(ABD40adj_h30[:,0])),\
             np.std(ABD50adj_h30[:,0])/np.sqrt(len(ABD50adj_h30[:,0]))]

width = 0.35
plt.rcParams.update({'font.size': 22})
fig06a, ax06a = plt.subplots(dpi=300, figsize=(10,6.2))
rects2 = ax06a.bar(xpos + width/2, Aadj_h15_mean, width,\
                  label=r'$h = 15\pm3~\mu m~(n =\ $'+ str(len(ABD70_h15)) + r'$)$',
                  yerr=Aadj_h15_std,\
                  align='center', alpha = 0.5, ecolor='black', capsize=10)
rects1 = ax06a.bar(xpos - width/2, Aadj_h30_mean, width,\
                  label=r'$h = 15\pm3~\mu m~(n =\ $'+ str(len(ABD70_h30)) + r'$)$',
                  yerr=Aadj_h30_std,\
                  align='center', alpha = 0.5, ecolor='black', capsize=10)
ax06a.set_ylabel(r'$A [N.s/m]$')
ax06a.set_title('Propulsion matrix A (adjusted to the water viscosity)')
ax06a.set_xticks(xpos)
ax06a.set_xticklabels(xaxis)
ax06a.legend()
# ax06a.set_ylim([0, 25]);
fig06a.tight_layout()

#%% Compute propulsion matrix

# h = 15 um
all_A_h15 = np.concatenate([ABD40adj_h15[:,0],ABD50adj_h15[:,0]]);
all_B_h15 = abs(np.concatenate([ABD40adj_h15[:,1],ABD50adj_h15[:,1]]));
all_D_h15 = np.concatenate([ABD40adj_h15[:,2],ABD50adj_h15[:,2]]);

A_mean_h15 = np.mean(all_A_h15)
B_mean_h15 = np.mean(all_B_h15)
D_mean_h15 = np.mean(all_D_h15)

A_std_h15 = np.std(all_A_h15)/np.sqrt(len(all_A_h15))
B_std_h15 = np.std(all_B_h15)/np.sqrt(len(all_B_h15))
D_std_h15 = np.std(all_D_h15)/np.sqrt(len(all_D_h15))

print('A, B, D: ', A_mean_h15, B_mean_h15, D_mean_h15, ' with n = ', len(all_A_h15))
print('A, B, D (std): ', A_std_h15, B_std_h15, D_std_h15)

# h = 30 um
all_A_h30 = np.concatenate([ABD40adj_h30[:,0],ABD50adj_h30[:,0]]);
all_B_h30 = abs(np.concatenate([ABD40adj_h30[:,1],ABD50adj_h30[:,1]]));
all_D_h30 = np.concatenate([ABD40adj_h30[:,2],ABD50adj_h30[:,2]]);

A_mean_h30 = np.mean(all_A_h30)
B_mean_h30 = np.mean(all_B_h30)
D_mean_h30 = np.mean(all_D_h30)

A_std_h30 = np.std(all_A_h30)/np.sqrt(len(all_A_h30))
B_std_h30 = np.std(all_B_h30)/np.sqrt(len(all_B_h30))
D_std_h30 = np.std(all_D_h30)/np.sqrt(len(all_D_h30))

print('A, B, D: ', A_mean_h30, B_mean_h30, D_mean_h30, ' with n = ', len(all_A_h30))
print('A, B, D (std): ', A_std_h30, B_std_h30, D_std_h30)

#%%
xaxis = [r'$h = 15~\mu m$', r'$h = 30~\mu m$']
xpos = np.arange(len(xaxis))
ABD70_h15_mean = [np.mean(ABD70_h15[:,0]), np.mean(abs(ABD70_h15[:,1])), np.mean(ABD70_h15[:,2])]
ABD70_h15_std = [np.std(ABD70_h15[:,0]), np.std(ABD70_h15[:,1]), np.std(ABD70_h15[:,2])]
ABD70_h30_mean = [np.mean(ABD70_h30[:,0]), np.mean(abs(ABD70_h30[:,1])), np.mean(ABD70_h30[:,2])]
ABD70_h30_std = [np.std(ABD70_h30[:,0]), np.std(ABD70_h30[:,1]), np.std(ABD70_h30[:,2])]

A70_mean = [ABD70_h15_mean[0], ABD70_h30_mean[0]]
A70_std = [ABD70_h15_std[0]/np.sqrt(len(ABD70_h15)), ABD70_h30_std[0]/np.sqrt(len(ABD70_h30))]
plt.rcParams.update({'font.size': 22})
fig08, ax08 = plt.subplots(dpi=300, figsize=(10,6.2))
ax08.bar(xpos, A70_mean,yerr=A70_std,\
                  align='center', alpha = 0.5, ecolor='black', capsize=10)
ax08.set_ylabel(r'$Dr$ [$rad^2$/sec]')
ax08.set_title('Resistence matrix A (70\% sucrose)')
ax08.set_xticks(xpos)
ax08.set_xticklabels(xaxis)
fig08.tight_layout()

B70_mean = [ABD70_h15_mean[1], ABD70_h30_mean[1]]
B70_std = [ABD70_h15_std[1]/np.sqrt(len(ABD70_h15)), ABD70_h30_std[1]/np.sqrt(len(ABD70_h30))]
plt.rcParams.update({'font.size': 22})
fig09, ax09 = plt.subplots(dpi=300, figsize=(10,6.2))
ax09.bar(xpos, B70_mean,yerr=B70_std,\
                  align='center', alpha = 0.5, ecolor='black', capsize=10)
ax09.set_ylabel(r'B [N.s]')
ax09.set_title('Resistence matrix B (70\% sucrose)')
ax09.set_xticks(xpos)
ax09.set_xticklabels(xaxis)
fig09.tight_layout()


D70_mean = [ABD70_h15_mean[2], ABD70_h30_mean[2]]
D70_std = [ABD70_h15_std[2]/np.sqrt(len(ABD70_h15)), ABD70_h30_std[2]/np.sqrt(len(ABD70_h30))]
plt.rcParams.update({'font.size': 22})
fig10, ax10 = plt.subplots(dpi=300, figsize=(10,6.2))
ax10.bar(xpos, D70_mean,yerr=D70_std,\
                  align='center', alpha = 0.5, ecolor='black', capsize=10)
ax10.set_ylabel(r'D [N.s.m]')
ax10.set_title('Resistence matrix D (70\% sucrose)')
ax10.set_xticks(xpos)
ax10.set_xticklabels(xaxis)
fig10.tight_layout()


# ABD_adjusted = f(distance)
xaxis = [r'$h = 15~\mu m$', r'$h = 30~\mu m$']
xpos = np.arange(len(xaxis))
ABD70_h15_mean = [np.mean(ABD70_h15[:,0]), np.mean(abs(ABD70_h15[:,1])), np.mean(ABD70_h15[:,2])]
ABD70_h15_std = [np.std(ABD70_h15[:,0]), np.std(ABD70_h15[:,1]), np.std(ABD70_h15[:,2])]
ABD70_h30_mean = [np.mean(ABD70_h30[:,0]), np.mean(abs(ABD70_h30[:,1])), np.mean(ABD70_h30[:,2])]
ABD70_h30_std = [np.std(ABD70_h30[:,0]), np.std(ABD70_h30[:,1]), np.std(ABD70_h30[:,2])]

A70_mean = [ABD70_h15_mean[0], ABD70_h30_mean[0]]
A70_std = [ABD70_h15_std[0]/np.sqrt(len(ABD70_h15)), ABD70_h30_std[0]/np.sqrt(len(ABD70_h30))]
plt.rcParams.update({'font.size': 22})
fig08, ax08 = plt.subplots(dpi=300, figsize=(10,6.2))
ax08.bar(xpos, A70_mean,yerr=A70_std,\
                  align='center', alpha = 0.5, ecolor='black', capsize=10)
ax08.set_ylabel(r'$Dr$ [$rad^2$/sec]')
ax08.set_title('Resistence matrix A (70\% sucrose)')
ax08.set_xticks(xpos)
ax08.set_xticklabels(xaxis)
fig08.tight_layout()

B70_mean = [ABD70_h15_mean[1], ABD70_h30_mean[1]]
B70_std = [ABD70_h15_std[1]/np.sqrt(len(ABD70_h15)), ABD70_h30_std[1]/np.sqrt(len(ABD70_h30))]
plt.rcParams.update({'font.size': 22})
fig09, ax09 = plt.subplots(dpi=300, figsize=(10,6.2))
ax09.bar(xpos, B70_mean,yerr=B70_std,\
                  align='center', alpha = 0.5, ecolor='black', capsize=10)
ax09.set_ylabel(r'B [N.s]')
ax09.set_title('Resistence matrix B (70\% sucrose)')
ax09.set_xticks(xpos)
ax09.set_xticklabels(xaxis)
fig09.tight_layout()


# D70_mean = [ABD70_h15_mean[2], ABD70_h30_mean[2]]
# D70_std = [ABD70_h15_std[2]/np.sqrt(len(ABD70_h15)), ABD70_h30_std[2]/np.sqrt(len(ABD70_h30))]
# plt.rcParams.update({'font.size': 22})
# fig10, ax10 = plt.subplots(dpi=300, figsize=(10,6.2))
# ax10.bar(xpos, D70_mean,yerr=D70_std,\
#                   align='center', alpha = 0.5, ecolor='black', capsize=10)
# ax10.set_ylabel(r'D [N.s.m]')
# ax10.set_title('Resistence matrix D (70\% sucrose)')
# ax10.set_xticks(xpos)
# ax10.set_xticklabels(xaxis)
# fig10.tight_layout()