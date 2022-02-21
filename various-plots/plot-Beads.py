#%% Import modules and files
import sys
sys.path.insert(0, './modules')
import glob
import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matmatrix import fitCDF_diff, gauss_cdf, gauss_pdf
from pathlib import Path

this_file_dir = os.path.dirname(os.path.abspath('./'))
path = os.path.join(this_file_dir,
                'DNA-Rotary-Motor', 'Helical-nanotubes',
                'Light-sheet-OPM', 'Result-data',
                '20220113_Beads_100nm')
whichDataSet = '\DataTrack*200nm*.csv'
result_dir = os.path.join(os.path.dirname(path), 'PDF')
result_dir_csv = os.path.join(Path('../6-DOF-Flagella').resolve())

files40Water = glob.glob(os.path.join(path, '40suc-water') + whichDataSet)
files50Water = glob.glob(os.path.join(path, '50suc-water') + whichDataSet)
files70Water = glob.glob(os.path.join(path, '70suc-water') + whichDataSet)
files40 = glob.glob(os.path.join(path, '40suc') + whichDataSet)
files50 = glob.glob(os.path.join(path, '50suc') + whichDataSet)
files70 = glob.glob(os.path.join(path, '70suc') + whichDataSet)
filesWater = glob.glob(os.path.join(path, 'water') + whichDataSet)
filesBuffer = glob.glob(os.path.join(path, 'buffer') + whichDataSet)
filesBuffer2 = glob.glob(os.path.join(path, 'buffer-2') + whichDataSet)
cutoff = 5
cutoffWater = 7.5

#%% import data
def extractD(fileList, cutoff):
    Dcoeff = []
    for j in range(len(fileList)):
        dataAll = np.concatenate(np.array(pd.DataFrame(pd.read_csv(fileList[j]),
                                 columns=['Diffusion coeff. (um^2/s)'])))
        Dcoeff.append(dataAll)
    Dcoeff = np.concatenate(np.array(Dcoeff,dtype=object))
    Dcoeff = Dcoeff[Dcoeff < cutoff]
    return Dcoeff

D40Water = extractD(files40Water, cutoff=5)
D50Water = extractD(files50Water, cutoff=5)
D70Water = extractD(files70Water, cutoff=5)
D40 = extractD(files40, cutoff=5)
D50 = extractD(files50, cutoff=5)
D70 = extractD(files70, cutoff=5)
DWater = extractD(filesWater, cutoff=5)
DBuffer = extractD(filesBuffer, cutoff=7.5)
DBuffer2 = extractD(filesBuffer2, cutoff=7.5)

#%% Extract to CSV   
Diff40 = D40
fmt = ",".join(["%s"] + ["%10.6e"] * (1-1))
np.savetxt(result_dir_csv + "/EPI-beads/Beads-40suc.csv", Diff40, fmt=fmt,
           header="Diffusion coefficients [um^2/sec]", 
           comments='')    

Diff50 = D50
fmt = ",".join(["%s"] + ["%10.6e"] * (1-1))
np.savetxt(result_dir_csv + "/EPI-beads/Beads-50suc.csv", Diff50, fmt=fmt,
           header="Diffusion coefficients [um^2/sec]", 
           comments='')    

Diff70 = D70
fmt = ",".join(["%s"] + ["%10.6e"] * (1-1))
np.savetxt(result_dir_csv + "/EPI-beads/Beads-70suc.csv", Diff70, fmt=fmt,
           header="Diffusion coefficients [um^2/sec]", 
           comments='')    

DiffWater = DWater
fmt = ",".join(["%s"] + ["%10.6e"] * (1-1))
np.savetxt(result_dir_csv + "/EPI-beads/Beads-Water.csv", DiffWater, fmt=fmt,
           header="Diffusion coefficients [um^2/sec]", 
           comments='')    


#%% Fit the diffusion coefficients    
amp40, mean40, sigma40 = fitCDF_diff(D40)
amp50, mean50, sigma50 = fitCDF_diff(D50)
amp70, mean70, sigma70 = fitCDF_diff(D70)

amp40Water, mean40Water, sigma40Water = fitCDF_diff(D40Water)
amp50Water, mean50Water, sigma50Water = fitCDF_diff(D50Water)
amp70Water, mean70Water, sigma70Water = fitCDF_diff(D70Water)

ampWater, meanWater, sigmaWater = fitCDF_diff(DWater)
ampBuffer, meanBuffer, sigmaBuffer = fitCDF_diff(DBuffer)
ampBuffer2, meanBuffer2, sigmaBuffer2 = fitCDF_diff(DBuffer2)

xplot = np.linspace(0,cutoff,1000, endpoint=False)
y40 = gauss_cdf(xplot, amp40, mean40, sigma40)
y50 = gauss_cdf(xplot, amp50, mean50, sigma50)
y70 = gauss_cdf(xplot, amp70, mean70, sigma70)

y40Water = gauss_cdf(xplot, amp40Water, mean40Water, sigma40Water)
y50Water = gauss_cdf(xplot, amp50Water, mean50Water, sigma50Water)
y70Water = gauss_cdf(xplot, amp70Water, mean70Water, sigma70Water)

xplotWater = np.linspace(0,cutoffWater,1000, endpoint=False)
yWater = gauss_cdf(xplotWater, ampWater, meanWater, sigmaWater)
yBuffer = gauss_cdf(xplotWater, ampBuffer, meanBuffer, sigmaBuffer)
yBuffer2 = gauss_cdf(xplotWater, ampBuffer2, meanBuffer2, sigmaBuffer2)

ypdf40 = gauss_pdf(xplot, amp40, mean40, sigma40)
ypdf50 = gauss_pdf(xplot, amp50, mean50, sigma50)
ypdf70 = gauss_pdf(xplot, amp70, mean70, sigma70)

ypdf40Water = gauss_pdf(xplot, amp40Water, mean40Water, sigma40Water)
ypdf50Water = gauss_pdf(xplot, amp50Water, mean50Water, sigma50Water)
ypdf70Water = gauss_pdf(xplot, amp70Water, mean70Water, sigma70Water)

ypdfWater = gauss_pdf(xplotWater, ampWater, meanWater, sigmaWater)
ypdfBuffer = gauss_pdf(xplotWater, ampBuffer, meanBuffer, sigmaBuffer)
ypdfBuffer2 = gauss_pdf(xplotWater, ampBuffer2, meanBuffer2, sigmaBuffer2)


#%% plot PDF: MBT
downSamplingBy = 1
plt.rcParams.update({'font.size': 15})
fig1,ax1 = plt.subplots(dpi=600, figsize=(6,5))

ax1.plot(xplotWater, ypdfWater,'C3--', alpha=0.5)
ax1.plot(xplotWater, ypdfBuffer2,'C3', alpha=0.5)
ax1.hist(DWater[::downSamplingBy], bins='fd', density=True,
         color='C3', alpha=0.3, label='_nolegend_')
ax1.hist(DBuffer2[::downSamplingBy], bins='fd', density=True,
         color='C3', alpha=0.3, label='_nolegend_')

ax1.plot(xplot, ypdf40,'C0', alpha=0.5)
ax1.plot(xplot, ypdf50,'C1', alpha=0.5)
ax1.plot(xplot, ypdf70,'C2', alpha=0.5)
ax1.hist(D40[::downSamplingBy], bins='fd', density=True,
         color='C0', alpha=0.3, label='_nolegend_')
ax1.hist(D50[::downSamplingBy], bins='fd', density=True,
         color='C1', alpha=0.3, label='_nolegend_')
ax1.hist(D70[::downSamplingBy], bins='fd', density=True,
         color='C2', alpha=0.3, label='_nolegend_')
ax1.set_xlabel(r'Diffusion coefficients [$\mu \rm{m}^2$/sec]');
ax1.set_ylabel(r'Probability density')
ax1.set_xlim([0, cutoffWater])
ax1.legend(['Water (n = ' + str(np.round(len(DWater)/10**6,2)) + 'M)',
            # '0%-buffer (n = ' + str(np.round(len(DBuffer2)/10**6,2)) + 'M)',
            '40%-buffer (n = ' + str(np.round(len(D40)/10**6,2)) + 'M)',
            '50%-buffer (n = ' + str(np.round(len(D50)/10**6,2)) + 'M)',
            '70%-buffer (n = ' + str(np.round(len(D70)/10**6,2)) + 'M)'],
            prop={'size': 10})

#%% plot PDF: Buffer vs No buffer
# 40% sucrose
downSamplingBy = 1
plt.rcParams.update({'font.size': 15})
fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))

ax1.plot(xplotWater, ypdfWater,'C3--', alpha=0.5)
ax1.plot(xplotWater, ypdfBuffer2,'C3', alpha=0.5)
ax1.hist(DWater[::downSamplingBy], bins='fd', density=True,
         color='C3', alpha=0.3, label='_nolegend_')
ax1.hist(DBuffer2[::downSamplingBy], bins='fd', density=True,
         color='C3', alpha=0.3, label='_nolegend_')

ax1.plot(xplot, ypdf40,'C0', alpha=0.5)
ax1.plot(xplot, ypdf40Water,'C0--', alpha=0.3)
ax1.hist(D40[::downSamplingBy], bins='fd', density=True,
          color='C0', alpha=0.3, label='_nolegend_')
ax1.hist(D40Water[::downSamplingBy], bins='fd', density=True,
          color='C0', alpha=0.3)
ax1.set_xlabel(r'Diffusion coefficients [$\mu \rm{m}^2$/sec]');
ax1.set_ylabel(r'Probability density')
ax1.set_xlim([0, cutoffWater])
ax1.legend(['Water (n = ' + str(np.round(len(DWater)/10**6,2)) + 'M)',
            '0%-buffer (n = ' + str(np.round(len(DBuffer2)/10**6,2)) + 'M)',
            '40%-no-buffer (n = ' + str(np.round(len(D40Water)/10**6,2)) + 'M)',
            '40%-buffer (n = ' + str(np.round(len(D40)/10**6,2)) + 'M)'],
            prop={'size': 10})

# 50% sucrose
downSamplingBy = 1
plt.rcParams.update({'font.size': 15})
fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))

ax1.plot(xplotWater, ypdfWater,'C3--', alpha=0.5)
ax1.plot(xplotWater, ypdfBuffer2,'C3', alpha=0.5)
ax1.hist(DWater[::downSamplingBy], bins='fd', density=True,
         color='C3', alpha=0.3, label='_nolegend_')
ax1.hist(DBuffer2[::downSamplingBy], bins='fd', density=True,
         color='C3', alpha=0.3, label='_nolegend_')

ax1.plot(xplot, ypdf50,'C1', alpha=0.5)
ax1.plot(xplot, ypdf50Water,'C1--', alpha=0.5)
ax1.hist(D50[::downSamplingBy], bins='fd', density=True,
         color='C1', alpha=0.3, label='_nolegend_')
ax1.hist(D50Water[::downSamplingBy], bins='fd', density=True,
         color='C1', alpha=0.3)
ax1.set_xlabel(r'Diffusion coefficients [$\mu \rm{m}^2$/sec]');
ax1.set_ylabel(r'Probability density')
ax1.set_xlim([0, cutoffWater])
ax1.legend(['Water (n = ' + str(np.round(len(DWater)/10**6,2)) + 'M)',
            '0%-buffer (n = ' + str(np.round(len(DBuffer2)/10**6,2)) + 'M)',
            '50%-no-buffer (n = ' + str(np.round(len(D50Water)/10**6,2)) + 'M)',
            '50%-buffer (n = ' + str(np.round(len(D50)/10**6,2)) + 'M)'],
            prop={'size': 10})

# 70% sucrose
downSamplingBy = 1
plt.rcParams.update({'font.size': 15})
fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))

ax1.plot(xplotWater, ypdfWater,'C3--', alpha=0.5)
ax1.plot(xplotWater, ypdfBuffer2,'C3', alpha=0.5)
ax1.hist(DWater[::downSamplingBy], bins='fd', density=True,
         color='C3', alpha=0.3, label='_nolegend_')
ax1.hist(DBuffer2[::downSamplingBy], bins='fd', density=True,
         color='C3', alpha=0.3, label='_nolegend_')

ax1.plot(xplot, ypdf70,'C2', alpha=0.5)
ax1.plot(xplot, ypdf70Water,'C2--', alpha=0.5)
ax1.hist(D70[::downSamplingBy], bins='fd', density=True,
         color='C2', alpha=0.3, label='_nolegend_')
ax1.hist(D70Water[::downSamplingBy], bins='fd', density=True,
         color='C2', alpha=0.3)
ax1.set_xlabel(r'Diffusion coefficients [$\mu \rm{m}^2$/sec]');
ax1.set_ylabel(r'Probability density')
ax1.set_xlim([0, cutoffWater])
ax1.legend(['Water (n = ' + str(np.round(len(DWater)/10**6,2)) + 'M)',
            '0%-buffer (n = ' + str(np.round(len(DBuffer2)/10**6,2)) + 'M)',
            '70%-no-buffer (n = ' + str(np.round(len(D70Water)/10**6,2)) + 'M)',
            '70%-buffer (n = ' + str(np.round(len(D70)/10**6,2)) + 'M)'],
            prop={'size': 10})

#%% Plot in buffer
downSamplingBy = 1
plt.rcParams.update({'font.size': 15})
fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))

ax1.plot(xplot, ypdf40,'C0', alpha=0.3)
ax1.plot(xplot, ypdf50,'C0', alpha=0.5)
ax1.plot(xplot, ypdf70,'C0', alpha=0.7)
ax1.hist(D40[::downSamplingBy], bins='fd', density=True,
         color='C0', alpha=0.3)
ax1.hist(D50[::downSamplingBy], bins='fd', density=True,
         color='C0', alpha=0.5)
ax1.hist(D70[::downSamplingBy], bins='fd', density=True,
         color='C0', alpha=0.7)

ax1.plot(xplot, ypdf40Water,'C1', alpha=0.3)
ax1.plot(xplot, ypdf50Water,'C1', alpha=0.5)
ax1.plot(xplot, ypdf70Water,'C1', alpha=0.7)
ax1.hist(D40Water[::downSamplingBy], bins='fd', density=True,
         color='C1', alpha=0.3)
ax1.hist(D50Water[::downSamplingBy], bins='fd', density=True,
         color='C1', alpha=0.5)
ax1.hist(D70Water[::downSamplingBy], bins='fd', density=True,
         color='C1', alpha=0.7)

ax1.plot(xplot, ypdfWater,'C2', alpha=0.3)
ax1.plot(xplot, ypdfBuffer,'C2', alpha=0.5)
ax1.plot(xplot, ypdfBuffer2,'C2', alpha=0.7)
ax1.hist(DWater[::downSamplingBy], bins='fd', density=True,
         color='C2', alpha=0.3)
ax1.hist(DBuffer[::downSamplingBy], bins='fd', density=True,
         color='C2', alpha=0.5)
ax1.hist(DBuffer2[::downSamplingBy], bins='fd', density=True,
         color='C2', alpha=0.7)

ax1.set_xlabel(r'Diffusion coefficients [$\mu \rm{m}^2$/sec]');
ax1.set_ylabel(r'Probability density')
ax1.set_xlim([0, cutoff])
# ax1.legend(['Water (n ~ ' + str(np.round(len(DWater)/10**6,2)) + 'M)',
#             'Buffer (n ~ ' + str(np.round(len(DBuffer2)/10**6,2)) + 'M)',
#             '40% (n ~ ' + str(np.round(len(D40)/10**6,2)) + 'M)',
#             '50% (n ~ ' + str(np.round(len(D50)/10**6,2)) + 'M)',
#             '50%-water (n ~ ' + str(np.round(len(D50Water)/10**6,2)) + 'M)',
#             '70% (n ~ ' + str(np.round(len(D70)/10**6,2)) + 'M)'])

#%% compute viscosity
def vis(diff):
    kB = 1.381e-23
    T = 273 + 20            # temperature in Kelvin
    r = (99 / 2) * 1e-9     # bead size: 99-nm diameter
    viscosity = 1e3 * kB * T / (6 * np.pi * diff*1e-12 * r) 
    return viscosity

visWater = vis(meanWater); sigmaVisWater = vis(meanWater+sigmaWater)
visBuffer = vis(meanBuffer); sigmaVisBuffer = vis(meanBuffer+sigmaBuffer)
visBuffer2 = vis(meanBuffer2); sigmaVisBuffer2 = vis(meanBuffer2+sigmaBuffer2)

vis40 = vis(mean40); sigmaVis40 = vis(mean40+sigma40)
vis50 = vis(mean50); sigmaVis50 = vis(mean50+sigma50)
vis70 = vis(mean70); sigmaVis70 = vis(mean70+sigma70)

vis40Water = vis(mean40Water)
sigmaVis40Water = vis(mean40Water+sigma40Water)
vis50Water = vis(mean50Water)
sigmaVis50Water = vis(mean50Water+sigma50Water)
vis70Water = vis(mean70Water)
sigmaVis70Water = vis(mean70Water+sigma70Water)

# Error bar 
mean_vis = [visBuffer2, vis40, vis50, vis70,
            visWater, vis40Water, vis50Water, vis70Water]
std_vis = [sigmaVisBuffer2/2,
           sigmaVis40/2, sigmaVis50/2, sigmaVis70/2,
           sigmaVisWater/2, 
           sigmaVis40Water/2, sigmaVis50Water/2, sigmaVis70Water/2]

xlabel = ["water", "40%-NB","50%-NB","70%-NB",
          "0%", "40%","50%","70%"]

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(dpi=300, figsize=(20,6.2))
ax.errorbar(xlabel, mean_vis, yerr=std_vis, marker="_", markersize=50,
            color='k', linestyle="none", capsize=10)
# ax.set_xticklabels(xlabel)
ax.set_title('Viscosity')
ax.set_ylabel(r'$\eta$ [mPa$\cdot$sec]')
ax.set_xlabel(r'solvent')
ax.set_ylim([0, 4.5])
plt.show()
# ax.figure.savefig(result_dir + '/vis-all.pdf')

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(dpi=300, figsize=(12,6.2))
ax.errorbar(xlabel[4:9], mean_vis[4:9], yerr=std_vis[4:9],
            marker="_", markersize=50,
            color='k', linestyle="none", capsize=10)
# ax.set_xticklabels(xlabel)
ax.set_title('Viscosity')
ax.set_ylabel(r'$\eta$ [mPa$\cdot$sec]')
ax.set_xlabel(r'%(w/w) sucrose in test tube')
ax.set_ylim([0, 4.5])
plt.show()
ax.figure.savefig(result_dir + '/vis-MBT.pdf')


#%% Plot CDF
downSamplingBy = 1
# plt.rcParams.update({'font.size': 15})
# fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))
# ax1.plot(xplot, yWater,'C0', alpha=0.5)
# ax1.plot(xplot, yBuffer,'C1', alpha=0.5)
# ax1.plot(xplot, yBuffer2,'C1', alpha=0.5)
# ax1.plot(xplot, y40,'C2', alpha=0.5)
# ax1.plot(xplot, y50,'C3', alpha=0.5)
# ax1.plot(xplot, y70,'C4', alpha=0.5)

# ax1.plot(xplot, y40Water,'k', alpha=0.5)
# ax1.plot(xplot, y50Water,'k', alpha=0.5)
# ax1.plot(xplot, y70Water,'k', alpha=0.5)

# ax1.plot(np.sort(DWater)[::downSamplingBy],
#          np.linspace(0,1,len(DWater[::downSamplingBy]),endpoint=False),
#          'C0o', ms=1, alpha=0.1)
# # ax1.plot(np.sort(DBuffer)[::downSamplingBy],
# #          np.linspace(0,1,len(DBuffer[::downSamplingBy]),endpoint=False),
# #          'C1o', ms=1, alpha=0.1)
# ax1.plot(np.sort(DBuffer2)[::downSamplingBy],
#          np.linspace(0,1,len(DBuffer2[::downSamplingBy]),endpoint=False),
#          'C1o', ms=1, alpha=0.1)
# ax1.plot(np.sort(D40)[::downSamplingBy],
#          np.linspace(0,1,len(D40[::downSamplingBy]),endpoint=False),
#          'C2o', ms=1, alpha=0.1)
# ax1.plot(np.sort(D50)[::downSamplingBy],
#          np.linspace(0,1,len(D50[::downSamplingBy]),endpoint=False),
#          'C3o', ms=1, alpha=0.1)
# ax1.plot(np.sort(D50Water)[::downSamplingBy],
#          np.linspace(0,1,len(D50Water[::downSamplingBy]),endpoint=False),
#          'C3o', ms=1, alpha=0.1)
# ax1.plot(np.sort(D70)[::downSamplingBy],
#          np.linspace(0,1,len(D70[::downSamplingBy]),endpoint=False),
#          'C4o', ms=1, alpha=0.1)
# ax1.legend(['Water (n ~ ' + str(np.round(len(DWater)/10**6,2)) + 'M)',
#             'Buffer (n ~ ' + str(np.round(len(DBuffer2)/10**6,2)) + 'M)',
#             '40% (n ~ ' + str(np.round(len(D40)/10**6,2)) + 'M)',
#             '50% (n ~ ' + str(np.round(len(D50)/10**6,2)) + 'M)',
#             '70% (n ~ ' + str(np.round(len(D70)/10**6,2)) + 'M)'])
# ax1.set_xlim([0, cutoff])