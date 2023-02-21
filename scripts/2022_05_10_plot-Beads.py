"""
Determine viscosity of mixture from data acquired on separate sample of diameter = 100nm beads prepared in
the same way as the samples
"""
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

this_file_dir = os.path.join(os.path.dirname(os.path.abspath("./")),
                            'Dropbox (ASU)','Research')
path = os.path.join(this_file_dir,
                'DNA-Rotary-Motor', 'Helical-nanotubes',
                'Light-sheet-OPM', 'Result-data',
                '20220516_TS_100nm')
result_dir = os.path.join(path, 'plot_beads')
result_dir_csv = os.path.join(Path('../6-DOF-Flagella').resolve())

files40 = glob.glob(os.path.join(path, '40suc') + '\DataTracking*.csv')
files50 = glob.glob(os.path.join(path, '50suc') + '\Track*.csv')
files70 = glob.glob(os.path.join(path, '70suc') + '\Track*.csv')
filesBuffer = glob.glob(os.path.join(path, 'buffer') + '\Track*.csv')
cutoff = 5
cutoffBuffer = 7.5

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

D40 = extractD(files40, cutoff=5)
D50 = extractD(files50, cutoff=5)
D70 = extractD(files70, cutoff=5)
DBuffer = extractD(filesBuffer, cutoff=7.5)

#%% Extract to CSV   
# Diff40 = D40
# fmt = ",".join(["%s"] + ["%10.6e"] * (1-1))
# np.savetxt(result_dir_csv + "/EPI-beads/Beads-40suc.csv", Diff40, fmt=fmt,
#            header="Diffusion coefficients [um^2/sec]", 
#            comments='')    

# Diff50 = D50
# fmt = ",".join(["%s"] + ["%10.6e"] * (1-1))
# np.savetxt(result_dir_csv + "/EPI-beads/Beads-50suc.csv", Diff50, fmt=fmt,
#            header="Diffusion coefficients [um^2/sec]", 
#            comments='')    

# Diff70 = D70
# fmt = ",".join(["%s"] + ["%10.6e"] * (1-1))
# np.savetxt(result_dir_csv + "/EPI-beads/Beads-70suc.csv", Diff70, fmt=fmt,
#            header="Diffusion coefficients [um^2/sec]", 
#            comments='')    

# DiffBuffer = DBuffer
# fmt = ",".join(["%s"] + ["%10.6e"] * (1-1))
# np.savetxt(result_dir_csv + "/EPI-beads/Beads-Water.csv", DiffBuffer, fmt=fmt,
#            header="Diffusion coefficients [um^2/sec]", 
#            comments='')    


#%% Fit the diffusion coefficients    
amp40, mean40, sigma40 = fitCDF_diff(D40)
amp50, mean50, sigma50 = fitCDF_diff(D50)
amp70, mean70, sigma70 = fitCDF_diff(D70)

ampBuffer, meanBuffer, sigmaBuffer = fitCDF_diff(DBuffer)

xplot = np.linspace(0,cutoff,1000, endpoint=False)
y40 = gauss_cdf(xplot, amp40, mean40, sigma40)
y50 = gauss_cdf(xplot, amp50, mean50, sigma50)
y70 = gauss_cdf(xplot, amp70, mean70, sigma70)

xplotBuffer = np.linspace(0,cutoffBuffer,1000, endpoint=False)
yBuffer = gauss_cdf(xplotBuffer, ampBuffer, meanBuffer, sigmaBuffer)

ypdf40 = gauss_pdf(xplot, amp40, mean40, sigma40)
ypdf50 = gauss_pdf(xplot, amp50, mean50, sigma50)
ypdf70 = gauss_pdf(xplot, amp70, mean70, sigma70)

ypdfBuffer = gauss_pdf(xplotBuffer, ampBuffer, meanBuffer, sigmaBuffer)

#%% plot PDF: MBT
downSamplingBy = 1
plt.rcParams.update({'font.size': 15})
fig1,ax1 = plt.subplots(dpi=600, figsize=(10,6.2))

ax1.plot(xplotBuffer, ypdfBuffer,'C3', alpha=0.5)
ax1.hist(DBuffer[::downSamplingBy], bins='fd', density=True,
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
ax1.set_xlim([0, cutoffBuffer])
ax1.legend(['Buffer (n = ' + str(np.round(len(DBuffer)/10**6,2)) + 'M)',
            # '0%-buffer (n = ' + str(np.round(len(DBuffer2)/10**6,2)) + 'M)',
            '40%-buffer (n = ' + str(np.round(len(D40)/10**6,2)) + 'M)'],
            # '50%-buffer (n = ' + str(np.round(len(D50)/10**6,2)) + 'M)',
            # '70%-buffer (n = ' + str(np.round(len(D70)/10**6,2)) + 'M)',
            prop={'size': 10})
ax1.figure.savefig(result_dir + '/pdf-diff.pdf')


#%% compute viscosity
def vis(diff):
    kB = 1.381e-23
    T = 273 + 20            # temperature in Kelvin
    r = (99 / 2) * 1e-9     # bead size: 99-nm diameter
    viscosity = 1e3 * kB * T / (6 * np.pi * diff*1e-12 * r) 
    return viscosity

visBuffer = vis(meanBuffer); sigmaVisBuffer = vis(meanBuffer+sigmaBuffer)

vis40 = vis(mean40); sigmaVis40 = vis(mean40+sigma40)
vis50 = vis(mean50); sigmaVis50 = vis(mean50+sigma50)
vis70 = vis(mean70); sigmaVis70 = vis(mean70+sigma70)

print('vis-buffer [mPa.sec] = %.2f' %visBuffer)
print('vis-40suc [mPa.sec] = %.2f' %vis40)
print('vis-50suc [mPa.sec] = %.2f' %vis50)
print('vis-70suc [mPa.sec] = %.2f' %vis70)

#%% Plot viscosity for all data
mean_vis = [visBuffer, vis40, vis50, vis70]
std_vis = [sigmaVisBuffer/2,
           sigmaVis40/2, sigmaVis50/2, sigmaVis70/2]

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(dpi=300, figsize=(12,6.2))
ax.errorbar([0,40,50,70], mean_vis, yerr=std_vis,
            marker="_", markersize=50,
            color='k', linestyle="none", capsize=10)
ax.set_title('Viscosity')
ax.set_ylabel(r'$\eta$ [mPa$\cdot$sec]')
ax.set_xlabel(r'%(w/w) sucrose in test tube')
ax.set_ylim([0, 5])
ax.set_xticks([0,40,50,70])
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