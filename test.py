import numpy as np
from msd import regMSD
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})
from scipy import stats, optimize

pxum = 0.115; 
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
# expTime = 1/ (sweep_um/stepsize_nm * camExposure_ms)
vol_exp = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 

Nframes = 50; pxum = 0.115; D = 1.5;
sigma_square = 2 * D * vol_exp
cmFluc = np.zeros([Nframes,3]);
for i in range(Nframes):
    cmFluc[i,0] = cmFluc[i-1,0] + np.random.normal(0, np.sqrt(sigma_square))

time_x, MSD_cm = regMSD(Nframes, cmFluc[:,0], vol_exp)

# Fit the MSDs curve
rData = 0.05;
nData = np.int32(rData*Nframes) # number of data fitted
def MSDfit(x, a):
    return a * x   
fitN = optimize.curve_fit(MSDfit, time_x[0:nData], MSD_cm[0:nData], p0=0.1)

print('D = ', fitN[0]/2)

fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
ax0.plot(time_x,MSD_cm,c='k',marker="s",mfc='none',ms=9,ls='None',alpha=0.5) 
ax0.plot(time_x,fitN[0]*time_x,c='k',alpha=0.2)   
# ax0.plot(time_x,time_x**2,c='b',alpha=0.2) 
ax0.set_xscale('log'); ax0.set_yscale('log'); 
ax0.set_title('MSD translation')
ax0.set_xlabel(r'Log($\tau$) [sec]');
ax0.set_ylabel(r'Log(MSD) [$\mu m^2$/sec]')
# ax0.set_ylim([np.exp(-0.5*10e-1),np.exp(10^4)])
ax0.legend(["parallel","perpendicular-1","perpendicular-2"])