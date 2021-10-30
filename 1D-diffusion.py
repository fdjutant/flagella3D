# Simulating 1D Brownian motion (Wiener process)
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})

Nframes = 10000
# deltaX = np.random.uniform(-0.5, 0.5, Nframes-1).astype(np.float64)
deltaX = np.random.normal(0, 0.25, Nframes-1).astype(np.float64)

cm = np.zeros(Nframes)
for i in range(Nframes):
    cm[i] = np.sum(deltaX[0:i+1])

# decide how long the lag time will be
NlagTime = np.round(0.8*Nframes,0).astype(int);

# compute MSD
time_x = np.linspace(0,NlagTime,NlagTime)
MSD = np.zeros(NlagTime)
j = 1;
while j < NlagTime:
    temp =[]; temp2 =[];
    i = 0;
    while i + j <= Nframes-1:
        temp.append((cm[i+j] - cm[i])**2)
        i += 1
    MSD[j] = np.mean(temp)
    j += 1

# compute step size
stepSize = np.diff(cm)

#%% Plot

# Plot the CDF
yaxis = np.linspace(0,1,len(stepSize),endpoint=False)
fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))    
ax0.plot(np.sort(stepSize),np.linspace(0,1,len(stepSize),endpoint=False),'ko',MarkerSize=1)
ax0.set_title('Step size')
ax0.set_xlabel(r'$\Delta x$');
ax0.set_ylabel(r'Cumulative Probability')

yaxis = np.linspace(0,1,len(cm),endpoint=False)
fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))    
ax0.plot(np.sort(cm),np.linspace(0,1,len(cm),endpoint=False),'ko',MarkerSize=1)
ax0.set_title('Position')
ax0.set_xlabel(r'$x$');
ax0.set_ylabel(r'Cumulative Probability')

# histogram
fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))    
weights = np.ones_like(deltaX)/len(deltaX)
n, bins, patches = ax0.hist(deltaX, weights=weights)
ax0.set_title('Step size')
ax0.set_xlabel(r'$\Delta x$');
ax0.set_ylabel(r'Probability')

fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))    
weights = np.ones_like(cm)/len(cm)
n, bins, patches = ax0.hist(cm, weights=weights)
ax0.set_title('Position')
ax0.set_xlabel(r'$x$');
ax0.set_ylabel(r'Probability')
    
fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
ax0.plot(time_x,MSD,c='k',marker="^",mfc='none',ls='None',alpha=0.5)   
ax0.set_xscale('log'); ax0.set_yscale('log'); 
ax0.set_title('MSD')
ax0.set_xlabel(r'Log($\tau$) [sec]');
ax0.set_ylabel(r'Log(MSD) [$\mu m^2$/sec]')
