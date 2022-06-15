import os
import sys
sys.path.insert(0, './modules')
import numpy as np
import time
import msd
from scipy import optimize
from matmatrix import fitCDF,  MSDfit
from movingHx import simulate_diff
import pickle

# time settings in the light sheet
pxum = 0.115
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
vol_exp = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) / 10

# input helix geometry
length = 8; radius = 0.3; pitchHx = 2.5 # all three in um
chirality = 1  # left-handed: 1, right-handed: -1
resol = 100    # number of points/resolutions

# input helix motion
Nframes = 3000 * 100 * 10
Dperp = 0.1
Dpar = 2*Dperp # in um^2/sec
Dpitch = 0.03
Droll = 1.5
Dyaw = Dpitch # in rad^2/sec
spin = 0
drift = 0

# input helix-and-noise level
hxInt = 200; hxVar = 0
noiseInt = 0; noiseVar = 0

start = time.perf_counter()

# set number of trajectories
nTMax = 100
# set separations to compute diffusion constant at
interval = np.array([1, 3, 10, 30, 100, 300, 1000,
                     3000, 10000, 30000, 100000, 300000]).astype('int')
interval = interval[interval < (Nframes / 10)]
nSepTotal = len(interval)

# variables to store results
nPoints, nDivider, errorTh = (np.zeros([nSepTotal]) for _ in range(3))
sigma2_N_MSD, sigma2_S_MSD, sigma2_S2_MSD = (np.zeros([nTMax,nSepTotal]) for _ in range(3))
sigma2_N, sigma2_S, sigma2_S2 = (np.zeros([nTMax,nSepTotal]) for _ in range(3))
sigma2_P, sigma2_R, sigma2_Y = (np.zeros([nTMax,nSepTotal]) for _ in range(3))
sigma2_NR = np.zeros([nTMax,nSepTotal]); sigma2_NR_MSD = np.zeros([nTMax,nSepTotal])
fitN_a, fitS_a, fitS2_a, fitNR_a = (np.zeros([nTMax,nSepTotal]) for _ in range(4))
fitP_a, fitR_a, fitY_a = (np.zeros([nTMax,nSepTotal]) for _ in range(3))
fitN_b, fitS_b, fitS2_b, fitNR_b = (np.zeros([nTMax,nSepTotal]) for _ in range(4))
fitP_b, fitR_b, fitY_b = (np.zeros([nTMax,nSepTotal]) for _ in range(3))
cm_traj = []
EuAng_traj = []
localAxes_traj = []

saveTraj = False
if saveTraj:
    transSS_traj = []
    rotSS_traj = []
    NRSS_traj = []
    MSD_N_traj = []
    MSD_S_traj = [] 
    MSD_S2_traj = []
    MSD_NR_traj = []
    MSD_P_traj = [] 
    MSD_R_traj = []
    MSD_Y_traj = []

print('Initialization time (sec)',
      np.round(time.perf_counter()-start,1))

# COMPUTE
nT =  0
while nT < nTMax:
    # Generate diffusion
    cmass, euler_angles, n1, n2, n3 = simulate_diff(Nframes, vol_exp,
                                                    Dpar, Dperp,
                                                    Dpitch, Droll, Dyaw)
    cm = cmass.reshape(3,Nframes).T
    EuAng = euler_angles.reshape(3,Nframes).T
    localAxes = np.zeros([Nframes,3,3])
    for i in range(Nframes):
        localAxes[i] = np.array([ n1[i], n2[i], n3[i] ])
        
    print('------------------------------------------')
    print('Generate trajectory #:', nT, 'time (sec):',
          np.round(time.perf_counter()-start,1))
    
    # generate the step size
    cm_sep = []; EuAng_sep = []; localAxes_sep = []
    transSS_sep = []; rotSS_sep = []; NRSS_sep = []
    MSD_N_sep = []; MSD_S_sep = []; MSD_S2_sep = []; MSD_NR_sep = []
    MSD_P_sep = []; MSD_R_sep = []; MSD_Y_sep = []
    
    for k in range(len(interval)):
        
        # Number divider & points
        nInt = interval[k]
        nPoints[k] = len(cm[::nInt])
        nDivider[k] = nInt
        errorTh[k] = np.sqrt(2/(len(cm[::nInt])-1))
    
        # Save cm, EuAng, localAxes
        cm_sep.append(cm[::nInt])
        EuAng_sep.append(EuAng[::nInt])
        localAxes_sep.append(localAxes[::nInt])
    
        # Step size technique
        # transSS = msd.trans_stepSize(cm[::nInt], localAxes[::nInt])
        n1 = localAxes[::nInt,0]
        n2 = localAxes[::nInt,1]
        n3 = localAxes[::nInt,2]
        transSS0, transSS1, transSS2 = msd.trans_stepSize_Namba(cm[::nInt],
                                                                n1, n2, n3)
        transSS = np.stack([transSS0, transSS1, transSS2],axis=0).T
        if saveTraj:
            transSS_sep.append(transSS)
        amp_N, mean_N, sigmaN = fitCDF(transSS[:,0])
        amp_S, mean_S, sigmaS = fitCDF(transSS[:,1])
        amp_S2, mean_S2, sigmaS2 = fitCDF(transSS[:,2])
        sigma2_N[nT,k] = sigmaN**2
        sigma2_S[nT,k] = sigmaS**2
        sigma2_S2[nT,k] = sigmaS2**2

        # rotSS = msd.rot_stepSize(EuAng[::nInt])
        rotSS0, rotSS1, rotSS2 = msd.rot_stepSize_Namba(EuAng[::nInt])
        rotSS = np.stack([rotSS0, rotSS1, rotSS2],axis=0).T
        if saveTraj:
            rotSS_sep.append(rotSS)
        amp_P, mean_P, sigmaP = fitCDF(rotSS[:,0])
        amp_R, mean_R, sigmaR = fitCDF(rotSS[:,1])
        amp_Y, mean_Y, sigmaY = fitCDF(rotSS[:,2])
        sigma2_P[nT,k] = sigmaP**2
        sigma2_R[nT,k] = sigmaR**2
        sigma2_Y[nT,k] = sigmaY**2

        NRSS = transSS[:,0] * rotSS[:,1]
        if saveTraj:
            NRSS_sep.append(NRSS)
        amp_NR, mean_NR, sigmaNR = fitCDF(NRSS)
        sigma2_NR[nT-1,k] = sigmaNR**2
        print('traj-#:', nT, 'SS-iteration-#:', k, 'time (sec):',
              np.round(time.perf_counter()-start,1))
        
        # MSD technique
        # either use first 10 points, or if don't have points use all that we do have
        nInterval = np.min([3, nPoints[k] - 1]).astype(int)        
        # fromMSD = msd.theMSD(len(cm[::nInt]), cm[::nInt], EuAng[:,1],
        #                       localAxes[::nInt], vol_exp*nInt, nInterval)
        # MSD_N, MSD_S, MSD_S2, MSD_NR = fromMSD.trans_combo_MSD()
        MSD_N, MSD_S, MSD_S2, MSD_NR = msd.trans_MSD_Namba(len(cm[::nInt]),
                                                  cm[::nInt], EuAng[::nInt,1],
                                                  n1, n2, n3,
                                                  vol_exp*nInt, nInterval)
        MSD_P = msd.regMSD(len(cm[::nInt]), EuAng[::nInt,0], vol_exp, nInterval)
        MSD_R = msd.regMSD(len(cm[::nInt]), EuAng[::nInt,1], vol_exp, nInterval)
        MSD_Y = msd.regMSD(len(cm[::nInt]), EuAng[::nInt,2], vol_exp, nInterval)
        print('traj-#:', nT, 'MSD-iteration-#:', k, 'time (sec):',
              np.round(time.perf_counter()-start,1))
        if saveTraj:
            MSD_N_sep.append(MSD_N)
            MSD_S_sep.append(MSD_S)
            MSD_S2_sep.append(MSD_S2)
            MSD_NR_sep.append(MSD_NR)
            MSD_P_sep.append(MSD_P)
            MSD_R_sep.append(MSD_R)
            MSD_Y_sep.append(MSD_Y)

        # Calculate variance directly (first point of MSD)
        sigma2_N_MSD[nT,k] = MSD_N[0]
        sigma2_S_MSD[nT,k] = MSD_S[0]
        sigma2_S2_MSD[nT,k] = MSD_S2[0]
        sigma2_NR_MSD[nT,k] = MSD_NR[0]
    
        # Fit 3 data points
        Ndata_a = np.min([3, nPoints[k] - 1]).astype(int)
        xtime = np.arange(1, Ndata_a + 1, dtype=float)
        fitN_a[nT-1,k],_  = optimize.curve_fit(MSDfit, xtime, MSD_N[0:Ndata_a])[0]
        fitS_a[nT-1,k],_  = optimize.curve_fit(MSDfit, xtime, MSD_S[0:Ndata_a])[0]
        fitS2_a[nT-1,k],_ = optimize.curve_fit(MSDfit, xtime, MSD_S2[0:Ndata_a])[0]
        fitNR_a[nT-1,k],_ = optimize.curve_fit(MSDfit, xtime, MSD_NR[0:Ndata_a])[0]
        fitP_a[nT-1,k],_  = optimize.curve_fit(MSDfit, xtime, MSD_P[0:Ndata_a])[0]
        fitR_a[nT-1,k],_  = optimize.curve_fit(MSDfit, xtime, MSD_R[0:Ndata_a])[0]
        fitY_a[nT-1,k],_  = optimize.curve_fit(MSDfit, xtime, MSD_Y[0:Ndata_a])[0]
     
        print('traj-#:', nT, 'FitMSD iteration-#:', k, 'time (sec):',
              np.round(time.perf_counter()-start,0))
    
        cm_traj.append(cm_sep)
        EuAng_traj.append(EuAng_sep)
        localAxes_traj.append(localAxes_sep)
    if saveTraj:
        transSS_traj.append(transSS_sep)
        rotSS_traj.append(rotSS_sep)
        NRSS_traj.append(NRSS_sep)
        MSD_N_traj.append(MSD_N_sep)
        MSD_S_traj.append(MSD_S_sep)
        MSD_S2_traj.append(MSD_S2_sep)
        MSD_NR_traj.append(MSD_NR_sep)
        MSD_P_traj.append(MSD_P_sep)
        MSD_R_traj.append(MSD_R_sep)
        MSD_Y_traj.append(MSD_Y_sep)
    
    del cm, EuAng, localAxes 
    nT += 1
    
#%% save to pickle
cm_traj = np.array(cm_traj,dtype=object)
EuAng_traj = np.array(EuAng_traj,dtype=object)
localAxes_traj = np.array(localAxes_traj,dtype=object)
if saveTraj:
    transSS_traj = np.array(transSS_traj,dtype=object)
    rotSS_traj = np.array(rotSS_traj,dtype=object)
    NRSS_traj = np.array(NRSS_traj,dtype=object)
    MSD_N_traj = np.array(MSD_N_traj,dtype=object)
    MSD_S_traj = np.array(MSD_S_traj,dtype=object)
    MSD_S2_traj = np.array(MSD_S2_traj,dtype=object)
    MSD_NR_traj = np.array(MSD_NR_traj,dtype=object)
    MSD_P_traj = np.array(MSD_P_traj,dtype=object)
    MSD_R_traj = np.array(MSD_R_traj,dtype=object)
    MSD_Y_traj = np.array(MSD_Y_traj,dtype=object)

data = {"Dpar": Dpar,
        "Dperp": Dperp,
        "Dpitch": Dpitch,
        "Droll": Droll,
        "Dyaw": Dyaw,
        "interval": interval,
        "vol_exp": vol_exp,
        "nPoints": nPoints,
        "nDivider": nDivider,
        "errorTh": errorTh,
        "cm_traj": cm_traj,
        "EuAng_traj": EuAng_traj,
        "localAxes_traj": localAxes_traj,
        "sigma2_N": sigma2_N,
        "sigma2_S": sigma2_S,
        "sigma2_S2": sigma2_S2,
        "sigma2_NR": sigma2_NR,
        "sigma2_N_MSD": sigma2_N_MSD,
        "sigma2_S_MSD": sigma2_S_MSD,
        "sigma2_S2_MSD": sigma2_S2_MSD,
        "sigma2_NR_MSD": sigma2_NR_MSD,
        "sigma2_P": sigma2_P,
        "sigma2_R": sigma2_R,
        "sigma2_Y": sigma2_Y,
        "Ndata_a": Ndata_a,
        "fitN_a": fitN_a,
        "fitS_a": fitS_a,
        "fitS2_a": fitS2_a,
        "fitNR_a": fitNR_a,
        "fitP_a": fitP_a,
        "fitR_a": fitR_a,
        "fitY_a": fitY_a,
        }
if saveTraj:
    dataTraj = {
            "transSS_traj": transSS_traj,
            "rotSS_traj": rotSS_traj,
            "NRSS_traj": NRSS_traj,
            "MSD_N_traj": MSD_N_traj,
            "MSD_S_traj": MSD_S_traj,
            "MSD_S2_traj": MSD_S2_traj,
            "MSD_NR_traj": MSD_NR_traj,
            "MSD_P_traj": MSD_P_traj,
            "MSD_R_traj": MSD_R_traj,
            "MSD_Y_traj": MSD_Y_traj,
         }   
    
# save data
try:
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    folderName = os.path.join(this_file_dir, "..", "..",
                              "DNA-Rotary-Motor", "Helical-nanotubes",
                              "Light-sheet-OPM", "Result-data",
                              "synthetic-data")
    
    if not os.path.exists(folderName):
        folderName = ""

except:
    folderName = ""


fName = "nT%d-nSep%d-nFrame%d.pkl" % (nTMax, nSepTotal, Nframes)
fdir = os.path.join(folderName, fName)
with open(fdir, "wb") as f:
     pickle.dump(data, f)
     if saveTraj:
         pickle.dump(dataTraj, f)
     