import sys
sys.path.insert(0, './modules')
import numpy as np
import time
import msd
from scipy import optimize
from matmatrix import fitCDF, trans_stepSize, rot_stepSize, MSDfit
import movingHx
import pickle

# time settings in the light sheet
pxum = 0.115; 
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
vol_exp = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) / 10

# input helix geometry
length = 8; radius = 0.3; pitchHx = 2.5 # all three in um
chirality = 1;  # left-handed: 1, right-handed: -1
resol = 100     # number of points/resolutions

# input helix motion
Nframes = 30000; spin = 0; drift = 0;
Dperp = 0.4; Dpar = 2*Dperp;            # in um^2/sec
Dpitch = 0.1; Droll = 5; Dyaw = 0.1;    # in rad^2/sec

# input helix-and-noise level
hxInt = 200; hxVar = 0;
noiseInt = 0; noiseVar = 0;

# Image analysis and curve fitting
xb = []; xb0 = []; xp = []; xp0 = []
blobSkel = []; blobBin =[]; blobSize = []
eigenvec = []; blobRaw = []; xb0 = []
    
lenfla = np.zeros(Nframes)
cmOutput = np.zeros([Nframes,3]); 
localAxesOutput = np.zeros([Nframes,3,3]);
eigenvec = np.zeros([Nframes,3,3]);
coord = [];
endpt = np.zeros([Nframes]).astype('int')
ep_ref = np.zeros([Nframes,3])

start = time.perf_counter()

nTMax = 100; nSepTotal = 20;
nPoints, nDivider, errorTh = (np.zeros([nSepTotal]) for _ in range(3));
sigma2_N_MSD, sigma2_S_MSD, sigma2_S2_MSD = (np.zeros([nTMax,nSepTotal]) for _ in range(3));
sigma2_N, sigma2_S, sigma2_S2 = (np.zeros([nTMax,nSepTotal]) for _ in range(3));
sigma2_P, sigma2_R, sigma2_Y = (np.zeros([nTMax,nSepTotal]) for _ in range(3));
sigma2_NR = np.zeros([nTMax,nSepTotal]); sigma2_NR_MSD = np.zeros([nTMax,nSepTotal]);
fitN_a, fitS_a, fitS2_a, fitNR_a = (np.zeros([nTMax,nSepTotal]) for _ in range(4));
fitP_a, fitR_a, fitY_a = (np.zeros([nTMax,nSepTotal]) for _ in range(3));
fitN_b, fitS_b, fitS2_b, fitNR_b = (np.zeros([nTMax,nSepTotal]) for _ in range(4));
fitP_b, fitR_b, fitY_b = (np.zeros([nTMax,nSepTotal]) for _ in range(3));

cm_traj = []; EuAng_traj = []; localAxes_traj = [];
transSS_traj = []; rotSS_traj = []; NRSS_traj = [];
time_x_traj = []; MSD_N_traj = [];
MSD_S_traj = []; MSD_S2_traj = [];
MSD_NR_traj = [];
MSD_P_traj = []; MSD_R_traj = [];
MSD_Y_traj = [];

print('Initialization time (sec)',\
      np.round(time.perf_counter()-start,1))

# COMPUTE
nT =  1;
while nT <= nTMax:
    # Generate diffusion
    fromHx = movingHx.createMovHx(length, radius, pitchHx,\
                                  chirality, resol, Nframes,\
                                  Dpar, Dperp, Dpitch, Droll, Dyaw,\
                                  spin, drift, hxInt, hxVar, noiseInt,\
                                  noiseVar, vol_exp)
    cm, EuAng, localAxes = fromHx.movHx()
    print('------------------------------------------')
    print('Generate trajectory #:', nT, 'time (sec):',\
          np.round(time.perf_counter()-start,1))
    
    # generate the step size
    cm_sep = []; EuAng_sep = []; localAxes_sep = [];
    transSS_sep = []; rotSS_sep = []; NRSS_sep = [];
    time_x_sep = [];
    MSD_N_sep = []; MSD_S_sep = []; MSD_S2_sep = []; MSD_NR_sep = [];
    MSD_P_sep = []; MSD_R_sep = []; MSD_Y_sep = []; 
    #for k in np.linspace(1,nSepTotal,nSepTotal)[::1].astype('int'):
    #for k in range(1, nSepTotal + 1):
    for k in [1, 3, 10, 30, 100, 300, 1000]:
    
        # Number divider & points
        nPoints[k-1] = len(cm[::k])
        nDivider[k-1] = k
        errorTh[k-1] = np.sqrt(2/(len(cm[::k])-1))
    
        # Save cm, EuAng, localAxes
        cm_sep.append(cm[::k]);
        EuAng_sep.append(EuAng[::k]);
        localAxes_sep.append(localAxes[::k]);
    
        # Step size technique
        transSS = trans_stepSize(cm[::k], localAxes[::k])
        transSS_sep.append(transSS)
        amp_N, mean_N, sigmaN = fitCDF(transSS[:,0])
        amp_S, mean_S, sigmaS = fitCDF(transSS[:,1])
        amp_S2, mean_S2, sigmaS2 = fitCDF(transSS[:,2])
        sigma2_N[nT-1,k-1] = sigmaN**2
        sigma2_S[nT-1,k-1] = sigmaS**2
        sigma2_S2[nT-1,k-1] = sigmaS2**2

        rotSS = rot_stepSize(EuAng[::k])
        rotSS_sep.append(rotSS)
        amp_P, mean_P, sigmaP = fitCDF(rotSS[:,0])
        amp_R, mean_R, sigmaR = fitCDF(rotSS[:,1])
        amp_Y, mean_Y, sigmaY = fitCDF(rotSS[:,2])
        sigma2_P[nT-1,k-1] = sigmaP**2
        sigma2_R[nT-1,k-1] = sigmaR**2
        sigma2_Y[nT-1,k-1] = sigmaY**2

        NRSS = transSS[:,0] * rotSS[:,1]
        NRSS_sep.append(NRSS)
        amp_NR, mean_NR, sigmaNR = fitCDF(NRSS)
        sigma2_NR[nT-1,k-1] = sigmaNR**2
        print('traj-#:', nT, 'SS-iteration-#:', k, 'time (sec):',\
              np.round(time.perf_counter()-start,1))
        
        # MSD technique
        nInterval = 10;
        fromMSD = msd.theMSD(len(cm[::k]), cm[::k], EuAng[:,1],
                              localAxes[::k], vol_exp*k, nInterval)
        time_x, MSD_N, MSD_S, MSD_S2, MSD_NR = fromMSD.trans_combo_MSD()
        time_xP, MSD_P = msd.regMSD(len(cm[::k]), EuAng[:,0], vol_exp, nInterval)
        time_xR, MSD_R = msd.regMSD(len(cm[::k]), EuAng[:,1], vol_exp, nInterval)
        time_xY, MSD_Y = msd.regMSD(len(cm[::k]), EuAng[:,2], vol_exp, nInterval)
        print('traj-#:', nT, 'MSD-iteration-#:', k, 'time (sec):',\
              np.round(time.perf_counter()-start,1))
        time_x_sep.append(time_x); MSD_N_sep.append(MSD_N);
        MSD_S_sep.append(MSD_S); MSD_S2_sep.append(MSD_S2);
        MSD_NR_sep.append(MSD_NR);
        MSD_P_sep.append(MSD_P); MSD_R_sep.append(MSD_R);
        MSD_Y_sep.append(MSD_Y);

        # Calculate variance directly (first point of MSD)
        sigma2_N_MSD[nT-1,k-1] = MSD_N[0]
        sigma2_S_MSD[nT-1,k-1] = MSD_S[0]
        sigma2_S2_MSD[nT-1,k-1] = MSD_S2[0]
        sigma2_NR_MSD[nT-1,k-1] = MSD_NR[0]
    
        # Fit 3 data points
        Ndata_a = 3;
        xtime = np.linspace(1,Ndata_a,Ndata_a)
        fitN_a[nT-1,k-1],_  = optimize.curve_fit(MSDfit, xtime, MSD_N[0:Ndata_a])[0]
        fitS_a[nT-1,k-1],_  = optimize.curve_fit(MSDfit, xtime, MSD_S[0:Ndata_a])[0]
        fitS2_a[nT-1,k-1],_ = optimize.curve_fit(MSDfit, xtime, MSD_S2[0:Ndata_a])[0]
        fitNR_a[nT-1,k-1],_ = optimize.curve_fit(MSDfit, xtime, MSD_NR[0:Ndata_a])[0]
        fitP_a[nT-1,k-1],_  = optimize.curve_fit(MSDfit, xtime, MSD_P[0:Ndata_a])[0]
        fitR_a[nT-1,k-1],_  = optimize.curve_fit(MSDfit, xtime, MSD_R[0:Ndata_a])[0]
        fitY_a[nT-1,k-1],_  = optimize.curve_fit(MSDfit, xtime, MSD_Y[0:Ndata_a])[0]
        
        # Fit 10 data points
        Ndata_b = 10;
        xtime = np.linspace(1,Ndata_b,Ndata_b)
        fitN_b[nT-1,k-1],_  = optimize.curve_fit(MSDfit, xtime, MSD_N[0:Ndata_b])[0]
        fitS_b[nT-1,k-1],_  = optimize.curve_fit(MSDfit, xtime, MSD_S[0:Ndata_b])[0]
        fitS2_b[nT-1,k-1],_ = optimize.curve_fit(MSDfit, xtime, MSD_S2[0:Ndata_b])[0]    
        fitNR_b[nT-1,k-1],_ = optimize.curve_fit(MSDfit, xtime, MSD_NR[0:Ndata_b])[0]
        fitP_b[nT-1,k-1],_  = optimize.curve_fit(MSDfit, xtime, MSD_P[0:Ndata_b])[0]
        fitR_b[nT-1,k-1],_  = optimize.curve_fit(MSDfit, xtime, MSD_R[0:Ndata_b])[0]
        fitY_b[nT-1,k-1],_  = optimize.curve_fit(MSDfit, xtime, MSD_Y[0:Ndata_b])[0]

        print('traj-#:', nT, 'FitMSD iteration-#:', k, 'time (sec):',\
              np.round(time.perf_counter()-start,0))
    
    cm_traj.append(cm_sep)
    EuAng_traj.append(EuAng_sep)
    localAxes_traj.append(localAxes_sep)
    transSS_traj.append(transSS_sep)
    rotSS_traj.append(rotSS_sep)
    NRSS_traj.append(NRSS_sep)
    time_x_traj.append(time_x_sep)
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
transSS_traj = np.array(transSS_traj,dtype=object)
rotSS_traj = np.array(rotSS_traj,dtype=object)
NRSS_traj = np.array(NRSS_traj,dtype=object)
time_x_traj = np.array(time_x_traj,dtype=object)
MSD_N_traj = np.array(MSD_N_traj,dtype=object)
MSD_S_traj = np.array(MSD_S_traj,dtype=object)
MSD_S2_traj = np.array(MSD_S2_traj,dtype=object)
MSD_NR_traj = np.array(MSD_NR_traj,dtype=object)
MSD_P_traj = np.array(MSD_P_traj,dtype=object)
MSD_R_traj = np.array(MSD_R_traj,dtype=object)
MSD_Y_traj = np.array(MSD_Y_traj,dtype=object)

data = {"Dperp": Dperp,
        "vol_exp": vol_exp,
        "nPoints": nPoints,
        "nDivider": nDivider,
        "errorTh": errorTh,
        "cm_traj": cm_traj,
        "EuAng_traj": EuAng_traj,
        "localAxes_traj": localAxes_traj,
        "transSS_traj": transSS_traj,
        "rotSS_traj": rotSS_traj,
        "NRSS_traj": NRSS_traj,
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
        "time_x_traj": time_x_traj,
        "MSD_N_traj": MSD_N_traj,
        "MSD_S_traj": MSD_S_traj,
        "MSD_S2_traj": MSD_S2_traj,
        "MSD_NR_traj": MSD_NR_traj,
        "MSD_P_traj": MSD_P_traj,
        "MSD_R_traj": MSD_R_traj,
        "MSD_Y_traj": MSD_Y_traj,
        "Ndata_a": Ndata_a,
        "Ndata_b": Ndata_b,
        "fitN_a": fitN_a,
        "fitS_a": fitS_a,
        "fitS2_a": fitS2_a,
        "fitNR_a": fitNR_a,
        "fitN_b": fitN_b,
        "fitS_b": fitS_b,
        "fitS2_b": fitS2_b,
        "fitNR_b": fitNR_b,
        "fitP_a": fitP_a,
        "fitR_a": fitR_a,
        "fitY_a": fitY_a,
        "fitP_b": fitP_b,
        "fitR_b": fitR_b,
        "fitY_b": fitY_b
        }
        
# save data
folderName = r"D:/Dropbox (ASU)/Research/DNA-Rotary-Motor/Helical-nanotubes/Light-sheet-OPM/Result-data/synthetic-data/"
fName = "nT" + str(nTMax) + "-nSep" + str(nSepTotal) +\
    "-nFrame" + str(Nframes) + '.pkl'
with open(folderName+fName, "wb") as f:
     pickle.dump(data, f)