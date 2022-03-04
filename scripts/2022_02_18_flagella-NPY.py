#%% Import all necessary libraries
import sys
sys.path.insert(0, './modules')
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})
from sklearn.decomposition import PCA
import time
import napari
from matmatrix import consistentPCA, hullAnalysis, EuAngfromN
import os.path
import glob
import pickle
import msd
from scipy import stats

# time settings in the light sheet
pxum = 0.115
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
exp3D_ms = 1e-3 * camExposure_ms * (sweep_um*1e3/stepsize_nm) 
pxum = 0.115
start = time.perf_counter()

# which end of flagella
# side40 = np.array([-1,-1,1,1,1,1,1,1,1])
# side50 = np.array([1,1,1,1,-1,-1,-1,1,-1])
# side70 = np.array([1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,1])
# side = np.array([side40,side50,side70],dtype=object)    

sucPer = ['suc40', 'suc50', 'suc70']

#%% Generate pickle files
for SP in range(len(sucPer)):
# for SP in range(1):

    this_file_dir = os.path.dirname(os.path.abspath("./"))
    folderName = "Flagella-all"
    
    path = os.path.join(this_file_dir,
                        "DNA-Rotary-Motor", "Helical-nanotubes",
                        "Light-sheet-OPM", "Result-data",
                        folderName, 'run-07',sucPer[SP])
    nFiles = len(glob.glob(path + '\*.npy'))
    
    ImgNameAll = []; LengthMean = []; LengthSTD = []
    fitN_All = []; fitS_All = []; fitNR_All = []; fitPY_All = []; fitR_All = []
    fitP_All = []; fitY_All = []; fitS1_All = []; fitS2_All = []
        
    for j in range(2):
        
        # which end of helix
        if j == 0:
            whEnd = -1
        else:
            whEnd = 1
        
        for i in range(nFiles):
        # for i in range(1):
            
            ThName = os.path.basename(glob.glob(path + '/*.npy')[i])[:-4]
            stackTh = np.array(da.from_npy_stack(os.path.join(path, ThName + ".npy")))
    
            Nframes = len(stackTh)
            
            # image analysis and curve fitting
            xb = []; xb0 = []; xp = []; xp0 = []
            eigenvec = []; xb0 = []
                
            lenfla = np.zeros(Nframes)
            cm = np.zeros([Nframes,3])
            eigenvec = np.zeros([Nframes,3,3])
            coord = []
            localAxes = np.zeros([Nframes,3,3])
            endpt = np.zeros([Nframes,3])
                     
            
            for frame in range(Nframes):
            # for frame in range(1):
            
                blob = stackTh[frame]
                           
                X0 = np.argwhere(blob).astype('float') 
                CM1 = np.array([sum(X0[:,j]) for j in range(X0.shape[1])])/X0.shape[0]
            
                xb.append(X0); cm[frame,:] = CM1
                       
                # Use PCA to find the rotation matrix
                X = X0 - CM1 # shift all the coordinates into origin
                pca = PCA(n_components=3)
                pca.fit(X)
                axes = whEnd*pca.components_
                xb0.append(X)
                
                # Make the PCA consistent
                if frame == 0:
                    axes_ref = axes
                else:
                    axes, axes_ref = consistentPCA(axes, axes_ref)
                
                # Find the flagella length and endpoint in the direction of n1
                lenfla[frame], endpt[frame] = hullAnalysis(X,axes)
                
                # Use Gram-Schmidt to find n2, then find n3 with the cross
                n1 = axes[0] / np.linalg.norm(axes[0])
                n2 = endpt[frame] - np.array([0,0,0])
                n2 -= n2.dot(n1) * n1 / np.linalg.norm(n1)**2
                n2 /= np.linalg.norm(n2)
                n3 = np.cross(n1,n2)
                n3 /= np.linalg.norm(n3)
                localAxes[frame,0] = n1
                localAxes[frame,1] = n2
                localAxes[frame,2] = n3
                
                # Rotate to the principal axes
                P0 = np.matmul(axes,X.T).T
                xp0.append(P0)  
                eigenvec[frame] = axes
            
                # Print each volume frame has finished
                end = time.perf_counter()
                print('Img-number = %d / %d \n' %(i, nFiles) + 
                      'frame#: %d\n' %frame +
                      'elapsed time (sec): %0.02f' %np.round(end-start,2))
            
            # compute pitch, roll, and yaw (Bernie's method)
            EuAng = EuAngfromN(localAxes)
            n1 = localAxes[:,0]; n2 = localAxes[:,1]; n3 = localAxes[:,2]
            
            # compute translation displacement
            disp_N, disp_S1, disp_S2 = msd.trans_stepSize_Namba(cm, n1, n2, n3)
            disp = np.stack([disp_N, disp_S1, disp_S2])
            
            # save each output to pickle
            fdir = os.path.join(path, ThName + '-' + str(j) + '.pkl')
            data = {"whEnd":whEnd,
                    "cm": cm,
                    "disp": disp,
                    "endpt": endpt,
                    "EuAng": EuAng,
                    "localAxes": localAxes,
                    "lenfla": lenfla}
            with open(fdir, "wb") as f:
                 pickle.dump(data, f)
    
            print('ImgName \t= %s\n' % ThName)
            
#%% Generate snapshots: take pickles and *threshold.npy
sucPer = ['suc40', 'suc50', 'suc70']
for SP in range(len(sucPer)):

    this_file_dir = os.path.dirname(os.path.abspath("./"))
    folderName = "Flagella-all"
    
    path = os.path.join(this_file_dir,
                        "DNA-Rotary-Motor", "Helical-nanotubes",
                        "Light-sheet-OPM", "Result-data",
                        folderName, 'run-07',sucPer[SP])
    pklPath = glob.glob(path + '\*.pkl')
    npyPath = glob.glob(path + '\*.npy')

    for i in range(len(pklPath)):
    # for i in range(1):
        
        ThName = os.path.basename(pklPath[i])[:-4]
        
        # load pickle
        with open(pklPath[i], "rb") as f:
              data_loaded = pickle.load(f)
        whEnd = data_loaded["whEnd"]
        cm = data_loaded["cm"]
        endpt = data_loaded["endpt"]
        EuAng = data_loaded["EuAng"]
        localAxes = data_loaded["localAxes"]
        lenfla = data_loaded["lenfla"] 
    
        # compute pitch, roll, and yaw (Bernie's method)
        EuAng = EuAngfromN(localAxes)
        n1 = localAxes[:,0]; n2 = localAxes[:,1]; n3 = localAxes[:,2]
        
        # compute translation displacement
        disp_N, disp_S1, disp_S2 = msd.trans_stepSize_Namba(cm, n1, n2, n3)
        disp = np.stack([disp_N, disp_S1, disp_S2])
        disp = np.hstack([np.zeros((3,1)),disp]).T
    
        # save folder for 3D points
        if whEnd == -1: symEnd = '0'
        else: symEnd = '1'
        savingFolder = os.path.join(path, ThName[:-2] +'-snapshots-'+ symEnd)
        if os.path.isdir(savingFolder) != True:
            os.mkdir(savingFolder) # create path    
    
        # load threshold
        stackTh = np.array(da.from_npy_stack(os.path.join(path,
                                                          ThName[:-2] +
                                                          ".npy")))
        # generate snapshots
        for frame in range(len(stackTh)):
        # for frame in range(2):
        
            blob = stackTh[frame]
            X0 = np.argwhere(blob).astype('float') 
            xb0 = X0 - cm[frame]
            
            fig = plt.figure(dpi=150, figsize = (7, 6))
            fig.suptitle('data: %s\n' %ThName +
                         'frame-num = ' + str(frame).zfill(3) + ', '
                         'length = %.3f $\mu$m\n' %np.round(lenfla[frame],3) +
                         '$\Delta_\parallel$ = %.3f $\mu$m, ' %np.round(disp[frame,0],3) +
                         '$\Delta_{\perp 1}$ = %.3f $\mu$m, ' %np.round(disp[frame,1],3) +
                         '$\Delta_{\perp 2}$ = %.3f $\mu$m\n' %np.round(disp[frame,2],3) +
                         '$\Delta_\psi$ = %.3f rad, ' %np.round(EuAng[frame,1],3) +
                         '$\Delta_\gamma$ = %.3f rad, ' %np.round(EuAng[frame,0],3) +
                         '$\Delta_\phi$ = %.3f rad\n' %np.round(EuAng[frame,2],3)
                         )
            ax0 = fig.add_subplot(221,projection='3d')
            ax2 = fig.add_subplot(222,projection='3d')
            ax3 = fig.add_subplot(223,projection='3d')
            ax4 = fig.add_subplot(224,projection='3d')
            pxum = 0.115
            
            ## plot 1
            x, y, z = np.array([[-40,0,0],[0,-40,0],[0,0,-40]])*pxum
            u, v, w = np.array([[60,0,0],[0,60,0],[0,0,60]])*pxum
            ax0.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
            edgePoint = 40
            ax0.set_ylim(-edgePoint*pxum,edgePoint*pxum)
            ax0.set_xlim(-edgePoint*pxum,edgePoint*pxum)
            ax0.set_zlim(-edgePoint*pxum,edgePoint*pxum)
            ax0.view_init(elev=30, azim=30)
            ax0.set_xlabel(r'x [$\mu m$]'); ax0.set_ylabel(r'y [$\mu m$]')
            ax0.set_zlabel(r'z [$\mu m$]')
            ax0.scatter(xb0[:,0]*pxum, xb0[:,1]*pxum,\
                        xb0[:,2]*pxum, c = 'k',alpha=0.1, s=10)
            ax0.scatter(endpt[frame,0]*pxum,\
                        endpt[frame,1]*pxum,\
                        endpt[frame,2]*pxum, c = 'r', alpha=0.5, s=50) 
            origin = [0,0,0]
            X, Y, Z = zip(origin)
            Un1, Vn1, Wn1 = zip(list(5*localAxes[frame,0])) 
            Un2, Vn2, Wn2 = zip(list(5*localAxes[frame,1])) 
            Un3, Vn3, Wn3 = zip(list(5*localAxes[frame,2]))
            ax0.quiver(X,Y,Z,Un1,Vn1,Wn1,color='r')
            ax0.quiver(X,Y,Z,Un2,Vn2,Wn2,color='g')
            ax0.quiver(X,Y,Z,Un3,Vn3,Wn3,color='b')
            
            ## plot 2
            x, y, z = np.array([[-40,0,0],[0,-40,0],[0,0,-40]])*pxum
            u, v, w = np.array([[60,0,0],[0,60,0],[0,0,60]])*pxum
            ax2.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
            edgePoint = 40
            ax2.set_ylim(-edgePoint*pxum,edgePoint*pxum)
            ax2.set_xlim(-edgePoint*pxum,edgePoint*pxum)
            ax2.set_zlim(-edgePoint*pxum,edgePoint*pxum)
            ax2.view_init(elev=0, azim=90)
            ax2.set_xlabel(r'x [$\mu m$]'); ax2.set_ylabel(r'y [$\mu m$]')
            ax2.set_zlabel(r'z [$\mu m$]')
            ax2.scatter(xb0[:,0]*pxum, xb0[:,1]*pxum,\
                        xb0[:,2]*pxum, c = 'k',alpha=0.1, s=10)
            ax2.scatter(endpt[frame,0]*pxum,\
                        endpt[frame,1]*pxum,\
                        endpt[frame,2]*pxum, c = 'r', alpha=0.5, s=50) 
            origin = [0,0,0]
            X, Y, Z = zip(origin)
            Un1, Vn1, Wn1 = zip(list(5*localAxes[frame,0])) 
            Un2, Vn2, Wn2 = zip(list(5*localAxes[frame,1])) 
            Un3, Vn3, Wn3 = zip(list(5*localAxes[frame,2]))
            ax2.quiver(X,Y,Z,Un1,Vn1,Wn1,color='r')
            ax2.quiver(X,Y,Z,Un2,Vn2,Wn2,color='g')
            ax2.quiver(X,Y,Z,Un3,Vn3,Wn3,color='b')
            
            ## plot 3
            x, y, z = np.array([[-40,0,0],[0,-40,0],[0,0,-40]])*pxum
            u, v, w = np.array([[60,0,0],[0,60,0],[0,0,60]])*pxum
            ax3.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
            edgePoint = 40
            ax3.set_ylim(-edgePoint*pxum,edgePoint*pxum)
            ax3.set_xlim(-edgePoint*pxum,edgePoint*pxum)
            ax3.set_zlim(-edgePoint*pxum,edgePoint*pxum)
            ax3.view_init(elev=0, azim=0)
            ax3.set_xlabel(r'x [$\mu m$]'); ax3.set_ylabel(r'y [$\mu m$]')
            ax3.set_zlabel(r'z [$\mu m$]')
            ax3.scatter(xb0[:,0]*pxum, xb0[:,1]*pxum,\
                        xb0[:,2]*pxum, c = 'k',alpha=0.1, s=10)
            ax3.scatter(endpt[frame,0]*pxum,\
                       endpt[frame,1]*pxum,\
                       endpt[frame,2]*pxum, c = 'r', alpha=0.5, s=50) 
            origin = [0,0,0]
            X, Y, Z = zip(origin)
            Un1, Vn1, Wn1 = zip(list(5*localAxes[frame,0])) 
            Un2, Vn2, Wn2 = zip(list(5*localAxes[frame,1])) 
            Un3, Vn3, Wn3 = zip(list(5*localAxes[frame,2]))
            ax3.quiver(X,Y,Z,Un1,Vn1,Wn1,color='r')
            ax3.quiver(X,Y,Z,Un2,Vn2,Wn2,color='g')
            ax3.quiver(X,Y,Z,Un3,Vn3,Wn3,color='b')
            
            ## plot 4
            x, y, z = np.array([[-40,0,0],[0,-40,0],[0,0,-40]])*pxum
            u, v, w = np.array([[60,0,0],[0,60,0],[0,0,60]])*pxum
            ax4.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
            edgePoint = 40
            ax4.set_ylim(-edgePoint*pxum,edgePoint*pxum)
            ax4.set_xlim(-edgePoint*pxum,edgePoint*pxum)
            ax4.set_zlim(-edgePoint*pxum,edgePoint*pxum)
            ax4.view_init(elev=90, azim=0)
            ax4.set_xlabel(r'x [$\mu m$]'); ax4.set_ylabel(r'y [$\mu m$]')
            ax4.set_zlabel(r'z [$\mu m$]')
            ax4.scatter(xb0[:,0]*pxum, xb0[:,1]*pxum,\
                        xb0[:,2]*pxum, c = 'k',alpha=0.1, s=10)
            ax4.scatter(endpt[frame,0]*pxum,\
                       endpt[frame,1]*pxum,\
                       endpt[frame,2]*pxum, c = 'r', alpha=0.5, s=50) 
            origin = [0,0,0]
            X, Y, Z = zip(origin)
            Un1, Vn1, Wn1 = zip(list(5*localAxes[frame,0])) 
            Un2, Vn2, Wn2 = zip(list(5*localAxes[frame,1])) 
            Un3, Vn3, Wn3 = zip(list(5*localAxes[frame,2]))
            ax4.quiver(X,Y,Z,Un1,Vn1,Wn1,color='r')
            ax4.quiver(X,Y,Z,Un2,Vn2,Wn2,color='g')
            ax4.quiver(X,Y,Z,Un3,Vn3,Wn3,color='b')
            ax4.figure.savefig(os.path.join(savingFolder, ThName + '-' + 
                                            str(frame).zfill(3) + '.png'))
            
            # Clear the whole plots
            ax0.clear(); ax0.remove()
            ax2.clear(); ax2.remove()
            ax3.clear(); ax3.remove()
            ax4.clear(); ax4.remove()

#%% Generate correlation
sucPer = ['suc40', 'suc50', 'suc70']
for SP in range(len(sucPer)):
# for SP in range(1):

    this_file_dir = os.path.dirname(os.path.abspath("./"))
    folderName = "Flagella-all"
    
    path = os.path.join(this_file_dir,
                        "DNA-Rotary-Motor", "Helical-nanotubes",
                        "Light-sheet-OPM", "Result-data",
                        folderName, 'run-07',sucPer[SP])
    pklPath = glob.glob(path + '\*.pkl')
    npyPath = glob.glob(path + '\*.npy')

    for i in range(len(pklPath)):
    # for i in range(1):
        
        ThName = os.path.basename(pklPath[i])[:-4]
        
        # load pickle
        with open(pklPath[i], "rb") as f:
              data_loaded = pickle.load(f)
        whEnd = data_loaded["whEnd"]
        cm = data_loaded["cm"]
        endpt = data_loaded["endpt"]
        EuAng = data_loaded["EuAng"]
        localAxes = data_loaded["localAxes"]
        lenfla = data_loaded["lenfla"] 
    
        # compute pitch, roll, and yaw (Bernie's method)
        EuAng = EuAngfromN(localAxes)
        n1 = localAxes[:,0]; n2 = localAxes[:,1]; n3 = localAxes[:,2]
        
        # compute translation displacement
        disp_N, disp_S1, disp_S2 = msd.trans_stepSize_Namba(cm, n1, n2, n3)
        disp = np.stack([disp_N, disp_S1, disp_S2])
        disp = np.hstack([np.zeros((3,1)),disp]).T
    
        # save folder for corellations
        if whEnd == -1: symEnd = '0'
        else: symEnd = '1'
        savingFolder = os.path.join(path, ThName[:-2] +'-correlations-'+ symEnd)
        if os.path.isdir(savingFolder) != True:
            os.mkdir(savingFolder) # create path    
            
        # loop through number of interval
        for numInt in range(1,11):
            
            # input data
            y = disp[:,0][::numInt]  # parallel displacement
            x = EuAng[:,1][::numInt] # roll
                
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
                        str(np.round(res[0],3)) + ' $\mu$m/rad'], prop={'size': 20})
            plt.ylabel(r'$\Delta_\parallel$ [$\mu$m]')
            plt.xlabel(r'$\Delta\psi$ [rad]')
            plt.grid(True, which='both')
            plt.xlim([-1.5,1.5])
            plt.ylim([-1.5,1.5])
            plt.title(ThName + ' (num-interval: %d)' %numInt)
            plt.savefig(os.path.join(savingFolder, ThName + '-' + 
                                            str(numInt).zfill(2) + '.png'))
            plt.clf()
            plt.close()
        # plt.show()

#%% Generate step size
sucPer = ['suc40', 'suc50', 'suc70']
for SP in range(len(sucPer)):
# for SP in range(1):

    this_file_dir = os.path.dirname(os.path.abspath("./"))
    folderName = "Flagella-all"
    
    path = os.path.join(this_file_dir,
                        "DNA-Rotary-Motor", "Helical-nanotubes",
                        "Light-sheet-OPM", "Result-data",
                        folderName, 'run-07',sucPer[SP])
    pklPath = glob.glob(path + '\*.pkl')
    npyPath = glob.glob(path + '\*.npy')

    for i in range(len(pklPath)):
    # for i in range(1):
        
        ThName = os.path.basename(pklPath[i])[:-4]
        
        # load pickle
        with open(pklPath[i], "rb") as f:
              data_loaded = pickle.load(f)
        whEnd = data_loaded["whEnd"]
        cm = data_loaded["cm"]
        endpt = data_loaded["endpt"]
        EuAng = data_loaded["EuAng"]
        localAxes = data_loaded["localAxes"]
        lenfla = data_loaded["lenfla"] 
    
        # compute pitch, roll, and yaw (Bernie's method)
        EuAng = EuAngfromN(localAxes)
        n1 = localAxes[:,0]; n2 = localAxes[:,1]; n3 = localAxes[:,2]
        
        # compute translation displacement
        disp_N, disp_S1, disp_S2 = msd.trans_stepSize_Namba(cm, n1, n2, n3)
        disp = np.stack([disp_N, disp_S1, disp_S2])
        disp = np.hstack([np.zeros((3,1)),disp]).T
    
        # save folder for corellations
        if whEnd == -1: symEnd = '0'
        else: symEnd = '1'
        savingFolder = os.path.join(path, ThName[:-2] +'-StepSize-'+ symEnd)
        if os.path.isdir(savingFolder) != True:
            os.mkdir(savingFolder) # create path    
            
        # loop through number of interval
        for numInt in range(1,11):
            
            # input data
            y = disp[:,0][::numInt]                 # parallel displacement
            x = EuAng[:,1][::numInt]                # roll
            yx = (disp[:,0]*EuAng[:,1])[::numInt]   # parallel-disp x roll
                    
            # plot it
            plt.figure(dpi=300, figsize=(10,6.2))
            plt.rcParams.update({'font.size': 22})
            plt.plot([np.mean(yx),np.mean(yx)],[0,2.5],'r')
            plt.hist(yx, bins='fd', density=True,
                     color='C0', alpha=0.3, label='_nolegend_')
            plt.ylabel(r'Probability density')
            plt.xlabel(r'$\Delta_\parallel \times \Delta\psi$ [$\mu$m $\times$ rad]')
            plt.grid(True, which='both')
            plt.xlim([-2.5,2.5])
            plt.ylim([0,2.5])
            plt.title(ThName + ' (num-interval: %d)' %numInt)
            plt.legend([str(np.round(np.mean(yx),3)) +
                       '$\mu$m x rad'], prop={'size': 20})
            plt.savefig(os.path.join(savingFolder, ThName + '-' + 
                                     str(numInt).zfill(2) + '.png'))
            plt.clf()
            plt.close()
            # plt.show()

#%% Reload movies to check
checknpy = 1
makeMov = 0
if checknpy:

    viewer = napari.Viewer(ndisplay=3)      
    # viewer.add_image(intensity, contrast_limits=[100,300],\
    #                   scale=[0.115,.115,.115],\
    #                   multiscale=False,colormap='gray',opacity=1)  
    viewer.add_image(stackTh, contrast_limits=[0,1],\
                        scale=[0.115,.115,.115],\
                        multiscale=False,colormap='green',opacity=0.5)
    viewer.scale_bar.visible=True
    viewer.scale_bar.unit='um'
    viewer.scale_bar.position='top_right'
    viewer.axes.visible = True
    napari.run()

if makeMov:
    from naparimovie import Movie
    
    viewer = napari.Viewer(ndisplay=3)
    # viewer.add_image(intensity, contrast_limits=[100,300],\
    #                  scale=[0.115,.115,.115],\
    #                   multiscale=False,colormap='gray',opacity=1)  
    viewer.add_image(stackTh, contrast_limits=[0,1],\
                      scale=[0.115,.115,.115],\
                      multiscale=False,colormap='green',opacity=0.5)
    viewer.scale_bar.visible=True
    viewer.scale_bar.unit='um'
    viewer.scale_bar.position='top_right'
    viewer.axes.visible = True
    movie = Movie(myviewer=viewer)
    
    movie.create_state_dict_from_script('./moviecommands/moviecommands4.txt')
    # movie.make_movie("synthetic.mov",fps=10)
    # movie.make_movie(os.path.join(path, "run-03", ImgName +
    #                               "-threshold.mov"),fps=10)  



#%% Plot the fluctuations
plotFluc = True
if plotFluc:    

    # pitch, roll, yaw
    fig01,ax01 = plt.subplots(dpi=300, figsize=(6,2))
    ax01.plot(np.linspace(0,Nframes-1,num=Nframes)*exp3D_ms,\
              (np.degrees(EuAng[:,0])),c='k',lw=0.5)
    ax01.set_ylim(-600,600)  
    ax01.set_xlabel(r'time [sec]');
    ax01.set_ylabel(r'pitch [deg]')    

    fig02,ax02 = plt.subplots(dpi=300, figsize=(6,2))
    ax02.plot(np.linspace(0,Nframes-1,num=Nframes)*exp3D_ms,\
              (np.degrees(EuAng[:,1])),c='k',lw=0.5)
    ax02.set_ylim(-100,1100)  
    ax02.set_xlabel(r'time [sec]');
    ax02.set_ylabel(r'roll [deg]')
    
    fig03,ax03 = plt.subplots(dpi=300, figsize=(6,2))
    ax03.plot(np.linspace(0,Nframes-1,num=Nframes)*exp3D_ms,\
             (np.degrees(EuAng[:,2])),c='k',lw=0.5)
    ax03.set_ylim(-600,600)  
    ax03.set_xlabel(r'time [sec]');
    ax03.set_ylabel(r'yaw [deg]')

#%% Plot projections (XY, XZ, YZ)
plotin2D = 1
if plotin2D:
    
    # Convert into np.array
    # xb = np.array(xb, dtype=object)
    # xp = np.array(xp, dtype=object)
    # xp0 = np.array(xp0, dtype=object)
    
    xb0 = []
    for i in range(len(xb)):
        xb0.append(xb[i] - cm[i,:])
    xb0 = np.array(xb0, dtype=object)
    
    iframe = 35
    
    # x-y plot
    fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
    ax0.axis('equal')
    ax0.scatter(xp0[iframe][:,0]*pxum,xp0[iframe][:,1]*pxum,c='k',alpha=0.3)
    ax0.scatter(xp0[iframe+55][:,0]*pxum,xp0[iframe+55][:,1]*pxum,c='r',alpha=0.3)
    ax0.set_xlabel(r'x [$\mu m$]'); ax0.set_ylabel(r'y [$\mu m$]')
    # fig0.savefig('filename.pdf')
    
    fig1,ax1 = plt.subplots(dpi=300, figsize=(6,5))
    ax1.axis('equal')
    ax1.scatter(xb0[iframe][:,0]*pxum,xb0[iframe][:,1]*pxum,c='k',alpha=0.3)
    ax1.scatter(xb0[iframe+55][:,0]*pxum,xb0[iframe+55][:,1]*pxum,c='r',alpha=0.3)
    ax1.set_xlabel(r'x [$\mu m$]'); ax1.set_ylabel(r'y [$\mu m$]')
    
    # y-z plot
    fig2,ax2 = plt.subplots(dpi=300, figsize=(6,5))
    ax2.axis('equal')
    ax2.scatter(xp0[iframe][:,1]*pxum,xp0[iframe][:,2]*pxum,c='k',alpha=0.3)
    ax2.scatter(xp0[iframe+55][:,1]*pxum,xp0[iframe+55][:,2]*pxum,c='r',alpha=0.3)
    ax2.set_xlabel(r'y [$\mu m$]'); ax2.set_ylabel(r'z [$\mu m$]')
    
    fig3,ax3 = plt.subplots(dpi=300, figsize=(6,5))
    ax3.axis('equal')
    ax3.scatter(xb0[iframe][:,1]*pxum,xb0[iframe][:,2]*pxum,c='k',alpha=0.3)
    ax3.scatter(xb0[iframe+55][:,1]*pxum,xb0[iframe+55][:,2]*pxum,c='r',alpha=0.3)
    ax3.set_xlabel(r'y [$\mu m$]'); ax3.set_ylabel(r'z [$\mu m$]')
    
    # x-z plot
    fig4,ax4 = plt.subplots(dpi=300, figsize=(6,5))
    ax4.axis('equal')
    ax4.scatter(xp0[iframe][:,0]*pxum,xp0[iframe][:,2]*pxum,c='k',alpha=0.3)
    ax4.scatter(xp0[iframe+55][:,0]*pxum,xp0[iframe+55][:,2]*pxum,c='r',alpha=0.3)
    ax4.set_xlabel(r'x [$\mu m$]'); ax4.set_ylabel(r'z [$\mu m$]')
    
    fig5,ax5 = plt.subplots(dpi=300, figsize=(6,5))
    ax5.axis('equal')
    ax5.scatter(xb0[iframe][:,0]*pxum,xb0[iframe][:,2]*pxum,c='k',alpha=0.3)
    ax5.scatter(xb0[iframe+55][:,1]*pxum,xb0[iframe+55][:,2]*pxum,c='r',alpha=0.3)
    ax5.set_xlabel(r'x [$\mu m$]'); ax5.set_ylabel(r'z [$\mu m$]')