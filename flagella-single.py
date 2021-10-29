# Import all necessary libraries
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})
from sklearn.decomposition import PCA
from scipy import stats, optimize
from scipy.ndimage import gaussian_filter
import pandas as pd
import time
import napari
from matmatrix import *
import helixFun
import imProcess
import msd
from msd import regMSD
import movingHx  
   
simORreal = False # True: simulation, False: actual data
if simORreal: # synthetic data

    # input helix geometry
    length = 8; radius = 0.3; pitchHx = 2.5 # all three in um
    chirality = 1;  # left-handed: 1, right-handed: -1
    resol = 200     # number of points/resolutions
    
    # input helix motion
    Nframes = 100; spin = 0; drift = 0;
    transRange = 2.0 # in um 
    pitchRange = 30.; rollRange = 60.; yawRange = 40.; # in degree
    
    # input helix-and-noise level
    hxInt = 200; hxVar = 0;
    noiseInt = 0; noiseVar = 0;
    
    # create movies of synthetic helix
    fromHx = movingHx.createMovHx(length, radius, pitchHx,\
                                  chirality, resol, Nframes, transRange,\
                                  pitchRange, rollRange, yawRange,\
                                  spin, drift,\
                                  hxInt, hxVar, noiseInt, noiseVar)
    
    intensity, cmInput, EuAngInput, dirAngInput, vectNInput = fromHx.movHx()
    
    Nframes = intensity.shape[0]
        
    print('--synthetic data creation is completed--')
else:
    path = r"../../Result-data/"
    fName = path + '20211022a_suc40_h15um/done/suc40-h15-03-A.npy'
    
    intensity0 = da.from_npy_stack(fName)
    intensity = intensity0[:,:,:,:] 
    Nframes = intensity.shape[0]
    frame0 = intensity[0].compute()
    print("Max pixel:", np.max(frame0[np.nonzero(frame0)]),"\n",\
          "Min pixel:", np.min(frame0[np.nonzero(frame0)]),"\n",\
          "Avg pixel:", np.mean(frame0[np.nonzero(frame0)]),"\n",\
          "Num frame:", len(intensity) )

# Image analysis and curve fitting
xb = []; xp = []; xp0 = []
blobSkel = []; blobBin =[]; blobSize = []
eigenvec = []; blobRaw = []; xb0 = []
    
lenfla = np.zeros(Nframes)
cm = np.zeros([Nframes,3]); 
eigenvec = np.zeros([Nframes,3,3]);
coord = [];
localAxes = np.zeros([Nframes,3,3]);
endpt = np.zeros([Nframes]).astype('int')

pxum = 0.115
start = time.perf_counter()

for frame in range(100):

    if simORreal:
        # Extract coordinates from the image
        # img_temp = intensity[frame]
        # coord.append(np.argwhere(img_temp))
        # X0 = coord[frame]
        # xb.append(X0)
        # CM1 = np.array([sum(X0[:,j]) for j in range(X0.shape[1])])/X0.shape[0]
        # cm[frame,:] = CM1
        
        # Using threshold
        thresvalue = 0.9; sizes = 1200;    
        fromImgPro = imProcess.ImPro(intensity[frame],thresvalue)
        img = fromImgPro.thresVol()                     # binary image
        skel,blob,sizes = fromImgPro.selectLargest()    # largest body only
        X0 = fromImgPro.extCoord()          # extract coordinates
        CM1 = fromImgPro.computeCM()        # compute center of mass
        blobBin.append(blob); blobSize.append(sizes);
        blobSkel.append(skel); blobRaw.append(img);
        xb.append(X0); cm[frame,:] = CM1
 
    else:
        # Image processing
        tstart_thresh = time.perf_counter()
        thresvalue = 0.7; sizes = 0;    
        fromImgPro = imProcess.ImPro(intensity[frame],thresvalue)
        img = fromImgPro.thresVol()                     # binary image
        sizes = max(fromImgPro.selectLargest()[0])    # largest body only
        
        att = 0;
        if sizes < 900:
            att = 1
            while sizes < 900 and att < 20: 
                thresvalue = thresvalue - 0.025
                fromImgPro = imProcess.ImPro(intensity[frame],thresvalue)
                img = fromImgPro.thresVol()                 
                sizes = max(fromImgPro.selectLargest()[0])
                att = att + 1
        thres_dur = time.perf_counter() - tstart_thresh
        print("thresholding took %0.2fs with %d attemps"\
              % (thres_dur, att))

        blob = fromImgPro.BlobAndSkel()
                   
        X0 = fromImgPro.extCoord()          # extract coordinates
        CM1 = fromImgPro.computeCM()        # compute center of mass

        # Store binary image, coordinates and center of mass
        blobBin.append(blob); blobSize.append(sizes);
        blobRaw.append(img);
        xb.append(X0); cm[frame,:] = CM1
           
    # Compute the flagella length    
    lenfla[frame] = flaLength(X0)*pxum
        
    # Use PCA to find n1
    X = X0 - CM1 # shift all the coordinates into origin
    pca = PCA(n_components=3)
    pca.fit(X)
    axes = pca.components_
    
    # Make the PCA consistent
    if frame == 0:
        axes_ref = axes
    else:
        axes, axes_ref = consistentPCA(axes, axes_ref)
    
    # Find the second vector orthogonal to the major axis using skeletonize
    if frame == 0:
        ep_ref = 0
        ep, Coord = endPoints(X0, CM1, axes)
    else:
        ep, Coord = endPoints(X0, CM1, axes)
        ep_ref = ep
    endpt[frame] = ep.astype('int')
    xb0.append(Coord)
    
    # Use Gram-Schmidt to find n2, then find n3 with the cross
    n1 = axes[0] / np.linalg.norm(axes[0])
    n2 = Coord[ep] - np.array([0,0,0])
    n2 -= n2.dot(n1) * n1 / np.linalg.norm(n1)**2
    n2 /= np.linalg.norm(n2)
    n3 = np.cross(n1,n2)
    n3 /= np.linalg.norm(n3)
    localAxes[frame,0] = n1; localAxes[frame,1] = n2; localAxes[frame,2] = n3;
    
    # Rotate to the principal axes
    P0 = np.matmul(axes,X.T).T
    xp0.append(P0)  
    eigenvec[frame] = axes

    # Print each volume frame has finished
    end = time.perf_counter()
    print('Processed done for frame#:',frame,
          '\n elapsed time (sec):',np.round(end-start,2),
          '\n flagella length (um):', lenfla[frame])

# write blobBin external file as Numpy array
blobBin = da.from_array(blobBin)
da.to_npy_stack(fName[:len(fName)-4] + '-threshold.npy',blobBin)  

# Reload movies to check
checknpy = 0
makeMov = 0
if checknpy:
    # blobBin = da.from_array(blobBin)

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(intensity, contrast_limits=[100,200],\
                     scale=[0.115,.115,.115],\
                      multiscale=False,colormap='gray',opacity=1)  
    viewer.add_image(test, contrast_limits=[0,1],\
                      scale=[0.115,.115,.115],\
                      multiscale=False,colormap='green',opacity=0.5) 
    viewer.scale_bar.visible=True
    viewer.scale_bar.unit='um'
    viewer.scale_bar.position='top_right'
    viewer.axes.visible = True
    napari.run()

if makeMov:
    from naparimovie import Movie
    # blobBin = da.from_array(blobBin)
    # blobSkel = da.from_array(blobSkel)
    
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(intensity, contrast_limits=[100,300],\
                     scale=[0.115,.115,.115],\
                      multiscale=False,colormap='gray',opacity=1)  
    viewer.add_image(blobBin, contrast_limits=[0,1],\
                      scale=[0.115,.115,.115],\
                      multiscale=False,colormap='green',opacity=0.5)
    viewer.add_image(blobSkel, contrast_limits=[0,1],\
                      scale=[0.115,.115,.115],\
                      multiscale=False,colormap='green',opacity=0.5)
    viewer.scale_bar.visible=True
    viewer.scale_bar.unit='um'
    viewer.scale_bar.position='top_right'
    viewer.axes.visible = True
    movie = Movie(myviewer=viewer)
    
    movie.create_state_dict_from_script('./moviecommands/moviecommands4.txt')
    # movie.make_movie("synthetic.mov",fps=10)
    movie.make_movie(fName+ "-skel.mov",fps=10)  


#%% Compute pitch, roll, and yaw (Bernie's method)
n1 = localAxes[:,0]; n2 = localAxes[:,1]; n3 = localAxes[:,2]
dpitch = np.zeros(Nframes); droll = np.zeros(Nframes); dyaw = np.zeros(Nframes)
for frame in range(Nframes-1):
    dpitch[frame] = np.dot(n2[frame], n1[frame+1] - n1[frame])
    droll[frame] = np.dot(n3[frame], n2[frame+1] - n2[frame])
    dyaw[frame] = np.dot(n1[frame], n3[frame+1] - n3[frame])
    # dpitch[frame] = np.dot((n2[frame]+n2[frame+1])/abs(n2[frame]+n2[frame+1]),\
    #                         n1[frame+1] - n1[frame])
    # droll[frame] = np.dot((n3[frame]+n3[frame+1])/abs(n3[frame]+n3[frame+1]),\
    #                       n2[frame+1] - n2[frame])
    # dyaw[frame] = np.dot(-(n3[frame]+n3[frame+1])/abs(n3[frame]+n3[frame+1]),\
    #                       n1[frame+1] - n1[frame])
EuAng = np.zeros([Nframes,3]);
for frame in range(Nframes):
    EuAng[frame,0] = np.sum(dpitch[0:frame+1])
    EuAng[frame,1] = np.sum(droll[0:frame+1])
    EuAng[frame,2] = np.sum(dyaw[0:frame+1])

# second-order numerics
dpitch2 = np.zeros(Nframes); droll2 = np.zeros(Nframes); dyaw2 = np.zeros(Nframes)
dpitch2[0] = np.dot(n2[0], n1[1] - n1[0])
droll2[0] = np.dot(n3[0], n2[1] - n2[0])
dyaw2[0] = np.dot(-n3[0], n1[1] - n1[0])
for frame in range(2,Nframes-2):
    dpitch2[frame] = np.dot(n2[frame], (n1[frame+1] - n1[frame-1])/2)
    droll2[frame] = np.dot(n3[frame], (n2[frame+1] - n2[frame-1])/2)
    dyaw2[frame] = np.dot(-n3[frame], (n1[frame+1] - n1[frame-1])/2)    
dpitch2[-2] = np.dot(n2[-2], n1[-1] - n1[-2])
droll2[-2] = np.dot(n3[-2], n2[-1] - n2[-2])
dyaw2[-2] = np.dot(-n3[-2], n1[-1] - n1[-2])
EuAng2 = np.zeros([Nframes,3]);
for frame in range(Nframes):
    EuAng2[frame,0] = np.sum(dpitch2[0:frame+1])
    EuAng2[frame,1] = np.sum(droll2[0:frame+1])
    EuAng2[frame,2] = np.sum(dyaw2[0:frame+1])
    
#%% Compute absolute angle relative to x, y,and z axis
n1 = localAxes[:,0]
dirAng = np.zeros([Nframes,3]); 
for frame in range(Nframes):
    dirAng[frame,0] = np.arccos(n1[frame,0])
    dirAng[frame,1] = np.arccos(n1[frame,1])
    dirAng[frame,2] = np.arccos(n1[frame,2])
  
#%% Plot in 3D space
plotin3D = True
if plotin3D:
    import matplotlib.animation as animation
    
    fig = plt.figure(dpi=300, figsize = (10, 7))
    ax = fig.add_subplot(111,projection='3d')
    
    # Make a 3D quiver plot
    x, y, z = np.array([[-30,0,0],[0,-30,0],[0,0,-30]])*pxum
    u, v, w = np.array([[60,0,0],[0,60,0],[0,0,60]])*pxum
    ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
    
    # plot data
    ax.set_ylim(-30*pxum,30*pxum); ax.set_xlim(-30*pxum,30*pxum);
    ax.set_zlim(-30*pxum,30*pxum); ax.view_init(elev=30, azim=30)

    ax.set_xlabel(r'x [$\mu m$]'); ax.set_ylabel(r'y [$\mu m$]')
    ax.set_zlabel(r'z [$\mu m$]')
    
    iframe = 10;
    # endpt = endpt.astype('int')
    ax.scatter(xb[iframe][:,0]*pxum, xb[iframe][:,1]*pxum,\
                xb[iframe][:,2]*pxum, c = 'k', alpha=0.1)
    ax.scatter(xb[iframe][endpt[iframe],0]*pxum,\
               xb[iframe][endpt[iframe],1]*pxum,\
               xb[iframe][endpt[iframe],2]*pxum, c = 'r', alpha=1)    
    # ax.scatter(xp0[iframe][:,0]*pxum, xp0[iframe][:,1]*pxum,\
    #             xp0[iframe][:,2]*pxum, c = 'g', alpha=0.1)
    # ax.scatter(xb0[iframe][:,0]*pxum, xb0[iframe][:,1]*pxum,\
    #             xb0[iframe][:,2]*pxum, c = 'r',alpha=0.1)
    # ax.scatter(xb0skel[iframe][:,0]*pxum, xb0skel[iframe][:,1]*pxum,\
    #             xb0skel[iframe][:,2]*pxum, c = 'r',alpha=0.2)     
    # ax.scatter(xb0skel[iframe][endpt[iframe],0]*pxum,\
    #             xb0skel[iframe][endpt[iframe],1]*pxum,\
    #             xb0skel[iframe][endpt[iframe],2]*pxum, c = 'g')    
    
    # origin = [0,0,0]
    # X, Y, Z = zip(origin)
    # U0, V0, W0 = zip(list(5*vectNInput[iframe,0]))
    # U1, V1, W1 = zip(list(5*vectNInput[iframe,1]))
    # U2, V2, W2 = zip(list(5*vectNInput[iframe,2]))
    # Uaux, Vaux, Waux = zip(xb0skel[iframe][endpt[iframe]]*pxum)
    # Un1, Vn1, Wn1 = zip(list(3*localAxes[iframe,0])) 
    # Un2, Vn2, Wn2 = zip(list(3*localAxes[iframe,1])) 
    # Un3, Vn3, Wn3 = zip(list(3*localAxes[iframe,2]))
    # Ut1, Vt1, Wt1 = zip(list(5*n1[iframe])) 
    # ax.quiver(X,Y,Z,Uaux,Vaux,Waux, color='b')
    # ax.quiver(X,Y,Z,U0,V0,W0,color='g')
    # ax.quiver(X,Y,Z,U1,V1,W1,color='g')
    # ax.quiver(X,Y,Z,U2,V2,W2,color='g',alpha=0.5)
    # ax.quiver(X,Y,Z,Ut1,Vt1,Wt1,color='g')
    # ax.quiver(X,Y,Z,Un1,Vn1,Wn1,color='g')
    # ax.quiver(X,Y,Z,Un2,Vn2,Wn2,color='g')
    # ax.quiver(X,Y,Z,Un3,Vn3,Wn3,color='g')

#%% plot the fluctuations
plotFluc = True; flucToPDF = False
expTime = 37.5e-3 # 37.5 ms
if plotFluc:    

    # pitch, roll, yaw
    fig01,ax01 = plt.subplots(dpi=300, figsize=(6,2))
    if simORreal:
        ax01.plot(np.linspace(0,Nframes-1,num=Nframes)*0.05,\
                  -np.degrees(EuAngInput[:,0]),\
                  c='k',marker="o",alpha=0.2)
        ax01.plot(np.linspace(0,Nframes-1,num=Nframes)*0.05,\
                  np.degrees(EuAng[:,0]),\
                  c='g',marker="o",mfc='none',alpha=0.4)
        # ax01.plot(np.linspace(0,Nframes-1,num=Nframes)*0.05,\
        #           np.degrees(EuAng2[:,0]),\
        #           c='r',marker="o",mfc='none',alpha=0.4)
    else:
        ax01.plot(np.linspace(0,Nframes-1,num=Nframes)*expTime,\
                  (np.degrees(EuAng[:,0])),c='k',lw=0.5)
        ax01.set_ylim(-600,600)  
    ax01.set_xlabel(r'time [sec]');
    ax01.set_ylabel(r'pitch [deg]')    
    if flucToPDF: fig01.savefig(r'./PDF/EuAng-pitch.pdf')

    fig02,ax02 = plt.subplots(dpi=300, figsize=(6,2))
    if simORreal:
        ax02.plot(np.linspace(0,Nframes-1,num=Nframes)*0.05,\
                  np.degrees(EuAngInput[:,1]),\
                  c='k',marker="o",alpha=0.2)
        ax02.plot(np.linspace(0,Nframes-1,num=Nframes)*0.05,\
                  np.degrees(EuAng[:,1]),\
                  c='g',marker="o",mfc='none',alpha=0.4)
        # ax02.plot(np.linspace(0,Nframes-1,num=Nframes)*0.05,\
        #           np.degrees(EuAng2[:,1]),\
        #           c='r',marker="o",mfc='none',alpha=0.4)
    else:
        ax02.plot(np.linspace(0,Nframes-1,num=Nframes)*expTime,\
                  (np.degrees(EuAng[:,1])),c='k',lw=0.5)
        ax02.set_ylim(-100,1100)  
    ax02.set_xlabel(r'time [sec]');
    ax02.set_ylabel(r'roll [deg]')
    if flucToPDF: fig02.savefig(r'./PDF/EuAng-roll.pdf')
    
    fig03,ax03 = plt.subplots(dpi=300, figsize=(6,2))
    if simORreal:
        ax03.plot(np.linspace(0,Nframes-1,num=Nframes)*0.05,\
                  -np.degrees(EuAngInput[:,2]),\
                  c='k',marker="o",alpha=0.2)
        ax03.plot(np.linspace(0,Nframes-1,num=Nframes)*0.05,\
                  np.degrees(EuAng[:,2]),\
                  c='g',marker="o",mfc='none',alpha=0.4)
        # ax03.plot(np.linspace(0,Nframes-1,num=Nframes)*0.05,\
        #           np.degrees(EuAng2[:,2]),\
        #           c='r',marker="o",mfc='none',alpha=0.4)
    else:
        ax03.plot(np.linspace(0,Nframes-1,num=Nframes)*expTime,\
                  (np.degrees(EuAng[:,2])),c='k',lw=0.5)
        ax03.set_ylim(-600,600)  
    ax03.set_xlabel(r'time [sec]');
    ax03.set_ylabel(r'yaw [deg]')
    if flucToPDF: fig03.savefig(r'./PDF/EuAng-yaw.pdf')
    
    # direction angles
    fig04,ax04 = plt.subplots(dpi=300, figsize=(6,2))
    if simORreal:
        ax04.plot(np.linspace(0,Nframes-1,num=Nframes)*0.05,\
                  np.degrees(dirAngInput[:,0]),\
                  c='k',marker="o",alpha=0.2)
        ax04.plot(np.linspace(0,Nframes-1,num=Nframes)*0.05,\
                  np.degrees(dirAng[:,0]),\
                  c='g',marker="o",mfc='none',ms=9,alpha=0.4)    
    else:
        ax04.plot(np.linspace(0,Nframes-1,num=Nframes)*expTime,\
                  np.degrees(dirAng[:,0]),c='k',lw=0.5)
        ax04.set_ylim(-100,200)  
    ax04.set_ylabel(r'angle to +y-axis [deg]')
    ax04.set_xlabel(r'time [sec]');
    if flucToPDF: fig04.savefig(r'./PDF/dirAng-y-axis.pdf')

    fig05,ax05 = plt.subplots(dpi=300, figsize=(6,2))
    if simORreal:
        ax05.plot(np.linspace(0,Nframes-1,num=Nframes)*0.05,\
                  np.degrees(dirAngInput[:,1]),\
                  c='k',marker="o",alpha=0.2)
        ax05.plot(np.linspace(0,Nframes-1,num=Nframes)*0.05,\
                  np.degrees(dirAng[:,1]),\
                  c='g',marker="o",mfc='none',ms=9,alpha=0.4)
    else:
        ax05.plot(np.linspace(0,Nframes-1,num=Nframes)*expTime,\
                  np.degrees(dirAng[:,1]),c='k',lw=0.5)
        ax05.set_ylim(-200,100)  
    ax05.set_xlabel(r'time [sec]');
    ax05.set_ylabel(r'angle to +x-axis [deg]')
    if flucToPDF: fig05.savefig(r'./PDF/dirAng-x-axis.pdf')
    
    fig06,ax06 = plt.subplots(dpi=300, figsize=(6,2))
    if simORreal:
        ax06.plot(np.linspace(0,Nframes-1,num=Nframes)*0.05,\
                  np.degrees(dirAngInput[:,2]),\
                  c='k',marker="o",alpha=0.2)
        ax06.plot(np.linspace(0,Nframes-1,num=Nframes)*0.05,\
                  np.degrees(dirAng[:,2]),\
                  c='g',marker="o",mfc='none',ms=9,alpha=0.4)
    else:
        ax06.plot(np.linspace(0,Nframes-1,num=Nframes)*expTime,\
                  np.degrees(dirAng[:,2]),c='k',lw=0.5)
        ax06.set_ylim(-200,100)  
    ax06.set_xlabel(r'time [sec]');
    ax06.set_ylabel(r'angle to +z-axis [deg]')
    if flucToPDF: fig06.savefig(r'./PDF/dirAng-z-axis.pdf')

#%% Compute and plot the MSD
allMSD = True; msdToPDF = False
if allMSD: 
        
    # All the MSD of interest
    fromMSD = msd.theMSD(0.8, Nframes, cm, dirAng, EuAng[:,1])
    time_x, MSD_N, MSD_S1, MSD_S2 = fromMSD.trans_MSD()
    time_x, MSD_combo = fromMSD.combo_MSD()
    time_x, MSD_pitch = regMSD(0.8, Nframes, EuAng[:,0])
    time_x, MSD_roll = regMSD(0.8, Nframes, EuAng[:,1])
    time_x, MSD_yaw = regMSD(0.8, Nframes, EuAng[:,2])

    # Fit the MSDs curve
    nData = np.int32(expTime*Nframes) # number of data fitted
    def MSDfit(x, a):
        return a * x   
    fitN = optimize.curve_fit(MSDfit, time_x[0:nData], MSD_N[0:nData],p0=0.1)
    fitS1 = optimize.curve_fit(MSDfit, time_x[0:nData], MSD_S1[0:nData],p0=0.1)
    fitS2 = optimize.curve_fit(MSDfit, time_x[0:nData], MSD_S2[0:nData],p0=0.1)
    fitPitch = optimize.curve_fit(MSDfit, time_x[0:nData], MSD_pitch[0:nData],p0=0.1)
    fitRoll = optimize.curve_fit(MSDfit, time_x[0:nData], MSD_roll[0:nData],p0=0.1)
    fitYaw = optimize.curve_fit(MSDfit, time_x[0:nData], MSD_yaw[0:nData],p0=0.1)
    fitCombo = optimize.curve_fit(MSDfit, time_x[0:nData], MSD_combo[0:nData],p0=0.1)
    
    if simORreal:
        
        # All the MSD of interest
        fromMSD_in = msd.theMSD(0.8, Nframes, cmInput, dirAngInput, EuAngInput[:,1])
        time_x, MSD_N_in, MSD_S1_in,  MSD_S2_in = fromMSD_in.trans_MSD()
        time_x, MSD_combo_in = fromMSD_in.combo_MSD()
        time_x, MSD_pitch_in = regMSD(0.8, Nframes, EuAngInput[:,0])
        time_x, MSD_roll_in = regMSD(0.8, Nframes, EuAngInput[:,1])
        time_x, MSD_yaw_in = regMSD(0.8, Nframes, EuAngInput[:,2])
        
        # Fit the MSDs curve
        nData = np.int32(0.05*Nframes) # number of data fitted
        def MSDfit(x, a):
            return a * x   
        fitN_in = optimize.curve_fit(MSDfit, time_x[0:nData],\
                                     MSD_N_in[0:nData],p0=0.1)
        fitS1_in = optimize.curve_fit(MSDfit, time_x[0:nData],\
                                      MSD_S1_in[0:nData],p0=0.1)
        fitS2_in = optimize.curve_fit(MSDfit, time_x[0:nData],\
                                      MSD_S2_in[0:nData],p0=0.1)
        fitRoll_in = optimize.curve_fit(MSDfit, time_x[0:nData],\
                                        MSD_roll_in[0:nData],p0=0.1)
        fitPitch_in = optimize.curve_fit(MSDfit, time_x[0:nData],\
                                         MSD_pitch_in[0:nData],p0=0.1)        
        fitYaw_in = optimize.curve_fit(MSDfit, time_x[0:nData],\
                                       MSD_yaw_in[0:nData],p0=0.1)
        fitCombo_in = optimize.curve_fit(MSDfit, time_x[0:nData],\
                                         MSD_combo_in[0:nData],p0=0.1)

    # Plot all the MSDs  
    fig0,ax0 = plt.subplots(dpi=300, figsize=(6,5))
    # ax0.plot(time_x,MSD_N_in,c='k',marker="^",mfc='none',ls='None',alpha=0.5)   
    # ax0.plot(time_x,MSD_S1_in,c='k',marker="s",mfc='none',ls='None',alpha=0.5)   
    # ax0.plot(time_x,MSD_S2_in,c='k',marker="o",mfc='none',ls='None',alpha=0.5) 
    ax0.plot(time_x,MSD_N,c='k',marker="^",mfc='none',ms=9,ls='None',alpha=0.5)   
    ax0.plot(time_x,MSD_S1,c='k',marker="s",mfc='none',ms=9,ls='None',alpha=0.5)   
    ax0.plot(time_x,MSD_S2,c='k',marker="o",mfc='none',ms=9,ls='None',alpha=0.5)  
    ax0.plot(time_x,fitN[0]*time_x,c='k',alpha=0.2)
    ax0.plot(time_x,fitS1[0]*time_x,c='k',alpha=0.2)
    ax0.plot(time_x,fitS2[0]*time_x,c='k',alpha=0.2)
    # ax0.plot(time_x,time_x**2,c='b',alpha=0.2) 
    ax0.set_xscale('log'); ax0.set_yscale('log'); 
    ax0.set_title('MSD translation')
    ax0.set_xlabel(r'Log($\tau$) [sec]');
    ax0.set_ylabel(r'Log(MSD) [$\mu m^2$/sec]')
    # ax0.set_ylim([np.exp(-0.5*10e-1),np.exp(10^4)])
    ax0.legend(["parallel","perpendicular-1","perpendicular-2"])
    if msdToPDF: fig0.savefig(r'./PDF/MSD-trans.pdf')

    fig3,ax3 = plt.subplots(dpi=300, figsize=(6,5))
    # ax3.plot(time_x,MSD_pitch_in,c='k',marker="^",mfc='none',ls='None',alpha=0.5)  
    # ax3.plot(time_x,MSD_roll_in,c='k',marker="s",mfc='none',ls='None',alpha=0.5)  
    # ax3.plot(time_x,MSD_yaw_in,c='k',marker="o",mfc='none',ls='None',alpha=0.5)
    ax3.plot(time_x,MSD_pitch,c='k',marker="^",mfc='none',ls='None',alpha=0.5)  
    ax3.plot(time_x,MSD_roll,c='k',marker="s",mfc='none',ls='None',alpha=0.5)  
    ax3.plot(time_x,MSD_yaw,c='k',marker="o",mfc='none',ls='None',alpha=0.5)
    ax3.plot(time_x,fitPitch[0]*time_x,c='k',alpha=0.2)
    ax3.plot(time_x,fitRoll[0]*time_x,c='k',alpha=0.2)
    ax3.plot(time_x,fitYaw[0]*time_x,c='k',alpha=0.2)
    ax3.set_xscale('log'); ax3.set_yscale('log'); 
    ax3.set_title('MSAD for pitch, yaw, and roll')
    ax3.set_xlabel(r'Log($\tau$) [sec]');
    ax3.set_ylabel(r'Log(MSAD) [rad$^2$/sec]')
    ax3.legend(["pitch","roll","yaw"])   
    if msdToPDF: fig3.savefig(r'./PDF/MSD-rot-both.pdf')
    
    fig4,ax4 = plt.subplots(dpi=300, figsize=(6,5))
    plotEnd = 20;
    # ax4.plot(time_x[0:plotEnd],MSD_combo_in[0:plotEnd],c='k',alpha=0.5)
    ax4.plot(time_x[0:plotEnd],MSD_combo[0:plotEnd],c='k',alpha=0.5)
    ax4.plot(time_x[0:plotEnd],fitCombo[0]*time_x[0:plotEnd],c='k',alpha=0.2)
    ax4.set_title('MSD combo')
    # ax4.axis('equal');
    ax4.set_xlabel(r'Log($\tau$) [sec]');
    ax4.set_ylabel(r'$\langle \Delta Y \Delta\psi\rangle [\mu m\cdot rad/sec]$') 
    if msdToPDF: fig4.savefig(r'./PDF/MSD-combo.pdf')



#%% Plot projections (XY, XZ, YZ)
plotin2D = 0
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
    

#%% Plot vectors in 3D to check
vecin3D = True
if vecin3D:
    import matplotlib.animation as animation
    
    fig = plt.figure(dpi=300, figsize = (10, 7))
    ax = fig.add_subplot(111,projection='3d')
    
    scb  = ax.scatter([],[],[], c = 'r',alpha=0.1)
    scep = ax.scatter([],[],[], c = 'g')
    
    # Make a 3D quiver plot
    x, y, z = np.array([[-30,0,0],[0,-30,0],[0,0,-30]])*pxum
    u, v, w = np.array([[60,0,0],[0,60,0],[0,0,60]])*pxum
    ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
    
    ax.set_ylim(-30*pxum,30*pxum); ax.set_xlim(-30*pxum,30*pxum);
    ax.set_zlim(-30*pxum,30*pxum); ax.view_init(elev=30, azim=30)
    ax.set_xlabel(r'x [$\mu m$]'); ax.set_ylabel(r'y [$\mu m$]');
    ax.set_zlabel(r'z [$\mu m$]')
            
    def get_arrow(iter):
        origin = [0,0,0]
        X, Y, Z = zip(origin)
        Un1, Vn1, Wn1 = zip(list(5*localAxes[iter,0])) 
        return X,Y,Z,Un1,Vn1,Wn1

    def get_arrow_n1(iter):
        origin = [0,0,0]
        X, Y, Z = zip(origin)
        Un1, Vn1, Wn1 = zip(list(3*localAxes[iter,0])) 
        return X,Y,Z,Un1,Vn1,Wn1

    def get_arrow_aux(iter):
        origin = [0,0,0]
        ep = endpt[iter]
        X, Y, Z = zip(origin)
        UnAux, VnAux, WnAux = zip(xb0skel[iter][ep]*pxum)
        return X,Y,Z,UnAux,VnAux,WnAux

    def get_arrow_n2(iter):
        origin = [0,0,0]
        X, Y, Z = zip(origin)
        Un2, Vn2, Wn2 = zip(list(3*localAxes[iter,1])) 
        return X,Y,Z,Un2,Vn2,Wn2

    def get_arrow_n3(iter):
        origin = [0,0,0]
        X, Y, Z = zip(origin)
        Un3, Vn3, Wn3 = zip(list(3*localAxes[iter,2])) 
        return X,Y,Z,Un3,Vn3,Wn3

    quiver_n1 = ax.quiver(*get_arrow_n1(0), color = 'g')
    # quiver_aux = ax.quiver(*get_arrow_aux(0), color = 'b')
    quiver_n2 = ax.quiver(*get_arrow_n2(0), color = 'g')
    quiver_n3 = ax.quiver(*get_arrow_n3(0), color = 'g')
    
    def all_plot(iter):
        ep = endpt[iter].astype('int')
        ep1 = ep + 1
        scb._offsets3d = (xb0skel[iter][:,0]*pxum,\
                          xb0skel[iter][:,1]*pxum,\
                              xb0skel[iter][:,2]*pxum)
        scep._offsets3d = (xb0skel[iter][ep:ep1,0]*pxum,\
                           xb0skel[iter][ep:ep1,1]*pxum,\
                              xb0skel[iter][ep:ep1,2]*pxum,)
        global quiver_n1
        global quiver_n2
        global quiver_n3
        # global quiver_aux
        quiver_n1.remove(); quiver_n2.remove(); quiver_n3.remove();
        # quiver_aux.remove();
        quiver_n1 = ax.quiver(*get_arrow_n1(iter), color = 'g')
        # quiver_aux = ax.quiver(*get_arrow_aux(iter), color = 'b')        
        quiver_n2 = ax.quiver(*get_arrow_n2(iter), color = 'g')
        quiver_n3 = ax.quiver(*get_arrow_n3(iter), color = 'g')

            
    anim = animation.FuncAnimation(fig, all_plot, len(xb0skel), interval=30,
                                   blit = False)
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, bitrate=1800,\
                    extra_args=['-vcodec', 'libx264'])
    anim.save('./simIO/synthetic-n1n2n3.mp4', writer=writer)
    
#%% Plot in 3D to check
movin3D = False
if movin3D:
    import matplotlib.animation as animation
    const = 0.115
    
    fig = plt.figure(dpi=300, figsize = (10, 7))
    ax = fig.add_subplot(111,projection='3d')
    
    # scR = ax.scatter([],[],[], c = 'k',alpha=0.1)
    scb = ax.scatter([],[],[], c = 'k',alpha=0.1)
    sc0 = ax.scatter([],[],[], c = 'g',alpha=0.1)
    sc = ax.scatter([],[],[], c = 'b',alpha=0.1)
    
    # Make a 3D quiver plot
    x, y, z = np.array([[-50,0,0],[0,-50,0],[0,0,-50]])*pxum
    u, v, w = np.array([[100,0,0],[0,100,0],[0,0,100]])*pxum
    ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
    
    ax.set_ylim(-50*pxum,50*pxum); ax.set_xlim(-50*pxum,50*pxum);
    ax.set_zlim(-50*pxum,50*pxum)
    ax.view_init(elev=30, azim=30)
    ax.set_xlabel(r'x [$\mu m$]');
    ax.set_ylabel(r'y [$\mu m$]');
    ax.set_zlabel(r'z [$\mu m$]',rotation=0)
    
    xb0 = []
    for i in range(len(coord)):
        xb0.append(coord[i] - cm[i,:])
    
    def all_plot(iter):
        scb._offsets3d = (xb0[iter][:,0]*pxum, xb0[iter][:,1]*pxum,\
                          xb0[iter][:,2]*pxum)
        sc0._offsets3d = (xp0[iter][:,0]*pxum, xp0[iter][:,1]*pxum,\
                          xp0[iter][:,2]*pxum)
        sc._offsets3d = (xp[iter][:,0]*pxum, xp[iter][:,1]*pxum,\
                         xp[iter][:,2]*pxum)
        
    # anim0 = animation.FuncAnimation(fig, only_raw, len(coord), interval=30)
    anim = animation.FuncAnimation(fig, all_plot, len(coord), interval=30)
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, bitrate=1800,\
                    extra_args=['-vcodec', 'libx264'])
    anim.save('./simIO/sim-Centered.mp4', writer=writer)
    
    # # LARGER space
    fig2 = plt.figure(dpi=300, figsize = (10, 7))
    ax2 = fig2.add_subplot(111,projection='3d')
    
    # scR = ax.scatter([],[],[], c = 'k',alpha=0.1)
    scb2 = ax2.scatter([],[],[], c = 'k',alpha=0.1)
    
    # Make a 3D quiver plot
    x, y, z = np.array([[-200,0,0],[0,-200,0],[0,0,-200]])*pxum
    u, v, w = np.array([[400,0,0],[0,400,0],[0,0,400]])*pxum
    ax2.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
    
    ax2.set_ylim(-200*pxum,200*pxum)
    ax2.set_xlim(-200*pxum,200*pxum)
    ax2.set_zlim(-200*pxum,200*pxum)
    ax2.view_init(elev=30, azim=30)
    ax2.set_xlabel(r'x [$\mu m$]');
    ax2.set_ylabel(r'y [$\mu m$]'); 
    ax2.set_zlabel(r'z [$\mu m$]',rotation=0)
    
    def only_raw(iter):
        scb2._offsets3d = (coord[iter][:,0]*const,\
                           coord[iter][:,1]*const,\
                               coord[iter][:,2]*const)
        
    anim0 = animation.FuncAnimation(fig2, only_raw, len(coord), interval=30)
    
    Writer2 = animation.writers['ffmpeg']
    writer2 = Writer(fps=10, bitrate=1800,\
                    extra_args=['-vcodec', 'libx264'])
    anim0.save('./simIO/sim-RawData.mp4', writer=writer2)

#%% Computing matrix A, B, D
computeD = True
if computeD:
    # Measure the radius and pitch length
    radfla = np.zeros(Nframes)
    pitfla = np.zeros(Nframes)
    for i in range(Nframes):
        radfla[i] =  (max(xp0[i][:,2])-min(xp0[i][:,2]) ) *pxum
        pitfla[i] = lenfla[i]/2.5
    
    print('SUMMARY for ' + fName + ' with total',Nframes,'frames')
    print("flagella radius [um] = ", np.mean(radfla),\
          " with std = ", np.std(radfla))
    print("flagella length [um] = ", np.mean(lenfla),\
          " with std = ", np.std(lenfla))
    print("flagella pitch-length [um] = ", np.mean(pitfla),\
          " with std = ", np.std(pitfla))
    print("Fit for parallel, perpen-1, perpen-2:",fitN[0], fitS1[0], fitS2[0])
    print("Fit for pitch, roll, yaw:",fitPitch[0], fitRoll[0], fitYaw[0])
    print("Fit for combo:",fitCombo[0])
    print("Matrix A, B, D for " + fName)
    A, B, D = BernieMatrix(fitN[0]*1e-12,fitRoll[0],fitCombo[0]*1e-6)
    A2, B2, D2 = BernieMatrix(fitN[0]*1e-12*(11.49),fitRoll[0]*(11.49),\
                              fitCombo[0]*1e-6*(11.49)) # 50% sucrose
    print("Propulsion-Matrix (A, B, D):", A, B, D)
    print("50% sucrose adjusted (A, B, D):", A2, B2, D2)
    
    # print to excel
    data = [['number of frames', Nframes],\
            ['radius [um]', np.mean(radfla),np.std(radfla)],\
            ['length [um]', np.mean(lenfla),np.std(lenfla)],\
            ['pitch [um]', np.mean(pitfla),np.std(pitfla)],\
            ['trans-fit [um^2/sec^2]',fitN[0][0], fitS1[0][0], fitS2[0][0]],\
            ['rotation-fit [rad^2/sec^2]',fitPitch[0][0], fitRoll[0][0], fitYaw[0][0]],\
            ['combo-fit [um.rad/sec^2]',fitCombo[0][0]],\
            ['A, B, D', A[0], B[0], D[0]],\
            ['A, B, D (adjusted 50% sucrose)', A2[0], B2[0], D2[0]]\
                ]
    df = pd.DataFrame(data)
    df.to_excel(fName + '.xlsx', index = False, header = False)  
