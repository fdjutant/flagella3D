import sys
sys.path.insert(0, './modules')
import napari
import dask.array as da
import os.path
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.max_open_warning': 0})
from sklearn.decomposition import PCA
from matmatrix import consistentPCA, hullAnalysis
# import glob

this_file_dir = os.path.dirname(os.path.abspath("./"))
folderName = "20211018b_suc50_h30um"

path = os.path.join(this_file_dir,
                     "DNA-Rotary-Motor", "Helical-nanotubes",
                     "Light-sheet-OPM", "Result-data",
                     folderName)
# nFiles = len(glob.glob(path + '\*.npy'))



ImgName = "suc50-h30-17-A"
ThName =  ImgName + "-threshold"
    
# input Zarr and convert to dask
stackImg = da.from_npy_stack(os.path.join(path, ImgName + ".npy"))
stackTh = da.from_npy_stack(os.path.join(path, "run-03", ThName + ".npy"))


#%% Plot in 3D space
# pixel to um
pxum = 0.115

# move thresholding to origin
xb0 = []; xp = []; xp0 = []
Nframes = len(stackTh)
lenfla = np.zeros(Nframes)
eigenvec = np.zeros([Nframes,3,3])
localAxes = np.zeros([Nframes,3,3])
endpt_idx = np.zeros([Nframes]).astype('int')
endpt_Hull = np.zeros([Nframes,3]).astype('int')
for frame in range(Nframes):
    X0 = np.argwhere(np.asarray(stackTh[frame])).astype('float')
    CM1 = np.array([sum(X0[:,j]) for j in range(X0.shape[1])])/X0.shape[0]
    
    # Use PCA to find the rotation matrix
    X = X0 - CM1 # shift all the coordinates into origin
    pca = PCA(n_components=3)
    pca.fit(X)
    axes = pca.components_
    xb0.append(X)
    
    # Make the PCA consistent
    if frame == 0:
        axes_ref = axes
    else:
        axes, axes_ref = consistentPCA(axes, axes_ref)
    
    # Find the flagella length and endpoint in the direction of n1
    lenfla[frame], endpt_Hull[frame] = hullAnalysis(X,axes)
    
    # Use Gram-Schmidt to find n2, then find n3 with the cross
    n1 = axes[0] / np.linalg.norm(axes[0])
    n2 = endpt_Hull[frame] - np.array([0,0,0])
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
    print('Frames-#: %d' %(frame))
    
#%% Save folder for 3D points
savingFolder = os.path.join(path, 'run-03', ThName+'-snapshots')
if os.path.isdir(savingFolder)!=True:
    os.mkdir(savingFolder) # create path

for iframe in range(Nframes):
    fig = plt.figure(dpi=150, figsize = (10, 7))
    ax = fig.add_subplot(111,projection='3d')
    
    # Make a 3D quiver plot
    x, y, z = np.array([[-30,0,0],[0,-30,0],[0,0,-30]])*pxum
    u, v, w = np.array([[60,0,0],[0,60,0],[0,0,60]])*pxum
    ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
    
    # plot data
    edgePoint = 30
    ax.set_ylim(-edgePoint*pxum,edgePoint*pxum);
    ax.set_xlim(-edgePoint*pxum,edgePoint*pxum);
    ax.set_zlim(-edgePoint*pxum,edgePoint*pxum);
    ax.view_init(elev=30, azim=30)
    
    ax.set_xlabel(r'x [$\mu m$]'); ax.set_ylabel(r'y [$\mu m$]')
    ax.set_zlabel(r'z [$\mu m$]')
    ax.set_title('length = %.3f $\mu$m' %np.round(lenfla[iframe],3))
    
    # iframe = 20
    endpt_idx = endpt_idx.astype('int')
    ax.scatter(xb0[iframe][:,0]*pxum, xb0[iframe][:,1]*pxum,\
                xb0[iframe][:,2]*pxum, c = 'k',alpha=0.1, s=30)
    # ax.scatter(hdist[:,0]*pxum, hdist[:,1]*pxum,\
    #            hdist[:,2]*pxum, c = 'b',alpha=1, s=10)
    ax.scatter(endpt_Hull[iframe,0]*pxum,\
               endpt_Hull[iframe,1]*pxum,\
               endpt_Hull[iframe,2]*pxum, c = 'r', alpha=0.5, s=100) 
    ax.scatter(xb0[iframe][endpt_idx[iframe],0]*pxum,\
               xb0[iframe][endpt_idx[iframe],1]*pxum,\
               xb0[iframe][endpt_idx[iframe],2]*pxum, c = 'b', alpha=0.2, s=100)   
    
    origin = [0,0,0]
    X, Y, Z = zip(origin)
    # U0, V0, W0 = zip(list(5*vectNInput[iframe,0]))
    # U1, V1, W1 = zip(list(5*vectNInput[iframe,1]))
    # U2, V2, W2 = zip(list(5*vectNInput[iframe,2]))
    # ax.quiver(X,Y,Z,Uaux,Vaux,Waux, color='b')
    # ax.quiver(X,Y,Z,U0,V0,W0,color='b')
    # ax.quiver(X,Y,Z,U1,V1,W1,color='g')
    # ax.quiver(X,Y,Z,U2,V2,W2,color='g',alpha=0.5)
    Un1, Vn1, Wn1 = zip(list(5*localAxes[iframe,0])) 
    Un2, Vn2, Wn2 = zip(list(5*localAxes[iframe,1])) 
    Un3, Vn3, Wn3 = zip(list(5*localAxes[iframe,2]))
    ax.quiver(X,Y,Z,Un1,Vn1,Wn1,color='r')
    ax.quiver(X,Y,Z,Un2,Vn2,Wn2,color='g')
    ax.quiver(X,Y,Z,Un3,Vn3,Wn3,color='g')
    ax.figure.savefig(os.path.join(savingFolder, ThName + '-' + 
                                    str(iframe).zfill(3) + '.png'))
    # ax.clear()
    # ax.remove()
    
#%% show in Napari    
# viewer = napari.Viewer(ndisplay=3)
# viewer.add_image(stackImg, contrast_limits=[100,300],
#                  scale=[0.115,.115,.115],colormap='gray',opacity=1)
# viewer.add_image(stackTh, contrast_limits=[0,1],
#                  scale=[0.115,.115,.115],colormap='green',opacity=0.5)
# viewer.scale_bar.visible=True
# viewer.scale_bar.unit='um'
# viewer.scale_bar.position='top_right'
# viewer.axes.visible = True
# napari.run()
