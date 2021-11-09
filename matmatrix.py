import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

#%% Define all necessary functions
def rotmat(angle): # forming rotation matrix
    theta, psi, phi = np.array(angle)

    R11 = np.cos(theta)*np.cos(phi)
    R12 = np.sin(psi)*np.sin(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi)
    R13 = np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi)
            
    R21 = np.cos(theta)*np.sin(phi)
    R22 = np.sin(psi)*np.sin(theta)*np.sin(phi) + np.cos(psi)*np.cos(phi)
    R23 = np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi)
            
    R31 = -np.sin(theta)
    R32 = np.sin(psi)*np.cos(theta)
    R33 = np.cos(psi)*np.cos(theta)      
    
    Mrot = np.array([[R11, R12, R13],[R21, R22, R23],[R31, R32, R33]])
    return Mrot

def retAngles(Mrot):
    
    if Mrot[2,0] != 1 and Mrot[2,0] != -1: 
        theta1 = -np.arcsin(Mrot[2,0])
        theta2 = np.pi - theta1
        
        psi1 = np.arctan2(Mrot[2,1]/np.cos(theta1), Mrot[2,2]/np.cos(theta1))
        psi2 = np.arctan2(Mrot[2,1]/np.cos(theta2), Mrot[2,2]/np.cos(theta2))
        
        phi1 = np.arctan2(Mrot[1,0]/np.cos(theta1), Mrot[0,0]/np.cos(theta1))
        phi2 = np.arctan2(Mrot[1,0]/np.cos(theta2), Mrot[0,0]/np.cos(theta2))
        
        angles1 = np.array([theta1,psi1,phi1])
        angles2 = np.array([theta2,psi2,phi2])
    else:
        phi = 0
        if Mrot[2,0] == -1:
            theta = np.pi/2
            psi = phi + np.arctan2(Mrot[0,1],Mrot[0,2])
            angles1 = np.array([theta,psi,phi])
            angles2 = angles1
        else:
            theta = -np.pi/2
            psi = -phi + np.arctan2(-Mrot[0,1],-Mrot[0,2])
            angles1 = np.array([theta,psi,phi])
            angles2 = angles1
        
    return angles1, angles2

#%% moving average
def movAverage(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

#%% find end-points using skeleton
def endPoints_old(skel,CM1,axes):
    
    skelCoord = [];
    skelCoord = np.argwhere(skel)-CM1
    disToOrigin = []    # find the longest distance from CM
    for i in range(len(skelCoord)):
        disToOrigin.append(np.linalg.norm(skelCoord[i]))
    disToOrigin = np.array(disToOrigin, dtype=object)
    
    # Find the end point in the major axes direction
    k = 1; lentest = -1;
    while lentest < 0:
        endpoint = np.where(disToOrigin == np.sort(disToOrigin)[-k])[0][0]
        k += 1
        lentest = np.dot(skelCoord[endpoint],axes[0])
    
    return endpoint, skelCoord

#%% find end-points
def endPoints(X0,CM1,axes):
    
    Coord = [];
    Coord = X0-CM1
    disToOrigin = []    # find the longest distance from CM
    for i in range(len(Coord)):
        disToOrigin.append(np.linalg.norm(Coord[i]))
    disToOrigin = np.array(disToOrigin, dtype=object)
    
    # Find the end point in the major axes direction
    k = 1; lentest = -1;
    while lentest < 0:
        endpoint = np.where(disToOrigin == np.sort(disToOrigin)[-k])[0][0]
        k += 1
        lentest = np.dot(Coord[endpoint],axes)
    
    return endpoint, Coord

#%% measure length (https://stackoverflow.com/a/60955825)
def flaLength(X0):

    hull = ConvexHull(X0)
    hullpoints = X0[hull.vertices,:] # Extract the points forming the hull
    hdist = cdist(hullpoints, hullpoints, metric='euclidean') # finding the best pair
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)

    thelength = np.linalg.norm([hullpoints[bestpair[0]][0]-hullpoints[bestpair[1]][0]])
    
    return thelength

#%% find n1 using end point
def findN1(X0, CM1, axes_ref):

    hull = ConvexHull(X0)
    hullpoints = X0[hull.vertices,:] # Extract the points forming the hull
    hdist = cdist(hullpoints, hullpoints, metric='euclidean') # finding the best pair
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)

    if np.mean(axes_ref) == 0:
        ep = np.where( hullpoints[bestpair[0]] == X0 )[0][0]
        axes = X0[ep] - CM1
    else:
        lentest1 = np.dot( (hullpoints[bestpair[0]][0] - CM1), axes_ref )
        lentest2 = np.dot( (hullpoints[bestpair[1]][0] - CM1), axes_ref ) 
        if lentest1 > 0:
            ep = np.where( hullpoints[bestpair[0]][0] == X0)[0][0]
            axes = X0[ep] - CM1
        else:
            ep = np.where( hullpoints[bestpair[1]][0] == X0)[0][0]
            axes = X0[ep] - CM1
                    
    return axes, ep

#%% phase unwrapping
def phaseUnwrap(angle):
    
    # angleUn = np.zeros(angle.shape)
    angleUn = angle
    for i in range(1,len(angle)):
        if abs(angleUn[i] - angleUn[i-1]) > np.pi and\
            angle[i] < angleUn[i-1]:
            angleUn[i:] += 2*np.pi
            # angleUn[i] = angleUn[i] + 2*np.pi
            # print('pos 1')
        elif abs(angleUn[i] - angleUn[i-1]) > np.pi and\
            angle[i] > angleUn[i-1]:
            # angleUn[i] = angleUn[i] - 2*np.pi
            angleUn[i:] -= 2*np.pi            
            # print('pos 2')
    
    return angleUn

#%% Ensure PCA order to be consisten
def consistentPCA(axes, axes_ref):
   
    # Normalization
    for k in range(3):
        axes[k] = axes[k]/np.linalg.norm(axes[k])
    
    # Compare axes with reference
    if np.dot(axes[0], axes_ref[0])<0:
        axes[0] *= -1
        
    # Switch around the aux. axes until found best match
    d1 = np.dot(axes[1],axes_ref[1])
    d2 = np.dot(axes[1],axes_ref[2])
    
    if np.abs(d1) > np.abs(d2):
        if d1<0:
            axes[1] *= -1
    else:
        if d2<0:
            axes[1] = -axes[2]
        else:
            axes[1] = axes[2]
    
    '''
    #Flip one of the aux. axes if dot prod. with ref. is negative
    if np.dot(axes[1], axes_ref[1]<0):
        axes[1] *= -1
        
        #Switch aux. axes if rotation is more than 45 deg. relative to ref.
        if np.dot(axes[1],axes_ref[1])<np.sqrt(2)/2:
            axes[1] = axes[2]
            
            #Second check
            if np.dot(axes[1], axes_ref[1]<0):
                axes[1] *= -1
    '''
    
    #Define the 3rd axis as the cross product of the first two for cyclic consistency
    axes[2] = np.cross(axes[0], axes[1])
    
    #Update reference axes
    axes_ref = axes
    
    return axes, axes_ref
#%% Computing A, B, D (Bernie's derivation)
def BernieMatrix(DY, DTh, DYTh):
    kB = 1.380649 * 1e-23  # unit: J/K
    T = 298                 # unit: K
    kBT = kB * T
    
    denom = DY * DTh - DYTh**2
    
    A = DTh * kBT / denom
    B = -DYTh * kBT / denom
    D = DY * kBT / denom
    
    return A, B, D

def invBernieMatrix(A, B, D):
    kB = 1.380649 * 1e-23  # unit: J/K
    T = 298                 # unit: K
    kBT = kB * T
    
    denom = A*D - B**2
    
    DY = D * kBT / denom
    DYTh = -B * kBT / denom
    DTh = A * kBT / denom
    
    return DY, DTh, DYTh
