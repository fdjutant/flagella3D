import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from lmfit import Model
from scipy.special import erf

#%% Fit CDF and PDF of Normal distribution
def MSDfit_old(x, a):
    return a * x  

def MSDfit(x, a, b):
    return b + a * x  

def gauss_cdf(x, amp, mu, sigma):
    Fx = amp/2.0 * (1. + erf( (x-mu)/ (sigma*np.sqrt(2.))))
    return Fx

def gauss_pdf(x, amp, mu, sigma):
    Fx = amp * np.exp(-0.5*(x-mu)**2/sigma**2) / (sigma * np.sqrt(2*np.pi))
    return Fx

def fitCDF_flexible(x, mean, sigma):
    model = Model(gauss_cdf, prefix='g1_')
    params = model.make_params()
    params['g1_amp'].set(1.,vary=False)
    params['g1_mu'].set(mean)
    params['g1_sigma'].set(1.)
    
    yaxis = np.linspace(0,1,len(x), endpoint=False)
    xaxis = np.sort(x)
    result = model.fit(yaxis,params,x=xaxis)
    amp = result.params['g1_amp'].value
    mean = result.params['g1_mu'].value
    sigma = result.params['g1_sigma'].value
    return amp, mean, sigma

def fitCDF_diff(x):
    model = Model(gauss_cdf, prefix='g1_')
    params = model.make_params()
    params['g1_amp'].set(1.,vary=False)
    params['g1_mu'].set(2.)
    params['g1_sigma'].set(1.)
    
    yaxis = np.linspace(0,1,len(x), endpoint=False)
    xaxis = np.sort(x)
    result = model.fit(yaxis,params,x=xaxis)
    amp = result.params['g1_amp'].value
    mean = result.params['g1_mu'].value
    sigma = result.params['g1_sigma'].value
    return amp, mean, sigma

def fitCDF(x):
    model = Model(gauss_cdf, prefix='g1_')
    params = model.make_params()
    params['g1_amp'].set(1., vary=False)
    params['g1_mu'].set(0., vary=False)
    params['g1_sigma'].set(0.5)
    
    yaxis = np.linspace(0,1,len(x), endpoint=False)
    xaxis = np.sort(x)
    result = model.fit(yaxis,params,x=xaxis)
    amp = result.params['g1_amp'].value
    mean = result.params['g1_mu'].value
    sigma = result.params['g1_sigma'].value
    return amp, mean, sigma

def gauss_two_pdf(x, amp1, mu1, mu2, sigma1, sigma2):
    Fx = amp1 * np.exp(-0.5*(x-mu1)**2/sigma1**2) / (sigma1 * np.sqrt(2*np.pi)) +\
         (1.-amp1) *np.exp(-0.5*(x-mu2)**2/sigma2**2) / (sigma2 * np.sqrt(2*np.pi))
    return Fx

def gauss_two_cdf(x, amp1, mu1, mu2, sigma1, sigma2):
    Fx = amp1/2.0 * (1. + erf( (x-mu1)/ (sigma1*np.sqrt(2.)))) +\
         (1. - amp1)/2.0 * (1. + erf( (x-mu2)/ (sigma2*np.sqrt(2.))))
    return Fx 

def fit2CDF(x):
    model = Model(gauss_two_cdf, prefix='g1_') 
    params = model.make_params()
    params['g1_amp1'].set(0.5, min=0., max=1.)
    params['g1_mu1'].set(0., vary=False)
    params['g1_mu2'].set(0., vary=False)
    params['g1_sigma1'].set(0.5)
    params['g1_sigma2'].set(0.5)    
    
    yaxis = np.linspace(0,1,len(x), endpoint=False)
    xaxis = np.sort(x)
    result = model.fit(yaxis,params,x=xaxis)
    if result.params['g1_amp1'].value > 0.5:
        amp1 = result.params['g1_amp1'].value
        sigma1 = result.params['g1_sigma1'].value
        sigma2 = result.params['g1_sigma2'].value
    else:
        amp1= 1 - result.params['g1_amp1'].value
        sigma1 = result.params['g1_sigma2'].value
        sigma2 = result.params['g1_sigma1'].value
    mean1 = result.params['g1_mu1'].value
    mean2 = result.params['g1_mu2'].value
    
    return amp1, mean1, mean2, sigma1, sigma2

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

#%% compute Euler angles from n1, n2, n3
def EuAngfromN(localAxes):
    
    n1 = localAxes[:,0]
    n2 = localAxes[:,1]
    n3 = localAxes[:,2]
    
    Nframes = len(localAxes)
    dpitch = np.zeros(Nframes)
    droll = np.zeros(Nframes)
    dyaw = np.zeros(Nframes)
    for frame in range(Nframes-1):
        dpitch[frame] = np.dot(n2[frame], n1[frame+1] - n1[frame])
        droll[frame] = np.dot(n3[frame], n2[frame+1] - n2[frame])
        dyaw[frame] = np.dot(n1[frame], n3[frame+1] - n3[frame])
    
    EuAng = np.zeros([Nframes,3])
    for frame in range(Nframes):
        EuAng[frame,0] = np.sum(dpitch[0:frame+1])
        EuAng[frame,1] = np.sum(droll[0:frame+1])
        EuAng[frame,2] = np.sum(dyaw[0:frame+1])
    
    return EuAng

#%% moving average
def movAverage(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

#%% find end-points (https://stackoverflow.com/a/60955825)
def hullAnalysis(X,axes):
    
    hull = ConvexHull(X)
    hullpoints = X[hull.vertices,:] # Extract the points forming the hull
    hdist = cdist(hullpoints, hullpoints, metric='euclidean') # finding the best pair
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
    
    # compute length
    pxum = 0.115
    thelength = np.linalg.norm([hullpoints[bestpair[0]]-
                                hullpoints[bestpair[1]]])*pxum
    
    # find end point that is in the direction of n1
    dotTest = np.dot(hullpoints[bestpair[0]], axes[0])
    if dotTest > 0:
        endpoint_Hull = hullpoints[bestpair[0]]
    else:
        endpoint_Hull = hullpoints[bestpair[1]]
        
    return thelength, endpoint_Hull

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
        # lentest2 = np.dot( (hullpoints[bestpair[1]][0] - CM1), axes_ref ) 
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
