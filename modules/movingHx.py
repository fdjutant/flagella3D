import numpy as np
# from matmatrix import *
from numba import jit

#%% Generate samples for cm, EuAng, localAxes
@jit(nopython=True)
def angle_sum(n1, n2, n3, das, nframes):
    for ii in range(1, nframes):
        n1[ii] = n1[ii - 1] + das[ii, 0] * n2[ii - 1] - das[ii, 2] * n3[ii - 1]
        n2[ii] = -das[ii, 0] * n1[ii - 1] + n2[ii - 1] + das[ii, 1] * n3[ii - 1]
        n3[ii] = np.cross(n1[ii], n2[ii])

        # normalize the vectors
        n1[ii] /= np.linalg.norm(n1[ii])
        n2[ii] /= np.linalg.norm(n2[ii])
        n3[ii] /= np.linalg.norm(n3[ii])

    return n1, n2, n3

def simulate_diff(nframes,vol_exp,Dpar,Dperp,d_pitch,d_roll,d_yaw):
    
    # parameters
    d_par = Dpar / (0.115**2)
    d_perp = Dperp / (0.115**2)
    
    # generate euler angles
    das = np.stack((np.random.normal(0, np.sqrt(2*vol_exp*d_pitch),size=nframes),
                    np.random.normal(0, np.sqrt(2*vol_exp*d_roll),size=nframes),
                    np.random.normal(0, np.sqrt(2*vol_exp*d_yaw),size=nframes)),axis=1)
    
    euler_angles = np.hstack((np.cumsum(das[:, 0]),
                              np.cumsum(das[:, 1]),
                              np.cumsum(das[:, 2])))

    # generate spatial steps
    dx_par = np.random.normal(0, np.sqrt(2*vol_exp*d_par),size=nframes)
    dx_perp1 = np.random.normal(0, np.sqrt(2*vol_exp*d_perp),size=nframes)
    dx_perp2 = np.random.normal(0, np.sqrt(2*vol_exp*d_perp),size=nframes)

    # generate first orientation
    ang1 = np.radians(31.)
    ang2 = np.radians(65.)
    ang3 = np.arccos(np.sqrt(1-np.cos(ang1)**2-np.cos(ang2)**2))
    n1_start = np.array([ np.cos(ang1),np.cos(ang2),np.cos(ang3) ])
    n2_start = np.array([1., 0., 0.]).astype(float)
    n2_start -= n2_start.dot(n1_start) * n1_start / np.linalg.norm(n1_start)**2
    n2_start /= np.linalg.norm(n2_start)
    n3_start = np.cross(n1_start,n2_start)
    n3_start /= np.linalg.norm(n3_start)
    n1_start /= np.linalg.norm(n1_start)

    # simulate
    ndims = 3
    n1 = np.zeros((nframes, ndims))
    n2 = np.zeros((nframes, ndims))
    n3 = np.zeros((nframes, ndims))
                     
    n1[0,:] = n1_start
    n2[0,:] = n2_start
    n3[0,:] = n3_start
    n1, n2, n3 = angle_sum(n1, n2, n3, das, nframes)
        
    d_cmass = n1 * np.expand_dims(dx_par, axis=1) + \
              n2 * np.expand_dims(dx_perp1, axis=1) + \
              n3 * np.expand_dims(dx_perp2, axis=1)
    #cmass = np.cumsum(d_cmass, axis=0)
    cmass = np.hstack((np.cumsum(d_cmass[:, 0]),
                        np.cumsum(d_cmass[:, 1]),
                        np.cumsum(d_cmass[:, 2])))
    
    return cmass, euler_angles, n1, n2, n3

#%% Create a class to create helix
class createMovHx:
    def __init__(self, length, radius, pitchHx, chirality, resol, Nframes,\
                 Dpar, Dperp,\
                 Dpitch, Droll, Dyaw,\
                 spin, drift,\
                 hxInt, hxVar, noiseInt, noiseVar, vol_exp):
        
        self.Nframes = Nframes
        self.spin = spin
        self.drift = UmToPx(drift)
        
        self.Dpar  = Dpar/(0.115**2)
        self.Dperp = Dperp/(0.115**2)
        
        self.Dpitch = Dpitch
        self.Droll = Droll
        self.Dyaw = Dyaw

        self.chirality = chirality
        self.resol = resol
        self.pitchHx = UmToPx(pitchHx)
        self.radius = UmToPx(radius)
        self.length= UmToPx(length)
        
        self.hxInt = hxInt
        self.hxVar = hxVar
        self.noiseInt = noiseInt
        self.noiseVar = noiseVar
        
        self.vol_exp = vol_exp
     
    def movHx(self):
               
        # Pitch, roll, and yaw (relative to the local axes)
        da1 = np.random.normal(0, np.sqrt(2*self.vol_exp*self.Dpitch), size=self.Nframes)
        da2 = np.random.normal(0, np.sqrt(2*self.vol_exp*self.Droll), size=self.Nframes)
        da3 = np.random.normal(0, np.sqrt(2*self.vol_exp*self.Dyaw), size=self.Nframes)
        
        EuAng = np.stack((np.cumsum(da1),
                          np.cumsum(da2),
                          np.cumsum(da3)), axis=1)
    
        # The three unit vectors
        n1 = np.zeros([self.Nframes,3])
        n2 = np.zeros([self.Nframes,3])
        n3 = np.zeros([self.Nframes,3])
        vectN = np.zeros([self.Nframes,3,3])

        # Initial orientation (n1 - relative to the positive lab axes)
        ang1 = np.radians(31.)
        ang2 = np.radians(65.)
        ang3 = np.arccos(np.sqrt(1-np.cos(ang1)**2-np.cos(ang2)**2))
        
        dpar =  np.random.normal(0, np.sqrt(2*self.vol_exp*self.Dpar), size=self.Nframes)
        dperp1 = np.random.normal(0, np.sqrt(2*self.vol_exp*self.Dperp), size=self.Nframes)
        dperp2 = np.random.normal(0, np.sqrt(2*self.vol_exp*self.Dperp), size=self.Nframes)
                
        cm = np.zeros([self.Nframes,3])
        points = self.createHx(0)  # generate the helix coordinates
        for i in range(self.Nframes):
            
            if i == 0:
                n1[i] = np.array([ np.cos(ang1),np.cos(ang2),np.cos(ang3) ])
                n2[i] = points[0,:]
                n2[i] -= n2[i].dot(n1[i]) * n1[i] / np.linalg.norm(n1[i])**2
                n2[i] /= np.linalg.norm(n2[i])
                n3[i] = np.cross(n1[i],n2[i])
                n3[i] /= np.linalg.norm(n3[i])
                
                n1[i] /= np.linalg.norm(n1[i])
                cm[i] = 0
            else:
                n1[i] = n1[i-1] + ( (EuAng[i,0]-EuAng[i-1,0]) * n2[i-1] -\
                                    (EuAng[i,2]-EuAng[i-1,2]) * n3[i-1])
                n2[i] = n2[i-1] + (-(EuAng[i,0]-EuAng[i-1,0]) * n1[i-1] +\
                                    (EuAng[i,1]-EuAng[i-1,1]) * n3[i-1])
                n3[i] = np.cross(n1[i],n2[i])
                # n3[i] = n3[i-1] + ( (EuAng[i,2]-EuAng[i-1,2]) * n1[i-1] -\
                #                     (EuAng[i,1]-EuAng[i-1,1]) * n2[i-1])
                               
                # normalize the vectors
                n1[i] /= np.linalg.norm(n1[i])
                n2[i] /= np.linalg.norm(n2[i])
                n3[i] /= np.linalg.norm(n3[i])
                
                # take step = position + n1 * step_par + ...
                cm[i,:] = cm[i-1,:] + (n1[i] * dpar[i]) + (n2[i] * dperp1[i]) + (n3[i] * dperp2[i])
            
            vectN[i] = np.array([ n1[i], n2[i], n3[i] ])
        
        return cm, EuAng, vectN
    
    def makeMov(self, CM, vectN):
        
        boxSize = self.length.astype('int') + 200
        img = np.zeros((boxSize,boxSize,boxSize),dtype='uint8')
        
        box = img.shape
        origin = np.array(box)/2
        
        points = self.createHx(0)     
        n1 = vectN[0]; n2 = vectN[1]; n3 = vectN[2]; 

        axes = np.array([ n2, n3, n1 ])
        frame = CM + origin + np.matmul(axes.T,points.T).T
        frame = self.digitize(frame,0)
        for f in frame:
            x, y, z = f
            if self.hxVar == 0:
                img[x,y,z] = self.hxInt
            else:
                img[x,y,z] = np.random.uniform(self.hxInt-self.hxVar,\
                                               self.hxInt+self.hxVar)

        img = dilation(img)
        img = dilation(img)
        
        # add salt-and-pepper noise 
        img = self.addNoise(img)
        
        return img
    
    def addNoise(self,img):
        totalNum = img.shape[0]*img.shape[1]*img.shape[2]
        saltpepper = np.reshape(np.random.normal(self.noiseInt,\
                                                 self.noiseVar,\
                                                     totalNum),img.shape)
        img = np.add(img,saltpepper)
        del saltpepper # conserve memory
        
        return img
        
    def createHx(self,phase):
        k = 2*np.pi/self.pitchHx
        points = []        
        for i in range(self.resol):
            z = (-0.5+i/(self.resol-1))*self.length
            x = self.radius*np.cos(k*z+self.chirality*phase)
            y = self.radius*np.sin(k*z+self.chirality*phase)
            
            points.append([x,y,z])
            
        points = np.array(points)
        CM = sum(points)/len(points)
        return points-CM

    def digitize(self,points,padding):
        points = points.astype('int')
        p = points.copy()

        if padding:
            for i in range(1, padding+1):
                for loc in range(1,27):
                    shift = np.zeros(3,dtype='int')
                    temp = loc
                    for j in range(3):
                        for k in range(j+1):
                            temp = temp//3
                        shift[2-j] = temp%3
                    shift += -1
                    p = np.append(p,points+i*shift,axis=0)
        
        return np.unique(p,axis=0)
            
    def rotateMainAxis(self,points,n1,n2,n3):
        
        axes = np.array([n1,n2,n3])
        
        return np.matmul(axes.T,points.T).T
        # return np.dot(points,axes.T)
    
#%% convert from um to px    
def UmToPx(inUM):
    inPX = np.round(inUM/0.115)#.astype('int')
    return inPX

#%% convert from um^2 to px^2
def UmToPx_sq(inUM):
    inPX = np.round(inUM/0.115**2).astype('int')
    return inPX