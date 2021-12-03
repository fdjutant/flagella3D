# Welcome to Bon's Virtual Flagella Factory!
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import dilation
from matmatrix import *
from sklearn.decomposition import PCA

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
        EuAng = np.zeros([self.Nframes,3]);
        for i in range(self.Nframes):
            EuAng[i,0] = EuAng[i-1,0] +\
                np.random.normal(0, np.sqrt(6*self.vol_exp*self.Dpitch)) 
            EuAng[i,1] = EuAng[i-1,1] +\
                np.random.normal(0, np.sqrt(6*self.vol_exp*self.Droll))
            EuAng[i,2] = EuAng[i-1,2] +\
                np.random.normal(0, np.sqrt(6*self.vol_exp*self.Dyaw))

        # The three unit vectors
        n1 = np.zeros([self.Nframes,3]); n2 = np.zeros([self.Nframes,3]);
        n3 = np.zeros([self.Nframes,3]); vectN = np.zeros([self.Nframes,3,3])

        # Initial orientation (n1 - relative to the positive lab axes)
        ang1 = np.radians(31.); ang2 = np.radians(65.)
        ang3 = np.arccos(np.sqrt(1-np.cos(ang1)**2-np.cos(ang2)**2))
        
        cm = np.zeros([self.Nframes,3])
        for i in range(self.Nframes):
            
            points = self.createHx(0)  # generate the helix coordinates
            
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
                
                par_step   = n1[i] *\
                    np.random.normal(0, np.sqrt(6*self.vol_exp*self.Dpar))
                perp1_step = n2[i] *\
                    np.random.normal(0, np.sqrt(6*self.vol_exp*self.Dperp))
                perp2_step = n3[i] *\
                    np.random.normal(0, np.sqrt(6*self.vol_exp*self.Dperp))
                
                cm[i,:] = cm[i-1,:] + par_step + perp1_step + perp2_step
            
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