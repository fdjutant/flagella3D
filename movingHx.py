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
        
        self.Dpar  = Dpar/0.115
        self.Dperp = Dperp/0.115
        
        self.Dpitch = np.radians(Dpitch)
        self.Droll = np.radians(Droll)
        self.Dyaw = np.radians(Dyaw)

        # self.Dpitch = Dpitch * (np.pi/180)**2
        # self.Droll = Droll * (np.pi/180)**2
        # self.Dyaw = Dyaw * (np.pi/180)**2

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
        # if self.transRange == 0:
        #     boxSize = self.length + 30
        # else:
        #     boxSize = self.length + np.round(0.5*self.transRange*\
        #                                      self.Nframes).astype('int')
        boxSize = self.length.astype('int') + 250
        img = np.zeros((self.Nframes,boxSize,\
                        boxSize,boxSize),dtype='uint8')
        
        box = img.shape[1:]
        origin = np.array(box)/2
        np.random.seed(13317043) # ensure repeatable random
        
        # Center of mass fluctuation
        # sig_par   = np.sqrt(2 * 3 * self.Dpar * self.vol_exp);
        # sig_perp1 = np.sqrt(2 * 3 * self.Dperp1 * self.vol_exp);
        # sig_perp2 = np.sqrt(2 * 3 * self.Dperp2 * self.vol_exp);
        sig_par = np.sqrt(self.Dpar)
        sig_perp = np.sqrt(self.Dperp)
        lengthwise = np.zeros(self.Nframes); 
        sidewise1 = np.zeros(self.Nframes); sidewise2 = np.zeros(self.Nframes);
        for i in range(1,self.Nframes):
            lengthwise[i] = lengthwise[i-1] + np.random.normal(0, sig_par)
            sidewise1[i] = sidewise1[i-1] + np.random.normal(0, sig_perp)
            sidewise2[i] = sidewise2[i-1] + np.random.normal(0, sig_perp)
        
        # Pitch, roll, and yaw (relative to the local axes)
        # sig_pitch = np.sqrt(2 * 3 * self.Dpitch * self.vol_exp);
        # sig_roll = np.sqrt(2 * 3 * self.Droll * self.vol_exp);
        # sig_yaw = np.sqrt(2 * 3 * self.Dyaw * self.vol_exp);
        sig_pitch = self.Dpitch;
        sig_roll = self.Droll;
        sig_yaw = self.Dyaw;
        EuAng = np.zeros([self.Nframes,3]);
        for i in range(1,self.Nframes):
            EuAng[i-1,0] = EuAng[i-1,0] + np.random.normal(0, sig_pitch)     
            EuAng[i-1,1] = EuAng[i-1,1] + np.random.normal(0, sig_roll)
            EuAng[i-1,2] = EuAng[i-1,2] + np.random.normal(0, sig_yaw)
            
        # Direction angle (relative to the lab axes)
        dirAng = np.zeros([self.Nframes,3])
        ang1 = np.radians(31.); ang2 = np.radians(65.)
        ang3 = np.arccos(np.sqrt(1-np.cos(ang1)**2-np.cos(ang2)**2))
        dirAng[0] = np.array([ang1,ang2,ang3])
        
        # The three unit vectors
        n1 = np.zeros([self.Nframes,3]); n2 = np.zeros([self.Nframes,3]);
        n3 = np.zeros([self.Nframes,3]); vectN = np.zeros([self.Nframes,3,3])
        
        CMS = []; frame = []; 
        for i in range(self.Nframes):
            
            points = self.createHx(0)  # generate the helix coordinates
            
            # set up the normal vectors
            if i == 0:
                n1[i] = np.array([ np.cos(ang1),np.cos(ang2),np.cos(ang3) ])
                n2[i] = points[0,:]
                n2[i] -= n2[i].dot(n1[i]) * n1[i] / np.linalg.norm(n1[i])**2
                n2[i] /= np.linalg.norm(n2[i])
                n3[i] = np.cross(n1[i],n2[i])
                n3[i] /= np.linalg.norm(n3[i])
                
            else:
                n1[i] = n1[i-1] + ( (EuAng[i,0]-EuAng[i-1,0]) * n2[i-1] -\
                                    (EuAng[i,2]-EuAng[i-1,2]) * n3[i-1])
                
                n2[i] = n2[i-1] + (-(EuAng[i,0]-EuAng[i-1,0]) * n1[i-1] +\
                                    (EuAng[i,1]-EuAng[i-1,1]) * n3[i-1])
                # n3[i] = np.cross(n1[i],n2[i])
                n3[i] = n3[i-1] + ( (EuAng[i,2]-EuAng[i-1,2]) * n1[i-1] -\
                                    (EuAng[i,1]-EuAng[i-1,1]) * n2[i-1])
                     
            # set the center of mass
            CM = origin + n1[i] * lengthwise[i] +\
                          n2[i] * sidewise1[i] +\
                          n3[i] * sidewise2[i]   
            CMS.append(CM)

            # prepare 3D images   
            n1[i] /= np.linalg.norm(n1[i])
            n2[i] /= np.linalg.norm(n2[i])
            n3[i] /= np.linalg.norm(n3[i])
            axes = np.array([ n2[i], n3[i], n1[i] ])
            frame = CM + np.matmul(axes.T,points.T).T
            frame = self.digitize(frame,0)
            for f in frame:
                x, y, z = f
                if self.hxVar == 0:
                    img[i,x,y,z] = self.hxInt
                else:
                    img[i,x,y,z] = np.random.uniform(self.hxInt-self.hxVar,\
                                                     self.hxInt+self.hxVar)
        
            # infer the direction angles from n1[i]
            for j in range(3):
                dirAng[i,j] = np.arccos(n1[i,j])
            
        # convert CMS to numpy array
        CMS = np.array(CMS)
    
        # dilate and plot frames
        for i in range(self.Nframes):
            img[i] = dilation(img[i])
            img[i] = dilation(img[i])
            # img[i] = dilation(img[i])
        
        # add salt-and-pepper noise 
        img = self.addNoise(img)
        
        # export all vector N
        for i in range(self.Nframes):
            vectN[i] = np.array([ n1[i], n2[i], n3[i] ])
        
        return img, CMS, EuAng, dirAng, vectN
    
    def addNoise(self,img):
        totalNum = img.shape[0]*img.shape[1]*img.shape[2]*img.shape[3]
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