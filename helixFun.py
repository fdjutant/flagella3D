# function for fitting helix after being aligned by PCA
import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from matmatrix import rotmat

# Basic functions for projection models
def f(xx,a): # model for "y" projection (on xz plane)
    return a[0]*np.cos(a[1]*xx+a[2])

def g(xx,a): # model for "z" projection (on xy plane)
    return a[0]*np.sin(a[1]*xx+a[2])

#%% Defining a class for the fit
class createFitImage:
    def __init__(self, img, xx, a, cm, RollChosen, pcaComp):
        self.img = img
        self.xx = xx
        self.a = a
        self.cm = cm
        self.RollChosen = RollChosen
        self.pcaComp = pcaComp

    def recon_vis(self):  # Construct helix with some padding
        x = np.linspace(min(self.xx),max(self.xx),5000)
        ym = f(x,self.a)                  # mid
        yt = f(x,self.a)+0.5*self.a[1]    # top
        yb = f(x,self.a)-0.5*self.a[1]    # bot
        zm = g(x,self.a)                  # mid
        zt = g(x,self.a)+0.5*self.a[1]    # top
        zb = g(x,self.a)-0.5*self.a[1]    # bot
            
        # Stack coordinates into 3D frames
        fit_P = np.array([x,yb,zb]).T 
        fit_P = np.append(fit_P, np.array([x,yb,zm]).T,axis=0)
        fit_P = np.append(fit_P, np.array([x,yb,zt]).T,axis=0)
        fit_P = np.append(fit_P, np.array([x,ym,zb]).T,axis=0)
        fit_P = np.append(fit_P, np.array([x,ym,zm]).T,axis=0)
        fit_P = np.append(fit_P, np.array([x,ym,zt]).T,axis=0)
        fit_P = np.append(fit_P, np.array([x,yt,zb]).T,axis=0)
        fit_P = np.append(fit_P, np.array([x,yt,zm]).T,axis=0)
        fit_P = np.append(fit_P, np.array([x,yt,zt]).T,axis=0)
        
        return fit_P

    def fit3Dimg(self): # Inverse transform
        M4 = rotmat(np.array([0,self.RollChosen,0]))

        fit_P = self.recon_vis()
        fit_X = np.matmul(M4,fit_P.T).T
        fit_X = np.matmul(np.linalg.inv(self.pcaComp),fit_X.T).T + self.cm
        fit_X = fit_X.astype('int')   # convert to integer for digitization
        fit_X = np.unique(fit_X,axis=0) # remove all the identical points
            
        # Prepare our model image
        fit_img = np.zeros(self.img.shape)
        for idx in fit_X:
            i,j,k = idx
            fit_img[i,j,k] = 1  # value of 1 for the fit
        
        return fit_img

    def fit3Dimg_noRot(self): # Inverse transform
        
        fit_P = self.recon_vis()
        fit_X = np.matmul(np.linalg.inv(self.pcaComp),fit_P.T).T + self.cm
        fit_X = fit_X.astype('int')   # convert to integer for digitization
        fit_X = np.unique(fit_X,axis=0) # remove all the identical points
            
        # Prepare our model image
        fit_img = np.zeros(self.img.shape)
        for idx in fit_X:
            i,j,k = idx
            fit_img[i,j,k] = 1  # value of 1 for the fit
        
        return fit_img
            
#%% Defining a class for the fit
class fitHelix:
    def __init__(self, xx, yy, zz, init_a=np.array([])):
        self.xx = xx
        self.yy = yy
        self.zz = zz
        
        self.init_a = init_a

    def exportFit(self):
        N = len(self.xx)
        
        # Initialize the guess values
        b = np.pi/10;
        a = np.array([1,b,0]) # a[0]: radius, a[1]: freq, a[2]: phase1
                
        r = 0
        Niter = 0
        
        if self.init_a.any():
            a0 = self.init_a
        else:
            a0 = np.array([1.,b,-np.pi/2])
        
        maxIter = 20
        while r<0.2 and Niter<maxIter:
            
            '''
            res = minimize(self.E, a0,
                           method='powell',options={'disp': False})
            '''
            res = minimize(self.E, a0, method='L-BFGS-B', 
                           jac=self.gradE,
                           options={'disp': False})    
            
            a = res.x       
            r = self.rsq(N,self.E(a)) # compute regression coefficient
            
            
            if r<0.2:
                
                a0[2] = (a0[2]+2*np.pi/maxIter)%(2*np.pi)
                
                Niter += 1
            
        return a, r, Niter
    
    def E(self,a): # cost function
        ydata = self.yy
        zdata = self.zz

        d = ((norm(f(self.xx,a)-ydata))**2 +\
             (norm(g(self.xx,a)-zdata))**2) 
        return d/len(self.xx)  # capital N: length of dataset
    
    def rsq(self,N,E): #Regression coef
        S = (norm(self.yy-self.yy.mean()))**2+(norm(self.zz-self.zz.mean()))**2
        return 1-N*E/S

    def gradE(self,a): #Cost gradient

        ydata = self.yy
        zdata = self.zz

    
        dy = f(self.xx,a) - ydata
        dz = g(self.xx,a) - zdata
        
        g0 = dy*np.cos(a[1]*self.xx+a[2]) +\
             dz*np.sin(a[1]*self.xx+a[2])
        g2 = -dy*np.sin(a[1]*self.xx+a[2]) +\
             dz*np.cos(a[1]*self.xx+a[2])
        g1 = self.xx*g2
        
        return np.array([g0.sum(),g1.sum(),g2.sum()])*2/len(self.xx)