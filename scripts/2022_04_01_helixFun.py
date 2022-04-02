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
    return  a[0]*np.sin(a[1]*xx+a[2])

# function which should match data
def model_fn(x, p):
    v = p[0] * np.sin(2*np.pi * p[1] * x + p[2]) + p[3]
    return v

# function which should be minimized ... or rather least_squares will take the sum of the squares to minimize it...
def min_fn(p): return model_fn(x, p) - y_noisy

def model_jacobian(x, p):
    dp0 = np.sin(2*np.pi * p[1] * x + p[2])
    dp1 = p[0] * (2 * np.pi * x) * np.cos(2*np.pi * p[1] * x + p[2])
    dp2 = p[0] * np.cos(2*np.pi * p[1] * x + p[2])
    dp3 = np.ones(x.shape)
    return np.stack((dp0, dp1, dp2, dp3), axis=1)

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
        
        maxIter = 30
        while r<0.2 and Niter<maxIter:
            
            res = minimize(self.E, a0,
                            method='powell',options={'disp': False})
            # res = minimize(self.E, a0, method='L-BFGS-B', 
            #                 jac=self.gradE,
            #                 bounds = [(0,5),(0,1),(-np.pi,np.pi)],
            #                 options={'disp': False})    
            
            a = res.x       
            r = self.rsq(N,self.E(a)) # compute regression coefficient
            
            a0 = a
            
            # update phase guess value
            a0[2] = (a0[2] + np.pi/8)%(2 * np.pi)
            
            # update wave number
            # a0[1] = a0[1]*1.1
                
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