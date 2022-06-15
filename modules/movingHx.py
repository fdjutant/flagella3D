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
