"""
Diffusion simulation function using numba
"""
import numpy as np
import numba
import timeit

setup_with_numba = """
import numpy as np
import numba
from numba import jit
# variables for simulation
vol_exp = 0.75e-3
d_pitch = 0.3
d_roll = 15
d_yaw = d_pitch
d_perp = 0.9
d_par = 2 * d_perp

@jit(nopython=True)
def simulate_diff(nframes):
    # generate euler angles
    das = np.stack((np.random.normal(0, np.sqrt(2 * vol_exp * d_pitch), size=nframes),
                    np.random.normal(0, np.sqrt(2 * vol_exp * d_roll), size=nframes),
                    np.random.normal(0, np.sqrt(2 * vol_exp * d_yaw), size=nframes)), axis=1)
    
    # euler_angles = np.cumsum(das, axis=0)
    # numba does not implement the keyword argument "axis", so have to rewrite code without it
    euler_angles = np.hstack((np.cumsum(das[:, 0]), np.cumsum(das[:, 1]), np.cumsum(das[:, 2])))

    # generate spatial steps
    dx_par = np.random.normal(0, np.sqrt(2 * vol_exp * d_par), size=nframes)
    dx_perp1 = np.random.normal(0, np.sqrt(2 * vol_exp * d_perp), size=nframes)
    dx_perp2 = np.random.normal(0, np.sqrt(2 * vol_exp * d_perp), size=nframes)

    # ############################
    # generate random starting directions
    # ############################
    cos_x = np.cos(np.random.uniform(0, 2*np.pi))
    cos_y = np.random.uniform(-np.sqrt(1 - cos_x**2), np.sqrt(1 - cos_x**2))
    cos_z = np.sqrt(1 - cos_x**2 - cos_y**2) * np.random.choice(np.array([1, -1]))
    n1_start = np.array([cos_x, cos_y, cos_z])

    # to generate n2, find two directions orthogonal to n1_start, then randomly mix them...
    # find one orthogonal direction
    n_orth_a = np.cross(n1_start, np.array([1, 0, 0]))
    if np.linalg.norm(n_orth_a) < 1e-12:
        n_orth_a = np.cross(n1_start, np.array([0, 1, 0]))
        if np.linalg.norm(n_orth_a) < 1e-12:
            n_orth_a = np.cross(n1_start, np.array([0, 1, 1]))

    n_orth_b = np.cross(n1_start, n_orth_a)

    theta_2 = np.random.uniform(0, 2*np.pi)
    n2_start = n_orth_a * np.cos(theta_2) + n_orth_b * np.sin(theta_2)
    # n3 orthogonal to n1, n2
    n3_start = np.cross(n1_start, n2_start)

    # ############################
    # simulate
    # ############################
    ndims = 3
    n1 = np.zeros((nframes, ndims), dtype=numba.float64)
    n2 = np.zeros((nframes, ndims), dtype=numba.float64)
    n3 = np.zeros((nframes, ndims), dtype=numba.float64)

    n1[0] = n1_start
    n2[0] = n2_start
    n3[0] = n3_start
    for ii in range(1, nframes):
        n1[ii] = n1[ii - 1] + das[ii, 0] * n2[ii - 1] - das[ii, 2] * n3[ii - 1]
        n2[ii] = -das[ii, 0] * n1[ii - 1] + n2[ii - 1] + das[ii, 1] * n3[ii - 1]
        n3[ii] = np.cross(n1[ii], n2[ii])

        # normalize the vectors
        n1[ii] /= np.linalg.norm(n1[ii])
        n2[ii] /= np.linalg.norm(n2[ii])
        n3[ii] /= np.linalg.norm(n3[ii])


    d_cmass = n1 * np.expand_dims(dx_par, axis=1) + \
              n2 * np.expand_dims(dx_perp1, axis=1) + \
              n3 * np.expand_dims(dx_perp2, axis=1)
    #cmass = np.cumsum(d_cmass, axis=0)
    cmass = np.hstack((np.cumsum(d_cmass[:, 0]),
                       np.cumsum(d_cmass[:, 1]),
                       np.cumsum(d_cmass[:, 2])))

    return cmass, euler_angles
"""

setup_without_numba = """
import numpy as np
# variables for simulation
vol_exp = 0.75e-3
d_pitch = 0.3
d_roll = 15
d_yaw = d_pitch
d_perp = 0.9
d_par = 2 * d_perp

def simulate_diff(nframes):
    # generate euler angles
    das = np.stack((np.random.normal(0, np.sqrt(2 * vol_exp * d_pitch), size=nframes),
                    np.random.normal(0, np.sqrt(2 * vol_exp * d_roll), size=nframes),
                    np.random.normal(0, np.sqrt(2 * vol_exp * d_yaw), size=nframes)), axis=1)

    # euler_angles = np.cumsum(das, axis=0)
    # numba does not implement the keyword argument "axis", so have to rewrite code without it
    euler_angles = np.hstack((np.cumsum(das[:, 0]), np.cumsum(das[:, 1]), np.cumsum(das[:, 2])))

    # generate spatial steps
    dx_par = np.random.normal(0, np.sqrt(2 * vol_exp * d_par), size=nframes)
    dx_perp1 = np.random.normal(0, np.sqrt(2 * vol_exp * d_perp), size=nframes)
    dx_perp2 = np.random.normal(0, np.sqrt(2 * vol_exp * d_perp), size=nframes)

    # ############################
    # generate random starting directions
    # ############################
    cos_x = np.cos(np.random.uniform(0, 2*np.pi))
    cos_y = np.random.uniform(-np.sqrt(1 - cos_x**2), np.sqrt(1 - cos_x**2))
    cos_z = np.sqrt(1 - cos_x**2 - cos_y**2) * np.random.choice(np.array([1, -1]))
    n1_start = np.array([cos_x, cos_y, cos_z])

    # to generate n2, find two directions orthogonal to n1_start, then randomly mix them...
    # find one orthogonal direction
    n_orth_a = np.cross(n1_start, np.array([1, 0, 0]))
    if np.linalg.norm(n_orth_a) < 1e-12:
        n_orth_a = np.cross(n1_start, np.array([0, 1, 0]))
        if np.linalg.norm(n_orth_a) < 1e-12:
            n_orth_a = np.cross(n1_start, np.array([0, 1, 1]))

    n_orth_b = np.cross(n1_start, n_orth_a)

    theta_2 = np.random.uniform(0, 2*np.pi)
    n2_start = n_orth_a * np.cos(theta_2) + n_orth_b * np.sin(theta_2)
    # n3 orthogonal to n1, n2
    n3_start = np.cross(n1_start, n2_start)

    # ############################
    # simulate
    # ############################
    ndims = 3
    n1 = np.zeros((nframes, ndims), dtype=float)
    n2 = np.zeros((nframes, ndims), dtype=float)
    n3 = np.zeros((nframes, ndims), dtype=float)

    n1[0] = n1_start
    n2[0] = n2_start
    n3[0] = n3_start
    for ii in range(1, nframes):
        n1[ii] = n1[ii - 1] + das[ii, 0] * n2[ii - 1] - das[ii, 2] * n3[ii - 1]
        n2[ii] = -das[ii, 0] * n1[ii - 1] + n2[ii - 1] + das[ii, 1] * n3[ii - 1]
        n3[ii] = np.cross(n1[ii], n2[ii])

        # normalize the vectors
        n1[ii] /= np.linalg.norm(n1[ii])
        n2[ii] /= np.linalg.norm(n2[ii])
        n3[ii] /= np.linalg.norm(n3[ii])


    d_cmass = n1 * np.expand_dims(dx_par, axis=1) + \
              n2 * np.expand_dims(dx_perp1, axis=1) + \
              n3 * np.expand_dims(dx_perp2, axis=1)
    #cmass = np.cumsum(d_cmass, axis=0)
    cmass = np.hstack((np.cumsum(d_cmass[:, 0]),
                       np.cumsum(d_cmass[:, 1]),
                       np.cumsum(d_cmass[:, 2])))

    return cmass, euler_angles
"""

setup_numpy_numba = """
import numpy as np
import numba
from numba import jit
# variables for simulation
vol_exp = 0.75e-3
d_pitch = 0.3
d_roll = 15
d_yaw = d_pitch
d_perp = 0.9
d_par = 2 * d_perp

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

def simulate_diff(nframes):
    # generate euler angles
    das = np.stack((np.random.normal(0, np.sqrt(2 * vol_exp * d_pitch), size=nframes),
                    np.random.normal(0, np.sqrt(2 * vol_exp * d_roll), size=nframes),
                    np.random.normal(0, np.sqrt(2 * vol_exp * d_yaw), size=nframes)), axis=1)

    # euler_angles = np.cumsum(das, axis=0)
    # numba does not implement the keyword argument "axis", so have to rewrite code without it
    euler_angles = np.hstack((np.cumsum(das[:, 0]), np.cumsum(das[:, 1]), np.cumsum(das[:, 2])))

    # generate spatial steps
    dx_par = np.random.normal(0, np.sqrt(2 * vol_exp * d_par), size=nframes)
    dx_perp1 = np.random.normal(0, np.sqrt(2 * vol_exp * d_perp), size=nframes)
    dx_perp2 = np.random.normal(0, np.sqrt(2 * vol_exp * d_perp), size=nframes)

    # ############################
    # generate random starting directions
    # ############################
    cos_x = np.cos(np.random.uniform(0, 2*np.pi))
    cos_y = np.random.uniform(-np.sqrt(1 - cos_x**2), np.sqrt(1 - cos_x**2))
    cos_z = np.sqrt(1 - cos_x**2 - cos_y**2) * np.random.choice(np.array([1, -1]))
    n1_start = np.array([cos_x, cos_y, cos_z])

    # to generate n2, find two directions orthogonal to n1_start, then randomly mix them...
    # find one orthogonal direction
    n_orth_a = np.cross(n1_start, np.array([1, 0, 0]))
    if np.linalg.norm(n_orth_a) < 1e-12:
        n_orth_a = np.cross(n1_start, np.array([0, 1, 0]))
        if np.linalg.norm(n_orth_a) < 1e-12:
            n_orth_a = np.cross(n1_start, np.array([0, 1, 1]))

    n_orth_b = np.cross(n1_start, n_orth_a)

    theta_2 = np.random.uniform(0, 2*np.pi)
    n2_start = n_orth_a * np.cos(theta_2) + n_orth_b * np.sin(theta_2)
    # n3 orthogonal to n1, n2
    n3_start = np.cross(n1_start, n2_start)

    # ############################
    # simulate
    # ############################
    ndims = 3
    n1 = np.zeros((nframes, ndims), dtype=float)
    n2 = np.zeros((nframes, ndims), dtype=float)
    n3 = np.zeros((nframes, ndims), dtype=float)

    n1[0] = n1_start
    n2[0] = n2_start
    n3[0] = n3_start
    n1, n2, n3 = angle_sum(n1, n2, n3, das, nframes)

    d_cmass = n1 * np.expand_dims(dx_par, axis=1) + \
              n2 * np.expand_dims(dx_perp1, axis=1) + \
              n3 * np.expand_dims(dx_perp2, axis=1)
    #cmass = np.cumsum(d_cmass, axis=0)
    cmass = np.hstack((np.cumsum(d_cmass[:, 0]),
                       np.cumsum(d_cmass[:, 1]),
                       np.cumsum(d_cmass[:, 2])))

    return cmass, euler_angles, n1, n2, n3
"""


nframes = 3000 * 100
t_numba = timeit.timeit(stmt="simulate_diff(%d)" % nframes, setup=setup_with_numba, number=1)
print("Running simulation for %d frames with numba took %0.2fs" % (nframes, t_numba))
t_no_numba = timeit.timeit(stmt="simulate_diff(%d)" % nframes, setup=setup_without_numba, number=1)
print("Running simulation for %d frames without numba took %0.2fs" % (nframes, t_no_numba))
t_numpy_numba = timeit.timeit(stmt="simulate_diff(%d)" % nframes, setup=setup_numpy_numba, number=1)
print("Running simulation for %d frames without numba took %0.2fs" % (nframes, t_numpy_numba))
print("speed up by a factor of %0.2f"% (t_no_numba/t_numba))