"""
Calculate RFT coefficients for a helix along arbitrary directions
Most interested in translation/rotation perpendicular to helix rotational symmetry axis, so can have
confirmation of expressions we derived but I have not seen elsewhere

For RFT values along helix axis, see (e.g.) eqs. [7]--[9] in
https://doi.org/10.1073/pnas.1219831110

Let Kn and Kt be the resistance coefficients along the normal and tangential directions
drag coefficient for motion along x
P_{x,x} = 0.5 * [(Kt - Kn) * sin^2(theta) + 2 * Kn] * L / cos(theta)
drag coefficient for rotation about x
P_{nx,nx} = [R^2 / 4 * (2 * Kn + (Kt - Kn) * 5 * cos^2(theta)) +
             L^2 /24 * (2 * kn + (Kt - Kn) * sin^2(theta)) ] * L / cos(theta)
"""

import numpy as np
from scipy.integrate import quad

# helix centerline, with helix axis pointing along z
def r(R, wlen, z):
    return np.array([R * np.cos(2*np.pi / wlen * z), R * np.sin(2*np.pi / wlen * z), z])

# unit vector in tangential direction
def t(R, wlen, z):
    theta = np.arctan(2*np.pi * R / wlen)
    return np.array([-2 * np.pi * R / wlen * np.sin(2*np.pi / wlen * z),
                      2 * np.pi * R / wlen * np.cos(2*np.pi / wlen * z),
                      1]) * np.cos(theta)

# velocity
# motion in z
# def v(R, wlen, len, z):
#     return np.array([0, 0, 1])

# rotation about z
# def v(R, wlen, len, z):
#     vel = np.cross(np.array([0, 0, 1]), r(R, wlen, z) - np.array([0, 0, len/2]))
#     return vel

# motion in x
# def v(R, wlen, len, z):
#     return np.array([1, 0, 0])

# rotation about x
def v(R, wlen, len, z):
    vel = np.cross(np.array([1, 0, 0]), r(R, wlen, z) - np.array([0, 0, len / 2]))
    return vel

# split velocity in tangential and normal components
def decompose_v(R, wlen, len, z):
    vt_mag = np.dot(v(R, wlen, len, z), t(R, wlen, z))
    vt = vt_mag * t(R, wlen, z)
    vn = v(R, wlen, len, z) - vt
    return vt, vn

# compute net force from local normal and tangential forces
def forces(R, wlen, len):

    theta = np.arctan(2 * np.pi * R / wlen)
    ft_x = quad(lambda z: decompose_v(R, wlen, len, z)[0][0] / np.cos(theta), 0, len)[0]
    ft_y = quad(lambda z: decompose_v(R, wlen, len, z)[0][1] / np.cos(theta), 0, len)[0]
    ft_z = quad(lambda z: decompose_v(R, wlen, len, z)[0][2] / np.cos(theta), 0, len)[0]

    fn_x = quad(lambda z: decompose_v(R, wlen, len, z)[1][0] / np.cos(theta), 0, len)[0]
    fn_y = quad(lambda z: decompose_v(R, wlen, len, z)[1][1] / np.cos(theta), 0, len)[0]
    fn_z = quad(lambda z: decompose_v(R, wlen, len, z)[1][2] / np.cos(theta), 0, len)[0]

    return np.array([ft_x, ft_y, ft_z]), np.array([fn_x, fn_y, fn_z])

# compute net torque from local normal and tangential forces
def torques(R, wlen, len):
    theta = np.arctan(2 * np.pi * R / wlen)

    def r_cross_v(R, wlen, len, z):
        rnow = r(R, wlen, z) - np.array([0, 0, len/2])
        vt, vn = decompose_v(R, wlen, len, z)
        return np.cross(rnow, vt), np.cross(rnow, vn)

    taut_x = quad(lambda z: r_cross_v(R, wlen, len, z)[0][0] / np.cos(theta), 0, len)[0]
    taut_y = quad(lambda z: r_cross_v(R, wlen, len, z)[0][1] / np.cos(theta), 0, len)[0]
    taut_z = quad(lambda z: r_cross_v(R, wlen, len, z)[0][2] / np.cos(theta), 0, len)[0]

    taun_x = quad(lambda z: r_cross_v(R, wlen, len, z)[1][0] / np.cos(theta), 0, len)[0]
    taun_y = quad(lambda z: r_cross_v(R, wlen, len, z)[1][1] / np.cos(theta), 0, len)[0]
    taun_z = quad(lambda z: r_cross_v(R, wlen, len, z)[1][2] / np.cos(theta), 0, len)[0]

    return np.array([taut_x, taut_y, taut_z]), np.array([taun_x, taun_y, taun_z])


# define helix parameters
R = 1
theta = 32 * np.pi / 180
wlen = 2 * np.pi * R / np.tan(theta)
len = 3 * wlen

# do computations ... net force = Kt * ft + Kn * fn
ft, fn = forces(R, wlen, len)
tt, tn = torques(R, wlen, len)

