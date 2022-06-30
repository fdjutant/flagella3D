"""
Simulate trajectories to understand non-gaussian statistics as found in https://www.science.org/doi/epdf/10.1126/science.1130146
"""

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.special import erf
from scipy.optimize import least_squares
from numba import jit


# %% Generate samples for cm, EuAng, localAxes
@jit(nopython=True)
def angles2axes(n1_start, n2_start, das, nframes):
    """

    :param n1_start:
    :param n2_start:
    :param das: euler angles
    :param nframes:
    :return:
    """
    n1 = np.zeros((nframes, 3))
    n2 = np.zeros((nframes, 3))
    n3 = np.zeros((nframes, 3))
    n1[0] = n1_start
    n2[0] = n2_start
    n3[0] = np.cross(n1_start, n2_start)

    # dn1/dt = (dn1/dphi_1) * (dphi_1 / dt) + (dn1/dphi_2) * (dphi_2 / dt) + (dn1 / dphi_3) * (dphi_3 / dt)
    for ii in range(1, nframes):
        n1[ii] = n1[ii - 1] + das[ii, 0] * 0          - das[ii, 1] * n3[ii - 1] + das[ii, 2] * n2[ii - 1]
        n2[ii] = n2[ii - 1] + das[ii, 0] * n3[ii - 1] + das[ii, 1] * 0          - das[ii, 2] * n1[ii - 1]
        # n3[ii] = n3[ii - 1] - das[ii, 0] * n2[ii - 1] + das[ii, 1] * n1[ii - 1] + das[ii, 2] * 0
        n3[ii] = np.cross(n1[ii], n2[ii])

        # normalize the vectors
        n1[ii] /= np.linalg.norm(n1[ii])
        n2[ii] /= np.linalg.norm(n2[ii])
        n3[ii] /= np.linalg.norm(n3[ii])

    return n1, n2, n3

@jit(nopython=True)
def msd_corr(xs, ys):
    """
    Compute MSD-like quantity in the two variables xs, ys

    C(dt) = mean( [x(t + dt) - x(t)] * [y(t + dt) - y(t)] )
    :param xs:
    :param ys:
    :return:
    """
    npts = len(xs)

    msd = np.zeros(npts - 1)
    unc_mean = np.zeros(npts - 1)
    for ii in range(1, npts):
        dx_steps = xs[ii:] - xs[:-ii]
        dy_steps = ys[ii:] - ys[:-ii]

        msd[ii - 1] = np.mean(dx_steps * dy_steps)
        unc_mean[ii - 1] = np.std(dx_steps * dy_steps) / np.sqrt(len(dx_steps))

    return msd, unc_mean

@jit(nopython=True)
def fourth_moment_fn(xs):
    npts = len(xs)

    moment = np.zeros(npts - 1)
    unc_moment = np.zeros(npts - 1)
    for ii in range(1, npts):
        dx_steps = xs[ii:] - xs[:-ii]

        moment[ii - 1] = np.mean(dx_steps**4)
        unc_moment[ii - 1] = np.std(dx_steps**4) / np.sqrt(len(dx_steps))

    return moment, unc_moment

@jit(nopython=True)
def moment_uncorr(xs, order):
    npts = len(xs)

    moment = np.zeros(npts - 1)
    unc_moment = np.zeros(npts - 1)
    for ii in range(1, npts):
        xs_now = xs[::ii]
        dx_steps = xs_now[1:] - xs_now[:-1]

        moment[ii - 1] = np.mean(dx_steps**order)
        unc_moment[ii - 1] = np.std(dx_steps**order) / np.sqrt(len(dx_steps))

    return moment, unc_moment

def get_cdf(dr):
    dr = np.sort(dr)
    dr_unique, counts = np.unique(dr, return_counts=True)
    csum = np.cumsum(counts)
    return dr_unique, csum / csum[-1]

# fit CDF's, p = [mu, sigma]
def cdf_gauss_fit_fn(p, dr):
    return 0.5 * (1 + erf((dr - p[0]) / (np.sqrt(2) * p[1])))

# diffusion constant matrix
# um^2/s, um*rad/s, and rad^2/s
diff_matrix = np.array([[0.2, 0, 0, 0.05, 0, 0],
                        [0, 0.1, 0, 0, 0, 0],
                        [0, 0, 0.1, 0, 0, 0],
                        [0.05, 0, 0, 1.5, 0, 0],
                        [0, 0, 0, 0, 0.03, 0],
                        [0, 0, 0, 0, 0, 0.03]
                       ])
diff_matrix[3, 0] = 0
diff_matrix[0, 3] = 0

# time step
dt = 2e-3 * (15 / 0.4) / 10 / 10
#
dtau23 = 1 / np.mean([diff_matrix[4, 4], diff_matrix[5, 5]])
#nframes = 30000000 #* 10
nframes = 300000000
times = np.arange(nframes) * dt


# ##################################
# do step simulation, drawing from multivariate normal distribution
# ##################################
tstart = time.perf_counter()

# draw steps from multivariate normal distribution
steps = np.random.multivariate_normal(np.zeros(len(diff_matrix)), 2 * diff_matrix * dt, size=nframes)
coords_body = np.cumsum(steps, axis=0)
d123 = steps[:, :3]
das = steps[:, 3:]

# generate first orientation
use_random = False
# see e.g. https://math.stackexchange.com/questions/44689/how-to-find-a-random-axis-or-unit-vector-in-3d

if use_random:
    phi = np.random.uniform(0, 2 * np.pi)
    z = np.random.uniform(-1, 1)
    n1 = np.array([np.sqrt(1 - z ** 2) * np.cos(phi), np.sqrt(1 - z ** 2) * np.sin(phi), z])
    n1 = n1 / np.linalg.norm(n1)

    m2 = np.cross(n1, np.array([1, 0, 0]))
    if np.linalg.norm(m2) > 1e-12:
        m2 = m2 / np.linalg.norm(m2)
    else:
        m2 = np.cross(n1, np.array([0, 1, 0]))
        m2 = m2 / np.linalg.norm(m2)
    m3 = np.cross(n1, m2)

    phi2 = np.random.uniform(0, 2 * np.pi)
    n2 = m2 * np.cos(phi2) + m3 * np.sin(phi2)
else:
    n1 = np.array([0, 0, 1])
    n2 = np.array([1, 0, 0])

# generate other orientations
n1s, n2s, n3s = angles2axes(n1, n2, das, nframes)
nvecs = np.stack((n1s, n2s, n3s), axis=-1)

dxyz = np.sum(nvecs * np.expand_dims(d123, axis=-2), axis=-1)
pos_xyz = np.cumsum(dxyz, axis=0)

print(f"simulated {nframes:d} steps in {time.perf_counter() - tstart:.2f}s")

# time dependent diffusion constants
ns = np.round(np.logspace(np.log10(dtau23 / (100*dt)), np.log10(len(times)), 100)).astype(int)
diff_versus_time_lab = np.zeros((len(ns), 3, 3))
diff_versus_time_lab_unc = np.zeros((len(ns), 3, 3))
non_gauss_param_time_lab = np.zeros((len(ns), 3))

diff_versus_time_body = np.zeros((len(ns), 6, 6))
diff_versus_time_body_unc = np.zeros((len(ns), 6, 6))
non_gauss_param_time_body = np.zeros((len(ns), 6))

for ii in range(len(ns)):
    pos_now = pos_xyz[:ns[ii]]
    steps_now = pos_now[1:] - pos_now[:-1]

    for aa in range(3):
        for bb in range(3):
            second_moment = np.mean(steps_now[:, aa] * steps_now[:, bb], axis=0)
            diff_versus_time_lab[ii, aa, bb] = second_moment / (2 * dt)
            diff_versus_time_lab_unc[ii, aa, bb] = np.std(steps_now[:, aa] * steps_now[:, bb], axis=0) / (2 * dt) / np.sqrt(ns[ii] - 1)

            if aa == bb:
                fourth_moment = np.mean(steps_now[:, aa]**4, axis=0)
                non_gauss_param_time_lab[ii, aa] = fourth_moment / (3 * second_moment**2) - 1

    pos_now_123 = coords_body[:ns[ii]]
    steps_now_123 = pos_now_123[1:] - pos_now_123[:-1]
    for aa in range(6):
        for bb in range(6):
            second_moment = np.mean(steps_now_123[:, aa] * steps_now_123[:, bb], axis=0)
            diff_versus_time_body[ii, aa, bb] = second_moment / (2*dt)
            diff_versus_time_body_unc[ii, aa, bb] = np.std(steps_now_123[:, aa] * steps_now_123[:, bb], axis=0) / (2 * dt) / np.sqrt(ns[ii] - 1)

            if aa == bb:
                fourth_moment = np.mean(steps_now_123[:, aa]**4, axis=0)
                non_gauss_param_time_body[ii, aa] = fourth_moment / (3* second_moment**2) - 1



figh = plt.figure()
ax = figh.add_subplot(1, 2, 1)
ax.set_title("diffusion constants, lab frame, versus time")
ax.set_xlabel(r"time ($\tau_{23}$)")
for ii in range(3):
    ax.errorbar(ns * dt / dtau23, diff_versus_time_lab[:, ii, ii], yerr=diff_versus_time_lab_unc[:, ii, ii])
ax.set_xscale('log')

ax = figh.add_subplot(1, 2, 2)
ax.set_title("non-gaussian paramter versus time")
ax.set_xlabel(r"time ($\tau_{23}$)")
for ii in range(3):
    ax.semilogx(ns * dt / dtau23, non_gauss_param_time_lab[:, ii])


figh = plt.figure()
ax = figh.add_subplot(1, 2, 1)
ax.set_title("diffusion constants, body frame, versus time")
ax.set_xlabel(r"time ($\tau_{23}$)")
ax.set_xscale('log')
for ii in range(6):
    ax.errorbar(ns * dt / dtau23, diff_versus_time_body[:, ii, ii], yerr=diff_versus_time_body_unc[:, ii, ii])

ax = figh.add_subplot(1, 2, 2)
ax.set_title("non-gaussian paramter versus time")
ax.set_xlabel(r"time ($\tau_{23}$)")
for ii in range(6):
    ax.semilogx(ns * dt / dtau23, non_gauss_param_time_body[:, ii])

# other computations...
compute_full_traj = False
if compute_full_traj:
    # compute moments in lab frame
    tstart_moments = time.perf_counter()

    second_moments_lab = np.zeros((len(pos_xyz) - 1, 3))
    fourth_moments_lab = np.zeros((len(pos_xyz) - 1, 3))

    for ii in range(3):
        second_moments_lab[:, ii], _ = msd_corr(pos_xyz[:, ii], pos_xyz[:, ii])
        fourth_moments_lab[:, ii], _ = fourth_moment_fn(pos_xyz[:, ii])
        # second_moments_lab[:, ii], _ = moment_uncorr(pos_xyz[:, ii], 2)
        # fourth_moments_lab[:, ii], _ = moment_uncorr(pos_xyz[:, ii], 4)

    nongauss_param_lab = fourth_moments_lab / (3 * second_moments_lab ** 2) - 1
    print(f"computed lab frame moments in {time.perf_counter() - tstart_moments:.2f}s")

    # compute moments in body frame
    tstart_moments_body = time.perf_counter()
    second_moments_body = np.zeros((len(coords_body) - 1, 6))
    fourth_moments_body = np.zeros((len(coords_body) - 1, 6))
    for ii in range(6):
        second_moments_body[:, ii], _ = msd_corr(coords_body[:, ii], coords_body[:, ii])
        fourth_moments_body[:, ii], _ = fourth_moment_fn(coords_body[:, ii])
        # second_moments_body[:, ii], _ = moment_uncorr(coords_body[:, ii], 2)
        # fourth_moments_body[:, ii], _ = moment_uncorr(coords_body[:, ii], 4)

    nongauss_param_body = fourth_moments_body / (3 * second_moments_body ** 2) - 1
    print(f"computed body frame moments in {time.perf_counter() - tstart_moments_body:.2f}s")

    # plot
    nmax_plot = 30000

    figh = plt.figure()
    ax = figh.add_subplot(1, 2, 1)
    ax.plot(times[1:nmax_plot + 1], nongauss_param_lab[:nmax_plot], marker=".")
    ax.set_xlabel("time")
    ax.set_title("nongauss-param lab")
    ax.set_ylim([-1, 1])

    ax = figh.add_subplot(1, 2, 2)
    ax.plot(times[1:nmax_plot+1], nongauss_param_body[:nmax_plot], marker=".")
    ax.set_xlabel("time")
    ax.set_title("nongauss-param body")
    ax.set_ylim([-1, 1])

# test CDF
if False:
    figh = plt.figure()

    ds = 100
    # steps_now = coords_body[ds:] - coords_body[:-ds]
    coords_now = coords_body[::ds]
    steps_now = coords_now[1:] - coords_now[:-1]
    ngp = np.mean(steps_now**4, axis=0) / (3 * np.mean(steps_now**2, axis=0)**2) - 1

    for ind in range(6):
        dr, cdf = get_cdf(steps_now[:, ind])
        dr_interp = np.linspace(dr.min(), dr.max(), 1000)
        def cost_fn(p): return cdf_gauss_fit_fn(p, dr) - cdf
        results = least_squares(cost_fn, [np.mean(steps_now[:, ind]), np.std(steps_now[:, ind])])

        ax = figh.add_subplot(1, 6, ind + 1)
        ax.plot(dr_interp, cdf_gauss_fit_fn(results["x"], dr_interp), 'r')
        ax.plot(dr, cdf, c=np.array([0, 0, 1, 0.5]), marker='.', linestyle="")

# test
# ntest = 100000
# ngps = np.zeros(ntest)
# for ii in range(ntest):
#     a = np.random.normal(0, 1.888, size=1000)
#     ngps[ii] = np.mean(a**4) / (3 * np.mean(a**2)**2) - 1
#
# dr, cdf = get_cdf(ngps)
# figh = plt.figure()
# plt.plot(dr, cdf, '.')
# plt.title(f"mean = {np.mean(ngps):.2f}")