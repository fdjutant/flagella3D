"""
Analyze diffusing flagella datasets
"""
# visualization
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 20})
# analysis
import numpy as np
from sklearn.decomposition import PCA
from skimage import measure
from scipy.optimize import least_squares
from scipy.special import erf
import pandas as pd
# utilities and io
import re
import time
import datetime
from pathlib import Path
import tifffile
import zarr
from numba import jit

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
def fourth_moment(xs):
    npts = len(xs)

    moment = np.zeros(npts - 1)
    unc_moment = np.zeros(npts - 1)
    for ii in range(1, npts):
        dx_steps = xs[ii:] - xs[:-ii]

        moment[ii - 1] = np.mean(dx_steps**4)
        unc_moment[ii - 1] = np.std(dx_steps**4) / np.sqrt(len(dx_steps))

    return moment, unc_moment

figsize = (25, 15)
plot_cdfs = False
fontsize_small = 20
fontsize_large = 30
plt.ioff()
plt.switch_backend("Agg")

# time-stamp analysis files
tstamp = datetime.datetime.now().strftime('%Y_%m_%d_%H;%M;%S')

# time settings in the light sheet
kb = 1.380649e-23
T = 273 + 25
dxyz_um = 0.115
camExposure_ms = 2
sweep_um = 15
step_size_um = 0.4
exp3D_sec = 1e-3 * camExposure_ms * (sweep_um / step_size_um)
nmax_msd_fit = 30

pattern_viscosity_dict = {"suc40.*": 1.77e-3,
                          "suc50.*": 1.99e-3,
                          "suc70.*": 2.84e-3
}

root_dir = Path(r"\\10.206.26.21\flagella_project")
thresholdFolder = root_dir / "threshold-labKit"
intensityFolder = root_dir / "TIF-files"
# data to save results
data_dir = root_dir / f"{tstamp:s}_processed_data"
data_dir.mkdir(exist_ok=True)

# files to process
fnames = list(thresholdFolder.glob("*-LabKit-*.tif"))

# summary table
summary_fname = data_dir / "summary.csv"
cnames = ["name", "number of frames", "mean length (um)", "std length (um)"]
for ii in range(6):
    for jj in range(6):
        if ii >= jj:
            cnames.append(f"D({ii:d}, {jj:d})")
summary = pd.DataFrame(columns=cnames)

tstart_all = time.perf_counter()
# chose which file to analyze
for file_index in range(len(fnames)):
    tstart_folder = time.perf_counter()

    threshold_fname = fnames[file_index]
    dset_prefix = threshold_fname.with_suffix("").name[:-17]

    imgs_thresh = tifffile.imread(threshold_fname).astype(bool)
    nt, nz, ny, nx = imgs_thresh.shape

    # analyze...
    frame_start = 0
    frame_end = nt

    # variables to store results
    blob_masks = np.zeros(imgs_thresh.shape, dtype=bool)
    centers_of_mass = np.zeros((nt, 3))
    n1s = np.zeros((nt, 3))
    n2s = np.zeros((nt, 3))
    n3s = np.zeros((nt, 3))
    m2s = np.zeros((nt, 3))
    m3s = np.zeros((nt, 3))
    flagella_len = np.zeros(nt)
    helix_fit_params = np.zeros([nt, 3])

    fitImage = np.zeros(imgs_thresh.shape)

    for frame in range(frame_start, frame_end):
        print(f"{dset_prefix:s}, file {file_index+1:d}/{len(fnames):d}, frame {frame+1:d}/{nt:d},"
              f" folder time: {time.perf_counter() - tstart_folder:.2f}s"
              f", full time: {time.perf_counter() - tstart_all:.2f}s", end="\r")

        # erosion: thinning the threshold
        # kernel = np.ones((4,4), np.uint8)
        # img = cv2.erode(img, kernel, iterations=1)

        # label and measure every clusters
        blobs = measure.label(imgs_thresh[frame], background=0)
        labels = np.arange(1, blobs.max() + 1, dtype=int)
        sizes = np.array([np.sum(blobs == l) for l in labels])

        # keep only the largest cluster
        max_ind = np.argmax(sizes)

        # mask showing which pixels ae in largest cluster
        blob = blobs == labels[max_ind]

        # store threshold/binarized image
        blob_masks[frame] = blob

        # ######################################
        # extract coordinates and center of mass
        # ######################################
        coords = np.argwhere(blob).astype('float')

        cm_pix_now = np.mean(coords, axis=0)

        centers_of_mass[frame, :] = cm_pix_now * dxyz_um  # store center of mass

        # ##############################################################
        # determine axis n1 from PCA and consistency with previous point
        # ##############################################################
        coords_cm = coords - cm_pix_now  # shift all the coordinates into origin
        pca = PCA(n_components=3)
        pca.fit(coords_cm)
        n1s[frame] = pca.components_[0]
        m2s[frame] = pca.components_[1]
        m3s[frame] = pca.components_[2]

        # choose the sign of current n1 so it is as close as possible to n1 at the previous timestep
        if frame > frame_start and np.linalg.norm(n1s[frame] - n1s[frame - 1]) > np.linalg.norm(n1s[frame] + n1s[frame - 1]):
            n1s[frame] = -n1s[frame]
            m2s[frame] = -m2s[frame]
            m3s[frame] = -m3s[frame]

        # ensure n1 x m2 = m3. Want to avoid possibility that n1 x m2 = -m3
        if np.linalg.norm(np.cross(n1s[frame], m2s[frame]) - m3s[frame]) > 1e-12:
            m3s[frame] = -m3s[frame]
            # check it worked
            assert np.linalg.norm(np.cross(n1s[frame], m2s[frame]) - m3s[frame]) < 1e-12

        # #####################################
        # get flagella coordinates along n1, m2, m3 directions
        # #####################################
        pts_on_n1 = n1s[frame, 0] * coords_cm[:, 0] + \
                    n1s[frame, 1] * coords_cm[:, 1] + \
                    n1s[frame, 2] * coords_cm[:, 2]
        pts_on_m2 = m2s[frame, 0] * coords_cm[:, 0] + \
                    m2s[frame, 1] * coords_cm[:, 1] + \
                    m2s[frame, 2] * coords_cm[:, 2]
        pts_on_m3 = m3s[frame, 0] * coords_cm[:, 0] + \
                    m3s[frame, 1] * coords_cm[:, 1] + \
                    m3s[frame, 2] * coords_cm[:, 2]
        coords_123 = np.stack([pts_on_n1, pts_on_m2, pts_on_m3], axis=1)

        # determine the flagella length along the n1
        flagella_len[frame] = (np.max(pts_on_n1) - np.min(pts_on_n1)) * dxyz_um

        # ##########################
        # Curve fit helix projection
        # ##########################
        def cosine_fn(pts_on_n1, a):  # model for "y" projection (on xz plane)
            return a[0] * np.cos(a[1] * pts_on_n1 + a[2])


        def sine_fn(pts_on_n1, a):  # model for "z" projection (on xy plane)
            return a[0] * np.sin(a[1] * pts_on_n1 + a[2])


        def cost_fn(a):
            cost = np.concatenate((cosine_fn(pts_on_n1, a) - pts_on_m2,
                                   sine_fn(pts_on_n1, a) - pts_on_m3)) / pts_on_n1.size
            return cost


        def jacobian_fn(a):  # Cost gradient
            dy = cosine_fn(pts_on_n1, a) - pts_on_m2
            dz = sine_fn(pts_on_n1, a) - pts_on_m3

            g0 = dy * np.cos(a[1] * pts_on_n1 + a[2]) + \
                 dz * np.sin(a[1] * pts_on_n1 + a[2])
            g2 = -dy * np.sin(a[1] * pts_on_n1 + a[2]) + \
                 dz * np.cos(a[1] * pts_on_n1 + a[2])
            g1 = pts_on_n1 * g2

            return np.array([g0.sum(), g1.sum(), g2.sum()]) * 2 / len(pts_on_n1)

        # [amplitude, wave_number, phase]
        init_params = np.array([1.65, 0.28, np.pi / 10])
        lower_bounds = [1.5, 0.25, -np.inf]
        upper_bounds = [2.5, 0.3, np.inf]

        # fit and store results
        results_fit = least_squares(lambda p: cost_fn([p[0], p[1], p[2]]),
                                    init_params[0:3],
                                    bounds=(lower_bounds[0:3], upper_bounds[0:3]))

        fit_params = results_fit["x"]
        helix_fit_params[frame, :] = fit_params

        # ########################################
        # compute n2 and n3 from phase information
        # ########################################
        phase = fit_params[2]
        n2s[frame] = m2s[frame] * np.cos(phase) - m3s[frame] * np.sin(phase)
        n2s[frame] = n2s[frame] / np.linalg.norm(n2s[frame])
        # n3s[frame] = m2s[frame] * np.sin(phase) + m3s[frame] * np.cos(phase)

        # generate n3 such that coordinate system is right-handed
        n3s[frame] = np.cross(n1s[frame], n2s[frame])
        n3s[frame] = n3s[frame] / np.linalg.norm(n3s[frame])

        # verify vectors
        assert np.abs(np.linalg.norm(n1s[frame]) - 1) < 1e-12
        assert np.abs(np.linalg.norm(n2s[frame]) - 1) < 1e-12
        assert np.abs(np.linalg.norm(n3s[frame]) - 1) < 1e-12
        assert n1s[frame].dot(n2s[frame]) < 1e-12
        assert n1s[frame].dot(n3s[frame]) < 1e-12

        # #################################################
        # Construct 3D matrix for the fit for visualization
        # #################################################
        # todo: move this outside of processing loop...only generate if desired
        # construct helix with some padding
        x = np.linspace(min(pts_on_n1), max(pts_on_n1), 5000)
        ym = cosine_fn(x, fit_params)  # mid
        yt = cosine_fn(x, fit_params) + 0.5 * fit_params[1]  # top
        yb = cosine_fn(x, fit_params) - 0.5 * fit_params[1]  # bottom
        zm = sine_fn(x, fit_params)  # mid
        zt = sine_fn(x, fit_params) + 0.5 * fit_params[1]  # top
        zb = sine_fn(x, fit_params) - 0.5 * fit_params[1]  # bottom

        # stack the coordinates
        fit_P = np.array([x, yb, zb]).transpose()
        fit_P = np.append(fit_P, np.array([x, yb, zm]).transpose(), axis=0)
        fit_P = np.append(fit_P, np.array([x, yb, zt]).transpose(), axis=0)
        fit_P = np.append(fit_P, np.array([x, ym, zb]).transpose(), axis=0)
        fit_P = np.append(fit_P, np.array([x, ym, zm]).transpose(), axis=0)
        fit_P = np.append(fit_P, np.array([x, ym, zt]).transpose(), axis=0)
        fit_P = np.append(fit_P, np.array([x, yt, zb]).transpose(), axis=0)
        fit_P = np.append(fit_P, np.array([x, yt, zm]).transpose(), axis=0)
        fit_P = np.append(fit_P, np.array([x, yt, zt]).transpose(), axis=0)

        # matrix rotation
        mrot = np.linalg.inv(np.vstack([n1s[frame], m2s[frame], m3s[frame]]))

        # inverse transform
        # fit_X = pca.inverse_transform(fit_P)+ CM1
        fit_X = np.matmul(mrot, fit_P.transpose()).transpose() + cm_pix_now
        fit_X = fit_X.astype('int')
        fit_X = np.unique(fit_X, axis=0)

        # prepare our model image
        for idx in fit_X:
            i, j, k = idx
            if i < nz and j < ny and k < nx:
                fitImage[frame, i, j, k] = 1  # value of 1 for the fit

        # ###################################
        # re-measure center of mass after fit
        # ###################################
        # extract coordinates
        # X0_post_fit = np.argwhere(fitImage[frame]).astype(float)  # coordinates

        # compute center of mass
        # todo: i don't think want to extract information from the fit image...
        # cm_now = np.array([sum(X0_post_fit[:, j]) for j in range(X0_post_fit.shape[1])]) / X0_post_fit.shape[0]
        # centers_of_mass[frame, :] = cm_now * dxyz_um  # store center of mass

    print("")

    # #############################
    # translational and angular displacements
    # #############################

    # compute translational displacement along lab axes
    delta_xyz = (centers_of_mass[1:] - centers_of_mass[:-1])
    # compute translational displacement along helix axes
    delta_123 = np.zeros(delta_xyz.shape)
    delta_123[:, 0] = np.sum(delta_xyz * 0.5 * (n1s[:-1] + n1s[1:]), axis=1)
    delta_123[:, 1] = np.sum(delta_xyz * 0.5 * (n2s[:-1] + n2s[1:]), axis=1)
    delta_123[:, 2] = np.sum(delta_xyz * 0.5 * (n3s[:-1] + n3s[1:]), axis=1)

    # add initial zero point so that np.diff(pos_123, axis=0) = delta_123
    pos_123 = np.concatenate((np.zeros((1, 3)), np.cumsum(delta_123, axis=0)), axis=0)

    # Compute translation displacements and angles
    # todo: is it better to use 0.5 * (n2s[frame + 1] + n2s[frame])
    # alpha_i be the rotation angle about vector i.
    # let ijk be even permutation, then d alpha_i = dn_j / dt \dot n_k
    dangles = np.stack((np.sum((n2s[1:] - n2s[:-1]) * 0.5 * (n3s[1:] + n3s[:-1]), axis=1),
                        np.sum((n3s[1:] - n3s[:-1]) * 0.5 * (n1s[1:] + n1s[:-1]), axis=1),
                        np.sum((n1s[1:] - n1s[:-1]) * 0.5 * (n2s[1:] + n2s[:-1]), axis=1)
                        ),
                       axis=1)

    # add initial zero point so that np.diff(EuAng, axis=0) = dangles
    EuAng = np.concatenate((np.zeros((1, 3)), np.cumsum(dangles, axis=0)), axis=0)

    generalized_coords = np.concatenate((pos_123, EuAng), axis=1)

    # combine all displacement...
    delta_all = np.concatenate((delta_123, dangles), axis=1)

    # #############################
    # look at step-size distributions
    # #############################
    if plot_cdfs:
        def get_cdf(dr):
            dr = np.sort(dr)
            dr_unique, counts = np.unique(dr, return_counts=True)
            csum = np.cumsum(counts)
            return dr_unique, csum / csum[-1]

        # fit CDF's, p = [mu, sigma]
        def cdf_gauss_fit_fn(p, dr):
            return 0.5 * (1 + erf((dr - p[0]) / (np.sqrt(2) * p[1])))

        def cdf_double_gauss_fit_fn(p, dr):
            g1 = p[4] * 0.5 * (1 + erf((dr - p[0]) / (np.sqrt(2) * p[1])))
            g2 = (1 - p[4]) * 0.5 * (1 + erf((dr - p[2]) / (np.sqrt(2) * p[3])))
            return g1 + g2

        nmax_step = 10
        for ss in range(1, nmax_step):
            steps_body = generalized_coords[ss:] - generalized_coords[:-ss]

            figh = plt.figure(figsize=(25, 6))
            figh.suptitle(f"step size distributions, body frame, {dset_prefix:s}, time step = {ss:d}")
            grid = figh.add_gridspec(nrows=1, ncols=6, hspace=0.5)

            for ii in range(6):
                dr_now, cdf_now = get_cdf(steps_body[:, ii])

                ax = figh.add_subplot(grid[0, ii])

                # gauss fit
                def cost_fn(p): return cdf_gauss_fit_fn(p, dr_now) - cdf_now
                init_params = [np.mean(steps_body[:, ii]), np.std(steps_body[:, ii])]
                results = least_squares(cost_fn, init_params)
                fit_params_gauss_cdf = results["x"]

                # double gauss fit
                def cost_fn(p): return cdf_double_gauss_fit_fn(p, dr_now) - cdf_now
                init_params = [0, np.std(steps_body[:, ii]),
                               np.mean(steps_body[:, ii]), np.std(steps_body[:, ii]),
                               0.85]
                results = least_squares(cost_fn, init_params, bounds=([-1e-5, -np.inf, -np.inf, -np.inf, 0],
                                                                      [1e-5, np.inf, np.inf, np.inf, 1]))
                if results["x"][4] >= 0.5:
                    fit_params_double_gauss_cdf = results["x"]
                else:
                    fit_params_double_gauss_cdf[0] = results["x"][2]
                    fit_params_double_gauss_cdf[1] = results["x"][3]
                    fit_params_double_gauss_cdf[2] = results["x"][0]
                    fit_params_double_gauss_cdf[3] = results["x"][1]
                    fit_params_double_gauss_cdf[4] = 1 - results["x"][4]


                # fit
                dr_interp = np.linspace(np.min(dr_now), np.max(dr_now), 1000)
                ax.plot(dr_interp, cdf_gauss_fit_fn(fit_params_gauss_cdf, dr_interp), c="r", marker="")
                ax.plot(dr_interp, cdf_double_gauss_fit_fn(fit_params_double_gauss_cdf, dr_interp), c="g", marker="")
                # ax.plot(dr_interp, cdf_gauss_fit_fn(fit_params_cdf, dr_interp), c="r", marker="")
                # data
                ax.plot(dr_now, cdf_now, c=np.array([0, 0, 1, 0.5]), marker=".", markersize=5, linestyle="")

                ax.set_xlabel("Step size")
                # ax.set_ylabel("CDF")
                ax.set_title(f"CDF, $dx_{{ {ii:d} }}$\n"
                             f"$\mu$={fit_params_double_gauss_cdf[0]:.3f}, $\sigma$={fit_params_double_gauss_cdf[1]:.3f}, fr={fit_params_double_gauss_cdf[4]:.2f}\n"
                             f"$\mu$={fit_params_double_gauss_cdf[2]:.3f}, $\sigma$={fit_params_double_gauss_cdf[3]:.3f}")

            figh.savefig(data_dir / f"{dset_prefix:s}_step_size_distribution_body_frame={ss:d}.png")
            plt.close(figh)

            # in lab coordinates
            steps_lab = centers_of_mass[ss:] - centers_of_mass[:-ss]

            figh = plt.figure(figsize=(25, 5))
            figh.suptitle(f"step size distributions, lab frame, {dset_prefix:s}, time step = {ss:d}")
            grid = figh.add_gridspec(nrows=1, ncols=3)

            for ii in range(3):
                dr_now, cdf_now = get_cdf(steps_lab[:, ii])

                ax = figh.add_subplot(grid[0, ii])

                # gauss fit
                def cost_fn(p): return cdf_gauss_fit_fn(p, dr_now) - cdf_now
                init_params = [np.mean(steps_lab[:, ii]), np.std(steps_lab[:, ii])]
                results = least_squares(cost_fn, init_params)
                fit_params_gauss_cdf = results["x"]

                # double gauss fit
                def cost_fn(p): return cdf_double_gauss_fit_fn(p, dr_now) - cdf_now
                init_params = [0, np.std(steps_body[:, ii]),
                               np.mean(steps_body[:, ii]), np.std(steps_body[:, ii]),
                               0.85]
                results = least_squares(cost_fn, init_params, bounds=([-1e-5, -np.inf, -np.inf, -np.inf, 0],
                                                                      [1e-5, np.inf, np.inf, np.inf, 1]))
                if results["x"][4] >= 0.5:
                    fit_params_double_gauss_cdf = results["x"]
                else:
                    fit_params_double_gauss_cdf[0] = results["x"][2]
                    fit_params_double_gauss_cdf[1] = results["x"][3]
                    fit_params_double_gauss_cdf[2] = results["x"][0]
                    fit_params_double_gauss_cdf[3] = results["x"][1]
                    fit_params_double_gauss_cdf[4] = 1 - results["x"][4]

                # fit
                dr_interp = np.linspace(np.min(dr_now), np.max(dr_now), 1000)
                ax.plot(dr_interp, cdf_gauss_fit_fn(fit_params_gauss_cdf, dr_interp), c="r", marker="")
                ax.plot(dr_interp, cdf_double_gauss_fit_fn(fit_params_double_gauss_cdf, dr_interp), c="g", marker="")
                # data
                ax.plot(dr_now, cdf_now, c=np.array([0, 0, 1, 0.5]), marker=".", markersize=5, linestyle="")

                ax.set_xlabel("Step size")
                # ax.set_ylabel("CDF")
                ax.set_title(f"CDF, $dx_{{ {ii:d} }}$\n"
                             f"$\mu$={fit_params_double_gauss_cdf[0]:.3f}, $\sigma$={fit_params_double_gauss_cdf[1]:.3f}, fr={fit_params_double_gauss_cdf[4]:.2f}\n"
                             f"$\mu$={fit_params_double_gauss_cdf[2]:.3f}, $\sigma$={fit_params_double_gauss_cdf[3]:.3f}")

            figh.savefig(data_dir / f"{dset_prefix:s}_step_size_distribution_lab_frame={ss:d}.png")
            plt.close(figh)


    # #############################
    # compute and fit MSDs in lab frame
    # #############################
    def msd_fn(p, t): return p[0] * t + p[1]

    time_lags = np.arange(1, len(delta_all) + 1) * exp3D_sec

    #  compute msds in lab-frame
    msds_lab = np.zeros((3, 3, len(delta_all)))
    msd_unc_lab = np.zeros(msds_lab.shape)
    msd_fit_params_lab = np.zeros((3, 3, nmax_msd_fit, 2)) * np.nan
    msd_fit_params_unc_lab = np.zeros((3, 3, nmax_msd_fit, 2)) * np.nan
    # fourth moments
    fourth_moments_lab = np.zeros((3, len(delta_all)))
    fourth_moments_unc_lab = np.zeros(fourth_moments_lab.shape)

    for ii in range(3):
        for jj in range(3):
            if ii >= jj:
                # 4th moments
                if ii == jj:
                    fourth_moments_lab[ii], fourth_moments_unc_lab[ii] = fourth_moment(centers_of_mass[:, ii])


                # msds
                msds_lab[ii, jj], msd_unc_lab[ii, jj] = msd_corr(centers_of_mass[:, ii], centers_of_mass[:, jj])
                msds_lab[jj, ii] = msds_lab[ii, jj]
                msd_unc_lab[jj, ii] = msd_unc_lab[ii, jj]

                for kk in range(2, nmax_msd_fit):
                    init_slope = np.mean(np.diff(msds_lab[ii, jj, :kk])) * exp3D_sec
                    init_offset = msds_lab[ii, jj, 0]

                    # fit model function including uncertainty
                    def model_fn(p): return (msd_fn(p, time_lags[:kk]) - msds_lab[ii, jj, :kk]) / msd_unc_lab[ii, jj, :kk]

                    results = least_squares(model_fn, [init_slope, init_offset])

                    msd_fit_params_lab[ii, jj, kk] = results["x"]
                    msd_fit_params_lab[jj, ii, kk] = results["x"]

                    # todo: check this
                    # set uncertainty
                    with np.errstate(divide="ignore", invalid="ignore"):
                        chi_sqr = 2 * results["cost"] / (kk - 2)
                        jacobian = results['jac']
                        cov = chi_sqr * np.linalg.inv(jacobian.transpose().dot(jacobian))

                        msd_fit_params_unc_lab[ii, jj, kk] = np.sqrt(np.diag(cov))
                        msd_fit_params_unc_lab[jj, ii, kk] = msd_fit_params_unc_lab[ii, jj, kk]



    # compute msds in 123 coordinates
    msds_body = np.zeros((6, 6, len(delta_all)))
    msd_unc_body = np.zeros((6, 6, len(delta_all)))
    msd_fit_params_body = np.zeros((6, 6, nmax_msd_fit, 2)) * np.nan
    msds_fit_params_unc_body = np.zeros(msd_fit_params_body.shape) * np.nan
    #
    fourth_moments_body = np.zeros((6, len(delta_all)))
    fourth_moments_unc_body = np.zeros(fourth_moments_body.shape)

    for ii in range(6):
        for jj in range(6):
            if ii >= jj:
                # 4th moments
                if ii == jj:
                    fourth_moments_body[ii], fourth_moments_unc_body[ii] = fourth_moment(generalized_coords[:, ii])

                # MSDs
                msds_body[ii, jj], msd_unc_body[ii, jj] = msd_corr(generalized_coords[:, ii], generalized_coords[:, jj])

                # set the symmetric points also
                msds_body[jj, ii] = msds_body[ii, jj]
                msd_unc_body[jj, ii] = msd_unc_body[ii, jj]

                for kk in range(2, nmax_msd_fit):
                    init_slope = np.mean(np.diff(msds_body[ii, jj, :kk])) * exp3D_sec
                    init_offset = msds_body[ii, jj, 0]

                    # fit model function including uncertainty
                    def model_fn(p): return (msd_fn(p, time_lags[:kk]) - msds_body[ii, jj, :kk]) / msd_unc_body[ii, jj, :kk]

                    results = least_squares(model_fn, [init_slope, init_offset])

                    msd_fit_params_body[ii, jj, kk] = results["x"]

                    # set the symmetric points also
                    msd_fit_params_body[jj, ii, kk] = results["x"]

                    # todo: check this
                    # set uncertainty
                    with np.errstate(divide="ignore", invalid="ignore"):
                        chi_sqr = 2 * results["cost"] / (kk - 2)
                        jacobian = results['jac']
                        cov = chi_sqr * np.linalg.inv(jacobian.transpose().dot(jacobian))

                        msds_fit_params_unc_body[ii, jj, kk] = np.sqrt(np.diag(cov))
                        msds_fit_params_unc_body[jj, ii, kk] = msds_fit_params_unc_body[ii, jj, kk]

    # plot single-step correlations
    # figh = plt.figure(figsize=figsize)
    # figh.suptitle(dset_prefix)
    # grid = figh.add_gridspec(nrows=6, ncols=6, hspace=0.4)
    #
    # for ii in range(6):
    #     for jj in range(6):
    #         if ii >= jj:
    #             ax = figh.add_subplot(grid[ii, jj])
    #             ax.plot(delta_all[:, ii], delta_all[:, jj], '.')
    #             ax.set_title(f"$dx_{ii:d}$ vs. $dx_{jj:d}$")

    # #############################
    # plot msds in 123 coordinates
    # #############################
    n_msd_fit_pts_to_plot = 10

    figh_msd = plt.figure(figsize=figsize)
    figh_msd.suptitle(f"{dset_prefix:s}, plotting MSD fit using first {n_msd_fit_pts_to_plot:d} points in body frame\n"
                      "$C_{ij}(\delta t) = \left \langle \left[ x_i(t + \delta t) - x_i(t) \\right] "
                      "\left[x_j(t + \delta t) - x_j(t) \\right] \\right \\rangle$", fontsize=fontsize_large)
    grid = figh_msd.add_gridspec(nrows=6, ncols=6, hspace=0.5, wspace=0.5)

    t_interp = np.linspace(0, (n_msd_fit_pts_to_plot + 1) * exp3D_sec, 1000)
    xlim = [-1 * exp3D_sec, (2 * n_msd_fit_pts_to_plot + 2) * exp3D_sec]

    for ii in range(6):
        for jj in range(6):
            if ii >= jj:
                ax = figh_msd.add_subplot(grid[ii, jj])
                ax.plot(xlim, [0, 0], c=np.array([0, 0, 0, 0.5]), linewidth=3)
                # plot fit
                ax.plot(t_interp, msd_fn(msd_fit_params_body[ii, jj, n_msd_fit_pts_to_plot], t_interp),
                        c=np.array([1, 0, 0, 0.5]), linewidth=3)
                # plot expt
                ax.errorbar(time_lags[:2 * n_msd_fit_pts_to_plot], msds_body[ii, jj, :2 * n_msd_fit_pts_to_plot],
                            yerr=msd_unc_body[ii, jj, :2 * n_msd_fit_pts_to_plot], c=np.array([0, 0, 1, 0.5]),
                            marker=".", markersize=8, linestyle="")

                ax.set_title(f"$C_{{ {ii:d},{jj:d} }} (\delta t)$")
                if ii == 5:
                    ax.set_xlabel("time (s)", fontsize=fontsize_small)

                if ii < 3 and jj < 3:
                    ax.set_ylabel("$\mu m^2$", fontsize=fontsize_small)
                elif ii >= 3 and jj >= 3:
                    ax.set_ylabel("rad$^2$", fontsize=fontsize_small)
                else:
                    ax.set_ylabel("rad $\cdot$ $\mu m$", fontsize=fontsize_small)

                ax.set_xlim(xlim)

    fname_fit = data_dir / f"{dset_prefix:s}_msd.png"
    figh_msd.savefig(fname_fit)
    plt.close(figh_msd)

    # #############################
    # plot msds in lab coordinates
    # #############################
    figh_msd = plt.figure(figsize=figsize)
    figh_msd.suptitle(f"{dset_prefix:s}, plotting MSD fit using first {n_msd_fit_pts_to_plot:d} points in lab frame\n"
                      "$C_{ij}(\delta t) = \left \langle \left[ x_i(t + \delta t) - x_i(t) \\right] "
                      "\left[x_j(t + \delta t) - x_j(t) \\right] \\right \\rangle$", fontsize=fontsize_large)
    grid = figh_msd.add_gridspec(nrows=3, ncols=3, hspace=0.5, wspace=0.5)

    t_interp = np.linspace(0, (n_msd_fit_pts_to_plot + 1) * exp3D_sec, 1000)
    for ii in range(3):
        for jj in range(3):
            if ii >= jj:
                ax = figh_msd.add_subplot(grid[ii, jj])
                ax.plot(xlim, [0, 0], c=np.array([0, 0, 0, 0.5]), linewidth=3)
                # plot fit
                ax.plot(t_interp, msd_fn(msd_fit_params_lab[ii, jj, n_msd_fit_pts_to_plot], t_interp),
                        c=np.array([1, 0, 0, 0.5]), linewidth=3)
                # plot expt
                ax.errorbar(time_lags[:2 * n_msd_fit_pts_to_plot],
                            msds_lab[ii, jj, :2 * n_msd_fit_pts_to_plot],
                            yerr=msd_unc_lab[ii, jj, :2 * n_msd_fit_pts_to_plot], c=np.array([0, 0, 1, 0.5]),
                            marker=".", markersize=8, linestyle="")

                ax.set_title(f"$C_{{ {ii:d},{jj:d} }} (\delta t)$", fontsize=fontsize_small)
                if ii == 2:
                    ax.set_xlabel("time (s)", fontsize=fontsize_small)

                ax.set_ylabel("$\mu m^2$", fontsize=fontsize_small)


    fname_fit = data_dir / f"{dset_prefix:s}_msd_lab_frame.png"
    figh_msd.savefig(fname_fit)
    plt.close(figh_msd)

    # #############################
    # save results
    # #############################
    fname_fit_data = (data_dir / dset_prefix).with_suffix(".zarr")
    if not data_dir.exists():
        data_dir.mkdir()

    fit_data = zarr.open(str(fname_fit_data), mode="w")
    fit_data.attrs["processing_timestamp"] = tstamp
    fit_data.attrs["processing_time"] = time.perf_counter() - tstart_folder
    fit_data.attrs["file_name"] = str(threshold_fname)
    fit_data.attrs["frame_range"] = [frame_start, frame_end]
    fit_data.attrs["dxyz_um"] = dxyz_um
    fit_data.attrs["exposure_time_ms"] = camExposure_ms
    fit_data.attrs["volumeteric_exposure_time_ms"] = exp3D_sec * 1e3

    visc = np.nan
    for k, v in pattern_viscosity_dict.items():
        if re.match(k, dset_prefix):
            visc = v

    fit_data.attrs["viscosity"] = visc
    fit_data.attrs["viscosity units"] = "Pascals * seconds"

    fit_data.create_dataset("img_thresholded", shape=blob_masks.shape, chunks=(1,) + blob_masks.shape[1:],
                            dtype=bool, compressor="none")
    fit_data.img_thresholded[:] = blob_masks

    fit_data.create_dataset("helix_fit", shape=fitImage.shape, chunks=(1,) + fitImage.shape[1:], dtype=bool, compressor="none")
    fit_data.helix_fit[:] = fitImage

    fit_data.create_dataset("lengths", shape=flagella_len.shape, dtype=float, compressor="none")
    fit_data.lengths[:] = flagella_len

    fit_data.create_dataset("helix_fit_params", shape=helix_fit_params.shape, dtype=float, compressor="none")
    fit_data.helix_fit_params[:] = helix_fit_params

    fit_data.create_dataset("n1", shape=n1s.shape, dtype=float, compressor="none")
    fit_data.n1[:] = n1s
    fit_data.create_dataset("n2", shape=n2s.shape, dtype=float, compressor="none")
    fit_data.n2[:] = n2s
    fit_data.create_dataset("n3", shape=n3s.shape, dtype=float, compressor="none")
    fit_data.n3[:] = n3s
    fit_data.create_dataset("m2", shape=m2s.shape, dtype=float, compressor="none")
    fit_data.m2[:] = m2s
    fit_data.create_dataset("m3", shape=m3s.shape, dtype=float, compressor="none")
    fit_data.m3[:] = m3s

    fit_data.create_dataset("center_of_mass", shape=centers_of_mass.shape, dtype=float, compressor="none")
    fit_data.center_of_mass[:] = centers_of_mass

    # lab coordinate MSDs
    fit_data.create_dataset("lag_times", shape=time_lags.shape, dtype="float", compressor="none")
    fit_data.lag_times[:] = time_lags

    fit_data.create_dataset("msds_lab", shape=msds_lab.shape, dtype=float, compressor="none")
    fit_data.msds_lab[:] = msds_lab

    fit_data.create_dataset("msds_unc_lab", shape=msd_unc_lab.shape, dtype=float, compressor="none")
    fit_data.msds_unc_lab[:] = msd_unc_lab

    fit_data.create_dataset("msd_fit_params_lab", shape=msd_fit_params_lab.shape, dtype=float, compressor="none")
    fit_data.msd_fit_params_lab[:] = msd_fit_params_lab

    fit_data.create_dataset("msd_fit_params_unc_lab", shape=msd_fit_params_unc_lab.shape, dtype=float, compressor="none")
    fit_data.msd_fit_params_unc_lab[:] = msd_fit_params_unc_lab

    fit_data.create_dataset("fourth_moments_lab", shape=fourth_moments_lab.shape, dtype=float, compressor="none")
    fit_data.fourth_moments_lab[:] = fourth_moments_lab

    fit_data.create_dataset("fourth_moments_unc_lab", shape=fourth_moments_unc_lab.shape, dtype=float, compressor="none")
    fit_data.fourth_moments_unc_lab[:] = fourth_moments_unc_lab

    # derived quantities xyz
    fit_data.create_dataset("diffusion_constants_lab", shape=msd_fit_params_lab.shape[:-1], dtype=float, compressor="none")
    fit_data.diffusion_constants_lab[:] = 0.5 * msd_fit_params_lab[..., 0]

    fit_data.create_dataset("diffusion_constants_unc_lab", shape=msd_fit_params_unc_lab.shape[:-1], dtype=float, compressor="none")
    fit_data.diffusion_constants_unc_lab[:] = 0.5 * msd_fit_params_unc_lab[..., 0]

    # 123 coordinates MSDs
    fit_data.create_dataset("generalized_coordinates", shape=generalized_coords.shape, dtype=float, compressor="none")
    fit_data.generalized_coordinates[:] = generalized_coords

    fit_data.create_dataset("msds_body", shape=msds_body.shape, dtype=float, compressor="none")
    fit_data.msds_body[:] = msds_body

    fit_data.create_dataset("msds_unc_body", shape=msd_unc_body.shape, dtype=float, compressor="none")
    fit_data.msds_unc_body[:] = msd_unc_body

    fit_data.create_dataset("msd_fit_params_body", shape=msd_fit_params_body.shape, dtype=float, compressor="none")
    fit_data.msd_fit_params_body[:] = msd_fit_params_body

    fit_data.create_dataset("msd_fit_params_unc_body", shape=msds_fit_params_unc_body.shape, dtype=float, compressor="none")
    fit_data.msd_fit_params_unc_body[:] = msds_fit_params_unc_body

    fit_data.create_dataset("fourth_moments_body", shape=fourth_moments_body.shape, dtype=float, compressor="none")
    fit_data.fourth_moments_body[:] = fourth_moments_body

    fit_data.create_dataset("fourth_moments_unc_body", shape=fourth_moments_unc_body.shape, dtype=float, compressor="none")
    fit_data.fourth_moments_unc_body[:] = fourth_moments_unc_body

    # derived quantities 123
    fit_data.create_dataset("diffusion_constants_body", shape=msd_fit_params_body.shape[:-1], dtype=float, compressor="none")
    fit_data.diffusion_constants_body[:] = 0.5 * msd_fit_params_body[..., 0]

    fit_data.create_dataset("diffusion_constants_unc_body", shape=msd_fit_params_body.shape[:-1], dtype=float, compressor="none")
    fit_data.diffusion_constants_unc_body[:] = 0.5 * msds_fit_params_unc_body[..., 0]

    fit_data.create_dataset("propulsion_matrix", shape=msd_fit_params_body.shape[:-1], dtype=float, compressor="none")
    # do permutations/transposes because linalg.inv acts on last two axes, but need to act on first two
    fit_data.propulsion_matrix[:] = kb * T * np.transpose(np.linalg.inv(0.5 * np.transpose(msd_fit_params_body[..., 0], [2, 0, 1])), [1, 2, 0])


    # #############################
    # store summary data
    # #############################
    summary_now = {"name": dset_prefix,
                   "number of frames": nt,
                   "mean length (um)": np.mean(flagella_len),
                   "std length (um)": np.std(flagella_len)}
    for ii in range(6):
        for jj in range(6):
            if ii >= jj:
                summary_now.update({f"D({ii:d}, {jj:d}) body": fit_data.diffusion_constants_body[ii, jj, n_msd_fit_pts_to_plot]})
    for ii in range(3):
        for jj in range(3):
            if ii >= jj:
                summary_now.update({f"D({ii:d}, {jj:d}) lab": fit_data.diffusion_constants_lab[ii, jj, n_msd_fit_pts_to_plot]})

    summary = pd.concat((summary, pd.DataFrame(summary_now, index=[0])), ignore_index=True)

summary.to_csv(summary_fname)
