"""
Analyze diffusing flagella datasets
"""
# visualization
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 10})
import napari
# analysis
import numpy as np
from sklearn.decomposition import PCA
from skimage import measure
from scipy.optimize import least_squares
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

figsize = (25, 15)
visualize_results = False
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
output_dir_pkl = root_dir / "PKL-files"
output_dir_img = root_dir / "TIF-reslice"
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
        # extract coordinates
        x0 = np.argwhere(blob).astype('float')  # coordinates

        # compute center of mass
        cm_now = np.array([sum(x0[:, j]) for j in range(x0.shape[1])]) / x0.shape[0]
        centers_of_mass[frame, :] = cm_now  # store center of mass

        # ##############################################################
        # determine axis n1 from PCA and consistency with previous point
        # ##############################################################
        coords = x0 - cm_now  # shift all the coordinates into origin
        pca = PCA(n_components=3)
        pca.fit(coords)
        n1s[frame] = pca.components_[0]
        m2s[frame] = pca.components_[1]
        m3s[frame] = pca.components_[2]

        # choose the sign of current n1 so it is as close as possible to n1 at the previous timestep
        if frame > frame_start and np.linalg.norm(n1s[frame] - n1s[frame - 1]) > np.linalg.norm(
                n1s[frame] + n1s[frame - 1]):
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
        pts_on_n1 = n1s[frame, 0] * coords[:, 0] + \
                    n1s[frame, 1] * coords[:, 1] + \
                    n1s[frame, 2] * coords[:, 2]
        pts_on_m2 = m2s[frame, 0] * coords[:, 0] + \
                    m2s[frame, 1] * coords[:, 1] + \
                    m2s[frame, 2] * coords[:, 2]
        pts_on_m3 = m3s[frame, 0] * coords[:, 0] + \
                    m3s[frame, 1] * coords[:, 1] + \
                    m3s[frame, 2] * coords[:, 2]
        coord_on_principal = np.stack([pts_on_n1,
                                       pts_on_m2,
                                       pts_on_m3], axis=1)

        # ##########################
        # Curve fit helix projection
        # ##########################
        # Fix amplitude and wave number, and set initial guess for phase
        amplitude = 1.65
        wave_number = 0.28
        phase = np.pi / 10

        def cosine_fn(pts_on_n1, a):  # model for "y" projection (on xz plane)
            return a[0] * np.cos(a[1] * pts_on_n1 + a[2])


        def sine_fn(pts_on_n1, a):  # model for "z" projection (on xy plane)
            return a[0] * np.sin(a[1] * pts_on_n1 + a[2])


        def cost_fn(a):
            cost = np.concatenate((cosine_fn(pts_on_n1, a) -
                                   pts_on_m2, sine_fn(pts_on_n1, a) - pts_on_m3)) / pts_on_n1.size
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


        init_params = np.array([amplitude, wave_number, phase])
        lower_bounds = [1.5, 0.25, -np.inf]
        upper_bounds = [2.5, 0.3, np.inf]

        # fix none
        results_fit = least_squares(lambda p: cost_fn([p[0], p[1], p[2]]),
                                    init_params[0:3],
                                    bounds=(lower_bounds[0:3], upper_bounds[0:3]))
        amplitude = results_fit["x"][0]
        wave_number = results_fit["x"][1]
        phase = results_fit["x"][2]

        # Save fit parameters
        fit_params = np.array([amplitude, wave_number, phase])
        helix_fit_params[frame, :] = fit_params

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
        fit_P = np.array([x, yb, zb]).T
        fit_P = np.append(fit_P, np.array([x, yb, zm]).T, axis=0)
        fit_P = np.append(fit_P, np.array([x, yb, zt]).T, axis=0)
        fit_P = np.append(fit_P, np.array([x, ym, zb]).T, axis=0)
        fit_P = np.append(fit_P, np.array([x, ym, zm]).T, axis=0)
        fit_P = np.append(fit_P, np.array([x, ym, zt]).T, axis=0)
        fit_P = np.append(fit_P, np.array([x, yt, zb]).T, axis=0)
        fit_P = np.append(fit_P, np.array([x, yt, zm]).T, axis=0)
        fit_P = np.append(fit_P, np.array([x, yt, zt]).T, axis=0)

        # matrix rotation
        mrot = np.linalg.inv(np.vstack([n1s[frame], m2s[frame], m3s[frame]]))

        # inverse transform
        # fit_X = pca.inverse_transform(fit_P)+ CM1
        fit_X = np.matmul(mrot, fit_P.T).T + cm_now
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
        X0_post_fit = np.argwhere(fitImage[frame]).astype(float)  # coordinates

        # compute center of mass
        cm_now = np.array([sum(X0_post_fit[:, j]) for j in range(X0_post_fit.shape[1])]) / X0_post_fit.shape[0]
        centers_of_mass[frame, :] = cm_now  # store center of mass

        # ##########################################
        # determine the flagella length along the n1
        # ##########################################
        flagella_len[frame] = (np.max(pts_on_n1) - np.min(pts_on_n1)) * dxyz_um

        # ########################################
        # compute n2 and n3 from phase information
        # ########################################

        n2s[frame] = m2s[frame] * np.cos(phase) - m3s[frame] * np.sin(phase)
        n2s[frame] = n2s[frame] / np.linalg.norm(n2s[frame])
        # n3s[frame] = m2s[frame] * np.sin(phase) + m3s[frame] * np.cos(phase)

        assert n1s[frame].dot(n2s[frame]) < 1e-12

        # generate n3 such that coordinate system is right-handed
        n3s[frame] = np.cross(n1s[frame], n2s[frame])
        n3s[frame] = n3s[frame] / np.linalg.norm(n3s[frame])

        assert n1s[frame].dot(n3s[frame]) < 1e-12

        # negative control: not tracking any points
        # n2s[frame] = m2s[frame]
        # n3s[frame] = m3s[frame]
    print("")

    # #############################
    # translational and angular displacements
    # #############################

    # compute translational displacement along lab axes
    delta_xyz = (centers_of_mass[1:] - centers_of_mass[:-1]) * dxyz_um
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
    # compute and fit MSDs
    # #############################
    def msd_fn(p, t): return p[0] * t + p[1]

    #  compute msds in lab-frame
    msds_lab = np.zeros((3, 3, len(delta_all)))
    msd_unc_lab = np.zeros(msds_lab.shape)
    msd_lab_fit_params = np.zeros((3, 3, nmax_msd_fit, 2)) * np.nan
    msd_lab_fit_params_unc = np.zeros((3, 3, nmax_msd_fit, 2)) * np.nan
    for ii in range(3):
        for jj in range(3):
            if ii >= jj:
                msds_lab[ii, jj], msd_unc_lab[ii, jj] = msd_corr(centers_of_mass[:, ii], centers_of_mass[:, jj])
                msds_lab[jj, ii] = msds_lab[ii, jj]
                msd_unc_lab[jj, ii] = msd_unc_lab[ii, jj]

                for kk in range(2, nmax_msd_fit):
                    init_slope = np.mean(np.diff(msds_lab[ii, jj, :kk])) * exp3D_sec
                    init_offset = msds_lab[ii, jj, 0]

                    # fit model function including uncertainty
                    def model_fn(p): return (msd_fn(p, np.arange(kk) * exp3D_sec) - msds_lab[ii, jj, :kk]) / msd_unc_lab[ii, jj, :kk]

                    results = least_squares(model_fn, [init_slope, init_offset])

                    msd_lab_fit_params[ii, jj, kk] = results["x"]
                    msd_lab_fit_params[jj, ii, kk] = results["x"]

                    # todo: check this
                    # set uncertainty
                    with np.errstate(divide="ignore", invalid="ignore"):
                        chi_sqr = 2 * results["cost"] / (kk - 2)
                        jacobian = results['jac']
                        cov = chi_sqr * np.linalg.inv(jacobian.transpose().dot(jacobian))

                        msd_lab_fit_params_unc[ii, jj, kk] = np.sqrt(np.diag(cov))
                        msd_lab_fit_params_unc[jj, ii, kk] = msd_lab_fit_params_unc[ii, jj, kk]




    # compute msds in 123 coordinates
    msds = np.zeros((6, 6, len(delta_all)))
    msd_unc_mean = np.zeros((6, 6, len(delta_all)))
    msd_fit_params = np.zeros((6, 6, nmax_msd_fit, 2)) * np.nan
    msds_fit_params_unc = np.zeros(msd_fit_params.shape) * np.nan
    for ii in range(6):
        for jj in range(6):
            if ii >= jj:
                msds[ii, jj], msd_unc_mean[ii, jj] = msd_corr(generalized_coords[:, ii], generalized_coords[:, jj])

                # set the symmetric points also
                msds[jj, ii] = msds[ii, jj]
                msd_unc_mean[jj, ii] = msd_unc_mean[ii, jj]

                for kk in range(2, nmax_msd_fit):
                    init_slope = np.mean(np.diff(msds[ii, jj, :kk])) * exp3D_sec
                    init_offset = msds[ii, jj, 0]

                    # fit model function including uncertainty
                    def model_fn(p): return (msd_fn(p, np.arange(kk) * exp3D_sec) - msds[ii, jj, :kk]) / msd_unc_mean[ii, jj, :kk]

                    results = least_squares(model_fn, [init_slope, init_offset])

                    msd_fit_params[ii, jj, kk] = results["x"]

                    # set the symmetric points also
                    msd_fit_params[jj, ii, kk] = results["x"]

                    # todo: check this
                    # set uncertainty
                    with np.errstate(divide="ignore", invalid="ignore"):
                        chi_sqr = 2 * results["cost"] / (kk - 2)
                        jacobian = results['jac']
                        cov = chi_sqr * np.linalg.inv(jacobian.transpose().dot(jacobian))

                        msds_fit_params_unc[ii, jj, kk] = np.sqrt(np.diag(cov))
                        msds_fit_params_unc[jj, ii, kk] = msds_fit_params_unc[ii, jj, kk]

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
    figh_msd.suptitle(f"{dset_prefix:s}, plotting MSD fit using first {n_msd_fit_pts_to_plot:d} points")
    grid = figh_msd.add_gridspec(nrows=6, ncols=6, hspace=0.5, wspace=0.5)

    t_interp = np.linspace(0, n_msd_fit_pts_to_plot * exp3D_sec, 1000)
    for ii in range(6):
        for jj in range(6):
            if ii >= jj:
                ax = figh_msd.add_subplot(grid[ii, jj])
                # plot fit
                ax.plot(t_interp, msd_fn(msd_fit_params[ii, jj, n_msd_fit_pts_to_plot], t_interp), 'r')
                # plot expt
                ax.errorbar(np.arange(2*n_msd_fit_pts_to_plot) * exp3D_sec, msds[ii, jj, :2*n_msd_fit_pts_to_plot],
                            yerr=msd_unc_mean[ii, jj, :2*n_msd_fit_pts_to_plot], fmt='b.')

                ax.set_title(f"$\langle \left[ x_{ii:d}(t + \delta t) - x_{ii:d}(t) \\right] "
                             f"\left[ x_{jj:d}(t + \delta t) - x_{jj:d}(t) \\right] \\rangle$")
                ax.set_xlabel("time (s)")

                if ii < 3 and jj < 3:
                    ax.set_ylabel("MSD ($\mu m^2$)")
                elif ii >= 3 and jj >= 3:
                    ax.set_ylabel("MSD (rad$^2$)")
                else:
                    ax.set_ylabel("MSD (rad $\cdot$ $\mu m$)")

    fname_fit = data_dir / f"{dset_prefix:s}_msd.png"
    figh_msd.savefig(fname_fit)
    plt.close(figh_msd)

    # #############################
    # plot msds in lab coordinates
    # #############################
    figh_msd = plt.figure(figsize=figsize)
    figh_msd.suptitle(f"{dset_prefix:s}, plotting MSD fit using first {n_msd_fit_pts_to_plot:d} points in lab coordinates")
    grid = figh_msd.add_gridspec(nrows=3, ncols=3, hspace=0.5, wspace=0.5)

    t_interp = np.linspace(0, n_msd_fit_pts_to_plot * exp3D_sec, 1000)
    for ii in range(3):
        for jj in range(3):
            if ii >= jj:
                ax = figh_msd.add_subplot(grid[ii, jj])
                # plot fit
                ax.plot(t_interp, msd_fn(msd_lab_fit_params[ii, jj, n_msd_fit_pts_to_plot], t_interp), 'r')
                # plot expt
                ax.errorbar(np.arange(2 * n_msd_fit_pts_to_plot) * exp3D_sec,
                            msds_lab[ii, jj, :2 * n_msd_fit_pts_to_plot],
                            yerr=msd_unc_lab[ii, jj, :2 * n_msd_fit_pts_to_plot], fmt='b.')

                ax.set_title(f"$\langle \left[ x_{ii:d}(t + \delta t) - x_{ii:d}(t) \\right] "
                             f"\left[ x_{jj:d}(t + \delta t) - x_{jj:d}(t) \\right] \\rangle$")
                ax.set_xlabel("time (s)")

                if ii < 3 and jj < 3:
                    ax.set_ylabel("MSD ($\mu m^2$)")
                elif ii >= 3 and jj >= 3:
                    ax.set_ylabel("MSD (rad$^2$)")
                else:
                    ax.set_ylabel("MSD (rad $\cdot$ $\mu m$)")

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
    fit_data.create_dataset("msds_lab", shape=msds_lab.shape, dtype=float, compressor="none")
    fit_data.msds_lab[:] = msds_lab

    fit_data.create_dataset("msds_lab_unc", shape=msd_unc_lab.shape, dtype=float, compressor="none")
    fit_data.msds_lab_unc[:] = msd_unc_lab

    fit_data.create_dataset("msd_lab_fit_params", shape=msd_lab_fit_params.shape, dtype=float, compressor="none")
    fit_data.msd_lab_fit_params[:] = msd_lab_fit_params

    fit_data.create_dataset("msd_lab_fit_params_unc", shape=msd_lab_fit_params_unc.shape, dtype=float, compressor="none")
    fit_data.msd_lab_fit_params_unc[:] = msd_lab_fit_params_unc

    # 123 coordinates MSDs
    fit_data.create_dataset("generalized_coordinates", shape=generalized_coords.shape, dtype=float, compressor="none")
    fit_data.generalized_coordinates[:] = generalized_coords

    fit_data.create_dataset("msds", shape=msds.shape, dtype=float, compressor="none")
    fit_data.msds[:] = msds

    fit_data.create_dataset("msds_unc", shape=msd_unc_mean.shape, dtype=float, compressor="none")
    fit_data.msds_unc[:] = msd_unc_mean

    fit_data.create_dataset("msd_fit_params", shape=msd_fit_params.shape, dtype=float, compressor="none")
    fit_data.msd_fit_params[:] = msd_fit_params

    fit_data.create_dataset("msd_fit_params_unc", shape=msds_fit_params_unc.shape, dtype=float, compressor="none")
    fit_data.msd_fit_params_unc[:] = msds_fit_params_unc

    fit_data.create_dataset("diffusion_constants", shape=msd_fit_params.shape[:-1], dtype=float, compressor="none")
    fit_data.diffusion_constants[:] = 0.5 * msd_fit_params[..., 0]

    fit_data.create_dataset("diffusion_constants_unc", shape=msd_fit_params.shape[:-1], dtype=float, compressor="none")
    fit_data.diffusion_constants_unc[:] = 0.5 * msds_fit_params_unc[..., 0]

    fit_data.create_dataset("propulsion_matrix", shape=msd_fit_params.shape[:-1], dtype=float, compressor="none")
    # do permutations/transposes because linalg.inv acts on last two axes, but need to act on first two
    fit_data.propulsion_matrix[:] = kb * T * np.transpose(np.linalg.inv(0.5 * np.transpose(msd_fit_params[..., 0], [2, 0, 1])), [1, 2, 0])

    fit_data.create_dataset("lengths", shape=flagella_len.shape, dtype=float, compressor="none")
    fit_data.lengths[:] = flagella_len
    fit_data.create_dataset("helix_fit_params", shape=helix_fit_params.shape, dtype=float, compressor="none")
    fit_data.helix_fit_params[:] = helix_fit_params

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
                summary_now.update({f"D({ii:d}, {jj:d})": fit_data.diffusion_constants[ii, jj, n_msd_fit_pts_to_plot]})
    summary = pd.concat((summary, pd.DataFrame(summary_now, index=[0])), ignore_index=True)

    # #############################
    # visualize
    # #############################
    if visualize_results:
        # %% View image, threshold, and fit together
        viewer = napari.Viewer(ndisplay=3)


        # imgs = tifffile.imread(intensityFiles[file_index])
        # viewer.add_image(imgs[frame_start:frame_end], name="deskewed intensity",
        #                  contrast_limits=[np.percentile(imgs[frame_start:frame_end], 1),
        #                                   np.percentile(imgs[frame_start:frame_end], 99.9)],
        #                  scale=[dxyz_um, dxyz_um, dxyz_um], blending='additive',
        #                  multiscale=False, colormap='gray', opacity=1)

        viewer.add_image(blob_masks[frame_start:frame_end], name="thresholded image",
                         contrast_limits=[0, 1],
                         scale=[dxyz_um, dxyz_um, dxyz_um], blending='additive',
                         multiscale=False, colormap='green', opacity=0.2)

        viewer.add_image(fitImage[frame_start:frame_end], name="helix fit",
                         contrast_limits=[0, 1],
                         scale=[dxyz_um, dxyz_um, dxyz_um], blending='additive',
                         multiscale=False, colormap='red', opacity=0.2)
        viewer.dims.axis_labels = ["time", "z", "y", "x"]
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = 'um'
        viewer.scale_bar.position = 'top_right'
        viewer.axes.visible = True
        napari.run()

summary.to_csv(summary_fname)
