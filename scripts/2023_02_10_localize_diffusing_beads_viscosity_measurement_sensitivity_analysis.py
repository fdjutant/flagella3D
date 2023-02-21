"""
Localize beads in data from Franky's experiment
"""
import time
import itertools
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import datetime
import napari
from pathlib import Path
import numpy as np
import dask.array as da
import zarr
from localize_psf import localize, affine, fit_psf
import tifffile
import pandas as pd
import trackpy as tp
from scipy.optimize import least_squares
from numpy.linalg import lstsq
import h5py

# ##############################
# load data
# ##############################
# root_dir = Path(r"\\10.206.26.21\flagella_project\Tetraspek_100nm_sucrose\20220114_Franky_2x_TS100nm_40suc_MBT")
# threshold = 250
# memories = np.array([0])
# search_ranges = np.linspace(0.75, 1.25, 3)
# min_trajs = np.array([2, 5])

# root_dir = Path(r"\\10.206.26.21\flagella_project\Tetraspek_100nm_sucrose\20220516_Franky_TS_diffusion_HILO_BufferOnly")
# root_dir = Path(r"\\10.206.26.21\flagella_project\Tetraspek_100nm_sucrose\20220516_Franky_TS_diffusion_HILO_50sucrose")
# root_dir = Path(r"\\10.206.26.21\flagella_project\Tetraspek_100nm_sucrose\20220516_Franky_TS_diffusion_HILO_70sucrose")
root_dir = Path(r"\\10.206.26.21\flagella_project\Tetraspek_100nm_sucrose\20220516_Franky_TS_diffusion_HILO_40sucrose")
threshold = 100
# memories = np.array([0, 2, 5])
# search_ranges = np.linspace(0.75, 2, 15)
# min_trajs = np.array([2, 5, 30])

memories = np.array([2])
search_ranges = np.array([1.5])
min_trajs = np.array([2])

fnames = list(root_dir.glob("*[!_locImage].tif"))
# fnames = [fnames[0]]

tstamp = datetime.datetime.now().strftime('%Y_%m_%d_%H;%M;%S')
save_dir = root_dir / f"{tstamp:s}_diffusion_coeff"
save_dir.mkdir(exist_ok=False)

dt = 10e-3  # 1 / 100.02286236854137
dz = 1
dxy = 0.11699999868869781
wavelength = 0.532
na = 1.4

show_napari = False
saving = True

for ff, fname in enumerate(fnames):
    img = tifffile.imread(fname)
    nt, ny, nx = img.shape

    dz_max_err = np.inf
    # dxy_max_err = 0.2
    dxy_max_err = 0.1
    z_min_spot_sep = 1
    xy_min_spot_sep = 0.2
    amp_min = 50.
    amp_max = 10000.
    sz_min = 0
    sz_max = np.inf
    sxy_min = 0.05
    sxy_max = 0.25
    dbounadry_z_min = 0
    dboundry_xy_min = 3 * dxy
    # roi_size = (1.75, 0.65, 0.65)
    roi_size = (1.75, 1, 1)

    # ##############################
    # localization
    # ##############################
    # simple filter
    # local coords
    coords_local = localize.get_coords((1, ny, nx), (dz, dxy, dxy))

    # model = fit_psf.gaussian3d_psf_model()
    # filter = localize.get_param_filter(coords_local,
    #                                    fit_dist_max_err=(dz_max_err, dxy_max_err),
    #                                    min_spot_sep=(z_min_spot_sep, xy_min_spot_sep),
    #                                    amp_bounds=(amp_min, amp_max),
    #                                    sigma_bounds=((sz_min, sxy_min), (sz_max, sxy_max)),
    #                                    dist_boundary_min=(dbounadry_z_min, dboundry_xy_min)
    #                                    )
    #
    model = fit_psf.gaussian3d_asymmetric_pixelated()


    class symmetry_filter(localize.filter):
        """
        Filter based on value being in a certain range
        """

        def __init__(self,
                     low: float,
                     high: float,
                     index1: int,
                     index2: int,
                     name: str):
            self.low = low
            self.high = high
            self.index1 = index1
            self.index2 = index2
            self.condition_names = [f"{name:s} too small", f"{name:s} too large"]

        def filter(self, fit_params, *args, **kwargs):
            conditions = np.stack((fit_params[:, self.index1] / fit_params[:, self.index2] >= self.low,
                                   fit_params[:, self.index1] / fit_params[:, self.index2] <= self.high), axis=1)

            return conditions


    # build filter for asymmetric gaussian
    z, y, x = coords_local
    filter_range = localize.range_filter(x.min() + dboundry_xy_min, x.max() - dboundry_xy_min, 1, "x-position") + \
                   localize.range_filter(y.min() + dboundry_xy_min, y.max() - dboundry_xy_min, 2, "y-position") + \
                   localize.range_filter(z.min() + dbounadry_z_min, z.max() - dbounadry_z_min, 3, "z-position") + \
                   localize.range_filter(sxy_min, sxy_max, 4, "x-size") + \
                   localize.range_filter(sxy_min, sxy_max, 5, "y-size") + \
                   localize.range_filter(sz_min, sz_max, 6, "z-size") + \
                   localize.range_filter(amp_min, amp_max, 0, "amplitude") + \
                   localize.proximity_filter((1, 2), 0, dxy_max_err, "xy") + \
                   localize.proximity_filter((3,), 0, dz_max_err, "z") + \
                   symmetry_filter(0.7, 1.4, 4, 5, "sx/sy")

    # full filter
    filter = localize.unique_filter(xy_min_spot_sep, z_min_spot_sep) * filter_range

    # filter = localize.no_filter()
    tstart = time.perf_counter()
    results_all = []
    imgs_filtered = []
    for ii in range(nt):
        if ii % 100 == 0:
            print(f"{ii + 1:d}/{nt:d} in {time.perf_counter() - tstart:.2f}s")

        _, r, img_filtered = localize.localize_beads_generic(img[ii],
                                                             (dz, dxy, dxy),
                                                             threshold=threshold,
                                                             roi_size=roi_size,
                                                             filter_sigma_small=(0, 0.5 * dxy, 0.5 * dxy),
                                                             filter_sigma_large=(0, 2., 2.),
                                                             min_spot_sep=(1., 0.4),
                                                             filter=filter,
                                                             model=model,
                                                             model_zsize_index=6,
                                                             use_gpu_fit=True,
                                                             use_gpu_filter=False,
                                                             return_filtered_images=True,
                                                             fit_filtered_images=False,
                                                             verbose=False)

        results_all.append(r)
        imgs_filtered.append(img_filtered)

    imgs_filtered = np.concatenate(imgs_filtered, axis=0)

    # collate localization data
    condition_names = results_all[0]["condition_names"]

    conditions_all_arr = np.concatenate([r["conditions"] for r in results_all])
    to_keep_all_arr = np.logical_and.reduce(conditions_all_arr, axis=1)

    param_names = ["frame"] + model.parameter_names

    init_params = [r["init_params"] for r in results_all]
    init_params = np.concatenate(
        [np.concatenate((ii * np.ones((len(p), 1)), p), axis=1) for ii, p in enumerate(init_params)], axis=0)

    params_all = [r["fit_params"] for r in results_all]
    params_all = np.concatenate(
        [np.concatenate((ii * np.ones((len(p), 1)), p), axis=1) for ii, p in enumerate(params_all)], axis=0)

    params_keep = params_all[to_keep_all_arr]
    centers_keep = params_keep[:, (0, 3, 2)]
    centers_not = init_params[np.logical_not(to_keep_all_arr)][:, (0, 3, 2)]

    rois_inds_all = np.concatenate([np.arange(len(r["to_keep"])) for r in results_all])
    roi_inds_kept = rois_inds_all[to_keep_all_arr]
    roi_inds_rejected = rois_inds_all[np.logical_not(to_keep_all_arr)]

    # ##############################
    # tracking
    # ##############################
    max_msd_points = 20
    max_imsd_points = 150
    nt_to_fit = 10
    kb = 1.38e-23
    T = 273 + 25
    R = 99e-3 / 2


    def insert_missing_frames(df):
        min_frame = df["frame"].min()
        max_frame = df["frame"].max()

        required_frames = np.arange(min_frame, max_frame + 1)
        missing_frames = np.array([f for f in required_frames if f not in df["frame"].values])

        new_dat = {}
        for c in df.columns:
            if c == "frame":
                new_dat.update({"frame": missing_frames})
            elif c == "particle":
                new_dat.update({"particle": np.ones(len(missing_frames)) * df["particle"].values[0]})
            else:
                new_dat.update({c: np.zeros(len(missing_frames)) * np.nan})

        df_new = pd.DataFrame(new_dat)

        df = pd.concat((df, df_new))

        return df


    def get_steps(df, n=1):
        # gaps = df["frame"].values[1:] - df["frame"].values[:-1]
        # not_to_use = gaps != 1
        # print(df["particle"].values[0])

        if len(df) < n:
            steps_x = np.ones(len(df)) * np.nan
            steps_y = np.ones(len(df)) * np.nan
        else:
            steps_x = df["x"].values[n:] - df["x"].values[:-n]
            steps_y = df["y"].values[n:] - df["y"].values[:-n]

            steps_x = np.concatenate((steps_x, np.ones(n) * np.nan))
            steps_y = np.concatenate((steps_y, np.ones(n) * np.nan))

        df[f"steps_x_n={n:d}"] = steps_x
        df[f"steps_y_n={n:d}"] = steps_y

        return df


    tstart = time.perf_counter()
    for ii, (search_range, memory, min_traj_len) in enumerate(itertools.product(search_ranges, memories, min_trajs)):
        id_str = f"roi={ff:d}_sensitivity={ii:d}_linked_{fname.name.strip('.tif'):s}"

        print(
            f"{ii + 1:d}/{len(search_ranges) * len(memories) * len(min_trajs):d} in {time.perf_counter() - tstart:.2f}s")

        linked = tp.link(pd.DataFrame({"frame": params_keep[:, 0],
                                       "amplitude": params_keep[:, 1],
                                       "x": params_keep[:, 2],
                                       "y": params_keep[:, 3],
                                       "z": params_keep[:, 4],
                                       "sxy": params_keep[:, 5],
                                       "sz": params_keep[:, 6],
                                       "bg": params_keep[:, 7]}),
                         search_range=search_range,
                         memory=memory,
                         link_strategy="drop",
                         # adaptive_step=0.9,
                         # adaptive_stop=0.5,
                         pos_columns=["y", "x"]
                         )

        # filter stubs first because afterwards will insert missing frames
        linked = tp.filter_stubs(linked, min_traj_len)

        # compute and fit MSD
        print("fitting MSD")
        tstart_msd = time.perf_counter()


        def line(x, p):
            return p[0] + p[1] * x


        if not linked.empty:
            msd = tp.emsd(linked, mpp=1.0, fps=1 / dt, max_lagtime=max_msd_points, pos_columns=["x", "y"])

            results_emsd = least_squares(lambda p: line(msd.index[:nt_to_fit], p) - msd.values[:nt_to_fit],
                                         np.array([0, 1]))
            emsd_fp = results_emsd["x"]

        else:
            msd = pd.DataFrame()
            emsd_fp = np.array([np.nan, np.nan])

        print(f'calculated ensemble MSD in {time.perf_counter() - tstart_msd:.2f}s')

        if not linked.empty:
            traj_lens = np.array([len(x["frame"]) for _, x in linked.reset_index(drop=True).groupby("particle")])

            lens_bin_edges = np.arange(np.round(np.percentile(traj_lens, 99)))
            lens_bin_centers = 0.5 * (lens_bin_edges[1:] + lens_bin_edges[:-1])
            lens_hist, _ = np.histogram(traj_lens, lens_bin_edges)
        else:
            traj_lens = []
            lens_bin_centers = np.array([np.nan])
            lens_hist = np.array([np.nan])

        track_arr = np.stack((linked["particle"].values,
                              linked["frame"].values,
                              linked["y"].values,
                              linked["x"].values), axis=1)

        # ##############################
        # compute and fit histogram of step-sizes
        # ##############################
        # add nan values for missed frames
        linked = linked.groupby("particle", group_keys=False).apply(insert_missing_frames)
        # sort for easy of printing
        linked = linked.sort_values(by=["particle", "frame"])
        # compute steps
        for n in range(1, nt_to_fit + 1):
            linked = linked.groupby("particle").apply(get_steps, n)

        nbins = 301
        step_bin_edges = np.linspace(-5, 5, nbins + 1)
        step_bin_centers = 0.5 * (step_bin_edges[1:] + step_bin_edges[:-1])

        step_fps_x = np.zeros((nt_to_fit, 2))
        step_fps_y = np.zeros((nt_to_fit, 2))
        xsteps_hists = np.zeros((nt_to_fit, nbins)) * np.nan
        ysteps_hists = np.zeros((nt_to_fit, nbins)) * np.nan
        steps_fps_msdx = np.zeros(2) * np.nan
        steps_fps_msdy = np.zeros(2) * np.nan
        if not linked.empty:

            for jj in range(1, nt_to_fit + 1):
                xsteps_hists[jj - 1], _ = np.histogram(linked[f"steps_x_n={jj:d}"], step_bin_edges)
                ysteps_hists[jj - 1], _ = np.histogram(linked[f"steps_y_n={jj:d}"], step_bin_edges)


                def gauss(x, p):
                    return p[0] * np.exp(-x ** 2 / (2 * p[1] ** 2))  # + p[2]


                initpx = [np.max(xsteps_hists[jj - 1]), np.std(xsteps_hists[jj - 1])]
                results_x = least_squares(lambda p: gauss(step_bin_centers, p) - xsteps_hists[jj - 1], initpx)
                step_fps_x[jj - 1] = results_x["x"]

                initpy = [np.max(ysteps_hists[jj - 1]), np.std(ysteps_hists[jj - 1])]
                results_y = least_squares(lambda p: gauss(step_bin_centers, p) - ysteps_hists[jj - 1], initpy)
                step_fps_y[jj - 1] = results_y["x"]

                if saving:
                    figh = plt.figure()
                    ax = figh.add_subplot(1, 1, 1)
                    ax.plot(step_bin_centers, xsteps_hists[jj - 1], 'r.', label="x")
                    ax.plot(step_bin_centers, gauss(step_bin_centers, step_fps_x[jj - 1]), 'r')
                    ax.plot(step_bin_centers, ysteps_hists[jj - 1], 'b.', label="y")
                    ax.plot(step_bin_centers, gauss(step_bin_centers, step_fps_y[jj - 1]), 'b')
                    ax.set_xlabel("position (um)")
                    ax.set_ylabel("histogram")
                    ax.set_title("step size histogram")
                    ax.legend()

                    figh.savefig(save_dir / f"{id_str:s}_steps={jj:d}.png")
                    plt.close(figh)

            results_stepsx_msd = least_squares(lambda p: line(msd.index[:nt_to_fit], p) - step_fps_x[:, 1] ** 2,
                                               np.array([0, 1]))
            steps_fps_msdx[:] = results_stepsx_msd["x"]

            results_stepsy_msd = least_squares(lambda p: line(msd.index[:nt_to_fit], p) - step_fps_y[:, 1] ** 2,
                                               np.array([0, 1]))
            steps_fps_msdy[:] = results_stepsy_msd["x"]

        # ##############################
        # save results
        # ##############################

        if saving:
            save_fname = save_dir / f"{id_str:s}.hdf5"
            linked.to_hdf(save_fname, "linked")
            msd.to_hdf(save_fname, "msd")

            with h5py.File(save_fname, "r+") as f:
                # todo: needlessly saving localization info repeatedly
                # localization data
                f["init params"] = init_params
                f["fit params"] = params_all
                f.attrs["param_names"] = param_names

                f["roi_inds"] = rois_inds_all

                f["conditions"] = conditions_all_arr
                f["to keep"] = to_keep_all_arr
                f.attrs["condition_names"] = condition_names

                f.attrs["threshold"] = threshold

                # step size data
                f["xsteps_hists"] = xsteps_hists
                f["ysteps_hists"] = ysteps_hists
                f["xsteps_hists_fps"] = step_fps_x
                f["ysteps_hists_fps"] = step_fps_y

                f.attrs["dt"] = dt
                f.attrs["dxy"] = dxy
                f.attrs["na"] = na
                f.attrs['wavelength'] = wavelength
                f.attrs['n_msd_points_to_fit'] = nt_to_fit
                f.attrs["search_radius_um"] = search_range
                f.attrs["memory"] = memory
                f.attrs["min_trajectory_length"] = min_traj_len
                f.attrs["diffusion_coeff_msd"] = emsd_fp[1] / 4
                f.attrs["diffusion_coeff_units"] = "um^2/s"
                f.attrs["viscosity_msd"] = kb * T / (6 * np.pi * emsd_fp[1] / 4 * 1e-6 ** 2 * R * 1e-6)
                f.attrs["viscosity_units"] = "mPa*s"
                # f.attrs["diffusion_coeff_steps"] = 0.5 * (step_fp_x[1]**2 / 2 / dt + step_fp_y[1]**2 / 2 / dt)
                f.attrs["diffusion_coeff_steps"] = 0.5 * (steps_fps_msdx[1] / 2 + steps_fps_msdy[1] / 2)
                f.attrs["viscosity_steps"] = kb * T / (
                            6 * np.pi * f.attrs["diffusion_coeff_steps"] / 4 * 1e-6 ** 2 * R * 1e-6)

        # ##############################
        # plot diffusion coefficient data
        # ##############################
        t_interp = np.linspace(0, nt_to_fit * dt, 1000)

        figh = plt.figure(figsize=(25, 10))
        figh.suptitle(f"{str(fname.name):s},"
                      f"search radius = {search_range:.2f}um,"
                      f" memory={memory:d},"
                      f" min traj len={min_traj_len:d},"
                      f" msd pts fit={nt_to_fit:d}")

        # plot MSD fits
        ax = figh.add_subplot(1, 5, 1)
        ax.set_title("ensemble MSD")
        ax.plot(msd.index, msd.values, 'r.', label="emsd")
        ax.plot(t_interp, line(t_interp, emsd_fp), 'r',
                label=f"emsd fit D={emsd_fp[1] / 4:.2f}+/-{np.nan:.2f} um^2/s,"
                      f" eta={kb * T / (6 * np.pi * emsd_fp[1] / 4 * 1e-6 ** 2 * R * 1e-6) * 1e3:.3f}mPa*s")

        ax.set_xlabel("lag time (s)")
        ax.set_ylabel("MSD (um^2)")
        ax.legend()

        # step sizes
        ax = figh.add_subplot(1, 5, 2)

        ax.axvline(search_range, color="r")
        ax.axvline(-search_range, color="r", label="search range")

        ax.semilogy(step_bin_centers, xsteps_hists[0] / np.sum(xsteps_hists[0]), label="x trackpy")
        ax.semilogy(step_bin_centers, ysteps_hists[0] / np.sum(ysteps_hists[0]), label="y trackpy")

        ax.set_xlim(-1.2 * search_range, 1.2 * search_range)
        ax.set_ylabel("fraction of steps")
        ax.set_xlabel("step size (um)")

        ax.legend()

        ax = figh.add_subplot(1, 5, 3)
        ax.set_title(
            f"step size\nDx={step_fps_x[0, 1] ** 2 / 2 / dt:.2f}um^2/s; Dy={step_fps_y[0, 1] ** 2 / 2 / dt:.2f}um^2/s")
        ax.axvline(search_range, color="r")
        ax.axvline(-search_range, color="r", label="search range")

        ax.plot(step_bin_centers, xsteps_hists[0] / np.sum(xsteps_hists[0]), 'r.', label="x trackpy")
        ax.plot(step_bin_centers, gauss(step_bin_centers, step_fps_x[0]) / np.sum(xsteps_hists[0]), 'r')

        ax.plot(step_bin_centers, ysteps_hists[0] / np.sum(ysteps_hists[0]), 'b.', label="y trackpy")
        ax.plot(step_bin_centers, gauss(step_bin_centers, step_fps_y[0]) / np.sum(ysteps_hists[0]), 'b')

        ax.set_xlim(-1.2 * search_range, 1.2 * search_range)
        ax.set_ylabel("fraction of steps")
        ax.set_xlabel("step size (um)")

        ax = figh.add_subplot(1, 5, 4)
        ax.set_title("step size $\sigma^2(t)$")
        ax.plot(msd.index[:len(step_fps_x)], step_fps_x[:, 1] ** 2, 'r.',
                label=f"Dx = {steps_fps_msdx[1] / 2:.2f}um^2/s")
        ax.plot(t_interp, line(t_interp, steps_fps_msdx), 'r')
        ax.plot(msd.index[:len(step_fps_y)], step_fps_y[:, 1] ** 2, 'b.',
                label=f"Dy = {steps_fps_msdy[1] / 2:.2f}um^2/s")
        ax.plot(t_interp, line(t_interp, steps_fps_msdy), 'b')
        ax.set_xlabel('lag time (s)')
        ax.set_ylabel('$\sigma^2(t)$ (um^2)')
        ax.legend()

        # plot trajectory lengths
        ax = figh.add_subplot(1, 5, 5)
        ax.set_title("trajectory length  histogram")

        ax.plot(lens_bin_centers, lens_hist, '.')
        ax.set_xlabel("trajectory length")
        ax.set_ylabel("number of trajectories")

        # plt.show(block=True)
        if saving:
            figh_fname = save_dir / f"{id_str:s}.png"
            figh.savefig(figh_fname)
            plt.close(figh)

    # ##############################
    # plot results
    # ##############################
    if show_napari:
        viewer = napari.Viewer()
        viewer.add_image(imgs_filtered, name="img filtered", scale=(dxy, dxy),
                         contrast_limits=[0, threshold], colormap="blue")
        viewer.add_image(img, name="imgs", scale=(dxy, dxy),
                         contrast_limits=[np.percentile(img, 5), np.percentile(img, 99.9)])

        viewer.add_points(centers_keep,
                          size=10 * dxy,
                          edge_width=dxy,
                          edge_color=[1, 0, 0, 1],
                          face_color=[0, 0, 0, 0],
                          name="kept",
                          features={"roi_ind": roi_inds_kept},
                          text={'string': '{roi_ind}',
                                'size': 15,
                                'color': 'red',
                                'translation': np.array([1, 1]),
                                },
                          )

        strs = ["\n".join([condition_names[aa] for aa, c in enumerate(cs) if not c])
                for ii, cs in enumerate(conditions_all_arr) if not to_keep_all_arr[ii]]

        viewer.add_points(centers_not,
                          symbol="disc",
                          name="rejected",
                          out_of_slice_display=False,
                          opacity=1,
                          face_color=[0, 0, 0, 0],
                          edge_color=[0, 1, 0, 1],
                          size=10 * dxy,
                          features={"rejection_reason": strs,
                                    "roi_ind": roi_inds_rejected},
                          text={'string': '{roi_ind}: {rejection_reason}',
                                'size': 15,
                                'color': 'green',
                                'translation': np.array([0, 0]),
                                },
                          visible=False)
        #
        # viewer.add_tracks(track_arr,
        #                   name="tracks",
        #                   tail_length=100,
        #                   head_length=0,
        #                   blending="opaque",
        #                   colormap="hsv")

        viewer.show(block=True)