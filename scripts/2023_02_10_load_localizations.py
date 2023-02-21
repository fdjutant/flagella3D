"""
Load bead localization results
"""
from pathlib import Path
import numpy as np
import napari
import tifffile
import pandas as pd
import h5py

fname_img = Path(r"\\10.206.26.21\flagella_project\Tetraspek_100nm_sucrose\20220516_Franky_TS_diffusion_HILO_40sucrose\Franky_TS_diffusion_HILO_40sucrose_dilution_100x-1.tif")
# fname_data = Path(r"\\10.206.26.21\flagella_project\Tetraspek_100nm_sucrose\20220516_Franky_TS_diffusion_HILO_40sucrose\2023_02_16_11;30;48_diffusion_coeff\roi=3_sensitivity=75_linked_Franky_TS_diffusion_HILO_40sucrose_dilution_100x-4.hdf5")
fname_data = Path(r"\\10.206.26.21\flagella_project\Tetraspek_100nm_sucrose\20220516_Franky_TS_diffusion_HILO_40sucrose\2023_02_16_16;42;41_diffusion_coeff\roi=0_sensitivity=0_linked_Franky_TS_diffusion_HILO_40sucrose_dilution_100x-1.hdf5")
fname_oni_data = Path(r"\\10.206.26.21\flagella_project\Tetraspek_100nm_sucrose\20220516_Franky_TS_diffusion_HILO_40sucrose\Track_TS_40suc_200nmExc_001.csv")


img = tifffile.imread(fname_img)

# ONI data
data_oni = pd.read_csv(fname_oni_data)

centers_oni = np.stack((data_oni["Frame"].values - 1,
                        data_oni["Y (nm)"].values / 1000,
                        data_oni[" X (nm)"].values / 1000),
                        axis=1)

track_arr_oni = np.concatenate((data_oni["Track ID"].values[:, None], centers_oni), axis=1)

# my localizations
data = h5py.File(fname_data, "r")
dxy = data.attrs["dxy"]
dt = data.attrs["dt"]

# parameters
init_params = np.array(data["init params"])
fit_params = np.array(data["fit params"])

# conditions
conditions_all_arr = np.array(data["conditions"])
to_keep_all_arr = np.logical_and.reduce(conditions_all_arr, axis=1)
condition_names = data.attrs["condition_names"]

# centers
centers_keep = fit_params[to_keep_all_arr][:, (0, 3, 2)]
centers_not = init_params[np.logical_not(to_keep_all_arr)][:, (0, 3, 2)]
params_keep = fit_params[to_keep_all_arr]


linked = pd.read_hdf(fname_data, "linked")
track_arr = np.stack((linked["particle"].values,
                      linked["frame"].values,
                      linked["y"].values,
                      linked["x"].values), axis=1)


condition_names = ['x-position too small',
 'x-position too large',
 'y-position too small',
 'y-position too large',
 'z-position too small',
 'z-position too large',
 'x-size too small',
 'x-size too large',
 'y-size too small',
 'y-size too large',
 'z-size too small',
 'z-size too large',
 'amplitude too small',
 'amplitude too large',
 'xy deviation too small',
 'xy deviation too large',
 'z deviation too small',
 'z deviation too large',
 'sx/sy too small',
 'sx/sy too large',
 'not unique']


viewer = napari.Viewer()

viewer.add_image(img, name=str(fname_img.name), scale=(dxy, dxy), contrast_limits=[300, 1500])

# roi_inds_kept = np.concatenate([r[tk] for r, tk in zip(rois_inds_all, to_keep_all)])
viewer.add_points(centers_keep,
                  size=10*dxy,
                  edge_width=dxy,
                  edge_color=[1, 0, 0, 1],
                  face_color=[0, 0, 0, 0],
                  name="kept",
                  # features={"roi_ind": roi_inds_kept},
                  # text={'string': '{roi_ind}',
                  #       'size': 15,
                  #       'color': 'red',
                  #       'translation': np.array([1, 1]),
                  #       },
                  )

viewer.add_points(centers_keep,
                  size=10*dxy,
                  edge_width=dxy,
                  edge_color=[1, 0, 0, 1],
                  face_color=[0, 0, 0, 0],
                  name="kept with params",
                  features={"amp": params_keep[:, 1],
                            "sx": params_keep[:, 5],
                            "sy": params_keep[:, 6],
                            "ratio": params_keep[:, 5] / params_keep[:, 6],
                            "bg": params_keep[:, -1]},
                  text={'string': 'amp={amp:.2f}\nsx={sx:.2f}\nsy={sy:.2f}\nsx/sy={ratio:.2f}\nbg={bg:.2f}',
                        'size': 15,
                        'color': 'red',
                        'translation': np.array([0, 0]),
                        },
                  visible=False
                  )

strs = ["\n".join([condition_names[aa] for aa, c in enumerate(cs) if not c])
        for ii, cs in enumerate(conditions_all_arr) if not to_keep_all_arr[ii]]
# roi_inds_rejected = np.concatenate([r[np.logical_not(tk)] for r, tk in zip(rois_inds_all, to_keep_all)])

viewer.add_points(centers_not,
                 symbol="disc",
                 name="rejected",
                 out_of_slice_display=False,
                 opacity=1,
                 face_color=[0, 0, 0, 0],
                 edge_color=[0, 1, 0, 1],
                 size=10*dxy,
                 features={"rejection_reason": strs,
                           # "roi_ind": roi_inds_rejected
                           },
                     text={'string': '{rejection_reason}',
                           'size': 15,
                           'color': 'green',
                           'translation': np.array([0, 0]),
                           },
                     visible=False)
#
viewer.add_tracks(track_arr,
                  name="tracks",
                  tail_length=10,
                  head_length=0,
                  blending="opaque",
                  colormap="bop orange")


viewer.add_points(centers_oni,
                  symbol="disc",
                  name="ONI",
                  out_of_slice_display=False,
                  opacity=1,
                  face_color=[0, 0, 0, 0],
                  edge_color=[0, 0, 1, 1],
                  size=10 * dxy,
                  )

viewer.add_tracks(track_arr_oni,
                  name="tracks ONI",
                  tail_length=10,
                  head_length=0,
                  blending="opaque",
                  colormap="bop blue")

viewer.show(block=True)
