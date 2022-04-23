#%% Import all necessary libraries
import sys
sys.path.insert(0, './modules')
import pickle
import os.path
from pathlib import Path
import pandas as pd
import numpy as np

# Loading pickle files
this_file_dir = os.path.join(os.path.dirname(os.path.abspath("./")),
                            'Dropbox (ASU)','Research')
thresholdFolder = os.path.join(this_file_dir,
                          'DNA-Rotary-Motor', 'Helical-nanotubes',
                          'Light-sheet-OPM', 'Result-data',
                          'Flagella-data', 'PKL-files')

pickle_files = list(Path(thresholdFolder).glob('*.pkl'))
total_file_number = len(pickle_files)

data_name = []
flagella_length_mean = np.zeros(total_file_number)
flagella_length_std = np.zeros(total_file_number)
number_of_frames = np.zeros(total_file_number)
D_trans = np.zeros( (total_file_number,3) )
D_rot = np.zeros( (total_file_number,3) )
D_co = np.zeros(total_file_number)
A_per_vis = np.zeros(total_file_number)
B_per_vis = np.zeros(total_file_number)
D_per_vis = np.zeros(total_file_number)

# Compute A, B, D
kB = 1.380649e-23  # J / K
T = 273 + 25       # K

# from bead measurement
vis70 = 2.84e-3
vis50 = 1.99e-3
vis40 = 1.77e-3

for index_files in range(total_file_number):
    
    print(index_files, pickle_files[index_files].name)
    
    with open(pickle_files[index_files], "rb") as f:
        individual_data = pickle.load(f)
    data_name.append(individual_data["data_name"])
    flagella_length_mean[index_files] = np.mean(individual_data["flagella_length"])
    flagella_length_std[index_files] = np.std(individual_data["flagella_length"])
    number_of_frames[index_files] = len(individual_data["flagella_length"])
    exp3D_sec = individual_data["exp3D_sec"]
    pxum = individual_data["pxum"]
    # cm = individual_data["cm"]
    # disp = individual_data["disp"]
    # disp_Ang = individual_data["disp_Ang"]
    # MSD = individual_data["MSD"]
    # MSAD = individual_data["MSAD"]
    # CO_MSD = individual_data["CO_MSD"]
    D_trans[index_files] = individual_data["D_trans"]
    D_rot[index_files] = individual_data["D_rot"]
    D_co[index_files] = individual_data["D_co"]
    
    if individual_data["data_name"][:5] == 'suc40':
        vis = vis40
    elif individual_data["data_name"][:5] == 'suc50':
        vis = vis50
    else:
        vis = vis70

    D_n1 = D_trans[index_files,0] * 1e-12
    D_n1_psi = D_co[index_files] * 1e-6
    D_psi = D_rot[index_files,0]

    A_per_vis[index_files] = D_psi * kB * T / (D_n1 * D_psi - D_n1_psi**2) / vis
    B_per_vis[index_files] = -D_n1_psi * kB * T / (D_n1 * D_psi - D_n1_psi**2) / vis
    D_per_vis[index_files] = D_n1 * kB * T / (D_n1 * D_psi - D_n1_psi**2) / vis
    
#%% create excel using pandas    
summary_data = {'data name': data_name,
                'number of frames': number_of_frames,
                'mean length [um]': np.round(flagella_length_mean * pxum, 2),
                'std length [um]': np.round(flagella_length_std * pxum, 2),
                'D_n1 [um^2/sec]': np.round(D_trans[:,0],4),
                'D_n2 [um^2/sec]': np.round(D_trans[:,1],4),
                'D_n3 [um^2/sec]': np.round(D_trans[:,2],4),
                'D_psi [rad^2/sec]': np.round(D_rot[:,0],4),
                'D_gamma [rad^2/sec]': np.round(D_rot[:,1],4),
                'D_beta [rad^2/sec]': np.round(D_rot[:,2],4),
                'D_n1_psi [um x rad/sec]': np.round(D_co,4),
                'A/eta [m]': A_per_vis,
                'B/eta [m^2]': B_per_vis,
                'D/eta [m^3]': D_per_vis
                }

df = pd.DataFrame(summary_data,
                  columns = ['data name', 'number of frames',
                             'mean length [um]', 'std length [um]',
                             'D_n1 [um^2/sec]','D_n2 [um^2/sec]',
                             'D_n3 [um^2/sec]', 'D_psi [rad^2/sec]',
                             'D_gamma [rad^2/sec]', 'D_beta [rad^2/sec]',
                             'D_n1_psi [um x rad/sec]',
                             'A/eta [m]', 'B/eta [m^2]',
                             'D/eta [m^3]'
                             ])
print(df)
excel_path = os.path.join(thresholdFolder, 'final-data-thesis.xlsx')
df.to_excel(excel_path, index = True, header = True)  
