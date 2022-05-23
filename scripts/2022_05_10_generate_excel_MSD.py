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
result_dir_csv = os.path.join(os.path.dirname(
                              os.path.dirname(
                              os.path.abspath("./"))), '6-DOF-Flagella',
                              'Flagella')

# x-axis for MSD/MSAD
nInterval = 50
xaxis = np.arange(1,nInterval+1)

# 70% sucrose
pickle_70 = list(Path(thresholdFolder).glob('suc70*.pkl'))
total_file_number_70 = len(pickle_70)
data_name_70 = []
MSD_n1_70 = []
MSD_n2_70 = []
MSD_n3_70 = []
MSAD_n1_70 = []
MSAD_n2_70 = []
MSAD_n3_70 = []
CO_MSD_70 = []
for index_files in range(total_file_number_70):
    print(index_files, pickle_70[index_files].name)
    with open(pickle_70[index_files], "rb") as f:
        individual_data = pickle.load(f)
    data_name_70.append(individual_data["data_name"])
    exp3D_sec = individual_data["exp3D_sec"]
    MSD_n1_70.append(individual_data["MSD"][:,0])
    MSD_n2_70.append(individual_data["MSD"][:,1])
    MSD_n3_70.append(individual_data["MSD"][:,2])
    MSAD_n1_70.append(individual_data["MSAD"][:,0])
    MSAD_n2_70.append(individual_data["MSAD"][:,1])
    MSAD_n3_70.append(individual_data["MSAD"][:,2])
    CO_MSD_70.append(individual_data["CO_MSD"])
data_name_70 = np.array(data_name_70)    
MSD_n1_70 = np.array(MSD_n1_70)
MSD_n2_70 = np.array(MSD_n2_70)
MSD_n3_70 = np.array(MSD_n3_70)
MSAD_n1_70 = np.array(MSAD_n1_70)
MSAD_n2_70 = np.array(MSAD_n2_70)
MSAD_n3_70 = np.array(MSAD_n3_70)
CO_MSD_70 = np.array(CO_MSD_70)

# 50% sucrose
pickle_50 = list(Path(thresholdFolder).glob('suc50*.pkl'))
total_file_number_50 = len(pickle_50)
data_name_50 = []
MSD_n1_50 = []
MSD_n2_50 = []
MSD_n3_50 = []
MSAD_n1_50 = []
MSAD_n2_50 = []
MSAD_n3_50 = []
CO_MSD_50 = []
for index_files in range(total_file_number_50):
    print(index_files, pickle_50[index_files].name)
    with open(pickle_50[index_files], "rb") as f:
        individual_data = pickle.load(f)
    data_name_50.append(individual_data["data_name"])
    exp3D_sec = individual_data["exp3D_sec"]
    MSD_n1_50.append(individual_data["MSD"][:,0])
    MSD_n2_50.append(individual_data["MSD"][:,1])
    MSD_n3_50.append(individual_data["MSD"][:,2])
    MSAD_n1_50.append(individual_data["MSAD"][:,0])
    MSAD_n2_50.append(individual_data["MSAD"][:,1])
    MSAD_n3_50.append(individual_data["MSAD"][:,2])
    CO_MSD_50.append(individual_data["CO_MSD"])
data_name_50 = np.array(data_name_50)    
MSD_n1_50 = np.array(MSD_n1_50)
MSD_n2_50 = np.array(MSD_n2_50)
MSD_n3_50 = np.array(MSD_n3_50)
MSAD_n1_50 = np.array(MSAD_n1_50)
MSAD_n2_50 = np.array(MSAD_n2_50)
MSAD_n3_50 = np.array(MSAD_n3_50) 
CO_MSD_50 = np.array(CO_MSD_50)

# 40% sucrose
pickle_40 = list(Path(thresholdFolder).glob('suc40*.pkl'))
total_file_number_40 = len(pickle_40)
data_name_40 = []
MSD_n1_40 = []
MSD_n2_40 = []
MSD_n3_40 = []
MSAD_n1_40 = []
MSAD_n2_40 = []
MSAD_n3_40 = []
CO_MSD_40 = []
for index_files in range(total_file_number_40):
    print(index_files, pickle_40[index_files].name)
    with open(pickle_40[index_files], "rb") as f:
        individual_data = pickle.load(f)
    data_name_40.append(individual_data["data_name"])
    exp3D_sec = individual_data["exp3D_sec"]
    MSD_n1_40.append(individual_data["MSD"][:,0])
    MSD_n2_40.append(individual_data["MSD"][:,1])
    MSD_n3_40.append(individual_data["MSD"][:,2])
    MSAD_n1_40.append(individual_data["MSAD"][:,0])
    MSAD_n2_40.append(individual_data["MSAD"][:,1])
    MSAD_n3_40.append(individual_data["MSAD"][:,2])
    CO_MSD_40.append(individual_data["CO_MSD"])
data_name_40 = np.array(data_name_40)    
MSD_n1_40 = np.array(MSD_n1_40)
MSD_n2_40 = np.array(MSD_n2_40)
MSD_n3_40 = np.array(MSD_n3_40)
MSAD_n1_40 = np.array(MSAD_n1_40)
MSAD_n2_40 = np.array(MSAD_n2_40)
MSAD_n3_40 = np.array(MSAD_n3_40) 
CO_MSD_40 = np.array(CO_MSD_40)

#%% Write to CSV
for i in range(len(data_name_70)):
    MSD_70suc = np.array([xaxis*exp3D_sec, MSD_n1_70[i],
                          0.5 * (MSD_n2_70[i] + MSD_n3_70[i]),
                          MSAD_n1_70[i],
                          0.5 * (MSAD_n2_70[i] + MSD_n3_70[i]),
                          CO_MSD_70[i]]).T
    fmt = ",".join(["%s"] + ["%10.6e"] * (MSD_70suc.shape[1]-1))
    np.savetxt(result_dir_csv + "/MSD/MSD-70suc-" + str(i).zfill(2) + ".csv",
               MSD_70suc, fmt=fmt,
               header="lag time [sec], MSD-longitudinal [um^2]," +
                      "MSD-transversal [um^2],"+
                      "MSAD-longitudinal [rad^2], MSAD-transversal [rad^2]," +
                      "MLAD [um x rad]", comments='')

for i in range(len(data_name_50)):
    MSD_50suc = np.array([xaxis*exp3D_sec, MSD_n1_50[i],
                          0.5 * (MSD_n2_50[i] + MSD_n3_50[i]),
                          MSAD_n1_50[i],
                          0.5 * (MSAD_n2_50[i] + MSD_n3_50[i]),
                          CO_MSD_50[i]]).T
    fmt = ",".join(["%s"] + ["%10.6e"] * (MSD_50suc.shape[1]-1))
    np.savetxt(result_dir_csv + "/MSD/MSD-50suc-" + str(i).zfill(2) + ".csv",
               MSD_50suc, fmt=fmt,
               header="lag time [sec], MSD-longitudinal [um^2]," +
                      "MSD-transversal [um^2],"+
                      "MSAD-longitudinal [rad^2], MSAD-transversal [rad^2]," +
                      "MLAD [um x rad]", comments='')
    
for i in range(len(data_name_40)):
    MSD_40suc = np.array([xaxis*exp3D_sec, MSD_n1_40[i],
                          0.5 * (MSD_n2_40[i] + MSD_n3_40[i]),
                          MSAD_n1_40[i],
                          0.5 * (MSAD_n2_40[i] + MSD_n3_40[i]),
                          CO_MSD_40[i]]).T
    fmt = ",".join(["%s"] + ["%10.6e"] * (MSD_40suc.shape[1]-1))
    np.savetxt(result_dir_csv + "/MSD/MSD-40suc-" + str(i).zfill(2) + ".csv",
               MSD_40suc, fmt=fmt,
               header="lag time [sec], MSD-longitudinal [um^2]," +
                      "MSD-transversal [um^2],"+
                      "MSAD-longitudinal [rad^2], MSAD-transversal [rad^2]," +
                      "MLAD [um x rad]", comments='')