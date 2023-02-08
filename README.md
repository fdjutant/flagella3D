[![preprint](https://img.shields.io/badge/preprint-bioRxiv-blue.svg)]()
[![paper](https://img.shields.io/badge/paper-journal%20name-blue.svg)]()
[![raw data](https://img.shields.io/badge/raw%20data-zenodo-blue.svg)](https://doi.org/10.5281/zenodo.6562089)
[![Github commit](https://img.shields.io/github/last-commit/QI2lab/mcSIM)](https://github.com/fdjutant/flagella3D)

# flagella3D
Code for analyzing helical flagellum diffusing in 3D acquired using the the OPM described [here](https://github.com/QI2lab/OPM)

See also [https://github.com/fdjutant/6-DOF-Flagella](https://github.com/fdjutant/6-DOF-Flagella)

## [scripts](scripts)
contains a record of data analysis and other calculations. 

#### [2022_04_19_diffusion_analysis.py](2022_04_19_diffusion_analysis.py)
The primary data analysis script

#### [2022_05_19_flagella_rft_calculations.py](2022_05_19_flagella_rft_calculations.py)
Extract non-dimensionalized propulsion matrices for the table in Fig. 4d

#### [2022_04_27_syn-diffusion.py](2022_04_27_syn-diffusion.py)
Simulation used in figure S5

## [modules](modules)
contains code for computing mean-squared displacement (MSD) values

## [moviecommands](moviecommands)
scripts for working with [naparimovie](https://github.com/guiwitz/naparimovie)
