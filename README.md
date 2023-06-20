[![paper](https://www.pnas.org/pb-assets/images/Logos/header-logo/logo-1624644560537.svg)](https://www.pnas.org/doi/10.1073/pnas.2220033120)
[![raw data](https://img.shields.io/badge/raw%20data-zenodo-blue.svg)](https://doi.org/10.5281/zenodo.6562089)
[![Github commit](https://img.shields.io/github/last-commit/QI2lab/mcSIM)](https://github.com/fdjutant/flagella3D)

# flagella3D
Code for analyzing helical flagellum diffusing in 3D acquired using the OPM described [here](https://github.com/QI2lab/OPM)

See also [https://github.com/fdjutant/6-DOF-Flagella](https://github.com/fdjutant/6-DOF-Flagella)

## [scripts](scripts)
contains a record of data analysis and other calculations. 

#### [2022_04_19_diffusion_analysis.py](2022_04_19_diffusion_analysis.py)
Old data analysis script

#### [2022_06_15_diffusion_analysis.py](scripts/2022_06_15_diffusion_analysis.py)
The primary data analysis script

#### [2023_02_13_plots_for_figures_3_and_4.py](figure_scripts/2023_02_13_plots_for_figures_3_and_4.py)
Extract data for figures 3, 4, and S6, as well as our experimental non-dimensionalized propulsion matrices for the table in Fig. 4d

### [2022_04_27_syn-diffusion.py](2022_04_27_syn-diffusion.py)
Simulation used in figure S5

### Comparing non-dimensionalized propulsion matrix values to previous work 
See [2022_07_04_compute_chattopadhyay_abd.py](scripts/2022_07_04_compute_chattopadhyay_abd.py),
[2022_07_04_rodenborn_abd_results.py](scripts/2022_07_04_rodenborn_abd_results.py), and
[2022_07_04_compute_purcell_abd_table1.py](scripts/2022_07_04_compute_purcell_abd_table1.py)

## [modules](modules)
contains code for computing mean-squared displacement (MSD) values

## [moviecommands](moviecommands)
scripts for working with [naparimovie](https://github.com/guiwitz/naparimovie)

# running
To run this code, first create a [conda](https://conda.io/projects/conda/en/latest/user-guide/) 
environment with python 3.9, activate your environment, then navigate to this directory and run
```commandline
pip install -e .
```
which will run the setup file and install all required packages to your current environment