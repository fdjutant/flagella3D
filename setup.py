from setuptools import setup, find_packages

required_pkgs = ['numpy (>=1.23.1)',
                 'scipy (>=1.9.0)',
                 'pandas (>=1.4.3)',
                 'scikit-image (>=0.19.3)',
                 'scikit-learn',
                 'numba (>=0.55.2)',
                 'napari (>=0.4.16)',
                 'matplotlib (>=3.5.2)',
                 'seaborn (>=0.11.2)',
                 'tifffile (>=2022.8.8)',
                 'zarr (>=2.12.0)',
                 'h5py (>=3.7.0)',
                 'trackpy (>=0.5.0)',
                 'opencv-python',
                 'lmfit',
                 'localize_psf @ git+https://git@github.com/qi2lab/localize-psf@master#egg=localize_psf'
                 ]

# requirements
setup(
    name='flagella3D',
    version='1.0.0',
    description="Code for analyzing diffusing flagella data",
    author='Hariadi and Shepherd labs',
    author_email='',
    packages=find_packages(include=['modules', 'modules.*']),
    python_requires='>=3.9',
    install_requires=required_pkgs)
