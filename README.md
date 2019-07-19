This repo contains scripts that work through the examples in Chapter 2 of the [Hands On Machine Learning](https://github.com/ageron/handson-ml) book.

## Scripts

- `fit_models.py` -- fits various regression models against the data and compares the results
- `visualize_data.py` -- generates data visualizations

## Initial project setup

1. If necessary, install [conda or miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Clone the repo
3. From a terminal shell, run `conda env create --file environment.yml`, then `conda activate hands-on-ml-housing`
4. Create a `data` folder and download [the housing data file](https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz) and extract it to the `data` folder.
5. You should now be able to run the scripts, for example by, `python visualize_data.py`

## Updating dependencies

1. To update your local dependencies to match the versions in the `environment.yml` file, run `conda env update --file environment.yml`

## Installing new dependencies

1. Install new package(s) by running `conda install <package_1> <package_2>`.
2. To update outdated packages, run `conda update --all`, which will show a list of packages that will be updated. Reply `N` to just see the list or `Y` to install the new packages.
3. After updating packages, run `conda env export > environment.yml` to update the `environment.yml` file; commit the changes.