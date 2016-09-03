# tropicalstorm-ml-analysis
Dataset generation and machine learning analysis of the IBTrACS-WMO tropical storm dataset

## Installation

1. Install anaconda3 (https://www.continuum.io/downloads).
1. Install requisite packages:

  ```
  conda install netCDF4
  conda install gdal
  conda install -c conda-forge ipyleaflet
  pip install liac-arff
  ```

1. Add the following line to your .bash_profile and source it:

  ```
  export GDAL_DATA=$(gdal-config --datadir)
  ```  

1. Optional: install jupyter-vim-binding

  ```
  conda install -c conda-forge jupyter_contrib_nbextensions
  jupyter-contrib nbextension install --user
  mkdir -p $(jupyter --data-dir)/nbextensions
  cd $(jupyter --data-dir)/nbextensions
  git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding
  chmod -R go-w vim_binding  
  ```

  The vim_binding extension can be enabled after you start up a notebook
  and clicking on the checkbox for "VIM binding" on the "Nbextensions" tab.

1. Run jupyter:

  ```
  jupyter notebook
  ```
