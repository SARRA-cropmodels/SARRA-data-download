
# SARRA data download

This repo is a collection of tools to download and prepare climate and weather files necessary for the use of [SARRA-O](https://gitlab.cirad.fr/sarrao/model/sarrao) and [SARRA-Py](https://github.com/SARRA-cropmodels/SARRA-Py) spatialized crop simulation models. Its rationale is to allow for easy download of time series datasets from different data providers for Africa.



## 1. How to install

You will need Python 3.9.6 or above.

Then, clone this repo and install its dependencies :

    git clone https://github.com/SARRA-cropmodels/SARRA_data-download
    cd SARRA_data-download
    pip install -r requirements.txt



## 2. How to use

### 2.1. Get satellite rainfall estimates

The `./notebooks/get_satellite_rainfall_estimated.ipynb` notebook allows for downloading of daily rainfall estimates product from different providers, and preprocessing for their direct use in crop simulation models of the SARRA family. The output format is then a series of geotiff files, one per day, with the same spatial resolution as the input data. 

Rainfall estimates products available :
- [TAMSAT](https://www.tamsat.org.uk/), resolution 0.037°
- [CHIRPS](https://www.chc.ucsb.edu/data/chirps), resolution 0.05°
- [IMERG](https://gpm.nasa.gov/data/imerg), resolution 0.1°

Note : before downloding IMERG data, you must have a NASA Earthdata account. Check [this document](https://gpm.nasa.gov/sites/default/files/2021-01/arthurhouhttps_retrieval.pdf) for more information about how to create an account and get your credentials.

To use, open the `./notebooks/get_satellite_rainfall_estimated.ipynb` notebook in Jupyter, JupyterLab or VSCode, modify the parameters in the appropriate section and run the cells. The output files will be stored in the `./data/3_output/` directory.



### 2.2. AgERA5 data download

The `./notebooks/get_AgERA5_data.ipynb` notebook allows for downloading of daily AgERA5 climate data from the [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/#!/home), and preprocessing for their direct use in crop simulation models of the SARRA family. The output format is a series of geotiff files, one per variable and per day.

The cdsapi package is used to download data from CDS. This has the advantage of magaging the caching of already passed requests, thus to speed up the downloading process.

AgERA5 data should be produced daily, with a 7-day lag, according to the [AgERA-5 frequency update](https://confluence.ecmwf.int/display/CUSF/AgERA-5+frequency+update) documentation.

Before running the notebook for the first time, [follow these instructions](https://cds.climate.copernicus.eu/api-how-to) to setup a Copernicus Climate Data Store API key on your machine, and accept [Copernicus Terms of Service](https://cds.climate.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products).

To use, open the `./notebooks/get_AgERA5_data.ipynb` notebook in Jupyter, JupyterLab or VSCode, modify the parameters in the appropriate section and run the cells. The output files will be stored in the `./data/3_output/` directory.