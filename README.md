
# SARRA data download
This repo is a collection of tools to download and prepare climate and weather files necessary for SARRA-O runs. So far, it can be used to retrieve AgERA5 daily data from the Copernicus Climate Data Store.
## How to install
You will need Python 3.9.6 or above.
First, [setup a Copernicus Climate Data Store API key](https://cds.climate.copernicus.eu/api-how-to) and accept [Copernicus Terms of Service](https://cds.climate.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products).
Then, clone this repo and install its dependencies :

    git clone https://github.com/SARRA-cropmodels/SARRA_data-download
    pip install -r requirements.txt
## How to use
**AgERA5 data download**

    cd SARRA_data-download
    python get_AgERA5_data.py

The previous command triggers the download of all the data of the month of the last available date in AgERA5, every day at 12:00 PM. 
Download will be performed by default on an extent covering all West Africa. Output files will be daily geotiffs, as required to run SARRA-O.

For example, if the current date is 2022-09-15, the last available date in AgERA5 will be 2022-09-07, thus, the script will download all data from 2022-09-01 to 2022-09-07.

The downloaded and prepared data will be stored in the `./data/3_output/` path.