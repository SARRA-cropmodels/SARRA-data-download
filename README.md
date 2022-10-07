# SARRA data download
This repo is a collection of tools to download and prepare climate and weather files necessary for SARRA-O runs. So far, it can be used to retrieve AgERA5 daily data  from the Copernicus Climate Data Store.

## How to use
**Installation**
You will need Python 3.9.6 or above.
First, [setup a Copernicus Climate Data Store API key](https://cds.climate.copernicus.eu/api-how-to) and accept [Copernicus Terms of Service](https://cds.climate.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products).

Then, clone this repo and install its dependencies :

    git clone https://github.com/SARRA-cropmodels/SARRA_data-download
    pip install -r requirements.txt
**AgERA5 data download**

    cd SARRA_data-download
    python get_AgERA5_data.py
