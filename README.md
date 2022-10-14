
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

    cd SARRA_data_download
    python get_AgERA5_data.py

This script automatically downloads AgERA5 data for the whole month of the last available date, every day at 12:00 PM. For example, if the current date is 2022-09-15, the last available date in AgERA5 will be 2022-09-07, thus, the script will download all data from 2022-09-01 to 2022-09-07.
Download is performed by default on an extent covering all West Africa (29°N,-20°E to 3.5°N,26°E).
Retrieved variables are Tmin, Tmax, solar radiation. 
ET0 is computed using Hargraeves formula.
Output files will be daily geotiffs, as required to run SARRA-O.

The downloaded and prepared data will be stored in the `./data/3_output/` path.

**TAMSAT data download**

    cd SARRA_data_download
    python get_TAMSAT_data.py

This script automatically downloads TAMSAT v3.1 data for the whole month of the last AgERA5 available date, every day at 12:00 PM. For example, if the current date is 2022-09-15, the last available date in AgERA5 will be 2022-09-07, thus, the script will download all TAMSAT data from 2022-09-01 to 2022-09-07.
Download is performed by default on an extent covering all West Africa (29°N,-20°E to 3.5°N,26°E).
Retrieved variables is `rfe_filled`. 
Output files will be daily geotiffs, as required to run SARRA-O.

The downloaded and cropped data will be stored in the `./data/3_output/` path.

**AgERA5 point data download**

    cd SARRA_data_download
    python get_AgERA5_data_point.py

This script automatically downloads AgERA5 data for a whole year, for the given coordinates.
Retrieved variables are Tmin, Tmax, Tmoy, solar radiation. 
ET0 is computed using Penman-Monteith as implemented in pcse.
Elevation info needed by pcse is retrieved from OpenElevation API.
Output files will be saved as csv, as required to run SARRA-H.

The downloaded and prepared data will be stored in the `./data/3_output/` path.

# ASSETS :
This repo also hosts a downscaled iSDAsoil soil texture class (USDA system) (https://zenodo.org/record/4094616#.Y0RBArTP1mN) to be used with SARRA. Downscaling was performed from the 0-20cm depth classification, at 4km resolution, aligned with TAMSAT raster files. Values were converted to SARRA-O soil type format, a 7 to 8 digit integer where the 6 last digits must be zeroes, and the first digits are the USDA soil code corresponding to the soil characteristics described in the `./data/csvTypeSol/` folder of SARRA-O executable file. Also, the null category value was replaced from 255 in iSDA to 0 in SARRA format.

This repo also hosts a downscaled ISRIC Africa SoilGrids soil texture class (USDA system) to be used with SARRA. Downscaling was performed from computing the mode of the different layers in the 0-60cm depth range of the classification, at 4km resolution, aligned with TAMSAT raster files. Values were converted to SARRA-O soil type format, a 7 to 8 digit integer where the 6 last digits must be zeroes, and the first digits are the USDA soil code corresponding to the soil characteristics described in the `./data/csvTypeSol/` folder of SARRA-O executable file. Also, the null category value was replaced from 255 in iSDA to 0 in SARRA format.

These assets can be found in `./soil_maps/` path.
To be used in SARRA-O, these files must be renamed `soil_africa_sarrah_tamsat.tif` and put into `./data/` folder of SARRA-O executable.

# TO-DO :
Calcul du ET0 par PM
Utiliser un MNT à la résolution des données pluie
TAMSAT resolution
sols actuellement à 9km
isda soil : à dégrader à la résolution de la pluie, ou a 3 km