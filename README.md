
# SARRA data download
This repo is a collection of tools to download and prepare climate and weather files necessary for the use of [SARRA-O](https://gitlab.cirad.fr/sarrao/model/sarrao) and [SARRA-Py](https://github.com/SARRA-cropmodels/SARRA-Py) spatialized crop simulation models. Its rationale is to allow for easy download of time series datasets from different data providers for Africa.

## How to install
You will need Python 3.9.6 or above.

Then, clone this repo and install its dependencies :

    git clone https://github.com/SARRA-cropmodels/SARRA_data-download
    cd SARRA_data-download
    pip install -r requirements.txt

## How to use
**Get satellite rainfall estimates**

This Jupyter notebook contains code to download and prepare daily satellite rainfall estimates for a given year from three data sources : TAMSAT (https://www.tamsat.org.uk/), CHIRPS (https://www.chc.ucsb.edu/data/chirps), and IMERG (https://gpm.nasa.gov/data/imerg).

Open the notebook, modify the parameters in the appropriate section and run the cells.



**AgERA5 data download**
First, [setup a Copernicus Climate Data Store API key](https://cds.climate.copernicus.eu/api-how-to) and accept [Copernicus Terms of Service](https://cds.climate.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products).

    cd SARRA_data_download
    python get_AgERA5_data.py

This script automatically downloads AgERA5 data for the whole month of the last available date, every day at 12:00 PM. For example, if the current date is 2022-09-15, the last available date in AgERA5 will be 2022-09-07, thus, the script will download all data from 2022-09-01 to 2022-09-07.
Download is performed by default on an extent covering all West Africa (29°N,-20°E to 3.5°N,26°E).
Retrieved variables are Tmin, Tmax, solar radiation. 
ET0 is computed using Hargraeves formula.
Output files will be daily geotiffs, as required to run SARRA-O.

The downloaded and prepared data will be stored in the `./data/3_output/` path.




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

**TAMSAT point data download**

This script automatically downloads TAMSAT tabular data format, for the given coordinates.
The downloaded and prepared data will be stored in the `./data/3_output/` path.

# ASSETS :
This repo also hosts a downscaled iSDAsoil soil texture class (USDA system) (https://zenodo.org/record/4094616#.Y0RBArTP1mN) to be used with SARRA. Downscaling was performed from the 0-20cm depth classification, at 4km resolution, aligned with TAMSAT raster files. Values were converted to SARRA-O soil type format, a 7 to 8 digit integer where the 6 last digits must be zeroes, and the first digits are the USDA soil code corresponding to the soil characteristics described in the `./data/csvTypeSol/` folder of SARRA-O executable file. Also, the null category value was replaced from 255 in iSDA to 0 in SARRA format.

https://dataverse.cirad.fr/dataset.xhtml?persistentId=doi:10.18167/DVN1/YSVTS2

This repo also hosts a downscaled ISRIC Africa SoilGrids soil texture class (USDA system) to be used with SARRA. Downscaling was performed from computing the mode of the different layers in the 0-60cm depth range of the classification, at 4km resolution, aligned with TAMSAT raster files. Values were converted to SARRA-O soil type format, a 7 to 8 digit integer where the 6 last digits must be zeroes, and the first digits are the USDA soil code corresponding to the soil characteristics described in the `./data/csvTypeSol/` folder of SARRA-O executable file. Also, the null category value was replaced from 255 in iSDA to 0 in SARRA format.

These assets can be found in `./soil_maps/` path.
To be used in SARRA-O, these files must be renamed `soil_africa_sarrah_tamsat.tif` and put into `./data/` folder of SARRA-O executable.

# TO-DO :
Calcul du ET0 par PM
Utiliser un MNT à la résolution des données pluie
TAMSAT resolution
sols actuellement à 9km
isda soil : à dégrader à la résolution de la pluie, ou a 3 km


https://www.catds.fr/Products/Available-products-from-CEC-SM/L4-Land-research-products/PrISM-precipitation-product
https://zenodo.org/record/5998113#.YzqpyITP02w
https://zenodo.org/record/5998113#.Y1k1BITP02w
https://earth.esa.int/eogateway/news/smos-data-improve-estimates-of-rainfall-in-africa

https://hal.inrae.fr/hal-02626156v2/document
https://www.mdpi.com/2072-4292/12/3/481
https://www.mdpi.com/2072-4292/14/3/746


