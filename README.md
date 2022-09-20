https://confluence.meteogroup.net/display/MA5/JRC+Marsop5%3A+Yearly+ERA5+processing
setup your cdsapi key https://cds.climate.copernicus.eu/api-how-to
accept TOS at https://cds.climate.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products

modify config.py to include out path in bdir variable

## 1. téléchargement des données
#### <br> run python main_0_download_ERA5_CDS_API_monthly.py

## 2. correction des données téléchargées 
#### <br> run python correction_cds_download_ncdf4_file_V1.0.py

## 3. aggregation journalière
####<br> run python main_2_aggregation.py

## 4. Calcul ETP journalier
#### <br> run python main_2_evapotranspiration.py

## 5. synthèse décadaire. 
#### <br> run python main_2_dekadly_aggregation.py

## 6. convertir ncdf4 en geotiff
####<br> run python convert_dekad_netcdf_to_geotiff_V1.1.py 

`