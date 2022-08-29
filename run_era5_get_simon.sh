#!/bin/bash

# Export python path

# conda activate environment
conda deactivate
conda activate sarrao



#run download data
python main_0_download_ERA5_CDS_API_monthly.py 2014-02 2014-03 "burkina"

#run aggregate_daily variable attention choisir la base #todo
python main_1_aggregation.py 2014-02 2014-03 "burkina"

#run compute daily evapotranspiration
python main_2_evapotranspiration.py 2014-02 2014-03 "burkina"

#run convert netcdf to geotiff
python main_4_convert_daily_netcdf_to_geotiff.py "burkina"

#run convert daily to dekad
python main5.py "burkina"