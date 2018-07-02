# MuseoToolBox / Sentinel 2
Sentinel-2 dedicated toolkit from RS toolkit.

## Compute SITS
`computeSITS.py`

Given list of zip files (from L2A Theia) or unzip files, generate the **Satellite Image Time Series** with the 4 bands at 10m, or by adding the 6 bands at 20m resampled at 10m.
Computation is made using Python Binding of the great Orfeo ToolBox. You can choose to not use the python binding but it will be less efficient.

## Generate custom sample time
` generateSampleTime.py`

Given list of unzip folders (if not will seek for zip), you can generate a X-day time resampling :
- with the start date and/or the last date,
- or only with your available data.

## Generate indices from SITS
`generateIndicesFromSITS.py`

Generate indice for each date using your gap-filled Satellite Image Time Series.
List of available indices : ['NDVI', 'ACORVI', 'SAVI', 'EVI', 'EVI2', 'PSRI', 'ARI2'].
Each result is multiplied by 100 to save space (int16 type instead of float type).
This will : 
- generate a indice for each date (tif format)
- generate a vrt of the indice for all dates

