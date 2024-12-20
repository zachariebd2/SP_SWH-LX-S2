# snow-swh



## Synopsis

This code implements a snow cover detection for SPOT4-SWH data.

The algorithm is based on neural networks trained on degraded Sentinel2 images and snow maps. This repository needs the model as requirement, it doesn't generate it. 


## Usage

### Script python "run\_snow\_swh.py"

Open a GPU node on HAL and activate the conda environment _/work/CAMPUS/etudes/Neige/Snow_SWH/conda_env/snow\_swh\_py38_. 

You may need to override the variable _LD\_LIBRARY\_PATH_ to import torch correctly :
```bash
export LD_LIBRARY_PATH=/work/CAMPUS/etudes/Neige/Snow_SWH/conda_env/snow_swh_py38/lib:$LD_LIBRARY_PATH
```

Then the easiest way to launch snow detection is by using the script "run\_snow\_swh.py" :
```bash
python run_snow_swh.py –j {launch_configuration_file.json}
```

with launch_configuration_file.json :
```json
{
	"input_path" : "XXX/SPOT4-HRVIR1-XS_20010614-110246-780_L1C_039-264-0_D_V1-0.zip",
	"output_dir" : "XXX/output",
	"models_dir" : "XXX/models",
	"model_type" : "mtn",
	"dem_dir"    : "XXX/DEM",
	"tcd_dir"    : "/datalake/static_aux/TreeCoverDensity",
	"buffer"     : 2000
}
```
All launch parameters can be overwritten by the following command line options :

```
* "-i", "--input_dir"           - Path to the zipped SWH product (e.g. SPOT4-HRVIR2-XS_20010530-105104-700_L1C_043-264-0_D_V1-0.zip)
* "-o", "--output_dir"          - Path to output directory
* "-models", "--model_dir"      - Path to models directory
* "-m", "--model_type"          - Type of model "tile" (local model) or "mtn" (model trained on the entire mountain range) to be used for inference. (optional, "mtn" by default)
* "-b", "buffer"                - Buffer size to be added around nodata pixels (optional, 2000 by default)
* "-dem", "--dem_dir"           - Path to dem file (optional)
* "-tcd", "--tcd_dir"           - Path to tree cover density file (optional)
```

In the following example, input directory and output directory are overwritten from launch_configuration file

```bash
python run_snow_swh.py –j {xx/launch_configuration_file.json} -i {xx/input_dir} -o {xx/output_dir}
```

It can also be launched like this :
```bash
python run_snow_swh.py -i {xx/input_dir} -o {xx/output_dir} -dem {xx/dem_dir} -tcd {xx/tcd_dir} -models {xx/models_dir}
```

**Example for HAL :**
```bash
python run_snow_swh.py –j snow_swh_launch.json
```

### Script shell "run\_snow\_swh\_job.sh"

To launch snow detection, you can also use the script "run\_snow\_swh\_job.sh".
Firstly, change some variables inside the shell script:
```bash
CONDA_ENV=/work/CAMPUS/etudes/Neige/Snow_SWH/conda_env/snow_swh_py38
WORKSPACE=/work/CAMPUS/etudes/Neige/Snow_SWH/snow-swh  #folder with snow-swh project
INPUT_PATH=XX/input_dir
INPUT_NAME=SPOT4-HRVIR1-XS_20120528-100939-327_L1C_040-264-0_D_V1-0.zip #input filename
OUTPUT_PATH=XX/output_dir
DEM_PATH=XX/DEM
TCD_PATH=/datalake/static_aux/TreeCoverDensity
MODELS_PATH=XX/models
```

Then you can submit a PBS job :
```bash
qsub run_snow_swh_job.sh
```

## Products format

Snow SWH generates the following files :
- A snow cover map + cloud of values 0 (ground), 1 (snow), 2 (clouds) and 255 (nodata). Raster: **<*satellite*>\_<*instrument*>\_<*YYYYMMDD*>\_<*HHMMSS*>\_<*milliseconds*>\_<*S2\_tilename*>\_SCA.tif**, 
- A pixel quality map in binary. The first bit indicates whether the SCA pixel is in the nodata buffer zone. The second bit indicates whether the pixel has a tcd value > 50%. Raster: **<*satellite*>\_<*instrument*>\_<*YYYYMMDD*>\_<*HHMMSS*>\_<*milliseconds*>\_<*S2\_tilename*>\_QA.tif**, 
- A copy of the original SWH product metadata file.


## Installation

### Dependencies

Following a summary of the required dependencies: 

* Python interpreter = 3.8
* Python libs
* Python packages:
  * torchvision
  * pyproj
  * gdal
  * numpy
* Python package dependencies:
  * os
  * glob
  * errno
  * json
  * pickle
  * sys
  * zipfile

### Conda environment on HAL

On HAL, you can create a PyTorch GPU environment (https://gitlab.cnes.fr/hpc/wikiHPC/-/wikis/conda_gpu) containing gdal and pyproj.

```bash
qsub -I -l walltime=04:00:00 -l select=1:ncpus=4:mem=92G:ngpus=1 -q qgpgpua100

module load conda/4.9.0

conda create -p /work/scratch/$USER/conda_env/snow_swh_py38/
conda activate /work/scratch/$USER/conda_env/snow_swh_py38
```

Modules installation via the Artifactory public repository of the CNES :
```bash
conda install python=3.8 gdal pytorch-gpu torchvision cudatoolkit=11.0 -c conda-pytorch-remote -c conda-forge-remote 
conda install -c conda-forge-remote pyproj
conda install -c conda-forge-remote pytorch-lightning
```

OR module installation via proxy :
```bash
conda install python=3.8 gdal pytorch-gpu torchvision cudatoolkit=11.0 -c pytorch -c conda-forge
conda install -c conda-forge pyproj
conda install -c conda-forge pytorch-lightning
```

