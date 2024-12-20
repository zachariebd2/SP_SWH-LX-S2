#!/bin/bash






#setting env#################################
CONDA_ENV=/work/scratch/$USER/conda_env/pl_lightning
WORKSPACE=/home/ad/barrouz/WORKSPACE/TIMESERIES_PROJECT/snow_code/snow-swh
OUTPUT_SWH_PATH=/work/CAMPUS/etudes/Neige/CoSIMS/zacharie/TIMESERIES_PROJECT/TMP
OUTPUT_SNOW_PATH=/work/CAMPUS/etudes/Neige/CoSIMS/zacharie/TIMESERIES_PROJECT/TMP
DEM_PATH=/work/CAMPUS/etudes/Neige/DEM
TCD_PATH=/datalake/static_aux/TreeCoverDensity
MODELS_PATH=/home/ad/barrouz/WORKSPACE/TIMESERIES_PROJECT/snow_code/models

INPUT_PATH=/home/ad/barrouz/WORKSPACE/TIMESERIES_PROJECT/snow_code/inputs
INPUT_NAME=test_SPOT1.txt

echo "start"
echo $INPUT_NAME


PBS_TMPDIR=$TMPDIR
unset TMPDIR
module unload conda
module load conda
conda activate $CONDA_ENV


INPUT_TMP_PATH=$PBS_TMPDIR/INPUT
mkdir -p $INPUT_TMP_PATH
OUTPUT_TMP_PATH=$PBS_TMPDIR/OUTPUT
mkdir -p $OUTPUT_TMP_PATH

while IFS= read -r line; do
  printf '%s\n' "$line"
  echo tata
  spotzip=$(basename ${line} | tr -d '\n')
  echo toto
  INPUT_TMP=$INPUT_TMP_PATH/$spotzip
  echo titi
  cp $line $INPUT_TMP
  echo $INPUT_TMP
  echo 'run script'
  python $WORKSPACE/run_snow_swh.py -i $INPUT_TMP\
                                    -o $OUTPUT_TMP_PATH\
                                    -dem $DEM_PATH\
                                    -m mtn\
                                    -tcd $TCD_PATH\
                                    -models $MODELS_PATH
  cp -r $INPUT_TMP $OUTPUT_SWH_PATH/
  cp -r $OUTPUT_TMP_PATH/* $OUTPUT_SNOW_PATH/
  rm  -rf $OUTPUT_TMP_PATH/*
  
done < $INPUT_PATH/$INPUT_NAME

