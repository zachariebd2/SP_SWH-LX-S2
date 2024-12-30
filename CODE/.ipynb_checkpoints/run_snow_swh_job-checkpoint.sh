#!/bin/bash


#setting env#################################
PBS_TMPDIR=$TMPDIR
CONDA_ENV=/work/scratch/env/$USER/conda_env/pl_lightning
unset TMPDIR
module unload conda
module load conda


SCRIPT_UNET=$PWD/CODE/app/run_snow_swh.py
SCRIPT_S3=$PWD/CODE/app/get_swh.py

if [ -z $INPUT_PATH ]; then
  echo "INPUT_PATH argument missing"
  exit 1
fi
echo INPUT_PATH $INPUT_PATH

if [ -z $OUTPUT_PATH ]; then
  echo "OUTPUT_PATH argument missing"
  exit 1
fi
echo OUTPUT_PATH $OUTPUT_PATH

if [ -z $DEM_PATH ]; then
  DEM_PATH=/work/CAMPUS/etudes/Neige/DEM
fi
echo DEM_PATH $DEM_PATH

if [ -z $TCD_PATH ]; then
  TCD_PATH=/work/datalake/static_aux/TreeCoverDensity
fi
echo TCD_PATH $TCD_PATH

if [ -z $MODELS_PATH ]; then
  echo "MODELS_PATH argument missing"
  exit 1
fi
echo MODELS_PATH $MODELS_PATH


if [ -z $KEEP_REFL ]; then
  echo "KEEP_REFL argument missing"
  exit 1
fi
echo KEEP_REFL $KEEP_REFL

if [ -z $MASK ]; then
  echo "MASK argument missing"
  exit 1
fi
echo MASK $MASK



echo "start"
echo $INPUT_PATH

INPUT_TMP_PATH=$PBS_TMPDIR/INPUT
mkdir -p $INPUT_TMP_PATH
OUTPUT_TMP_PATH=$PBS_TMPDIR/OUTPUT
mkdir -p $OUTPUT_TMP_PATH

while IFS= read -r line; do
  printf '%s\n' "$line"

  spotzip=$(basename ${line} | tr -d '\n')
  INPUT_TMP=$INPUT_TMP_PATH/$spotzip
  echo 'activate s3 conda env'
  conda activate /softs/rh8/conda-envs/s3-env
  echo "start s2 script"
  python $SCRIPT_S3 -swh $line\
                    -out $INPUT_TMP_PATH
  ls $INPUT_TMP_PATH
  echo $INPUT_TMP
  echo 'run script'
  conda activate $CONDA_ENV
  python $SCRIPT_UNET -i $INPUT_TMP\
                 -tmp $OUTPUT_TMP_PATH\
                 -o $OUTPUT_PATH\
                 -dem $DEM_PATH\
                 -msk $MASK\
                 -m mtn\
                 -k $KEEP_REFL\
                 -tcd $TCD_PATH\
                 -models $MODELS_PATH

  #cp -r $OUTPUT_TMP_PATH/* $OUTPUT_PATH/
  rm  -rf $INPUT_TMP


done < $INPUT_PATH

echo end
