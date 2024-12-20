#!/bin/bash


#setting env#################################
PBS_TMPDIR=$TMPDIR
CONDA_ENV=/work/scratch/env/$USER/conda_env/pl_lightning
unset TMPDIR
module unload conda
module load conda

SCRIPT=/home/ad/barrouz/WORKSPACE/TIMESERIES_PROJECT/synthesis_code/safran_median_aspect_MK.py
conda activate $CONDA_ENV

if [ -z $MTN ]; then
  echo "MTN argument missing"
  exit 1
fi
echo MTN $MTN

if [[ -z $MASSIF ]]; then
  echo "MASSIF argument missing"
  exit 1
fi
echo MASSIF $MASSIF

if [ -z $N ]; then
  echo "N argument missing"
  exit 1
fi
echo N $N

python $SCRIPT -massif "$MASSIF"\
               -mtn $MTN\
               -N $N
