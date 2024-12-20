#!/bin/bash


#setting env#################################
PBS_TMPDIR=$TMPDIR
CONDA_ENV=/work/scratch/env/$USER/conda_env/pl_lightning
unset TMPDIR
module unload conda
module load conda


conda activate $CONDA_ENV



gdalwarp $options $in $out