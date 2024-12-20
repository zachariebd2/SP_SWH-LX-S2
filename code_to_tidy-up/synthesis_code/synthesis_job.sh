#!/bin/bash



module load lis
echo 'config' $config
let_it_snow_synthesis.py -j $config
rm -rf $out/tmp