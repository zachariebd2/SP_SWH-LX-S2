#!/bin/bash
#PBS -N BULK
#PBS -j oe
#PBS -l select=1:ncpus=8:mem=40G:ngpus=1
#PBS -l walltime=00:59:59
#PBS -k n
#PBS -o /home/ad/barrouz/WORKSPACE/TIMESERIES_PROJECT/job_logs
#PBS -e /home/ad/barrouz/WORKSPACE/TIMESERIES_PROJECT/job_logs
#PBS -q qgpgpu
#PBS -m abe

echo 'bulk download'
cd /home/ad/barrouz/WORKSPACE/TIMESERIES_PROJECT/bulk-downloader

python download_espa_order.py -e zacharie.barrou-dumont@univ-tlse3.fr -o "075420-341" -d /home/ad/barrouz/zacharie/TIMESERIES_PROJECT/USGS_DOWNLOAD/LANDSAT4/ -u zacharie -p "Earth_0b5ervat10n" -v 

