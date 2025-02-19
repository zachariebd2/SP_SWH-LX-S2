import os
import boto3
from botocore.exceptions import ClientError
import logging
from assumerole import assumerole
import zipfile
import glob
import json
import subprocess
import s3fs
import shutil
import argparse
import tempfile
import argparse
import os.path as op
from argparse import Action


arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

arg_parser.add_argument('-swh', action='store', default="", nargs='?', const='', dest='swh')
arg_parser.add_argument('-out', action='store', default="", nargs='?', const='', dest='out')
swh = arg_parser.parse_args().swh
out = arg_parser.parse_args().out

ENDPOINT_URL="https://s3.datalake.cnes.fr"
print("get credentials")
credentials = assumerole.getCredentials("put credentials here", Duration=7200)

print("get s3filesystem")
s3 = s3fs.S3FileSystem(
      client_kwargs={
                      'aws_access_key_id': credentials['AWS_ACCESS_KEY_ID'],
                      'aws_secret_access_key': credentials['AWS_SECRET_ACCESS_KEY'],
                      'aws_session_token': credentials['AWS_SESSION_TOKEN'],
         'endpoint_url': 'https://s3.datalake.cnes.fr'
      }
    
   )
print("download")
s3.get(swh, out, recursive=True)
