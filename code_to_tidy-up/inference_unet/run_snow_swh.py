#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
import argparse
import os.path as op
from argparse import Action
from src.snow_swh import detect_snow


class PathAction(Action):
    """
    Manage argument as path
    """

    def __call__(self, _, namespace, values, __=None):
        abs_path = op.realpath(values)
        if not op.exists(abs_path):
            raise argparse.ArgumentTypeError("Path '{}' does not exist".format(abs_path))
        setattr(namespace, self.dest, abs_path)

def create_argument_parser():
    """
    Create argument parser

    This parser contains :
    - the input file,
    - the output directory,
    - the log level,
    - the models directory,
    - the model type
    - the dem directory,
    - the tree cover density directory,
    - the buffer size
    :return: argument parser
    :rtype: argparse.ArgumentParser
    """
    description = "This script is used to run the snow detector module that computes snow cover from SWH products"

    arg_parser = argparse.ArgumentParser(description=description,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument("-i", "--input_file", help="Path to input file", action=PathAction, default=None)
    arg_parser.add_argument("-o", "--output_dir", help="Path to output directory; which will contains snow maps", default=None)
    arg_parser.add_argument("-tmp", "--job_dir", help="Path to tmp output directory", default=None)
    arg_parser.add_argument("-models", "--model_dir", help="Path to models directory", action=PathAction, default=None)
    arg_parser.add_argument("-m", "--model_type", help="Type of model to be used for inference", choices=['mtn', 'tile'])
    arg_parser.add_argument("-dem", "--dem_dir", help="Path to dem directory", action=PathAction, default=None)
    arg_parser.add_argument("-tcd", "--tcd_dir", help="Path to tree cover density directory", action=PathAction, default=None)
    arg_parser.add_argument("-b", "--buffer", help="Buffer size to be added around nodata pixels", default=None)
    arg_parser.add_argument("-k", "--keep", help="keep swh bands", default=None)
    arg_parser.add_argument("-msk", "--mask", help="apply mask directly to inference", default=None)
    arg_parser.add_argument("-j", "--json_config_file", help="Path to json config file", action=PathAction, default=None)
    arg_parser.add_argument("-l", "--log_level", help="Log level", choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])
    
    return arg_parser

def main(args):
    if args.json_config_file is not None:
        with open(args.json_config_file) as json_config_file:
            global_config = json.load(json_config_file)
            input_file = global_config.get("input_file", None)
            output_dir = global_config.get("output_dir", None)
            job_dir = global_config.get("job_dir", None)
            model_dir = global_config.get("model_dir", None)
            model_type = global_config.get("model_type", None)
            dem_dir = global_config.get("dem_dir", None)
            tcd_dir = global_config.get("tcd_dir", None)
            buffer = global_config.get("buffer", None)
            keep = global_config.get("keep", None)
            mask = global_config.get("mask", None)
            log_level = global_config.get("log", "INFO")
    else:
        input_file = None
        output_dir = None
        job_dir = None
        model_dir = None
        model_type = None
        dem_dir = None
        tcd_dir = None
        buffer = None
        keep=None
        mask=None
        log_level = None

    if args.output_dir is not None:
        output_dir = args.output_dir
        
    if args.job_dir is not None:
        job_dir = args.job_dir

    if output_dir is None:
        sys.exit(OUTPUT_UNDEFINED)

    if not os.path.exists(output_dir):
        logging.warning("Output directory product does not exist.")
        logging.info("Create directory " + output_dir + "...")
        os.makedirs(output_dir)

    # Create tmp dir
    os.makedirs(output_dir + "/tmp", exist_ok=True)
    tmp_dir = os.path.join(output_dir, "tmp")
    log_file = os.path.join(tmp_dir, "snow_swh.log")

    # init logger
    logger = logging.getLogger()
    if args.log_level is not None:
        log_level = args.log_level
    if log_level is None:
        log_level = "INFO"

    level = logging.getLevelName(log_level)
    logger.setLevel(level)

    # file handler
    file_handler = RotatingFileHandler(log_file, 'a', 1000000, 1)
    formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d    %(levelname)s:%(filename)s::%(funcName)s:%(message)s',
                                  datefmt='%Y-%m-%dT%H:%M:%S')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logging.info("output directory : %s", output_dir)
    
    if args.input_file is not None:
        logging.info("input file : %s", args.input_file)
        input_file = args.input_file
    if args.dem_dir is not None:
        logging.info("dem directory : %s", args.dem_dir)
        dem_dir = args.dem_dir
    if args.tcd_dir is not None:
        logging.info("tree cover density directory : %s", args.tcd_dir)
        tcd_dir = args.tcd_dir
    if args.model_dir is not None:
        logging.info("model directory : %s", args.model_dir)
        model_dir = args.model_dir
    if args.model_type is not None:
        logging.info("model type : %s", args.model_type)
        model_type = args.model_type
    if args.buffer is not None:
        logging.info("no data buffer : %s", args.buffer)
        buffer = args.buffer
    if args.keep is not None:
        logging.info("keep merged bands : %s", args.buffer)
        keep = args.keep
    if args.mask is not None:
        logging.info("mask inference : %s", args.buffer)
        mask = args.mask
    if model_type is None:
        model_type = "mtn"
        logging.info("model type : %s", model_type)
    if buffer is None:
        buffer = 2000
        logging.info("no data buffer : %s", buffer)
          
    # Run the snow detector
    try:
        logging.info("Launch snow detection.")
        detect_snow(input_file, output_dir,job_dir, model_type, model_dir, buffer, dem_dir, tcd_dir,keep,mask)
    except Exception as e:
        logging.error(e)
        sys.exit(1)

    sys.exit(0)
 
        
if __name__ == "__main__":
    args_parser = create_argument_parser()
    args = args_parser.parse_args(sys.argv[1:])
    print(args)
    main(args)
