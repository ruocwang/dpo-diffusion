import transformers
import os
import logging
import torch
transformers.logging.set_verbosity(transformers.logging.ERROR)
import sys
import shutil
from pprint import pformat
from utils.exp_utils import *


def create_logs(args):
    ## override path
    if os.path.exists(args.save):
        if input('{} exists, override? [y/n]'.format(args.save)) == 'y': shutil.rmtree(args.save)
        else: exit()
    create_exp_dir(args.save)
    ## output files
    args.save_image_path = os.path.join(args.save, 'gen_images')
    args.save_plot_path = os.path.join(args.save, 'plots')
    args.save_json_path = os.path.join(args.save.replace(f'/{args.prompt_id}', ''), 'all_results.json')
    args.save_cand_path = os.path.join(args.save.replace(f'/{args.prompt_id}', ''), 'cand_history.json')
    os.mkdir(args.save_image_path)
    os.mkdir(args.save_plot_path)
    # logging
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    log_file = 'log.txt'
    log_path = os.path.join(args.save, log_file)
    if os.path.exists(log_path) and input(f'log: {log_file} exists, override? [y/n]') != 'y':
        exit(0)
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('\n================== Args ==================\n')
    logging.info(pformat(vars(args)))