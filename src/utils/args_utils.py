import os
import argparse
from prompts.paths import *
from utils.exp_utils import *


def parse_and_process_arguments():
    parser = argparse.ArgumentParser(description="DPO-DIFFUSION Optimizer.")

    #### management
    parser.add_argument('--save', type=str, default='cmd', help='saving directory / expid')
    parser.add_argument('--gpu', type=str, default='auto')
    #### model and data
    parser.add_argument("--version", default='v1-4', type=str, help="Stable Diffusion Version")
    parser.add_argument('--prompt_id', type=int, default=0)
    parser.add_argument('--path', type=str, default='substitutes-dev', help='substitutes json file path')
    parser.add_argument('--num_seeds', type=int, default=1, help='num images optimized (one per seed)')
    
    parser.add_argument('--algo_config', type=str, default='hybrid.yaml')
    parser.add_argument('--task_config', type=str, default='improve-antonym.yaml')

    args = parser.parse_args()

    args.algo_config = load_config(os.path.join('src/configs/algo', args.algo_config))
    args.task_config = load_config(os.path.join('src/configs/task', args.task_config))

    #### args augment
    args.gpu = str(pick_gpu_lowest_memory()) if args.gpu == 'auto' else args.gpu
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    ## output dir
    script_name = args.save
    exp_id = '{}'.format(script_name)
    args.save = os.path.join('experiments/', exp_id, f'{args.prompt_id}')
    ## args transform
    if 'data' not in args.path[:6]: args.path = os.path.join('data', args.path)
    if 'nfs' not in args.path: args.path = os.path.join(PROJECT_DIR, args.path)
    if '.json' not in args.path: args.path += '.json'

    return args