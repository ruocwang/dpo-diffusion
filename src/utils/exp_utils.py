import os
import time
import json
import logging
import torch
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt


def create_exp_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    script_path = os.path.join(path, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path)


def pick_gpu_lowest_memory(wait_on_full=0):
    import gpustat

    while True:
        print('queueing for GPU...')
        stats = gpustat.GPUStatCollection.new_query()
        ids = list(map(lambda gpu: int(gpu.entry['index']), stats))
        ratios = list(map(lambda gpu: float(gpu.memory_used)/float(gpu.memory_total), stats))
        bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
        print(stats)
        
        if not wait_on_full:
            break

        bestGPUratio = ratios[bestGPU]
        if bestGPUratio < 0.05:
            break
        else:
            time.sleep(1)
    print('found available GPU: {}'.format(bestGPU))
    return bestGPU


def deterministic_mode(seed=0):
    logging.info('===> Deterministic mode with seed: {}'.format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_exp_stats(all_results):
    if not isinstance(all_results, list):
        all_results = [all_results]
    all_best = []
    for result in all_results:
        all_best.append(result['best_avg_loss'])
    
    avg_best, std_best = np.mean(all_best), np.std(all_best)
    
    return (avg_best, std_best)


def save_image(save_image_path, image_pil, prompt, seed, avg_loss, loss, prefix):
    path = f"{save_image_path}/{prefix}-{np.round(abs(avg_loss), 2)}-{prompt}-{seed}-{np.round(abs(loss), 2)}.png"
    image_pil.save(path)
    return path


def save_concat_image(save_image_path, image_pil, prefix):
    path = f"{save_image_path}/{prefix}-paper.png"
    image_pil.save(path)
    return path


def plot(points, seed, t, save_plot_path):
    plt.plot(points)
    plt.savefig(f'{save_plot_path}/loss-t={t}-seed={seed}.png')
    plt.close()


def plot_learning_curve(info_list, save_plot_path):
    points = []
    max_score = 0
    for info in info_list:
        points.append(max(abs(info['avg_loss']), max_score))
        max_score = max(points)

    plt.plot(points)
    plt.savefig(f'{save_plot_path}/best_score.png')
    plt.close()


def json_save(obj, file_path):
    json_object = json.dumps(obj, indent=4)
    json_object = unindent_list(json_object)
    with open(file_path, "w") as outfile:
        outfile.write(json_object)


def json_load(file_path, report_error=False):
    if not os.path.exists(file_path) and not report_error:
        print('load an empty json object')
        return {}
    with open(file_path, "r") as outfile:
        obj = json.load(outfile)
    return obj


def text_save(obj_list, file_path):
    with open(file_path, 'w') as f:
        f.writelines(obj + '\n' for obj in obj_list)


def text_load(file_path):
    with open(file_path, 'r') as f:
        obj_list = [line.strip() for line in f.readlines()]
    return obj_list


def print_args(args):
    args_dict = vars(args)
    for arg_name, arg_value in sorted(args_dict.items()):
        print(f"\t{arg_name}: {arg_value}")


def unindent_list(json_object):
    res = ''
    inside_list = False
    idx = 0
    while idx < len(json_object):
        c = json_object[idx]
        if c == '[':
            inside_list = True
        if c == ']':
            inside_list = False

        if inside_list and c == '\n':
            num_spaces = len(json_object[idx + 1:]) - len(json_object[idx + 1:].lstrip())
            idx += num_spaces

        res += json_object[idx]
        idx += 1

    return res


def result_to_json(all_results):
    if not isinstance(all_results, list):
        all_results = [all_results]
    ## loss of all seeds
    best_avg_loss_list = []
    pts_best_avg_loss_list = []
    for result in all_results:
        best_avg_loss_list.append(result['best_avg_loss'])
        if 'pts_best_avg_loss' in result:
            pts_best_avg_loss_list.append(result['pts_best_avg_loss'])
    ## ori loss
    ori_avg_loss = result['ori_avg_loss']
    ori_prompt = result['ori_prompt']
    
    ret = {
        'ori_avg_loss': ori_avg_loss,
        'best_avg_loss': np.mean(best_avg_loss_list),
        'best_avg_loss_list': best_avg_loss_list,
        'ori_prompt': ori_prompt
    }

    if len(pts_best_avg_loss_list) > 0:
        ret['pts_best_avg_loss'] = np.mean(pts_best_avg_loss_list),

    return ret


def load_config(file_path):
    """
    Load a YAML configuration file.
    
    Args:
        file_path (str): Path to the YAML file.
    
    Returns:
        dict: Configuration settings loaded from the YAML file.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        return config