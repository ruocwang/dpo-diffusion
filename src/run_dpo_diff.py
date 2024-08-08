import transformers
import logging
import numpy as np
from PIL import Image
transformers.logging.set_verbosity(transformers.logging.ERROR)

from optimizers.evolution.epo import EvolutionPromptOptimizer
from optimizers.gradient.gpo import GradientPromptOptimizer
from optimizers.common.prompt_optimizer import PromptOptimizer
from optimizers.common.search_space import validate_synonyms, load_word_substitutes

from model.sd import StableDiffusion
from utils.task_utils import get_constraint_fn
from utils import parse_and_process_arguments, create_logs, get_constraint_fn, save_concat_image, compute_exp_stats, deterministic_mode


def main(args, device):
    #### load model
    sd = StableDiffusion(args.version, device=device)
    text_encoder = sd.text_encoder
    tokenizer = sd.tokenizer

    #### load pregen word substitutes
    subs, ori_prompt = load_word_substitutes(args.path, args.prompt_id, args.task_config)
    logging.info('ori prompt: {}'.format(ori_prompt))

    #### load constraints (if any)
    constraint = args.task_config.get('constraint', None)
    if constraint is not None and constraint['thresh'] > 0:
        constraint['fn'] = get_constraint_fn(constraint['fn'], ori_prompt, args.device)
        subs = validate_synonyms(subs, constraint['thresh'], constraint['fn'])

    #### load base optimizer (manages search space, sampling, and gradient updates)
    prompt_opt = PromptOptimizer(sd, text_encoder, tokenizer, subs, args)


    ######## main search
    num_evals = 0
    #### gradient-based optimization
    algo_config = args.algo_config
    if 'gpo' in algo_config:
        logging.info('='*20 + ' GPO ' + '='*20)
        gpo = GradientPromptOptimizer(prompt_opt, args)
        _, res_list = gpo.search()

    #### evolution-baed optimization
    if 'epo' in algo_config:
        logging.info('='*20 + ' EPO ' + '='*20)
        epo = EvolutionPromptOptimizer(args, prompt_opt)
        res_list = epo.search()


    ######## logging
    #### original image
    logging.info('original prompt...')
    ori_avg_loss, ori_image_pil_loss = prompt_opt.eval_cand(ori_prompt, None, list(range(args.num_seeds)))
    ori_im = Image.fromarray(np.concatenate([it[0] for it in ori_image_pil_loss], axis=1))
    save_concat_image(args.save_image_path, ori_im, prefix='ori')
    #### optimized images
    all_avg_losses = prompt_opt.save_cand(res_list)

    result = {
        'ori_prompt': ori_prompt,
        'ori_avg_loss': ori_avg_loss,
        'avg_losses': all_avg_losses,
        'best_avg_loss': all_avg_losses[0],
        'num_evals': num_evals
    }

    return result


if __name__ == "__main__":
    args = parse_and_process_arguments()
    create_logs(args)

    deterministic_mode()
    result = main(args, args.device)

    (avg_best, std_best) = compute_exp_stats(result)
    logging.info('Best:  {:.4f} \u00B1 {:.4f}'.format(avg_best, std_best))
    logging.info('Ori: {:.4f}'.format(result['ori_avg_loss']))