import numpy as np
import logging


def get_t_schedule(gpo_config):
    num_iters = gpo_config['num_iters']
    t_mode_sche = gpo_config['t']
    mode = t_mode_sche.split('-')[0]
    sche = [int(x) for x in t_mode_sche.split('-')[1:]]

    if num_iters == 0:  ## ES only
        sche = []
        avg_t = 0
    elif mode == 'rand':  ## rand-15-25
        min_t, max_t = sche
        avg_t = (max_t + min_t) // 2
        sche = [np.random.randint(min_t, max_t) for i in range(num_iters)]
    elif mode == 'fix':  ## fix-25
        fix_t = sche[0]
        avg_t = fix_t
        sche = [fix_t] * num_iters
    elif mode == 'step':  ## step-10-5
        start_t, repeat = sche
        sche = [start_t + (i // repeat) for i in range(num_iters)]
        avg_t = (sche[0] + sche[-1]) // 2

    return sche, avg_t


class GradientPromptOptimizer(object):

    def __init__(self,
                 prompt_opt,
                 args):
        self.args = args
        self.prompt_opt = prompt_opt
        self.config = args.algo_config['gpo']
        self.train_seed_list = list(range(args.num_seeds))

        constraint = args.task_config.get('constraint', None)
        if constraint is not None and constraint['thresh'] > 0:
            self.constraint_fn = constraint['fn']
            self.constraint_thresh = constraint['thresh']
        else:
            self.constraint_fn = None

    def search(self, num_evals=0):
        ts, avg_t = get_t_schedule(self.config)
        best_log_coeffs = self.prompt_opt.log_coeffs

        for i in range(self.config['num_iters']):
            ## sample mixed embedding
            mixed_embeds_pos, mixed_embeds_neg = self.prompt_opt.get_mixed_embeds(1)

            #### compute text gradient
            self.prompt_opt.model.inference_with_grad(self.prompt_opt.ori_prompt,
                                                      mixed_embeds_pos, mixed_embeds_neg,
                                                      self.train_seed_list,
                                                      ts[i],
                                                      self.args)

            self.prompt_opt.step()
            self.prompt_opt.display_log_coeffs()

        num_evals += (self.config['num_iters'] * avg_t / 50)
        num_evals += self.config['num_samples']

        ## register best log_coeffs
        logging.info('RESULT LOGITS')
        self.prompt_opt.set_log_coeffs(best_log_coeffs)
        self.prompt_opt.display_log_coeffs()

        ## sample from learned distribution
        res_list = self.sample_and_eval(self.config['num_samples'])

        return num_evals, res_list
    
    def sample_and_eval(self, N):
        res_list = []
        visited = {}
        while N > 0:
            cand_prompt = self.prompt_opt.sample_prompt()
            prompt_pos, prompt_neg = cand_prompt
            
            if cand_prompt in visited: continue
            visited[cand_prompt] = True

            ## validate against constraint
            if self.constraint_fn is not None:
                import pdb, rlcompleter; pdb.Pdb.complete = rlcompleter.Completer(locals()).complete; pdb.set_trace()
                if self.constraint_fn(cand_prompt) < self.constraint_thresh:
                    continue
            
            ## evaluate valid prompts
            N -= 1
            avg_loss, image_pil_loss = self.prompt_opt.eval_cand(prompt_pos, prompt_neg, self.train_seed_list)
            res_list.append({'cand_prompt': cand_prompt, 'avg_loss': avg_loss, 'loss_list': image_pil_loss})

        res_list = sorted(res_list, key=lambda x: x['avg_loss'])
        return res_list
