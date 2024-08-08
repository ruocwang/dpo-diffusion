import sys
import numpy as np
import logging
from collections import OrderedDict
from copy import deepcopy
import sys
sys.setrecursionlimit(10000)
import functools
print = functools.partial(print, flush=True)


choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))


class EvolutionPromptOptimizer(object):

    def __init__(self, args, prompt_opt):
        self.args = args
        self.algo_config = self.args.algo_config
        self.train_seed_list = list(range(args.num_seeds))
        if 'constraint' in args.task_config and args.task_config['constraint']['thresh'] > 0:
            self.constraint_fn = args.task_config['constraint']['fn']
            self.thresh = args.task_config['constraint']['thresh']
        else:
            self.constraint_fn = None

        ## EA hyper-params
        config = args.algo_config['epo']
        self.select_num = config['select_num']
        self.population_num = config['population_num']
        self.m_prob = config['m_prob']
        self.crossover_num = config['crossover_num']
        self.mutation_num = config['mutation_num']

        self.vis_dict = OrderedDict()  #### wrc comment: keep all information about visited candidates
        self.keep_top_k = {self.select_num: [], self.population_num: []}
        self.epoch = 0
        self.candidates = []

        self.sspace = prompt_opt.sspace
        self.cand_choices = self.generate_cand_choices()
        
        ## initial distribution to generate population
        self.prompt_opt = prompt_opt
        self.num_samples = config['num_samples']
        self.explore = args.explore if 'explore' in args else 1.0
        self.max_iter_exceeded = False

    def generate_cand_choices(self):
        cand_choices = []
        for _type in self.sspace:
            sub_dict = self.sspace[_type]
            for sub_word_dict in sub_dict['all_words']:
                do_sub = sub_word_dict['do_sub']
                if do_sub:
                    sub_words = sub_word_dict['sub_words']
                    cand_choices.append(len(sub_words))
                    print(_type, sub_words)
        return cand_choices

    def sample_from_learned(self):
        cand = self.prompt_opt.sample_prompt(argmax=self.first_gumbel_sample, return_ids=True)
        self.first_gumbel_sample = False
        return cand

    def sample_from_uniform(self):
        cand = []
        for cand_choice in self.cand_choices:
            cand.append(np.random.randint(0, cand_choice))
        return cand

    def sample_cand_fn(self):
        explore = np.random.uniform() < self.explore
        if explore:
            cand = self.sample_from_uniform()
        else:
            cand = self.sample_from_learned()
        return tuple(cand)
    
    def cand_id2prompt(self, cand):
        """ core function of transforming DNA/word_id (in this file) to prompt (in main.py) """
        all_prompt = {'pos':None, 'neg':None}
        idx = 0
        for _type in self.sspace:
            sub_dict = self.sspace[_type]

            prompt = []
            for sub_word_dict in sub_dict['all_words']:
                if sub_word_dict['do_sub']:
                    sub_words = sub_word_dict['sub_words']
                    prompt.append(sub_words[cand[idx]]); idx += 1
                else:
                    sub_word = sub_word_dict['ori_word']
                    prompt.append(sub_word)

            prompt = ' '.join(prompt)
            if _type == 'pos':
                all_prompt['pos'] = prompt
            elif _type == 'neg':
                all_prompt['neg'] = prompt.replace(' ', ',')
            else:
                raise ValueError(_type)
        
        if all_prompt['pos'] is None:
            all_prompt['pos'] = self.prompt_opt.ori_prompt
        assert idx == len(cand)
        return all_prompt['pos'], all_prompt['neg']

    def validate_and_eval(self, cand):
        assert isinstance(cand, tuple)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False

        info['visited'] = True

        ## valid cands
        info['cand_prompt'] = self.cand_id2prompt(cand)
        if self.constraint_fn is not None:
            info['constraint'] = self.constraint_fn(info['cand_prompt'][0])
            if info['constraint'] < self.thresh:
                info['avg_loss'] = 1e5
                return False
        
        info['visited'] = True
        avg_loss, loss_list = self.prompt_opt.eval_cand(*info['cand_prompt'], self.train_seed_list); self.budget += 1
        info['avg_loss'] = avg_loss
        info['loss_list'] = loss_list

        return True

    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        ori_t = deepcopy(t)
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

        if t[:k] != ori_t[:k]:
            logging.info(f'---> Top {k} updated')
        if len(ori_t) > 0 and t[0] != ori_t[0]:
            logging.info(f'---> Best updated')

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random(self, num):
        print('random select ........')
        max_iters = num * 10

        cand_iter = self.stack_random_cand(self.sample_cand_fn)
        while len(self.candidates) < num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.validate_and_eval(cand):
                continue
            self.candidates.append(cand)
            print('random {}/{}'.format(len(self.candidates), num))
        print('random_num = {}'.format(len(self.candidates)))
        if max_iters == 0:
            self.max_iter_exceeded = True

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        max_iters = mutation_num * 10

        def random_select_and_mutate_func():  ## randomly pick a candidate and mutate
            cand = list(choice(self.keep_top_k[k]))
            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    cand[i] = np.random.randint(self.cand_choices[i])
            return tuple(cand)

        cand_iter = self.stack_random_cand(random_select_and_mutate_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.validate_and_eval(cand):
                continue
            res.append(cand)
        
        print('mutation_num = {}'.format(len(res)))
        if max_iters == 0:
            self.max_iter_exceeded = True
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        max_iters = 10 * crossover_num

        def random_parent_crossover_func():
            p1 = choice(self.keep_top_k[k])
            p2 = choice(self.keep_top_k[k])
            return tuple(choice([i, j]) for i, j in zip(p1, p2))

        cand_iter = self.stack_random_cand(random_parent_crossover_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.validate_and_eval(cand):
                continue
            res.append(cand)

        print('crossover_num = {}'.format(len(res)))
        if max_iters == 0:
            self.max_iter_exceeded = True
        return res

    def get_topk(self, k):
        topk_cands = []
        for cand in self.keep_top_k[k]:
            topk_cands.append(self.vis_dict[cand])
        return topk_cands

    def search(self):
        logging.info('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} budget = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.num_samples))
    
        self.budget = 0
        self.first_gumbel_sample = True

        #### init population
        self.get_random(self.population_num)

        #### search
        while self.budget < self.num_samples and not self.max_iter_exceeded:  ## the first epoch is random init
            logging.info('epoch = {}'.format(self.epoch))

            ## register top k
            self.update_top_k(self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['avg_loss'])
            self.update_top_k(self.candidates, k=self.population_num, key=lambda x: self.vis_dict[x]['avg_loss'])

            logging.info('epoch = {} : top {} result'.format(self.epoch, len(self.keep_top_k[self.population_num])))
            for i, cand in enumerate(self.keep_top_k[self.population_num]):
                logging.info('No.{} {} Top-1 err = {}'.format(i + 1, cand, self.vis_dict[cand]['avg_loss']))
                ops = [i for i in cand]
                logging.info(ops)

            ## skip the last mutation crossover (redundant runs)
            mutation = self.get_mutation(self.select_num, self.mutation_num, self.m_prob)
            if self.budget >= self.num_samples:
                break
            crossover = self.get_crossover(self.select_num, self.crossover_num)
            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

        self.update_top_k(self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['avg_loss'])
        self.update_top_k(self.candidates, k=self.population_num, key=lambda x: self.vis_dict[x]['avg_loss'])
        return self.get_topk(k=self.population_num)

    def get_cand_history(self):
        cand_history = []
        for cand in self.vis_dict:
            info = self.vis_dict[cand]
            if 'loss_list' in info:
                cand_history.append(info)
        return cand_history
