import logging
import torch
import numpy as np
import torch.nn.functional as F

from optimizers.common.search_space import build_search_space
from optimizers.common.rmsprop import RMSprop
from utils.exp_utils import save_concat_image
from PIL import Image


def padding(inputs_embeds, pad_inputs_embeddings, max_length):
    pad_length = max_length - inputs_embeds.shape[1]
    if pad_length > 0:
        pad_inputs_embeddings = pad_inputs_embeddings.repeat((1, pad_length, 1))
        inputs_embeds = torch.cat([inputs_embeds, pad_inputs_embeddings], dim=1)
    else:
        inputs_embeds = inputs_embeds[:, :max_length, :]

    return inputs_embeds


def get_init_log_coeff(sub_embeddings, init_coeff=1):
    log_coeff = torch.zeros(1, len(sub_embeddings)).to(sub_embeddings.device)
    log_coeff[0, 0] = init_coeff
    log_coeff.requires_grad = True
    return log_coeff


class PromptOptimizer():
    """ a class to manage 1. embeddings 2. prompt distribution """
    def __init__(self, model, text_encoder, tokenizer, subs, args):
        self.device = text_encoder.device
        self.args = args
        
        ## set model
        self.model = model
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        ## extract embeds
        with torch.no_grad():
            self.embeddings = text_encoder.get_input_embeddings()(
                torch.arange(0, tokenizer.vocab_size).long().to(self.device))
        self.bos_input_embeds = self.embeddings[tokenizer.bos_token_id]
        self.eos_input_embeds = self.embeddings[tokenizer.eos_token_id]
        self.model_max_length = tokenizer.model_max_length
        self.ori_prompt = subs['prompt']
        input_ids = tokenizer(self.ori_prompt, padding='max_length', truncation=True, return_tensors='pt').input_ids
        self.ori_embeds = text_encoder(input_ids.to(self.device))[0]

        ## init search space
        self.sspace = build_search_space(subs, self.tokenizer, self.embeddings)
        self.log_coeffs = self._init_log_coeffs(self.sspace)
        self.optimizer = RMSprop(self.log_coeffs, lr=0.1, momentum=0.5)

    #### TODO under dev #### here
    def _init_log_coeffs(self, sspace):
        log_coeffs = []
        for _type in sspace:
            sub_dict = sspace[_type]

            for sub_word_dict in sub_dict['all_words']:
                do_sub = sub_word_dict['do_sub']
                sub_embeddings = sub_word_dict['embeddings']
                if do_sub:  ## substitute (1, sub_choices)
                    log_coeff = get_init_log_coeff(sub_embeddings)
                    sub_word_dict['log_coeff'] = log_coeff
                    log_coeffs.append(log_coeff)
                    assert id(sub_word_dict['log_coeff']) == id(log_coeffs[-1])
        return log_coeffs

    def _get_log_coeffs_grads(self):
        return [log_coeff.grad for log_coeff in self.log_coeffs]

    def _inplace_clip(self, params, min, max):
        for param in params:
            param.data.copy_(torch.clip(param, min, max))
    
    def _inplace_sub(self, ps1, ps2):
        assert len(ps1) == len(ps2)
        for p1, p2 in zip(ps1, ps2):
            p1.data.sub_(p2)

    def step(self, min_coeff=0, max_coeff=3):
        grads = self._get_log_coeffs_grads()

        self._inplace_clip(grads, -0.025, 0.025)
        updates = self.optimizer.get_update() ## include lr
        self._inplace_sub(self.log_coeffs, updates)
        self._inplace_clip(self.log_coeffs, min_coeff, max_coeff)

        self.optimizer.zero_grad()

    def set_log_coeffs(self, new_log_coeffs):
        if isinstance(new_log_coeffs[0], list):  # multiple log_coeffs
            new_log_coeffs = list(map(list, zip(*new_log_coeffs)))
            assert len(self.log_coeffs) == len(new_log_coeffs)
            for idx, new_log_coeffs in enumerate(new_log_coeffs):
                self.log_coeffs[idx] = torch.cat([*new_log_coeffs], dim=0)
            ## update sub_dict as well
            idx = 0
            for _type in self.sspace:
                sub_dict = self.sspace[_type]
                for sub_word_dict in sub_dict['all_words']:
                    if sub_word_dict['do_sub']:
                        assert self.log_coeffs[idx].shape[-1] == sub_word_dict['log_coeff'].shape[-1]
                        sub_word_dict['log_coeff'] = self.log_coeffs[idx]; idx += 1
            assert idx == len(self.log_coeffs)
        else:
            assert len(self.log_coeffs) == len(new_log_coeffs)
            for log_coeff, new_log_coeff in zip(self.log_coeffs, new_log_coeffs):
                log_coeff.data.copy_(new_log_coeff)

    def get_log_coeffs(self):
        assert not isinstance(self.log_coeffs[0], list), 'not implemented'
        log_coeffs_copy = []
        for log_coeff in self.log_coeffs:
            log_coeffs_copy.append(log_coeff.clone())
        return log_coeffs_copy

    def display_log_coeffs(self):
        for log_coeff in self.log_coeffs: logging.info(log_coeff.cpu().detach())

    def sample_prompt(self, argmax=False, return_ids=False):
        ## sample sub words
        all_prompt = {'pos': self.ori_prompt, 'neg': None}
        all_sub_ids = []  ## flattened, for EA
        for _type in self.sspace:
            sub_dict = self.sspace[_type]['all_words']

            prompt = []
            sub_ids = []
            for sub_word_dict in sub_dict:
                do_sub = sub_word_dict['do_sub']
                if do_sub:
                    sub_words = sub_word_dict['sub_words']
                    sub_log_coeff = sub_word_dict['log_coeff']
                    if argmax:
                        max_indices = torch.where(sub_log_coeff[0] == torch.max(sub_log_coeff))[0]
                        sub_id = np.random.choice(max_indices.cpu().data)
                    else:
                        sub_coeff = F.gumbel_softmax(sub_log_coeff.unsqueeze(0).repeat(1, 1, 1), tau=1)
                        sub_id = sub_coeff.argmax(dim=-1).item()
                    sub_word = sub_words[sub_id]
                    sub_ids.append(sub_id)
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
        
            all_sub_ids += sub_ids

        if return_ids:
            return all_sub_ids
        return tuple([all_prompt['pos'], all_prompt['neg']])

    def get_mixed_embeds(self, temp=1):
        all_mixed_embeds = {'pos':None, 'neg':None}
        for _type in self.sspace:
            sub_dict = self.sspace[_type]

            mixed_embeds = self.bos_input_embeds.repeat(1, 1, 1)
            for sub_word_dict in sub_dict['all_words']:
                do_sub = sub_word_dict['do_sub']
                sub_embeddings = sub_word_dict['embeddings']
                if do_sub:  ## substitute
                    sub_log_coeff = sub_word_dict['log_coeff']
                    sub_coeff = F.gumbel_softmax(sub_log_coeff.unsqueeze(0).repeat(1, 1, 1), tau=temp)
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        sub_mixed_inputs_embeds = sub_coeff @ sub_embeddings
                else:
                    sub_mixed_inputs_embeds = sub_embeddings.repeat(1, 1, 1)
                mixed_embeds = torch.cat([mixed_embeds, sub_mixed_inputs_embeds], dim=1)
            mixed_embeds = torch.cat([mixed_embeds, self.eos_input_embeds.repeat(1, 1, 1)], dim=1)
            mixed_embeds = padding(mixed_embeds, self.eos_input_embeds, self.model_max_length)

            if _type == 'pos':
                all_mixed_embeds['pos'] = mixed_embeds
            elif _type == 'neg':
                all_mixed_embeds['neg'] = mixed_embeds
            else:
                raise ValueError(_type)

        if all_mixed_embeds['pos'] is None:
            all_mixed_embeds['pos'] = self.ori_embeds

        return all_mixed_embeds['pos'], all_mixed_embeds['neg']

    def eval_cand(self, prompt_pos, prompt_neg, train_seed_list):
        avg_loss, image_pil_loss = self.model.inference(self.ori_prompt,
                                                        prompt_pos,
                                                        train_seed_list,
                                                        self.args,
                                                        negative_prompt=prompt_neg)
    
        return avg_loss, image_pil_loss

    def save_cand(self, cand_list, topk=5):
        logging.info('RESULT IMAGE')
        success_opt_prompts = []
        for cand_info in cand_list:
            success_opt_prompts.append([cand_info['cand_prompt'], cand_info['avg_loss'], cand_info['loss_list']])
        success_opt_prompts = success_opt_prompts[:topk]

        for cand_idx, (opt_prompt, avg_loss, image_pil_loss) in enumerate(success_opt_prompts):
            logging.info(f'opt prompt: {opt_prompt} - {abs(avg_loss)}')
            if opt_prompt[1] is None:
                opt_prompt = (opt_prompt[0], '') # positive, negative

            opt_im = Image.fromarray(np.concatenate([it[0] for it in image_pil_loss], axis=1))
            save_concat_image(self.args.save_image_path, opt_im, f'dpo-{cand_idx}')

        all_avg_losses = [item[1] for item in success_opt_prompts]

        return all_avg_losses