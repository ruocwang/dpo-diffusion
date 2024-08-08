import torch
import copy
import numpy as np

from utils.exp_utils import json_load, text_load
from prompts.paths import NEG_PROMPTS_LIBRARY, NEG_PROMPTS_LIBRARY_V2


def load_word_substitutes(path, prompt_id, task_config):
    all_pregen_subs = json_load(path, report_error=True)
    ori_prompt = list(all_pregen_subs.keys())[prompt_id]
    pregen_subs = all_pregen_subs[ori_prompt]
    task_name = task_config['task_name'].lower()
    sspace = task_config['sspace'].lower()

    subs = {'prompt': ori_prompt, 'keywords': pregen_subs['keywords']}
    if 'attack' in task_name:
        subs['pos'] = pregen_subs['sub']
    elif 'improve' in task_name:
        if 'synoym' in sspace:
            subs['pos'] = pregen_subs['sub']
        if 'antonym' in sspace:
            subs['neg'] = pregen_subs['opp']
        if 'nplib' in sspace:
            nplib = load_negative_prompt_library(ori_prompt, length=4)
            if 'neg' in subs:
                subs['neg'].update(nplib['sub'])
            else:
                subs['neg'] = nplib['sub']

        ## add empty string for prompt improvement task
        for _type in subs:
            if _type in ['prompt', 'keywords']: continue
            for ori_word in subs[_type]:
                subs[_type][ori_word].insert(0, "")
    else:
        raise ValueError(f"task_name: {task_name} not supported")


    return subs, ori_prompt


def build_search_space(subs, tokenizer, embeddings):
    ori_prompt = subs['prompt']
    sspace = {}
    if 'pos' in subs:
        sspace['pos'] = positive_space(ori_prompt, subs['pos'], tokenizer, embeddings)
    if 'neg' in subs:
        sspace['neg'] = negative_space(ori_prompt, subs['neg'], tokenizer, embeddings)

    return sspace


def negative_space(ori_prompt, subs, tokenizer, embeddings):
    """ neg opp version, only keep matched words and do not use ori_word """
    """ map json substitute (one prompt) to sub_dict """
    """ will also remove duplicate subs """

    ## get the sub_words list
    all_words = []
    for ori_word in subs:
        sub_words = subs[ori_word]
        effective_sub_words = [w for w in sub_words \
                               if w.lower() != ori_word.lower() and w != '']
        if len(effective_sub_words) <= 0:
            continue

        ## remove words with more than one token
        sub_words = [w for w in sub_words if len(tokenizer(w).input_ids) <= 3]
        sub_ids = [tokenizer(w).input_ids[1] for w in sub_words]
        sub_embeds = [embeddings[id] for id in sub_ids]
        
        all_words.append({
            'ori_word': ori_word,
            'do_sub': True,
            'sub_words': sub_words,
            'sub_ids': sub_ids,
            'embeddings': torch.stack(sub_embeds)
        })

    return {'prompt': ori_prompt, 'all_words': all_words}


def positive_space(ori_prompt, subs, tokenizer, embeddings):
    """ map json substitute (one prompt) to sub_dict """
    """ will also remove duplicate subs """
    import re
    # Split the sentence into words, treating punctuation as separate words
    words = re.findall(r'\w+|[^\w\s]', ori_prompt)

    all_words = []
    for word in words:
        ori_id = tokenizer(word).input_ids[1:-1]
        
        # if the word contains spaces, or is a punctuation, or more than one token, do not sub
        if ' ' in word or word in '.,!?;' or len(ori_id) > 1 or word not in subs:
            sub_words = [word]
            sub_ids = [ori_id]
            sub_embeds = embeddings[ori_id]
            do_sub = False
        else:
            ## got assigned more than 1 tokens, do not substitute
            sub_words = [word] + [w for w in subs[word]
                                  if w.lower() != word.lower() and len(tokenizer(w).input_ids) == 3]
            sub_ids = [tokenizer(w).input_ids[1] for w in sub_words]
            sub_embeds = torch.stack([embeddings[id] for id in sub_ids])
            do_sub = True
        
        all_words.append({
            'ori_word': word,
            'do_sub': do_sub,
            'sub_words': sub_words,
            'sub_ids': sub_ids,
            'embeddings': sub_embeds
        })
    
    return {'prompt': ori_prompt, 'all_words': all_words}


def load_negative_prompt_library(ori_prompt, length, version='v2'):
    """ length: length of negative prompt """
    def load_neg_prompt(path):
        words = []
        with open(path, 'r') as f:
            for line in f.readlines():
                words.append(line.strip())
        return words

    ## load negative prompts
    words_v1 = load_neg_prompt(NEG_PROMPTS_LIBRARY)
    words_v2 = load_neg_prompt(NEG_PROMPTS_LIBRARY_V2)

    ## make substitutes
    substitutes = {
        'sub': {},
        'prompt_id': None,
        'prompt': ori_prompt,
    }
    if 'v1' in version:  ## copy
        for wid in range(length):
            substitutes['sub'][f'NPLib-{wid}'] = copy.deepcopy(words_v1)
    elif 'v2' in version:
        words_splits = np.array_split(words_v2, length)
        for wid, words in enumerate(words_splits):
            substitutes['sub'][f'NPLib-{wid}'] = words.tolist()
    elif 'v3' in version:
        ## load v2
        words_splits = np.array_split(words_v2 + words_v1, len(words_v2 + words_v1 * 3) // 50)
        for wid, words in enumerate(words_splits):
            substitutes['sub'][f'NPLib-{wid}'] = words.tolist()
    return substitutes


def validate_synonyms(substitutes, thresh, constraint_fn):
    """ use bert score to filter wrong synonyms in subs """
    ori_prompt = substitutes['prompt']
    subs = substitutes['pos']

    subs_filtered = {}
    for ori_word in subs: ## each position
        sub_words = subs[ori_word]

        sub_words_filtered = []
        for sub_word in sub_words:
            adv_prompt = ori_prompt.lower().replace(ori_word, sub_word)
            if constraint_fn(adv_prompt) >= thresh:
                sub_words_filtered.append(sub_word)
        
        if len(sub_words_filtered) > 0:
            subs_filtered[ori_word] = sub_words_filtered

    substitutes['pos'] = subs_filtered

    return substitutes

