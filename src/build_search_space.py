"""
prompt -> subtitutes using either
chatgpt WEB (manual)
chatgpt API
"""
import argparse
from utils.exp_utils import json_save
from prompts import load_prompts, prefix_dict, get_perturbations, extract_sub, get_chatgpt_response_content


def remove_empty_sub(sub):
    new_sub = {}
    for ori_word in sub:
        if len(sub[ori_word]) > 0:
            new_sub[ori_word] = sub[ori_word]
    return new_sub


def validate_results(sub, args):
    ## empty sub
    if len(sub) == 0:
        return False
    
    ## too many single word sub (may not be a negative point)
    cnt = 0
    for ori_word in sub:
        sub_words = sub[ori_word]
        if len(sub_words) == 1:
            cnt += 1
    if cnt / len(sub) > 0.4:
        return False

    ## wrong format (not parsed well)
    return True


def build_search_space(prompts, chatgpt_prefix, model, save_path, space_name):
    save_dict = {}
    prompt_id = 0
    for prompt in prompts:
        ## query chatgpt for initial results
        max_query = 5
        while max_query > 0:
            valid = True
            ## keywords for human evaluation
            response = get_perturbations(prompt, chatgpt_prefix['keyword'], model=model)
            keywords = get_chatgpt_response_content(response[0])
            keywords = keywords.replace(', ', ',').split(',')

            ## synonyms
            response = get_perturbations(prompt, chatgpt_prefix['synonym'], model=model)
            synonym = extract_sub(response)
            synonym = remove_empty_sub(synonym)
            valid = valid and validate_results(synonym, args)

            ## antonyms
            antonym = {}
            if space_name.lower() == 'antonym':
                response = get_perturbations(prompt, chatgpt_prefix['antonym'], model=model)
                antonym = extract_sub(response)
                antonym = remove_empty_sub(antonym)
                valid = valid and validate_results(antonym, args)

            if valid:
                save_dict[prompt] = {
                    'synonym': synonym,
                    'antonym': antonym,
                    'keywords': keywords,
                    'prompt_id': prompt_id,
                    'prompt': prompt
                }
                prompt_id += 1
                break
            else:
                max_query -= 1

        ## failed to generate valid subtitutes for this prompt
        if prompt not in save_dict:
            print(f'Failed to generate sub for prompt: {prompt}')

        ## active saving
        print(f'Saving to {save_path}')
        json_save(save_dict, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--prompt_path", default="none", type=str)
    parser.add_argument("--model", default="gpt-4", type=str)
    parser.add_argument("--space_name", default="antonym", type=str, choices=['antonym', 'synonym'])
    parser.add_argument("--mode", default="default", type=str, choices=['default', 'keyword'])
    parser.add_argument("--tag", default="none", type=str)
    args = parser.parse_args()

    chatgpt_prefix = {
        'synonym': prefix_dict[f'synonyms_{args.mode}'],
        'antonym': prefix_dict[f'antonyms_{args.mode}'],
        'keyword': prefix_dict['human_eval']
    }

    save_path = args.prompt_path.replace('.json', f'-{args.space_name}.json')
    if args.model != 'gpt-3.5-turbo':
        save_path = save_path.replace('.json', f'-{args.model}.json')
    if args.mode != 'default':
        save_path = save_path.replace('.json', f'-{args.mode}.json')
    save_path = save_path.replace('.json', '.json')

    prompts = load_prompts(args.prompt_path)

    build_search_space(prompts, chatgpt_prefix, args.model, save_path, args.space_name)
