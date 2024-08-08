"""Use ChatGPT to generate perturbations."""
import json
from openai import OpenAI
import time
import os

from nltk.tokenize import sent_tokenize


with open('./files/openai_key.txt') as file:
    openai_key = file.read()

client = OpenAI(
  api_key=openai_key,  # this is also the default, it can be omitted
)


def get_chatgpt_response(post, model="gpt-3.5-turbo",
                         verbose=False,
                         presence_penalty=0, frequency_penalty=0,
                         num_retries=20, wait=5):
    if verbose:
        print(f'Calling ChatGPT. Input length: {len(post)}')
    while True:
        try:
            ret = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": post}],
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            break
        except Exception as e:
            if num_retries == 0:
                raise RuntimeError
            num_retries -= 1
            print(f'[ERROR] {e}.\nWait for {wait} seconds and retry...')
            time.sleep(wait)
            wait = 20

    return ret


def get_chatgpt_response_content(response):
    assert len(response.choices) == 1
    return response.choices[0].message.content.strip()


def get_perturbations(text, prefix, model="gpt-3.5-turbo", verbose=True):
    results = []

    sentences = sent_tokenize(text)
    count_words = 0
    max_words = 100
    buffer = []

    def _call_lm():
        paragraph = ' '.join(buffer)
        post = prefix + paragraph
        ret = get_chatgpt_response(post, model, verbose=verbose)
        results.append(ret)

    for sent in sentences:
        words = sent.split(' ')
        if count_words > 0 and count_words + len(words) > max_words:
            _call_lm()
            count_words = 0
            buffer = []
        count_words += len(words)
        buffer.append(sent)

    if buffer:
        _call_lm()

    return results


def save_perturbations(results, idx):
    with open(f'results/chatgpt/{idx}.json', 'w') as file:
        file.write(json.dumps(results))


def parse_mapping(line):
    line_ori = line
    line = line.strip()
    if not line:
        return
    if line.startswith('- '):
        line = line[2:].strip()
    tokens = line.split(' ')
    if tokens[0].endswith('.') and (
            tokens[0][0] >= '0' and tokens[0][0] <= '9'):
        # The line begins with "x."
        line = ' '.join(tokens[1:]).strip()
    tokens = line.split(',')
    if not '->' in tokens[0]:
        print(f'Failed to parse: {line_ori}')
        return
    first_tokens = tokens[0].split('->')
    assert len(first_tokens) == 2
    original = first_tokens[0].strip()
    substitution = [first_tokens[1]] + tokens[1:]
    substitution = [item.strip() for item in substitution]
    substitution = [item for item in substitution if ' ' not in item]
    return original, substitution


def extract_content(response):
    assert len(response.choices) == 1
    content = response.choices[0].message.content.strip()
    return content

def parse_response(responses=None, path=None, lower=False):
    mapping = {}
    if responses is None and path is not None:
        with open(path) as f:
            responses = json.loads(f.read())
    for response in responses:
        content = extract_content(response)
        if lower:
            content = content.lower()
        try:
            lines = content.split('\n')
            for line in lines:
                item = parse_mapping(line)
                if item is None:
                    continue
                mapping[item[0]] = item[1]
        except:
            print('==========Failed to parse=========')
            print(content)
            print('==================================')
            print()
    return mapping


# Serializing json
def json_save(obj, file_path):
    json_object = json.dumps(obj, indent=4)
    with open(file_path, "w") as outfile:
        outfile.write(json_object)

def json_load(file_path):
    with open(file_path, "r") as outfile:
        obj = json.load(outfile)
    return obj

def load_save_dict(path):  # prompt -> results
    if os.path.exists(path):
        return json_load(path)
    else:
        return {}

def extract_sub(response):
    sub = parse_response(response, lower=True)
    sub = {key.lower(): value for key, value in sub.items()}
    # item = {'sub': sub, 'prompt_id': None, 'prompt': None}
    return sub
