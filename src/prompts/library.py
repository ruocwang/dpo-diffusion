SYNONYMS_DEFAULT = 'Read the next paragraph. For each word, give 5 substitution words that do not change the meaning. Use the format of "A -> B".\n\n'
SYNONYMS_KEYWORD = 'Read the next paragraph. For each key word, give 5 substitution words that do not change the meaning. Use the format of "A -> B".\n\n'

ANTONYMS_DEFAULT = 'Read the next paragraph. For each word, give 5 opposite words if it has any. Use the format of "A -> B".\n\n'
ANTONYMS_DEFAULT_V2 = 'Read the next paragraph. For each word, give 5 opposite qualities if it has any. Use the format of "A -> B".\n\n'
ANTONYMS_KEYWORD = 'Read the next paragraph. For each key word, give 5 opposite words if it has any. Use the format of "A -> B".\n\n'
ANTONYMS_KEYWORD_V2 = 'Read the next paragraph. For each key word, give 5 opposite qualities if it has any. Use the format of "A -> B".\n\n'

TRIM_DEFAULT = 'I will give you a sentence describing an image, followed by a set of trailing generic aesthetic modifiers irrelevant to the semantices of the image. Output the image description part without any modifiers. Do not return anything else.\n\n'

NON_IMAGE_PROMPT_FILTER_DEFAULT = 'Read the following sentence. Does it describe an image? Answer yes/no'

HUMAN_EVAL_PROTOCOL = 'If you are asked to evaluate how well a picture can be described by the following sentence, list the important key words or concepts from this sentence that you will pay attention to during evaluation, ordered from most to least important. Use comma separated list and do not return anything else.\n\n'

prefix_dict = {
    'synonyms_default': SYNONYMS_DEFAULT,
    'synonyms_keyword': SYNONYMS_KEYWORD,

    'antonyms_default': ANTONYMS_DEFAULT,
    'antonyms_default_v2': ANTONYMS_DEFAULT_V2,
    'antonyms_keyword': ANTONYMS_KEYWORD,
    'antonyms_keyword_v2': ANTONYMS_KEYWORD_V2,

    'human_eval':  HUMAN_EVAL_PROTOCOL,

}