import torch


class SentenceT5():
    def __init__(self, device):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('sentence-transformers/sentence-t5-base', cache_folder='nfs/data/ruocwang/models', device=device)

    def get_embedding(self, sentences):
        if not isinstance(sentences, list):
            sentences = [sentences]

        embeddings = self.model.encode(sentences)
        embeddings = torch.tensor(embeddings)[0]
        return embeddings
    

def get_constraint_fn(constraint, ori_prompt, device):

    if constraint == 'none':
        constraint_fn = lambda p: 100
    elif constraint == 'cos_t5':
        ref_model = SentenceT5(device)
        constraint_fn = lambda p: torch.nn.functional.cosine_similarity(
            ref_model.get_embedding(ori_prompt), ref_model.get_embedding(p), dim=0).item()

    return constraint_fn
