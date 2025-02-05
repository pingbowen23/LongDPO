# from utils.zhipu_embedding import ZhipuEmbeddings
from transformers import AutoTokenizer
import torch
import re
from copy import deepcopy
# from nltk.tokenize import PunktSentenceTokenizer
import randomß

def text_split_by_punctuation(original_text, return_dict=False):
    # text = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', original_text)  # separate period without space
    text = original_text
    custom_sent_tokenizer = PunktSentenceTokenizer(text)
    punctuations = r"([。；！？])"  # For Chinese support

    separated = custom_sent_tokenizer.tokenize(text)
    separated = sum([re.split(punctuations, s) for s in separated], [])
    for i in range(1, len(separated)):
        if re.match(punctuations, separated[i]):
            separated[i-1] += separated[i]
            separated[i] = ''

    separated = [s for s in separated if s != ""]
    if len(separated) == 1:
        separated = original_text.split('\n\n')
    separated = [s.strip() for s in separated if s.strip() != ""]
    if not return_dict:
        return separated
    else:
        pos = 0
        res = []
        for i, sent in enumerate(separated):
            st = original_text.find(sent, pos)
            assert st != -1, sent
            ed = st + len(sent)
            res.append(
                {
                    'c_idx': i,
                    'content': sent,
                    'start_idx': st,
                    'end_idx': ed,
                }
            )
            pos = ed
        return res

def text_split(content, chunk_size=128, overlap=0, return_token_ids=False):
    texts = []
    chunk_size -= 2*overlap
    tokenized_content = tokenizer.encode(content, add_special_tokens=False)
    num_tokens = len(tokenized_content)
    for i in range(0, len(tokenized_content), chunk_size):
        start_idx, end_idx = max(0, i-overlap), min(i+chunk_size+overlap, len(tokenized_content))
        split_content = tokenizer.decode(tokenized_content[start_idx:end_idx])
        texts.append(
            {
                'c_idx': len(texts),
                'content': split_content,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'total_token': num_tokens,
                'token_ids': tokenized_content[start_idx:end_idx] if return_token_ids else None,
            }
        )
    return texts

def cat_chunks(chunks, remove_head_tail=0):
    token_ids = sum([c['token_ids'] for c in chunks], [])
    if remove_head_tail > 0:
        token_ids = token_ids[remove_head_tail:-remove_head_tail]
    return tokenizer.decode(token_ids, add_special_tokens=False)
            

def batch_embed(texts, rank=0):
    if isinstance(texts, dict) and 'embed' in texts:
        return texts
    assert isinstance(texts, list)
    embeddings = ZhipuEmbeddings(
        url="https://open.bigmodel.cn/api/paas/v4/embeddings",
        embedding_proc=8,
        embedding_batch_size=8, 
    )  
    if isinstance(texts[0], str):
        embed = embeddings.embed_documents(texts)
    elif isinstance(texts[0], dict):
        embed = embeddings.embed_documents([x['content'] for x in texts])
    else:
        raise NotImplementedError
    try:
        embed = torch.tensor(embed, device=rank, dtype=torch.bfloat16)
    except:
        embed = torch.tensor(embed, device='cpu', dtype=torch.bfloat16)
    return {
        'docs': texts,
        'embed': embed,
    }

def mean_pooling(hidden, attention_mask):
    s = torch.sum(hidden * attention_mask.unsqueeze(-1).float(), dim=1)
    d = attention_mask.sum(dim=1, keepdim=True).float()
    reps = s / d
    return reps

@torch.inference_mode()
def encode(input_texts,model):
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt', return_attention_mask=True).to("cuda")
    
    outputs = model(**batch_dict)
    attention_mask = batch_dict["attention_mask"]
    hidden = outputs.last_hidden_state

    reps = mean_pooling(hidden, attention_mask)   
    embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
    return embeddings

def get_embedding(texts,model):
    INSTRUCTION = "Query: "
    queries = [INSTRUCTION + query for query in texts]
    embeddings = encode(queries,model)
    return embeddings


@torch.inference_mode()
def batch_search(queries, contexts, k=20):
    if isinstance(queries, str):
        queries = [queries]
    rank = random.choice([0,1,2,3,4,5,6,7])
    c_res = batch_embed(contexts, rank)
    q_res = batch_embed(queries, rank)
    c, q = c_res['embed'], q_res['embed']
    if c.device != q.device:
        c, q = c.cpu(), q.cpu()
    c = c / c.norm(dim=1, keepdim=True)
    q = q / q.norm(dim=1, keepdim=True)
    score = q @ c.T
    idxs = torch.argsort(score, dim=-1, descending=True)
    res = []
    for i in range(len(idxs)):
        chunks = []
        for j, idx in enumerate(idxs[i][:k]):
            doc = deepcopy(c_res['docs'][idx])
            chunks.append(doc)
        res.append({
            'q_idx': i,
            'query': queries[i],
            'retrieve_results': chunks
        })
    return res