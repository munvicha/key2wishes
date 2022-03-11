import config
import torch
import numpy as np
import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rank_bm25 import BM25Okapi

MODEL = T5ForConditionalGeneration.from_pretrained(config.MODEL_PATH)
DEVICE = config.DEVICE
TOKENIZER = T5Tokenizer.from_pretrained(config.MODEL_PATH)

def have_kw(corpus, kw):
    have_kw = []
    for t in corpus:
        tl = t.lower()
        if any(k.lower() in tl for k in kw):
            have_kw.append(t)
    return have_kw

def bm25(query, corpus):
    tokenized_corpus = [doc.split(' ') for doc in corpus]
    bm_25 = BM25Okapi(tokenized_corpus)
    a = bm_25.get_top_n(query, corpus, n=1)
    return a[0]

def keywords_to_input(keywords_list):
    text = str(keywords_list)
    text = text.replace(",", " ")
    text = text.replace("'", "")
    text = text.replace("[", "")
    text = text.replace("]", "")
    return text

def generate(keywords, model=MODEL, tokenizer=TOKENIZER):
    input = 'k2w: ' + keywords_to_input(keywords_list = keywords)
    input = torch.tensor(tokenizer.encode(input)).unsqueeze(0).to(DEVICE)
    sample_outputs = model.generate(input, 
                                    do_sample=True,    
                                    max_length=config.MAX_LEN,
                                    top_k=30,                                 
                                    top_p=0.7,        
                                    temperature=0.6,
                                    no_repeat_ngram_size=2,
                                    repetition_penalty=3.0,
                                    num_return_sequences=25,
                                    early_stopping = True
                                    )
    corpus = []
    for s in sample_outputs:
        text = tokenizer.decode(s, skip_special_tokens=True)
        corpus.append(text)
    corpus_kw = have_kw(corpus = corpus, kw = keywords)
    if len(corpus_kw) == 0:
        a = f'Không thể sinh câu với các từ khóa: {str(keywords)}'
        return a
    else:
        sen = bm25(query = keywords, corpus = corpus_kw)
        return sen
