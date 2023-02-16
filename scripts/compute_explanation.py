from multiprocessing.sharedctypes import Value
from pyrsistent import v
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import torch
import pickle
from datasets import load_dataset
import pickle
from tqdm import tqdm
from lime.lime_text import LimeTextExplainer
import shap
import os, copy

# for post-processing attributions
from flair.data import Sentence
from itertools import groupby
from utils import *
import argparse
import nltk
from operator import itemgetter
from lime.lime_text import IndexedString

###############################################################################
## Helper functions for BART model
###############################################################################

def construct_input_ref_pair(text, summary, tokenizer, ref_token_id, sep_token_id, cls_token_id, device='cpu'):
    tokenized_input = tokenizer([text, summary], 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True, 
                                max_length=512)
    input_ids = tokenized_input['input_ids']
    text_ids = tokenized_input['input_ids'][0]
    summary_ids = tokenized_input['input_ids'][1]

    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids[1:-1]) + [sep_token_id]
    ref_summary_input_ids = [cls_token_id] + [ref_token_id] * len(summary_ids[1:-1]) + [sep_token_id]
    ref_input_ids = np.vstack((ref_input_ids, ref_summary_input_ids))

    return torch.tensor(input_ids, device=device), torch.tensor(ref_input_ids, device=device)

def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)

def construct_whole_embeddings(input_ids, ref_input_ids):
    input_embeddings = model.shared.embeddings(input_ids)
    ref_input_embeddings = model.shared.embeddings(ref_input_ids)
    
    return input_embeddings, ref_input_embeddings

## Similarity computation
def compute_similarity(model, inputs, no_grad=False):
    # run forward pass from the token ids
    cc = torch.nn.CosineSimilarity(dim=0)
    if no_grad:
        with torch.no_grad():
            output = model.model(**inputs).last_hidden_state
    else:
        output = model.model(**inputs).last_hidden_state
        
    output = torch.einsum('ijk,ij->ik', output, inputs['attention_mask'].float())
    output /= torch.sum(inputs['attention_mask'], axis=1)[:, None]
    score = cc(output[0], output[1])
    return score.reshape(1,)

def compute_similarity_from_embeds(model, input_embeds, attention_mask=None):
    # run foward pass from embedding inputs
    cc = torch.nn.CosineSimilarity(dim=0)
    output = model.model(inputs_embeds=input_embeds, 
                         decoder_inputs_embeds=input_embeds,
                         attention_mask=attention_mask).last_hidden_state
    output = torch.einsum('ijk,ij->ik', output, attention_mask.float())
    output /= torch.sum(attention_mask, axis=1)[:, None]
    score = cc(output[0], output[1])
    return score.reshape(1,)

def truncate_doc(text, tokenizer, max_length=512):
    text_sent, tok_s, tok_e = get_sentence_token_ids(text, tokenizer)

    # get sentences up to 512 token length
    valid = []
    for i, e in enumerate(tok_e):
        if e <= max_length:
            valid.append(i)
    text_sent = [text_sent[i] for i in valid]
    return ' '.join(text_sent), text_sent

def break_down_hyphens(lst):
    out = []
    for i, w in enumerate(lst):
        if '-' in w and '--' not in w:
            ss = w.split('-')
            for s in ss[:-1]:
                out.append(s)
                out.append('-')
            out.append(ss[-1])
        else:
            out.append(w)
    return out

def rescale_pos_neg_attrs(arr):
    # rescale positive and negative attributions to be between -1 and 1 
    # while keeping 0 at 0.
    out = []
    if np.where(arr < 0)[0].shape[0] == 0:
        # no negative value
        neg_max = 0
        neg_min = 0
    else:
        neg_max, neg_min = np.max(arr[arr < 0]), np.min(arr[arr < 0])
    
    if np.where(arr > 0)[0].shape[0] == 0:
        # no positive value
        pos_max = 0
        pos_min = 0
    else:
        pos_max, pos_min = np.max(arr[arr > 0]), np.min(arr[arr > 0])

    for a in arr:
        if a < 0:
            out.append( (a - neg_min) / (neg_max - neg_min) -1 ) 
        elif a > 0:
            out.append( (a - pos_min) / (pos_max - pos_min) ) 
        else:
            out.append(a)
    return out

###############################################################################
## POST HOC EXPLANATION METHODS
###############################################################################

def get_random_attributions(tokenized_input, thr=0.2):
    '''
    HEURISTIC RANDOM METHOD
    '''
    output = np.zeros(len(tokenized_input))
    pp = post_process_token_arr(tokenized_input)
    for i, t in enumerate(pp):
        if np.random.random() < thr:
            output[i] = np.random.random() * 2 - 1
    output = output / np.linalg.norm(output)
    return output

def get_input_gradients(model,
                        tokenizer, 
                        text,
                        summary,
                        device='cpu'):
    '''
    INPUT GRADIENT METHOD
    '''
    # input x gradients
    model.zero_grad()
    inputs = tokenizer([text, summary], 
                       return_tensors="pt", 
                       padding=True, 
                       truncation=True, 
                       max_length=512).to('cuda:0')
    text1_token_ids = inputs['input_ids'][0].cpu().numpy()
    summary_token_ids = inputs['input_ids'][1].cpu().numpy()
    
    # get gradients
    d = compute_similarity(model, inputs)
    d.backward()
    gr = model.model.shared.weight.grad.cpu().detach().numpy()

    attr1 = np.zeros((text1_token_ids.shape[0], gr.shape[1]))
    attr2 = np.zeros((summary_token_ids.shape[0], gr.shape[1]))
    for i, token_id in enumerate(text1_token_ids):
        attr1[i,:] = gr[int(token_id),:]
    for i, token_id in enumerate(summary_token_ids):
        attr2[i,:] = gr[int(token_id),:]

    # get word embeddings of the input
    model.zero_grad()
    word_embeddings = model.model.shared(inputs['input_ids']).cpu().detach().numpy()
    
    # input x gradients
    attr1 = np.mean(attr1 * word_embeddings[0], axis=1)
    attr1 /= np.linalg.norm(attr1)
    attr2 = np.mean(attr2 * word_embeddings[1], axis=1)
    attr2 /= np.linalg.norm(attr2)

    del inputs, text1_token_ids, summary_token_ids, word_embeddings
    torch.cuda.empty_cache()
    
    return attr1, attr2

def get_integrated_gradients(model, 
                             tokenizer, 
                             text, 
                             summary, 
                             no_steps=50, 
                             device='cpu'):
    '''
    INTEGRATED GRADIENT METHOD
    '''
    def _linear_interpolate_embedding(ref, inp, no_steps):
        alphas = torch.linspace(0, 1, no_steps+1)
        alphas = alphas[:, None, None]
        delta = inp - ref
        out = ref + alphas * delta
        return out

    # integrated gradients
    cls_token_id = tokenizer.cls_token_id
    ref_token_id = tokenizer.pad_token_id
    sep_token_id = tokenizer.sep_token_id
    _, ref_id = construct_input_ref_pair(text, 
                                                summary, 
                                                tokenizer,
                                                ref_token_id, 
                                                sep_token_id, 
                                                cls_token_id,
                                                device=device)
    
    inputs = tokenizer([text, summary], 
                       return_tensors="pt", 
                       padding=True, 
                       truncation=True, 
                       max_length=512).to('cuda:0')
    
    tmp = model.model.shared(inputs['input_ids']).cpu().detach()
    text_embed = tmp[0]
    summary_embed = tmp[1]
    tmp = model.model.shared(ref_id).cpu().detach()
    ref_text_embed = tmp[0]
    ref_summary_embed = tmp[1]
    text_spectrum = _linear_interpolate_embedding(ref_text_embed, text_embed, no_steps)
    summary_spectrum = _linear_interpolate_embedding(ref_summary_embed, summary_embed, no_steps)
    
    text_grads = 0
    summ_grads = 0
    for i in range(no_steps+1):
        embeds_input = torch.stack((text_spectrum[i], summary_spectrum[i]), 0).to(device)
        embeds_input.requires_grad = True
        model.zero_grad()
        score = compute_similarity_from_embeds(model, embeds_input, attention_mask=inputs['attention_mask'])
        score.backward()
        g = embeds_input.grad.cpu().detach().numpy()
        text_grads += g[0]
        summ_grads += g[1]
        del embeds_input
        
    ig_attr_text = np.mean((text_embed.numpy() - ref_text_embed.numpy()) * text_grads / no_steps, axis=-1)
    ig_attr_text /= np.linalg.norm(ig_attr_text)

    ig_attr_summ = np.mean((summary_embed.numpy() - ref_summary_embed.numpy()) * summ_grads / no_steps, axis=-1)
    ig_attr_summ /= np.linalg.norm(ig_attr_summ)
    
    return ig_attr_text, ig_attr_summ

def get_shap(model, tokenizer, text1, text2, summary, device='cpu'):
    """
    SHAP METHOD
    """
    def _wrapped_forward_function(x):
        sim = torch.nn.CosineSimilarity(dim=0)
        if type(x) == list or isinstance(x, np.ndarray):
            preds = np.zeros(len(x))
            for i, t in enumerate(x):
                tt = tokenizer([t, summary], 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True, 
                                max_length=512).to('cuda:0')
                preds[i] = compute_similarity(model, tt, no_grad=True).cpu().detach().numpy()[0]
        else:
            tt = tokenizer([x, summary], 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True, 
                            max_length=512).to('cuda:0')
            with torch.no_grad():
                out = model.model(**tt).last_hidden_state
                out1 = out[0].mean(axis=0)
                out2 = out[1].mean(axis=0)
            preds = sim(out1, out2).detach().cpu().numpy()
        return preds

    tokenizer.mask_token = '<pad>'
    explainer = shap.Explainer(_wrapped_forward_function, tokenizer)
    exp = explainer([text1] + text2)
    text_attr1 = exp.values[0]
    text_attr2 = exp.values[1:]
    summary_attr = np.zeros(512)
    
    return text_attr1, text_attr2, summary_attr

def get_lime(model, tokenizer, text, summary, device='cpu'):
    """
    LIME METHOD
    """
    def _compute_similarity_batch(model, inputs):
        # compute similiarity from batch of text and one summary
        ll = len(inputs['input_ids'])
        cc = torch.nn.CosineSimilarity(dim=1)
        with torch.no_grad():
            output = model.model(**inputs).last_hidden_state
            text_rep = output[:ll-1].mean(axis=1)
            summary_rep = output[-1].repeat(ll-1, 1, 1).mean(axis=1)
            scores = cc(text_rep, summary_rep)
        return scores
    def _wrapped_predict_fn(texts, batch=False):
        if type(texts) == list: 
            if not batch:
                preds = np.zeros(len(texts))
                for i, t in enumerate(texts):
                    inputs = tokenizer([t, summary],
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=512).to('cuda:0')
                    preds[i] = compute_similarity(model, inputs, no_grad=True).cpu().detach().numpy()[0]
                    del inputs
            else:
                # NOTE Depreciated for long texts: prone to GPU memory issues
                inputs = tokenizer(texts + [summary],
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=512).to('cuda:0')
                preds = _compute_similarity_batch(model, inputs).cpu().detach().numpy()
        else:
            inputs = tokenizer([texts, summary],
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512).to('cuda:0')
            preds = compute_similarity(model, inputs, no_grad=True).cpu().detach().numpy()
            del inputs
        return np.vstack((preds, preds)).T
    
    def _process_lime_output_attrs(html_raw, text):
        # internal function for processing lime outputs
        start, end = html_raw.find('exp.show_raw_text('), html_raw.find('], 1, "')
        dd = html_raw[start+19:end].replace('"', '').replace('[', '').replace(']', '').replace(' ', '').split(',')
        word_score = []
        for i in range(0, len(dd), 3):
            word = dd[i]
            char_start_idx = int(dd[i+1])
            score = float(dd[i+2])
            word_score.append([word, char_start_idx, score])

        # sort based on the char_start_idx info
        word_score = sorted(word_score, key=itemgetter(1))
        words_to_include = [x[0] for x in word_score]
        scores = [x[2] for x in word_score]

        dd = IndexedString(text, bow=False)
        all_words_text = dd.as_list

        curr = 0
        text_attr = np.zeros(len(all_words_text))
        target_word = words_to_include[0]
        for i, w in enumerate(all_words_text):
            if w == target_word:
                #print(target_word)
                text_attr[i] = scores[curr]
                curr += 1
                if curr == len(words_to_include):
                    break
                target_word = words_to_include[curr]
        #assert(np.where(text_attr != 0)[0].shape[0] == len(words_to_include))
        
        return all_words_text, text_attr
    
    class_names = ['scores', 'scores']
    bow = False # whether to use bag of words
    explainer = LimeTextExplainer(class_names=class_names, bow=bow)
    exp = explainer.explain_instance(
        text, 
        _wrapped_predict_fn, 
        num_features=30, 
        num_samples=1000 
    )
    
    # NOTE extract word-score pair from the results in html format
    html_raw = exp.as_html()
    #all_words, word_attrs = _process_lime_output_attrs(html_raw, text)

    if bow: # if using bag of words, all words have the same score
        start, end = html_raw.find('exp.show('), html_raw.find('], 1, exp_div);')
        dd = html_raw[start+10:end].replace('"', '').replace('[', '').replace(']', '').replace(' ', '').split(',')
        word_score = dict()
        for i in range(0, len(dd), 2):
            word_score[dd[i]] = float(dd[i+1])

        # mark the words to highlight according to the score
        all_words_text = break_down_hyphens(nltk.word_tokenize(truncate_doc(text, tokenizer)[0]))
        all_words_summary = break_down_hyphens(nltk.word_tokenize(summary))
        text_attr = np.zeros(len(all_words_text))
        summ_attr = np.zeros(len(all_words_summary))
        words_to_include = list(word_score.keys())
        
        for i, w in enumerate(all_words_text):
            if w in words_to_include:
                text_attr[i] += word_score[w]

    else: # if not using bag of words, words at different location have different scores.
        all_words_text, text_attr = _process_lime_output_attrs(html_raw, truncate_doc(text, tokenizer)[0])
        all_words_summary = break_down_hyphens(nltk.word_tokenize(summary))
        summ_attr = np.zeros(len(all_words_summary))
            
    # normalize the score
    text_attr = np.array(rescale_pos_neg_attrs(text_attr))
    return text_attr, summ_attr

###############################################################################
## POST PROCESSING
###############################################################################

def post_select_attr(attr, top_k=16, scale=0.8):
    # select few top attributes and assign them equal weights
    out = np.zeros(len(attr))
    sorted_idx = np.argsort(attr)
    for i in sorted_idx[-int(top_k/2):]:# positive words
        out[i] = 1
    for i in sorted_idx[:int(top_k/2)]:# negative words
        out[i] = -1
    return out*scale

def get_sentence_token_ids(text, tokenizer, sentences=None):
    # Split the text into tokens by mapping the index of original text to index of tokens
    from nltk.tokenize import sent_tokenize 
    sents = sent_tokenize(text)
    # get start and end index for each sentence from the plain text
    start = []
    for s in sents:
        if len(start) != 0:
            start.append(text[start[-1]:].index(s) + start[-1]) # add sentences sequentially
        else:
            start.append(text.index(s))
    end = [start[i] for i in range(1, len(start))] + [len(text)]
        
    encoded = tokenizer(text, max_length=512, truncation=True) 

    tok_s = []
    tok_e = []
    for s, e in zip(start, end):
        for i, t in enumerate(encoded.tokens()):
            try:
                ids = encoded.token_to_chars(i) # start token
            except TypeError:
                ids = None
            if ids == None:
                continue
            ss = ids.start
            ee = ids.end
            if ss == s:
                tok_s.append(i)
                break
    tok_e = [s for s in tok_s[1:]] + [len(encoded.tokens())]

    assert(len(tok_s) == len(tok_e))
    #assert(len(tok_s) == len(sents))
    
    if sentences is not None: # select starts and ends for specific sentences
        tok_ss = []
        tok_ee = []
        sents = sentences
        start_sel = [text.index(s) for s in sents]
        for s in start_sel:
            for i, t in enumerate(encoded.tokens()):
                try:
                    ids = encoded.token_to_chars(i)
                except TypeError:
                    ids = None
                if ids == None:
                    continue
                ss = ids.start
                ee = ids.end
                if ss == s:
                    tok_ss.append(i)
        tok_ee = [tok_e[tok_s.index(s)] for s in tok_ss]
        tok_s = tok_ss
        tok_e = tok_ee
        assert(len(tok_s) == len(tok_e))
    assert(len([x for x in tok_s if x > 512]) == 0)
    assert(len([x for x in tok_e if x > 512]) == 0)
    return sents, tok_s, tok_e

def post_select_sentences(text, tok_attrs, tokenizer, agg='sq-avg', top_k=3, scale=0.8):
    # select a sentence based on the attribution values
    # mark the ids for end of sentence
    
    sents, sos, eos  = get_sentence_token_ids(text, tokenizer)
    sents_attr = [] 
    
    for start, end in zip(sos, eos):
        # range of a sentence is from start to i+1
        if agg == 'sq-sum':
            attr = np.sum(tok_attrs[start:end] ** 2) 
        elif agg == 'abs-sum':
            attr = np.sum(np.abs(tok_attrs[start:end]))
        elif agg == 'sq-avg':
            attr = np.mean(tok_attrs[start:end] * 1e3 ** 2) 
        elif agg == 'abs-avg':
            attr = np.mean(np.abs(tok_attrs[start:end]) )
        else:
            raise ValueError('agg type not defined')
        sents_attr.append(attr)
    #assert(len(sents_attr) == len(sents))
    
    # sort and select top k    
    out_attr = np.zeros(tok_attrs.shape)
    sent_idx_sorted = np.argsort(sents_attr)[::-1]
    sent_idx_sel = sent_idx_sorted[:top_k] 

    # select tokens that belong to important sentences and score them
    for sidx in sent_idx_sel:
        start = sos[sidx]
        end = eos[sidx]
        out_attr[start:end] = scale
        
    # return the post-processed token attributions
    return out_attr, sents, sent_idx_sorted

def post_select_attr_only_pos(words, 
                              attr, 
                              top_k=5, 
                              scale=0.8, 
                              remove_duplicate=True, 
                              chunking=False, 
                              remove_nonwords=False, 
                              tagger=None):
    
    # select only the positive attribution words, followed by removing non-words and redundant ones
    if tagger is None and remove_nonwords:
        raise ValueError('define the tagger to remove non-words')
    
    # predefine allowed POS tags
    allowed_tags = ['FW', 'JJ', 'JJR', 'CD', 'JJS', 'NN', 'RB', 'RBR', 'RBS', 'NNP', 'NNPS', 'NNS', 'SYM', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    # predefine disallowed verbs
    be_verbs = ['am', 'are', 'is', 'was', 'were', 'be', 'been', 'being', 'have', 'had', 'has', 'do', 'does', 'did', 'doing', 'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would']
    
    assert(len(words) == len(attr))
    out = np.zeros(len(attr)) # same as the number of words
    sorted_idx = np.argsort(attr)[::-1]
    if remove_nonwords:
        allowed = []
        for i, w in enumerate(words):
            s = Sentence(w)
            try:
                tagger.predict(s)
                postag = s.to_dict()['all labels'][0]['value']
            except IndexError:
                print(w)
                postag = 'XX'
            if postag in allowed_tags and w not in be_verbs:  # remove words that are not part of POS allowed or meaningless
                allowed.append(i)
        sorted_idx = [i for i in sorted_idx if i in allowed]
    
    # chunk the index list for consecutive words
    def _group_chunks(sorted_idx):
        seen_chunks = []
        current_chunk = []
        for i, x in enumerate(sorted_idx):
            if len(current_chunk) == 0:
                current_chunk.append(x)
            else:
                if current_chunk[-1] != x:
                    current_chunk.append(x)
            try:
                sorted_chunk = sorted(current_chunk + [sorted_idx[i+1]])
                start = sorted_chunk[0]
                end = sorted_chunk[-1]
                if end - start + 1 != len(sorted_chunk):
                    seen_chunks.append(sorted(current_chunk)) 
                    current_chunk = []
            except IndexError:
                sorted_chunk = sorted(current_chunk)
                seen_chunks.append(sorted_chunk)
                current_chunk = []
        return seen_chunks

    if chunking:  # chunk consecutive highlights into one
        sorted_idx = _group_chunks(sorted_idx) 
    else:
        sorted_idx = [[i] for i in sorted_idx]   
        
    if not remove_duplicate:
        for c in sorted_idx[:top_k]:# top positive words
            for i in c:
                out[i] = 1
    else:
        # remove duplicate words
        seen = []
        count = 0
        for c in sorted_idx:
            if len(c) == 1: # for individual words, check for duplicate
                if words[c[0]] not in seen:
                    seen.append(words[c[0]])
                    out[c[0]] = 1
                    count += 1
            else: # for chunks, don't check for duplicates
                for i in c:
                    out[i] = 1
                count += 1
            if count == top_k:
                break
    return out*scale

# Wrapper function to generat post hoc explanations
def generate_post_hoc_explanations(
    text_exp, 
    model, 
    tokenizer, 
    text1, 
    text2_lst, 
    summary1, 
    text_idx, 
    name='manual', 
    include_word_lvl=True,
    device='cuda:0',
    **kwargs):

    print(' --- %s'%name)
    model.eval()
    model.zero_grad()
    
    text_exp[text_idx]['%s_attr_tok'%name] = dict() # raw attributions on tokens
    text_exp[text_idx]['%s_attr'%name] = dict() # post-processed attributions

    # get tokens
    inp = [text1, summary1] + text2_lst
    tokenized_input = tokenizer(inp, 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True, 
                                max_length=512)
    text1_ids = tokenized_input['input_ids'][0].detach().tolist()
    summary1_ids = tokenized_input['input_ids'][1].detach().tolist()
    text2_ids = [
        tokenized_input['input_ids'][i].detach().tolist() \
        for i in range(2,len(inp))
    ]
    
    all_tokens_text1 = tokenizer.convert_ids_to_tokens(text1_ids)
    all_tokens_sum1 = tokenizer.convert_ids_to_tokens(summary1_ids)
    all_tokens_text2 = [tokenizer.convert_ids_to_tokens(x) for x in text2_ids]
    
    # get words
    all_words_text1, _ = token2word_attr(tokenizer, text1, np.zeros(512))
    all_words_text2 = [token2word_attr(tokenizer, text2, np.zeros(512))[0] for text2 in text2_lst]
    all_words_sum1, _ = token2word_attr(tokenizer, summary1, np.zeros(512))

    if name == 'shap':
        attr_text1, attr_text2, attr_sum1 = get_shap(model, tokenizer, text1, text2_lst, summary1, device=device)

    elif name == 'lime':
        all_words_text1 = break_down_hyphens(nltk.word_tokenize(truncate_doc(text1, tokenizer)[0]))
        all_words_summary = break_down_hyphens(nltk.word_tokenize(summary1))
        all_words_text2 = [break_down_hyphens(nltk.word_tokenize(truncate_doc(t, tokenizer)[0])) for t in text2_lst]
        
        # NOTE for LIME, get the word-level output first then turn it into token-level
        attr_words_text1, attr_words_sum1 = get_lime(
            model, 
            tokenizer, 
            truncate_doc(text1,tokenizer)[0], 
            summary1, 
            device=device)

        attr_words_text2 = [
            get_lime(model, tokenizer, truncate_doc(text2, tokenizer)[0], summary1, device=device)[0] \
                for text2 in text2_lst
        ]
        if include_word_lvl:
            text_exp[text_idx]['lime_attr_words'] = dict() # raw attributions on words
            text_exp[text_idx]['lime_attr_words']['correct'] = [
                (all_words_text1, attr_words_text1),
                (all_words_sum1, attr_words_sum1)
            ]
            text_exp[text_idx]['lime_attr_words']['wrong'] = [
                (w2, attrs) for w2, attrs in zip(all_words_text2, attr_words_text2)
            ]
        
    elif name == 'int_grad':
        attr_text1, attr_sum1 = get_integrated_gradients(model, tokenizer, text1, summary1, device=device)
        attr_text2 = [
            get_integrated_gradients(model, tokenizer, text2, summary1, device=device)[0] \
                for text2 in text2_lst
        ]
        
    elif name == 'input_grad':
        attr_text1, attr_sum1 = get_input_gradients(model, tokenizer, text1, summary1, device=device)
        attr_text2 = [
            get_input_gradients(model, tokenizer, text2, summary1, device=device)[0] \
            for text2 in text2_lst
        ]

    elif name == 'random':
        attr_text1 = get_random_attributions(all_tokens_text1)
        attr_sum1 = get_random_attributions(all_tokens_sum1)
        attr_text2 = [
            get_random_attributions(t2_tok) for t2_tok in all_tokens_text2
        ]
        
    else:
        raise ValueError('unsupported method name')

    # save token-level to dictionary
    if name != 'lime':
        text_exp[text_idx]['%s_attr_tok'%name]['correct'] = [
                (all_tokens_text1, attr_text1),
                (all_tokens_sum1, attr_sum1)
            ]
        text_exp[text_idx]['%s_attr_tok'%name]['wrong'] = [
            (t2_tok, attrs) for t2_tok, attrs in zip(all_tokens_text2, attr_text2)
        ] 
    
    # save word-level info 
    if include_word_lvl and name != 'lime':
        text_exp[text_idx]['%s_attr_word'%name] = dict() # raw attributions on words

        # get word-level attribution from token results
        _, attr_words_text1 = token2word_attr(tokenizer, text1, attr_text1)
        attr_words_text2 = [
            token2word_attr(tokenizer, t2, attr_t2)[1] \
                for t2, attr_t2 in zip(text2_lst, attr_text2)
        ]
        
        _, attr_words_sum1 = token2word_attr(tokenizer, summary1, attr_sum1)
        
        text_exp[text_idx]['%s_attr_word'%name]['correct'] = [
            (all_words_text1, attr_words_text1),
            (all_words_sum1, attr_words_sum1)
        ]
        text_exp[text_idx]['%s_attr_word'%name]['wrong'] = [
            (w2, attrs) for w2, attrs in zip(all_words_text2, attr_words_text2)
        ]
        
    return text_exp

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cnn_dailymail')
    parser.add_argument('--fname', type=str, default='samples')
    parser.add_argument('--model', type=str, default='sshleifer/distilbart-cnn-12-6')
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--ops', type=str, default='scratch')
    parser.add_argument('--method', type=str, default='random')
    parser.add_argument('--just', type=str, default='')

    # NOTE methods to run
    methods = ['random', 'shap', 'input_grad', 'int_grad']

    args = parser.parse_args()
    just = args.just # specify the index to run 
    
    # NOTE specify the indices of texts to compute explanations for
    fname = args.fname
    print(fname)
    dataset_name = args.data
    output_name = 'output/text_exp_bb_%s.pkl'%(fname)
    top_k = args.K
    ops = args.ops
    added_method = args.method

    if dataset_name == 'cnn_dailymail':
        dataset = load_dataset(args.data, "3.0.0")
        model_name = args.model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        text_info = pickle.load(open('%s-cnndm.pkl'%fname, 'rb'))
    else:
        raise ValueError('unsupported dataset')
            
    print(model_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model.zero_grad()

    cls_token_id = tokenizer.cls_token_id
    ref_token_id = tokenizer.pad_token_id
    sep_token_id = tokenizer.sep_token_id
    
    text_idx_lst = text_info.keys()

    if os.path.exists(output_name):
        print('loading from cache')
        text_exp = pickle.load(open(output_name, 'rb'))
        if ops == 'add':  # load what is already there and just replace or add additional methods
            print('---adding/updating method [%s]'%added_method)
            for text_idx in tqdm(text_idx_lst):
                print(text_idx)
                
                text1 = text_exp[text_idx]['original_text']
                text2_lst = text_exp[text_idx]['wrong_text']
                summary1 = text_exp[text_idx]['correct_summary']

                text_exp = generate_post_hoc_explanations(text_exp, model, tokenizer, text1, text2_lst, summary1, text_idx, name=added_method, include_word_lvl=True, device='cuda:0')
            # save after every idx
            pickle.dump(text_exp, open(output_name, 'wb'))

        if just != '': # if there is subset of objects to update
            updates_to_make = [int(x) for x in just.split(' ')]
            print('---updating results for specific entries: %s'%updates_to_make)
            for i in updates_to_make:
                text_exp[i]
    else:
        print('running everything from scratch')
        text_exp = dict()
        for text_idx in tqdm(text_idx_lst):
            print(text_idx)
            text_exp[text_idx] = text_info[text_idx]
            text1 = text_info[text_idx]['original_text'].replace('\n', ' ')
            text2_lst = text_info[text_idx]['wrong_text']
            text2_lst = [x.replace('\n', ' ') for x in text2_lst]
            summary1 = text_info[text_idx]['correct_summary']
            
            if 'score_t1_s1' in text_info[text_idx].keys():
                text_exp[text_idx]['score_t1_s1'] = text_info[text_idx]['score_t1_s1']
                text_exp[text_idx]['score_t2_s1'] = text_info[text_idx]['score_t2_s1']
            else: 
                # if there is no pre-computed score, compute scores
                scores = []
                for t in [text1] + text2_lst:
                    inputs = tokenizer([t, summary1], 
                                        return_tensors="pt", 
                                        padding=True, 
                                        truncation=True, 
                                        max_length=512).to('cuda:0')
                    score = compute_similarity(model, inputs, no_grad=True)
                    scores.append(score.cpu().detach().numpy())
                del inputs
                text_exp[text_idx]['score_t1_s1'] = scores[0]
                text_exp[text_idx]['score_t2_s1'] = scores[1:]
            
            # run methods for token-level / word-level attributions
            for m in methods:
                text_exp = generate_post_hoc_explanations(text_exp, model, tokenizer, text1, text2_lst, summary1, text_idx, name=m, include_word_lvl=True, device='cuda:0')

            # save after every idx
            pickle.dump(text_exp, open(output_name, 'wb'))
            
    # get sentence-level info for post-hoc methods    
    print('post-processing for sentence-level')
    for text_idx in tqdm(text_idx_lst):
        print(text_idx)
        for m in methods:
            print('--%s'%m)
            if m != 'lime':
                attrs = [text_exp[text_idx]['%s_attr_tok'%m]['correct'][0]]
                attrs += text_exp[text_idx]['%s_attr_tok'%m]['wrong']
            else:
                attrs = [text_exp[text_idx]['%s_attr_words'%m]['correct'][0]]
                attrs += text_exp[text_idx]['%s_attr_words'%m]['wrong']
            tok_attrs = [x[1] for x in attrs]
            #import pdb; pdb.set_trace()
            text_exp[text_idx]['%s_attr_sent'%m] = dict()
            texts = [text_exp[text_idx]['original_text']] + text_exp[text_idx]['wrong_text']
            sents = [
                post_select_sentences(tt, ta, tokenizer, agg='sq-avg', top_k=top_k, scale=0.8)\
                    for tt, ta in zip(texts, tok_attrs)
            ]
            text_exp[text_idx]['%s_attr_sent'%m]['correct'] = sents[0]
            text_exp[text_idx]['%s_attr_sent'%m]['wrong'] = sents[1:]

            # get sentence -> word for visualization
            text_exp[text_idx]['%s_attr_sent_viz'%m] = dict()

            word_list, word_attr = token2word_attr(tokenizer, text_exp[text_idx]['original_text'], text_exp[text_idx]['%s_attr_sent'%m]['correct'][0])
            word_attr[word_attr != 0] = 0.8
            word_list2, word_attr2 = [], []
            for i, t in enumerate(text_exp[text_idx]['wrong_text']):
                wl, wa = token2word_attr(tokenizer, t, text_exp[text_idx]['%s_attr_sent'%m]['wrong'][i][0])
                wa[wa != 0] = 0.8
                word_list2.append(wl)
                word_attr2.append(wa)

            text_exp[text_idx]['%s_attr_sent_viz'%m]['correct'] = [(word_list, word_attr)]
            text_exp[text_idx]['%s_attr_sent_viz'%m]['wrong'] = [
                (wl, wa) for wl, wa in zip(word_list2, word_attr2)
            ]
            
    pickle.dump(text_exp, open(output_name, 'wb'))
