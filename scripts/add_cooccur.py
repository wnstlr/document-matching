import pickle
import os
import numpy as np
import argparse
from transformers import AutoTokenizer
from utils import token2word_attr
import copy
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize, ngrams

from rouge_score import rouge_scorer
from add_semantic import *

# pick sentences based on their individual similarity to the individual summary sentence
def pick_sent_rouge(tokenizer, text, text_lst, summary, top_k=1):
    def _get_sents(text):
        return sent_tokenize(text)
    
    def _compute_rouge_pairs(selected_text, target_text, measure=['rougeL']): # list of sentences, string of target text
        scorer = rouge_scorer.RougeScorer(measure, use_stemmer=True)
        scores = scorer.score(selected_text, target_text)
        return scores, measure
    
    def _greedy_pick(sentences, target_text, top_k=top_k):
        out = []
        for s in sentences:
            scores, m = _compute_rouge_pairs(s, target_text)
            out.append(scores[m[0]].fmeasure) 
        out = np.array(out)
        sorted_idx = np.argsort(out)[::-1]
        return sorted_idx[:top_k], out[sorted_idx[:top_k]]
        
    summ_sents = _get_sents(summary)
    all_texts = [text] + text_lst
    # truncate the documents
    all_texts = [truncate_doc(tt, tokenizer)[0] for tt in all_texts] # truncate the doc to valid length
    assert(len(all_texts) >= 2) # must contain at least one alternate
    text_sents = [_get_sents(t) for t in all_texts]
    
    selected = []
    for t in text_sents:
        tmp = []
        for s in summ_sents:
            selected_id, score = _greedy_pick(t, s, top_k=top_k)
            sents = [t[x] for x in selected_id]
            tmp.append((selected_id, score, sents))
        selected.append(tmp)
        
    return selected # list of list of tuples (axis0 across candidate docs, 
                    # axis1 across summary sentences)
    
def get_phrase_info_rouge(selected, summary):
    # obtain overlapping phrases from the document, return the phrase_info in the same format
    phrase_info = {
        'summary': summary,
        'summary_sents': sent_tokenize(summary),
        'correct': dict(),
        'wrong_1': dict(), 
        'wrong_2': dict(),
        0: {
            'sent': None,
            'correct': {
                'sent_ids' : None,
                'sent_score': None,
                'phrases' : None
            },
            'wrong_1': {
                'sent_ids' : None,
                'sent_score': None,
                'phrases' : None
            },
            'wrong_2': {
                'sent_ids' : None,
                'sent_score': None,
                'phrases' : None
            }
        },
        1: {
            'sent': None,
            'correct': {
                'sent_ids' : None,
                'sent_score': None,
                'phrases' : None
            },
            'wrong_1': {
                'sent_ids' : None,
                'sent_score': None,
                'phrases' : None
            },
            'wrong_2': {
                'sent_ids' : None,
                'sent_score': None,
                'phrases' : None
            }
        },
        2: {
            'sent': None,
            'correct': {
                'sent_ids' : None,
                'sent_score': None,
                'phrases' : None
            },
            'wrong_1': {
                'sent_ids' : None,
                'sent_score': None,
                'phrases' : None
            },
            'wrong_2': {
                'sent_ids' : None,
                'sent_score': None,
                'phrases' : None
            }
        }
    } 
    
    # merge info for each doc 
    for i, docs in enumerate(['correct', 'wrong_1', 'wrong_2']):
        score_lst = []
        sent_ids_aggregated = []
        phrases_aggregated = []
        for summ_i in range(len(phrase_info['summary_sents'])):
            sent_ids, scores, sents = selected[i][summ_i]
            phrase_info[summ_i]['sent'] = phrase_info['summary_sents'][summ_i]
            phrase_info[summ_i][docs]['sent'] = sents
            phrase_info[summ_i][docs]['sent_ids'] = sent_ids
            phrase_info[summ_i][docs]['sent_score'] = scores[0]
            phrase_info[summ_i][docs]['phrases'] = []
            for j, si in enumerate(sent_ids):
                overlaps = get_ngram_match(phrase_info['summary_sents'][summ_i], sents[j])
                overlaps = filter_trivial_pos(overlaps) # filter uselss pos
                overlaps = filter_subset_overlaps(overlaps) # filter repetivie overlaps
                phrase_info[summ_i][docs]['phrases'].append(overlaps)
            
            score_lst.append(scores[0])
            sent_ids_aggregated.append(list(sent_ids))
            phrases_aggregated.append(phrase_info[summ_i][docs]['phrases'])
            
        phrase_info[docs]['sent_scores'] = score_lst
        phrase_info[docs]['sent_ids_aggregated'] = sent_ids_aggregated
        phrase_info[docs]['phrases_aggregated'] = phrases_aggregated
    
    return phrase_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='samples')
    parser.add_argument('--model', type=str, default='sshleifer/distilbart-cnn-12-6')
    parser.add_argument('--no_sent', type=int, default=3)
    parser.add_argument('--just', type=str, default='')

    args = parser.parse_args()
    model_name = args.model
    fname = args.fname
    just = args.just
    # maximum sentence to pick per summary sentence
    max_sent_pick = args.no_sent

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    #text_exp = pickle.load(open('text_exp_cnn_dailymail.pkl', 'rb'))
    if just == '': # run everything from scratch
        print('--runnning from scratch')
        text_exp = {
            **pickle.load(open('data/easy-ans-%s-cnndm.pkl'%fname, 'rb')),
            **pickle.load(open('data/hard-ans-%s-cnndm.pkl'%fname, 'rb')),
        } 
        text_exp_target = text_exp
        ids = text_exp.keys()
    else: # load the existing results, just change the explanations of ids specified
        text_exp = {
            **pickle.load(open('data/easy-ans-%s-cnndm.pkl'%fname, 'rb')),
            **pickle.load(open('data/hard-ans-%s-cnndm.pkl'%fname, 'rb')),
        } 
        text_exp_target = pickle.load(open('output/text_exp_cooccur.pkl', 'rb'))
        ids = [int(x) for x in just.split(' ')]
        print('--updating specific entries: %s'%ids)

    for idx in tqdm(ids):
        print(idx)
        text = text_exp[idx]['original_text']
        text_lst = text_exp[idx]['wrong_text']
        summary = text_exp[idx]['correct_summary']
        
        sent_info = pick_sent_rouge(tokenizer, text, text_lst, summary, top_k=max_sent_pick)
        phrase_info = get_phrase_info_rouge(sent_info, summary)
        phrase_info = update_color_overlaps(phrase_info)
        phrase_attrs = get_phrase_attributions(phrase_info, text, text_lst, summary, tokenizer)
        
        text_exp_target[idx]['rouge_phrase_info'] = phrase_info
        text_exp_target[idx]['rouge_phrase_attrs'] = phrase_attrs
        
    # add the results to the files and save
    pickle.dump(text_exp_target, open('output/text_exp_cooccur.pkl', 'wb'))