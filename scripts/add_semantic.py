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

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

from sentence_transformers import SentenceTransformer, util

from compute_explanation import get_sentence_token_ids

# pick sentences based on their individual similarity to the individual summary sentence
def pick_sent_pairs_cover(sent_model, tokenizer, text, text_lst, summary, max_pick=3):
    # greedily select sentences so that they contain as much information 
    # about the target sentence as possible
    # specifically, pick sentences in decreasing score order
    # until the combined score does not increase
    
    def _get_sents(text):
        return sent_tokenize(text)
    
    def _greedy_pick_coverage(sentences, score, target_sent, sent_model):
        assert(len(sentences) == score.shape[0])
        sorted_idx = np.argsort(score)[::-1]
        target_emb = sent_model.encode(target_sent)
        out = [sentences[sorted_idx[0]]]
        ss = score[sorted_idx[0]]
        ids = [sorted_idx[0]]
        for s in sorted_idx[1:]:
            tmp_emb = sent_model.encode(' '.join(out + [sentences[s]]))
            score_ = util.cos_sim(tmp_emb, target_emb)
            score_ = score_.flatten()[0]
            if score_ > ss and len(ids) < max_pick:
                out.append(sentences[s]) # add only if adding it increases the score
                ids.append(s)
                ss = score_
            else:
                break
                
        assert(len(ids) <= max_pick)
        return out, ids, ss

    summ_sents = _get_sents(summary)
    summary_embeddings = sent_model.encode(summ_sents, convert_to_tensor=True)
    
    all_texts = [text] + text_lst
    all_texts = [truncate_doc(tt, tokenizer)[0] for tt in all_texts] # truncate the doc to valid length
    assert(len(all_texts) >= 2) # must contain at least one alternate
    text_sents = [_get_sents(t) for t in all_texts]
    selected = []
    for text_sent in text_sents:
        text_embeddings = sent_model.encode(text_sent, convert_to_tensor=True)
        score = util.cos_sim(text_embeddings, summary_embeddings)
        entry = []
        for dd in range(len(summ_sents)):
            sent = summ_sents[dd]
            out, ids, ss = _greedy_pick_coverage(text_sent, score[:,dd].cpu().numpy(), sent, sent_model)
            entry.append((out, ids, ss))
        selected.append(entry)
        
    return selected # list of list of tuples (axis0 across candidate docs, 
                    # axis1 across summary sentences)
    
def truncate_doc(text, tokenizer, max_length=512):
    text_sent, tok_s, tok_e = get_sentence_token_ids(text, tokenizer)

    # get sentences up to 512 token length
    valid = []
    for i, e in enumerate(tok_e):
        if e <= max_length:
            valid.append(i)
    text_sent = [text_sent[i] for i in valid]
    return ' '.join(text_sent), text_sent
    
def remove_punct(s):
    # remove punctuation from the string
    new_string = s.translate(str.maketrans('', '', string.punctuation))
    return new_string

def get_ngram_match(s1, s2):
    # return the matching ngram part with n value maximum
    n = 1
    overlapped = dict()
    while True:
        tmp = []
        ngram_picked = ngrams(s1.split(), n)
        ngram_summary = ngrams(s2.split(), n)
        ngram_picked = [' '.join(x).lower() for x in ngram_picked]
        #ngram_picked = [remove_punct(x) for x in ngram_picked]
        ngram_summary = [' '.join(x).lower() for x in ngram_summary]
        #ngram_summary = [remove_punct(x) for x in ngram_summary]
        #print(ngram_picked, ngram_summary)
        for grams in ngram_picked:
            if grams in ngram_summary:
                if grams != '':
                    tmp.append(grams)
                #print(grams)
        #print('--')
        if len(tmp) == 0:
            break
        else:
            overlapped[n] = tmp
            n += 1
    if n == 1:
        return []
    else:
        # remove overlapped elements and return only the largest ngrams
        over = []
        unigrams = copy.copy(overlapped[1])
        for n in range(len(overlapped), 0, -1):
            target = overlapped[n]
            comps = [unigrams[i:i+n] for i in range(len(unigrams)-n+1)]
            #print(comps)
            for c in comps:
                if ' '.join(c) in target:
                    over.append(' '.join(c))
                    for cc in c:
                        #print(unigrams, cc, over)
                        try:
                            unigrams[unigrams.index(cc)] = ''  
                        except ValueError:
                            pass
        return over

def collect_phrase_info(output, text, text_lst, summary, srl_predictor):
    # need a sematic role predictor
    all_text = [text] + text_lst
    summ_sents = sent_tokenize(summary)
    out = dict()
    out['summary'] = summary
    out['summary_sents'] = summ_sents
    for i, t in enumerate(all_text):
        if i == 0:
            out['doc_correct'] = dict()
            out['doc_correct']['text'] = t
        else:
            out['doc_wrong'] = dict()
            out['doc_wrong']['text'] = t
    for sum_i, sum_sent in enumerate(summ_sents):
        out[sum_i] = dict()
        out[sum_i]['sent'] = sum_sent
        for doc_i in range(len(all_text)):
            if doc_i == 0:
                key = 'correct'
            else:
                key = 'wrong_%d'%doc_i
            picked_sentences, ids, score = output[doc_i][sum_i]
            out[sum_i][key] = dict()
            out[sum_i][key]['sents'] = picked_sentences
            out[sum_i][key]['sent_ids'] = ids 
            out[sum_i][key]['sent_score'] = score
            out[sum_i][key]['phrases'] = []
            for doc_sent in picked_sentences:
                phrases = get_phrase_from_sentence(
                    sum_sent, doc_sent, srl_predictor
                ) 
                
                # filter the phrases
                phrases = filter_trivial_pos(phrases) # filter uselss pos
                phrases = filter_subset_overlaps(phrases) # filter repetivie overlaps
                out[sum_i][key]['phrases'].append(phrases)
                
    # doc info
    for doc_i in range(len(all_text)):

        if doc_i == 0:
            key = 'correct'
        else:
            key = 'wrong_%d'%doc_i
            
        ## Aggregate all information for each doc
        all_sent_ids = []
        all_sent_scores = []
        all_phrases = []
        for i in range(len(out['summary_sents'])): # for each summary sentence
            all_sent_ids.append(out[i][key]['sent_ids'])
            all_sent_scores.append(out[i][key]['sent_score'])
            all_phrases.append(out[i][key]['phrases'])
                
        # add aggregated results for each document
        try:
            out[key]['sent_scores'] = all_sent_scores
        except KeyError:
            out[key] = dict()
            out[key]['sent_scores'] = all_sent_scores
        out[key]['sent_ids_aggregated'] = all_sent_ids
        out[key]['phrases_aggregated'] = all_phrases
    return out

def filter_trivial_pos(lst):
    # filter tirival phrases with specific POS
    out = []
    trivial = ['CC', 'IN', 'TO', 'PRP', 'DT', 'AT', 'DET', ',', '.', ':'] # POS that are trivial
    punct = ['"', "'"]
    for l in lst:
        dd = nltk.pos_tag([l]) # use POS tagger to filter trivial ones
        if dd[0][1] not in trivial and l not in punct:
            out.append(l)
    return out

def filter_subset_overlaps(lst):
    lst.sort(key=lambda s: len(s), reverse=True)
    out = []
    for s in lst:
        if not any([s in o for o in out]):
            out.append(s)
    return out

def get_phrase_from_sentence(sum_sent, doc_sent, srl_predictor):
    # INPUT: single sentence from summary and doc
    
    def get_args(sentence):
        
        def _search_tags(lst, tag='ARG0'):
            # search the list of tags, and return the start, end idx corresponding to the tag
            # return -1 for both if the tag is not found
            start = -1
            end = -1
            count = 0
            for i, l in enumerate(lst):
                if l == 'B-%s'%tag:
                    start = i
                elif l == 'I-%s'%tag:
                    count += 1
            if start != -1:
                end = start + count + 1
            return start, end

        # get args from the semantic role parser
        semantic_role = srl_predictor.predict(sentence)
        words = semantic_role['words']
        verbs = [tt['verb'] for tt in semantic_role['verbs']]
        tags = [tt['tags'] for tt in semantic_role['verbs']]
        arg0 = []
        arg1 = []
        arg2 = []
        arg3 = []
        for tag_list in tags:
            s0, e0 = _search_tags(tag_list, tag='ARG0')
            s1, e1 = _search_tags(tag_list, tag='ARG1')
            s2, e2 = _search_tags(tag_list, tag='ARG2')
            s3, e3 = _search_tags(tag_list, tag='ARGM-TMP')
            arg0.append(' '.join(words[s0:e0]))
            arg1.append(' '.join(words[s1:e1]))
            arg2.append(' '.join(words[s2:e2]))
            arg3.append(' '.join(words[s3:e3]))

        return arg0, arg1, arg2, arg3, verbs, words
    
    summ_arg0, summ_arg1, summ_arg2, summ_arg3, summ_verbs, summ_words = get_args(sum_sent)
    arg0, arg1, arg2, arg3, verbs, words = get_args(doc_sent)
    
    # find the overlapping phrase between the summary sentence and document sentence
    overlaps = []
    summ_args = summ_arg0 + summ_arg1 + summ_arg2 + summ_arg3
    doc_args = arg0 + arg1 + arg2 + arg3
    #print(summ_args, doc_args)
    summ_args = list(set(summ_args))
    doc_args = list(set(doc_args))
    #print(summ_args, doc_args)

    for s0 in summ_args:
        for a0 in doc_args:
            overlap = get_ngram_match(a0, s0)
            if len(overlap) != 0:
                overlaps.append(overlap) 
    #print(overlaps)
    out = []
    # remove duplicates
    for x in overlaps:
        for xx in x:
            if xx != '' and xx not in out:
                out.append(xx)
    return out

def search_and_mark_word_attrs(
    words_sent, 
    words_attr, 
    ids, 
    phrases, 
    value=0):
    # look for phrases in each sentence of specific ids,
    # mark the location with the assigned value
    # return the attribution
    #assert(len(ids) == len(phrases))
    def _search(phrase, word_list):
        #return the word indices from the word_list that matches phrase
        if ' ' not in phrase:
           if '-' in phrase:
               pp = [phrase.split('-')[0], '-', phrase.split('-')[1]]
           else:
               pp = [phrase]
        else:
            #only tokenize if there is a space in the phrase (composed of multiple words)
            pp = break_down_hyphens(nltk.word_tokenize(phrase))

        m = 0
        stack = []
        
        for i, w in enumerate(word_list):
            if w == pp[m]: 
                stack.append((w, i))
                m += 1
            else:
                if len(stack) == 0:
                    pass
                else:
                    stack.pop()
                    m = 0
            if m == len(pp):
                break
        widx = [s[1] for s in stack]
        #print(widx, stack)
        if len(widx) != 0:
            assert(word_list[widx[0]] == stack[0][0] and word_list[widx[-1]] == stack[-1][0])
        return widx
    
    for i, p in zip(ids, phrases):
        words_in_sentence = [x.lower() for x in words_sent[i]]
        
        # search for phrases 
        widx = []
        for phrase in p:
            widx += _search(phrase, words_in_sentence)
        for wi in widx:
            if words_attr[i][wi] > 10: # the word is already colored as the important phrase, so skip
                continue
            elif words_attr[i][wi] > 0 and value < 10: 
                # the word is already contained within a different-colored sentence, so skip
                # (we do not want to overwrite the already-colored sentence to the sentence with new color)
                continue
            words_attr[i][wi] = value
        
    return words_attr

def resolve_color_overlap(bg_level, fg_level):
    assert(len(bg_level) == len(fg_level))
    joint_attrs = []
    for bg, fg in zip(bg_level, fg_level):
        if bg == fg: # if the sentence and phrase has the same color
            joint_attrs.append(2*bg)
        else: # if the sentence and phrase has different color, follow the phrase color
            if fg == 0: # if just the background color
                joint_attrs.append(bg)
            else: # if both background and phrase color, follow phrase color
                joint_attrs.append(2*fg)
    return joint_attrs

# resolve the overlapping colors within the same document
def update_color_overlaps(phrase_info):
    for k in ['correct', 'wrong_1', 'wrong_2']:
        dd = phrase_info[k]
        def check_overlap_id(ref, lst):
            ff = []
            for r in ref: 
                ff += r
            out = []
            removed = []
            for i, l in enumerate(lst):
                if l in ff:
                    removed.append(i)
                else:
                    out.append(l)
            return out, removed

        ordering = np.argsort(dd['sent_scores'])[::-1]
        sent_ids_aggregated = []
        phrases_aggregated = []
        for i, o in enumerate(ordering):
            if len(sent_ids_aggregated) == 0:
                sent_ids_aggregated.append(dd['sent_ids_aggregated'][o])
                phrases_aggregated.append(dd['phrases_aggregated'][o])
            else:
                # check for overlapping
                out, removed = check_overlap_id(sent_ids_aggregated, dd['sent_ids_aggregated'][o])
                sent_ids_aggregated.append(out)
                phrases_aggregated.append([dd['phrases_aggregated'][o][x] for x in range(len(dd['phrases_aggregated'][o])) if x not in removed])
            assert(len(sent_ids_aggregated) == len(phrases_aggregated))

        for x, z in zip(sent_ids_aggregated, phrases_aggregated):
            assert(len(x) == len(z))
        #assert(len(sent_scores) == len(sent_ids_aggregated))
        
        # rearrange to original order
        tmp_s = []
        tmp_p = []
        for i in range(len(ordering)):
            idx = np.where(ordering == i)[0][0]
            tmp_s.append(sent_ids_aggregated[idx])
            tmp_p.append(phrases_aggregated[idx])
        
        dd['sent_ids_aggregated_post'] = tmp_s
        dd['phrases_aggregated_post'] = tmp_p
        
    return phrase_info

def break_down_hyphens(lst):
    out = []
    for i, w in enumerate(lst):
        if w == '-':  # if just the hyphen, add it.
            out.append(w)
        else:
            if '-' in w and '--' not in w: # if hyphen is part of the word, separate it
                ss = w.split('-')
                for s in ss[:-1]:
                    out.append(s)
                    out.append('-')
                out.append(ss[-1])
            else:
                out.append(w)
    return out

def flatten(arr):
    out = []
    for a in arr:
        out += a
    return out, len(out)

# given the list of text, tokenizer, and the phrase_info
def get_phrase_attributions(
    phrase_info, 
    text, 
    text_lst, 
    summary, 
    tokenizer,
    rescale_score=True
):
    all_text = [text] + text_lst
    all_text_raw = [truncate_doc(t, tokenizer)[0] for t in all_text]
    all_text_sents = [truncate_doc(t, tokenizer)[1] for t in all_text]
    #phrase_info = text_exp[idx]['phrase_info']
    
    attr_info = dict()
    
    # summary information
    summ_sents = phrase_info['summary_sents']
    summ_words = [break_down_hyphens(word_tokenize(t)) for t in summ_sents]
    summ_words_flat = flatten(summ_words)[0]
    summ_words_attrs = [[0] * len(x) for x in summ_words]
    summ_ids = list(range(len(summ_sents)))
    tmp = []
    for si, x in enumerate(summ_words_attrs):
        tmp.append([float(si+2)] * len(x))
    summ_words_attrs = tmp

    attr_info['summary'] = {
        'joint_words': summ_words_flat,
        'joint_attrs': None, # needs to be updated later with phrases
    }
    
    def _rescale_scores(scores, max_val=.999, min_val=0.001):
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores)) * (max_val - min_val) + min_val
    
    # keep the attribution data in a dictionary to be updated
    doc_types = ['correct', 'wrong_1', 'wrong_2']
    for i, k in enumerate(doc_types):
        words_sent = [break_down_hyphens(word_tokenize(t)) for t in all_text_sents[i]] # list of list
        words_flat, no_words = flatten(words_sent)
        words_attrs = [[0.0] * len(x) for x in words_sent]

        attr_info[k] = {
            'joint_words' : words_flat,
            'joint_attrs' : words_attrs,
            'words_sent' : words_sent   
        }
    assert(len(words_flat) == len(flatten(words_attrs)[0]))

    for summary_id in range(len(summ_sents)):
        # procedure for a single summary sentence (loop over multiple sents)

        #summary_sent = summ_sents[summary_id]
        #print(summary_sent)
        #summary_words = word_tokenize(summary_sent)
        #print(summary_words)
        color_offset = summary_id

        # aggregate all info across doc for the chosen color (summary_sentence)
        dd = phrase_info[summary_id]
        scores = np.array([dd[k]['sent_score'] for k in ['correct', 'wrong_1', 'wrong_2']]) #scores across the doc
        
        # NOTE: normalize the scores so that max is 1, min is 0 
        # (this is for visualization purposes -- the values will be used for coloring)
        if rescale_score:
            scores = _rescale_scores(scores)
            
        assert(np.all(scores >= 0))

        # for each document, process the attributions
        for i, k in enumerate(doc_types):

            sent_ids = phrase_info[k]['sent_ids_aggregated_post']  # non-overlapped version : use this for the sentence highlights
            phrases = phrase_info[k]['phrases_aggregated_post'] # non-overlappedn version

            sent_ids_ = phrase_info[k]['sent_ids_aggregated']  # overlapped version 
            phrases_ = phrase_info[k]['phrases_aggregated'] # overlappedn version : use this for phrase highlights

            # #print(sent_ids[summary_id], phrases[summary_id])
            # words_sent = [break_down_hyphens(word_tokenize(t)) for t in all_text_sents[i]] # list of list
            # #print(words_sent[1])
            # words_flat, no_words = flatten(words_sent)
            # words_attrs = [[0.0] * len(x) for x in words_sent]
            
            words_attrs = attr_info[k]['joint_attrs']
            words_sent = attr_info[k]['words_sent']

            sent_color = scores[i] + color_offset + 1
            phrase_color = scores[i] + (color_offset + 1) * 10

            ## fill in the attributions

            # color sentences in the doc (assuming there is no overlap)
            #tmp = []
            for si, x in enumerate(words_attrs): #i is sentence id, x is a list of attrs
                if si in sent_ids[summary_id]:
                    words_attrs[si] = [sent_color] * len(x)  #update (replace) the sentence values
                    #tmp.append([sent_color] * len(x))
                #else:
                    #tmp.append(x)
            #assert(len(words_attrs) == len(tmp))
            #words_attrs = tmp

            # color phrases in the doc
            #print(words_attrs)
            words_attrs = search_and_mark_word_attrs(
                words_sent, 
                words_attrs, 
                sent_ids[summary_id], 
                phrases[summary_id], 
                value=phrase_color
            )
            
            # color phrases in the doc that may lie within already-colored sentence but is relevant to a different summary
            words_attrs = search_and_mark_word_attrs(
                words_sent, 
                words_attrs, 
                sent_ids_[summary_id], 
                phrases_[summary_id], 
                value=phrase_color
            )

            # update summary info based on the phrases
            summ_words_attrs = search_and_mark_word_attrs(
                summ_words, 
                summ_words_attrs, 
                [summary_id]*len(phrases[summary_id]),  # format the summary sent ids properly
                phrases[summary_id], 
                value=phrase_color
            )
            
            # update summary info based on the phrases that may lie within already-colored sentence but is relevant to a different part
            summ_words_attrs = search_and_mark_word_attrs(
                summ_words, 
                summ_words_attrs, 
                [summary_id]*len(phrases_[summary_id]), 
                phrases_[summary_id], 
                value=phrase_color
            )

    attr_info['summary']['joint_attrs'] = np.array(flatten(summ_words_attrs)[0])
    for k in doc_types:
        attr_info[k]['joint_attrs'] = np.array(flatten(attr_info[k]['joint_attrs'])[0])
    return attr_info

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
    
    # load the semantic role parser model
    srl_predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
    )
    
    # load the sentence model
    sent_model = SentenceTransformer('all-MiniLM-L6-v2')

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # NOTE Specify the file to load and update with new resuls
    #text_exp = pickle.load(open('text_exp_cnn_dailymail.pkl', 'rb'))
    #ids = text_exp.keys()
    
    if just == '': # run everything from scratch
        print('--runnning from scratch')
        text_exp = {
            **pickle.load(open('data/easy-ans-%s-cnndm.pkl'%fname, 'rb')),
            **pickle.load(open('data/hard-ans-%s-cnndm.pkl'%fname, 'rb')),
        } 
        text_exp_target = text_exp
        ids = text_exp.keys()
    else: # load the existing results, just change the ids specified
        text_exp = {
            **pickle.load(open('data/easy-ans-%s-cnndm.pkl'%fname, 'rb')),
            **pickle.load(open('data/hard-ans-%s-cnndm.pkl'%fname, 'rb')),
        } 
        text_exp_target = pickle.load(open('output/text_exp_semantic.pkl', 'rb'))
        ids = [int(x) for x in just.split(' ')]
        print('--updating specific entries: %s'%ids)
    

    for idx in tqdm(ids):
        text = text_exp[idx]['original_text']
        text_lst = text_exp[idx]['wrong_text']
        summary = text_exp[idx]['correct_summary']

        # first get group of sentences that best covers the content in the summary
        sent_info = pick_sent_pairs_cover(
            sent_model, 
            tokenizer, 
            text, 
            text_lst, 
            summary, 
            max_pick=max_sent_pick
        )

        # then focus on the phrases within the sentences
        phrase_info = collect_phrase_info(
            sent_info, 
            text,
            text_lst,
            summary, 
            srl_predictor
        )
        
        # resolve color overlaps among sentence
        phrase_info = update_color_overlaps(phrase_info)

        # get attributions from the raw phrase info
        phrase_attrs = get_phrase_attributions(
            phrase_info, 
            text,
            text_lst, 
            summary, 
            tokenizer
        )
        
        text_exp_target[idx]['phrase_info'] = phrase_info
        text_exp_target[idx]['phrase_attrs'] = phrase_attrs
        
    # add the results to the files and save
    pickle.dump(text_exp_target, open('output/text_exp_semantic.pkl', 'wb'))
