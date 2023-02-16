import pickle
import os
import numpy as np
import argparse
from transformers import AutoTokenizer
from utils import token2word_attr
import copy
import tqdm

from nltk import sent_tokenize

from compute_explanation import get_sentence_token_ids

'''
Convert the extracted sentences from BERTSum into the metadata format we have for visualization.
'''

# NOTE specify the directory where the BERTSUm results are saved
BERTSUM_RESULT_DIR = '' 

def truncate_doc(text, tokenizer):
    text_sent, tok_s, tok_e = get_sentence_token_ids(text, tokenizer)

    # get sentences up to 512 token length
    valid = []
    for i, e in enumerate(tok_e):
        if e <= 512:
            valid.append(i)
    text_sent = [text_sent[i] for i in valid]
    return ' '.join(text_sent), text_sent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='samples')
    parser.add_argument('--model', type=str, default='sshleifer/distilbart-cnn-12-6')

    args = parser.parse_args()
    
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    fname = args.fname
    level = fname.split('-')[0] # samples, easy, hard
    ptype = fname.split()

    # load and overwrite (add a dictionary element)
    print(fname)
    #target = 'text_exp_%s_cnn_dailymail.pkl'%fname
    #text_exp = pickle.load(open(target, 'rb'))
    
    text_exp = {
        **pickle.load(open('data/easy-ans-%s-cnndm.pkl'%fname, 'rb')),
        **pickle.load(open('data/hard-ans-%s-cnndm.pkl'%fname, 'rb')),
    } 
    
    for level in ['easy-ans','hard-ans']:
        print('processing [%s]'%level)
        #posthoc_out = pickle.load(open('text_exp_%s-samples_cnn_dailymail.pkl'%level, 'rb'))
        dir_name = level + '-cnndm'
        BERTSUM_RESULT_DIR = BERTSUM_RESULT_DIR + dir_name
        print(BERTSUM_RESULT_DIR)
        files = sorted(os.listdir(BERTSUM_RESULT_DIR))
        assert(len(files) != 0)
        files = [x for x in files if '_' in x]
        ids = [x for x in text_exp.keys() if text_exp[x]['ptype'] == level]
        
        for idx in tqdm.tqdm(ids):
            #print(idx)
            # get the sentneces for each candidate
            cand_files = [ff for ff in files if (int(ff.split('_')[0]) == idx) and (ff.split('.')[-1]=='candidate')]
            sent_out = []
            for c in cand_files:
                with open(os.path.join(BERTSUM_RESULT_DIR, c), 'r') as f:
                    sent_out.append([x.replace('\n', '') for x in f.readlines()[0].split('<q>')])
                    
            # add results to the file 
            text1 = text_exp[idx]['original_text']
            text2 = text_exp[idx]['wrong_text']
            all_texts = [text1] + text2
            
            # truncate the document
            all_texts = [truncate_doc(t, tokenizer)[0] for t in all_texts]
            
            # select sentences that are highlighted 
            sent_ids = []
            sents = []
            sent_attrs = []
            for i, curr_text in enumerate(all_texts):
                text_sents = sent_tokenize(curr_text)
                sents.append(text_sents)
                sent_attr = np.zeros(len(text_sents))

                sents_to_search = sent_out[i]
                tmp = []
                for sid, s in enumerate(text_sents):
                    if s in sents_to_search:
                        tmp.append(sid)
                        sent_attr[sid] = 1.0
                if len(tmp) != 3:
                    import pdb
                    pdb.set_trace()
                    assert(False)
                    
                sent_ids.append(tmp)
                sent_attrs.append(sent_attr)
            
            text_exp[idx]['presum_attr_sent_viz'] = dict()
            text_exp[idx]['presum_attr_sent_viz']['correct'] = [(sents[0], sent_attrs[0])]
            text_exp[idx]['presum_attr_sent_viz']['wrong'] = [
                (wl, wa) for wl, wa in zip(sents[1:], sent_attrs[1:])
            ]
            
    pickle.dump(text_exp, open('text_exp_presum_only_main.pkl', 'wb'))