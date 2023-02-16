import os
import pickle
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

'''
Format the summary article pairs in the modified dataset
so that they can be fed to the BERTSum model.
'''

def get_sentence_token_ids(text, tokenizer, sentences=None):
    # Split the text into tokens by mapping the index of original text to index of tokens
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
                ids = encoded.token_to_chars(i)
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
    fnames = [
        'easy-ans-samples-cnndm.pkl',
        'hard-ans-samples-cnndm.pkl',
    ]

    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    for fname in fnames:
        print('-- processing file: {}'.format(fname))
        output_dir = os.path.join('..', fname.split('.')[0])
        print('---> saving output in : {}'.format(output_dir))
        
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        text_info = pickle.load(open(fname, 'rb'))
        idx = list(text_info.keys())
        for i in idx:
            s0 = text_info[i]['original_text']
            s1 = text_info[i]['wrong_text'][0]
            s2 = text_info[i]['wrong_text'][1]
            
            # truncate document to match the max token length
            s0, s0_sents = truncate_doc(s0, tokenizer)
            s1, s1_sents = truncate_doc(s1, tokenizer)
            s2, s2_sents = truncate_doc(s2, tokenizer)

            ## preprocess for BERTSum input format

            # split into sentences with specific format required by the method
            # sentence split by specific tokens
            par = lambda s : ' [CLS] [SEP] '.join(s)
            s0 = par(s0_sents)
            s1 = par(s1_sents)
            s2 = par(s2_sents)

            # write each document a a file (labeled with 0-2)
            ff = 'sample-%d_0.txt'%i
            with open(os.path.join(output_dir, ff), 'w') as f:
                f.write(s0)
            ff = 'sample-%d_1.txt'%i
            with open(os.path.join(output_dir, ff), 'w') as f:
                f.write(s1)
            ff = 'sample-%d_2.txt'%i
            with open(os.path.join(output_dir, ff), 'w') as f:
                f.write(s2)
