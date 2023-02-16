from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import numpy as np
from sklearn import metrics
import torch
import torch.nn.functional as F
import pickle
import scipy
from datasets import load_dataset
from scipy import spatial
from tqdm import tqdm
import pickle 
import os

"""
Code used to collect data points from CNN/DailyMail or XSum dataset
Data points consist of (summary, candidate articles) pairs.
"""

# Curated data points will be stored in the directory specified below
DATA_DIR = '../data'

# NOTE Below specify dataset name 
dataset_name = 'cnn'


if dataset_name == 'xsum':
    dataset = load_dataset('xsum')
    model_name = "sshleifer/distilbart-xsum-12-1"
    keyval = 'document'
    keyval_s = 'summary'
elif dataset_name == 'cnn':
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    model_name = "sshleifer/distilbart-cnn-12-6"
    keyval = 'article'
    keyval_s = 'highlights'

if 'pegasus' in model_name: # pegasus model
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
model.zero_grad()

print(dataset_name)

# NOTE Below specify document index range to process
doc_range = range(0, 500)

# First, compute the similarity information, search for the most similar pairs in the dataset
fname = os.path.join(DATA_DIR, 'sum_results_%s_%d_%d.pkl'%(dataset_name, doc_range[0], doc_range[-1]))
if os.path.exists(fname):
    print('loading from existing file')
    similarity_results = pickle.load(open(fname, 'rb'))
else: # batch comparison
    similarity_results = dict()
    for target_idx in tqdm(doc_range):
        print(target_idx)
        with torch.no_grad():
            no_data = len(dataset['test'])
            target_text = dataset['test'][target_idx][keyval].replace('\n', ' ')
            target_token = tokenizer(target_text,return_tensors="pt",
                                            padding=True,
                                            truncation=True,
                                            max_length=512).to(device)
            target_rep = model.model(**target_token).last_hidden_state.cpu().detach().numpy()
            target_rep = target_rep.mean(axis=1).flatten()
            out = []
            for i in tqdm(range(no_data)):
                raw_text = dataset['test'][i][keyval].replace('\n', ' ')
                tokenized_input = tokenizer(raw_text,
                                            return_tensors="pt",
                                            padding=True,
                                            truncation=True,
                                            max_length=512).to(device)
                rep = model.model(**tokenized_input).last_hidden_state.cpu().detach().numpy()
                rep = rep.mean(axis=1).flatten()
                sim = 1 - scipy.spatial.distance.cosine(target_rep, rep)
                out.append(sim)
                del tokenized_input
                torch.cuda.empty_cache()

        sorted_idx = np.argsort(out)[::-1]
        similarity_results[target_idx] = dict()
        similarity_results[target_idx]['sorted_idx'] = sorted_idx
        similarity_results[target_idx]['scores'] = out
        pickle.dump(similarity_results, open(fname, 'wb'))

# save multiple candidates for each postive examples
text_information = dict()
wrong_count = 0
with torch.no_grad():
    for text_idx in tqdm(doc_range):
        text_information[text_idx] = dict()
        original_text = dataset['test'][text_idx][keyval].replace('\n', ' ')
        text_information[text_idx]['original_text'] = original_text
        correct_summary = dataset['test'][text_idx][keyval_s].replace('\n', ' ')
        text_information[text_idx]['correct_summary'] = correct_summary
        # Select the alternate text with the highest similarity by sorting them
        sorted_idx = int(similarity_results[text_idx]['sorted_idx'][1])
        wrong_summary = dataset['test'][sorted_idx][keyval_s].replace('\n', ' ')
        wrong_text = dataset['test'][sorted_idx][keyval].replace('\n', ' ')
        text_information[text_idx]['wrong_summary'] = wrong_summary
        text_information[text_idx]['wrong_text'] = wrong_text

        rep1 = model.model(
            **tokenizer(
                original_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to('cuda:0')
        ).last_hidden_state.cpu().detach().numpy()
        rep2 = model.model(
            **tokenizer(
                wrong_text, 
                return_tensors="pt",
                padding=True, 
                truncation=True, 
                max_length=512
            ).to('cuda:0')
        ).last_hidden_state.cpu().detach().numpy()
        rep_s1 = model.model(
            **tokenizer(
                correct_summary, 
                return_tensors="pt",
                padding=True,
                truncation=True, 
                max_length=512
            ).to('cuda:0')
        ).last_hidden_state.cpu().detach().numpy()
        rep_s2 = model.model(
            **tokenizer(
                wrong_summary, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to('cuda:0')
        ).last_hidden_state.cpu().detach().numpy()
        
        # compute scores from the representations 
        rep1 = rep1.mean(axis=1).flatten()
        rep2 = rep2.mean(axis=1).flatten()
        rep_s1 = rep_s1.mean(axis=1).flatten()
        rep_s2 = rep_s2.mean(axis=1).flatten()
        t1s1 = 1 - spatial.distance.cosine(rep1, rep_s1)
        t2s1 = 1 - spatial.distance.cosine(rep2, rep_s1)
        t1s2 = 1 - spatial.distance.cosine(rep1, rep_s2)
        t2s2 = 1 - spatial.distance.cosine(rep2, rep_s2)
        text_information[text_idx]['score_t1_s1'] = t1s1
        text_information[text_idx]['score_t1_s2'] = t1s2 
        text_information[text_idx]['score_t2_s1'] = t2s1
        text_information[text_idx]['score_t2_s2'] = t2s2

        # obtain correctness information
        if t1s1 < t2s1:
            wrong_count += 1
            text_information[text_idx]['correct'] = False
        else:
            text_information[text_idx]['correct'] = True

        torch.cuda.empty_cache()
    
    print(wrong_count)
    pickle.dump(text_information,
                open(os.path.join(DATA_DIR, 'text_info_%s_%d_%d.pkl'%(dataset_name, doc_range[0], doc_range[-1]), 'wb')))
