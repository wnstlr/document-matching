"""
Script for automatically curating set of questions to use for the user study.
"""

## NOTE: this should be run after generating samples for all question entries via generate_samples.py
import numpy as np
import os
import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--v', type=str, default='study_v0')

    args = parser.parse_args()

    OUT_DIR = 'output/samples'
    IN_DIR = 'output/samples'

    # NOTE fill in these: SOURCE FOLDER TO SAMPLE FROM
    fname = 'samples'

    # NOTE define the group of methods to include in the study
    groups = ['phrase', 'none', 'rouge_phrase', 'presum', 'shap']

    # get the ids
    types = ['easy-ans', 'hard-ans']
    tids = dict()
    ids = []
    for t in types:
        dd = pickle.load(open('data/%s-samples-cnndm.pkl'%t, 'rb'))
        tids[t] = list(dd.keys())
        ids += list(dd.keys())

    # NOTE define the output directory name
    out_fname = args.v
    
    out_fdir = os.path.join(OUT_DIR, out_fname) # output file dir
    if not os.path.exists(out_fdir):
        os.mkdir(out_fdir)
        overwrite = 'yes'
    else:
        overwrite = input('output directory already exists. overwrite?')
        if overwrite.lower() not in ['y', 'yes']:
            raise ValueError('overwriting not allowed.')

    folders = [x for x in os.listdir(IN_DIR) if fname in x.split('-')[-1:]] 
    print('Searching files under :', folders)
    answers = dict()
    for f in folders: # for every level
        subdir = os.path.join(IN_DIR, f)
        print('loading from {}'.format(subdir))
        diff = '-'.join(f.split('-')[:2])
        if diff in types:
            text_ids = tids[diff]
            for g in groups: # for each group
                subsubdir = os.path.join(subdir, 'text-score-%s'%g)
                print(subsubdir)
                for i in text_ids:
                    filename = os.path.join(subsubdir, 'id%s.html'%i)
                    targetname = os.path.join(out_fdir, 'id%s-%s.html'%(i, g))
                    os.system("cp %s %s"%(filename, targetname))
                    with open(os.path.join(subsubdir, 'answers.txt'), 'r') as f:
                        for d in f.readlines():
                            x = d.replace('\n', '').split('\t')
                            if int(x[0]) == i:
                                answers[i] = int(x[1])
                
    answer_file = os.path.join(out_fdir, 'answers.txt')
    with open(answer_file, 'w') as f:
        for k, v in tids.items():
            for i in v:
                f.write('%d\t%d\n'%(i, answers[i]))
                
    print('Creating Problem Sets')
    types = ['easy-ans', 'hard-ans']
    no_qs_per_type = [4,12]  # set the number of questions for easy and hard

    tids = dict()
    ids = []
    for t in types:
        dd = pickle.load(open('%s-samples-cnndm.pkl'%t, 'rb'))
        tids[t] = list(dd.keys())
        ids += list(dd.keys())

    num_pset = 5
    pset_entries = dict()

    # get answers
    answers = dict()
    with open(answer_file, 'r') as f:
        raw = f.readlines()
        ids = [int(x.split('\t')[0]) for x in raw]
        ans = [int(x.split('\t')[1].replace('\n', '')) for x in raw]
    for i, a in zip(ids, ans):
        answers[i] = a
        
    for i in range(num_pset):
        pset_entries[i] = dict()
        all_probs = []
        all_probs_type = []
        all_probs_corr = []
        for j, t in enumerate(types):
            noq = no_qs_per_type[j]
            ids = tids[t]
            dd = pickle.load(open('data/%s-samples-cnndm.pkl'%t, 'rb'))
            np.random.seed(i*7)
            selected = list(np.random.choice(ids, size=noq, replace=False))
            pset_entries[i][t] = selected
            all_probs += selected
            all_probs_type += [t] * len(selected)
            all_probs_corr += [dd[x]['correct'][0] * 1 for x in selected]
            
            
        pset_entries[i]['ids'] = all_probs # all ids of the problems
        pset_entries[i]['types'] = all_probs_type # all types of the problems
        pset_entries[i]['ans'] = [answers[x] for x in all_probs] # all answers
        pset_entries[i]['corr'] = all_probs_corr # all score correctness
        
    pickle.dump(pset_entries, open(os.path.join(out_fdir, 'pset_info_main.pkl'), 'wb'))