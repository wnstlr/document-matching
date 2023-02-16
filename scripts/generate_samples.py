import os
import pickle

from utils import *

import argparse
import tqdm

"""
Geenerates HTML files with or without highlights, used for the user study.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='samples')

    args = parser.parse_args()
    
    # generate samples and save them for each group, along with answers

    # NOTE specify the directory names here before running
    # NOTE name this the same as the text_exp file
    fname = args.fname

    print('Generating samples for [%s]' % fname.upper())
    sample_dir = 'output/samples/%s'%fname
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
        overwrite = 'y'
    else:
        overwrite = input("overwriting the current dir. ok? ")
    
    if overwrite not in ['y', 'yes', 'Yes', 'Y']:
        raise ValueError('overwriting not allowed.')
    
    nota = 'nota' in fname  # none of the above flag
        
    # types of questions        
    types = ['easy-ans', 'hard-ans']

    # NOTE conditions for dividing up the groups (show_exp, show_score, exp_type)
    if fname in [f + '-samples' for f in types]:
    #if fname == 'samples':
    #if fname == 'samples-test' or fname == 'easy-samples-test' or fname == 'hard-samples-test': 
        exp_types = ['none', 'shap', 'presum', 'phrase', 'rouge_phrase']
        sub_dirs = ['text-score-%s'%s for s in exp_types]
        control_sets = [(True, True) if s != 'none' else (False, True) for s in exp_types] 
        show_legend = False
    else:
        raise ValueError('undefined file name')

    assert(len(control_sets) == len(sub_dirs))
    assert(len(exp_types) == len(sub_dirs))

    # NOTE specify the file containing explanation outputs 
    text_exp = pickle.load(open('explanations-all-main.pkl', 'rb'))
    idx_list = text_exp.keys()

    for cc, ss, ee in zip(control_sets, sub_dirs, exp_types):
        print(cc, ss, ee)
        curr_dir = os.path.join(sample_dir, ss)
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)
        answers = []
        for idx in tqdm.tqdm(idx_list):
            # create samples
            out, answer = generate_question(text_exp, 
                                            idx,
                                            show_exp=cc[0],
                                            show_score=cc[1],
                                            exp_type=ee,
                                            shuffle=True,
                                            none_of_above=nota,
                                            legend=show_legend)
            
            # save html base file
            with open(os.path.join(curr_dir, "id%d.html"%(idx)), "w") as f:
                f.write(out.data)
                
            # save ground-truth answers
            answers.append((idx, answer))
        with open(os.path.join(curr_dir, "answers.txt"), 'w') as f:
            for i, a in answers:
                f.write('%d\t%d\n'%(i, a))