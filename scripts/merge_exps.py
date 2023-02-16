import pickle
import numpy as np

# merge the results from compute_explanation
fname = 'samples' 

# post hoc
posthoc_exp = {
    **pickle.load(open('output/text_exp_bb_easy-ans-%s.pkl'%fname, 'rb')),
    **pickle.load(open('output/text_exp_bb_hard-ans-%s.pkl'%fname, 'rb')),
}

# bertsum
bertsum_exp = pickle.load(open('output/text_exp_bertsum.pkl', 'rb'))

# semantic 
semantic_exp = pickle.load(open('output/text_exp_semantic.pkl', 'rb'))

# rouge
rouge_exp = pickle.load(open('output/text_exp_cooccur.pkl', 'rb'))

# merge everything to the post-hoc file
for idx in posthoc_exp.keys():
    posthoc_exp[idx] = {
        **posthoc_exp[idx],
        **bertsum_exp[idx],
        **rouge_exp[idx],
        **semantic_exp[idx]
    }
    
pickle.dump(posthoc_exp, open('output/explanations-all.pkl', 'wb'))