import os, pickle

'''
Get outputs from BERTSum, collect and organize the selected sentneces 
into a single text file, named with their question id. 
'''

RESULT_DIR = '../results/cnndm' # directory to save the outputs
delim = '<q>'

# clean up the bertsum output and add to the results
fname = 'text_exp_samples_cnn_dailymail_distill.pkl'
text_exp = pickle.load(open(fname, 'rb'))
idx = list(text_exp.keys())
files = sorted(os.listdir(RESULT_DIR))
print(files)

for i in idx:
    summary = text_exp[i]['correct_summary']
    cand_files = [ff for ff in files if (int(ff.split('_')[0]) == i) and (ff.split('.')[-1]=='candidate')]
    print(cand_files)
    cands = []
    for c in cand_files:
        print(c)
        with open(os.path.join(RESULT_DIR, c), 'r') as f:
            cands.append('\n'.join(f.readlines()[0].split(delim)))
    
    with open(os.path.join(RESULT_DIR, '%d.txt'%i), 'w') as f:
        f.write(summary + '\n--\n')
        for t in cands:
            f.write(t + '\n--\n')
