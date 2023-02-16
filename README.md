# Assisting Human Decisions in Document Matching

This is a repository of code used in paper Assisting Human Decisions in Document Matching (link TBD).

## Base Dataset

The data points used for the task is sampled from [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail).

### Preprocessing 

`python collect_data.py` to collect (summary, candidate articles) pairs where the articles are picked according to their similarity to the summary. The collected information will be saved in the `data/` directory.

### Modified Dataset

We manually modified the a subset of collected summary and candidate articles above so that a single ground-truth answer is guaranteed for each questions (refer to Section 3 of the paper for more details on how we modified the existing data points to better fit the task requirements). 

We include two datasets consisting of multiple questions for easy (`easy-ans-samples-cnndm.pkl`) and hard (`data/hard-ans-samples-cnndm.pkl`) types [here](https://drive.google.com/drive/folders/1_E2hOGZEnX3LwMCKpQiWvoo4Pv23Hg4V?usp=share_link) under the `data/` folder.  Questions sampled from these datasets were used for the user study. 

Each dataset has the following format.
```
{
    question_id (int) : {
        'original_text' : ground-truth article (str),
        'correct_summary' : query summary (str), 
        'wrong_text' : wrong candidate articles (list of str)
        'score_t1_s1' : affinity score for the ground-truth article (numpy array),
        'score_t2_s1' : affinity scores for the wrong candidate articles (numpy array),
        'correct' : whether the ground-truth article has the highest affinity score (bool)
        'ptype' : question type, either 'easy' or 'hard' (str)
    }
}
```
## Methods

Running each method gives a metadata used for visualization of highlights that will be saved in `output/` directory. You can download the outputs used for the experiments [here](https://drive.google.com/drive/folders/1_E2hOGZEnX3LwMCKpQiWvoo4Pv23Hg4V?usp=share_link) under the `output/` folder.

### Black-box Model Explanations

We used the implementation of SHAP [here](https://github.com/slundberg/shap). For other methods (input-gradient, integrated gradients), refer to `scripts/compute_explanations.py`. 

### Text Summarization

We used the implementation of BERTSum [here](https://github.com/nlpyang/PreSumm) to extract summaries for the articles. `scripts/add_bertsum.py` converts the raw output into the metadata format we use for visualization. Refer to the scripts in `bertsum_scripts/` directory on how BERTSum was run. 

### Task-specific Methods

#### Co-occurrence Method

Co-occurrence method computes the similarity between sentences based on the F1-score of ROUGE-L metric. Refer to `scripts/add_cooccur.py`. 

#### Semantic Method

Semantic method computes the similarity between sentences based on the representation learned by SentenceBERT. Refer to `scripts/add_semantic.py`. 

## Visualize Highlights

Using the metadata generated by each methods above, `scripts/generate_samples.py` generates HTML files that visualize the summary article pairs and highlights. These files are used to present the users with assistive information in the user study.

## Notebooks for Plots

- `notebooks/eval.ipynb` : evaluates black-box model explanations based using EM distances.
- `notebooks/power-analysis.pynb` : Monte-Carlo power anlaysis for finding the effective sample size for the user study
- `notebooks/analysis.ipynb` : test hypotheses, analyze, and plot the results based on user study data (To be added).

## Run

Instructions for running each part of the code is described in steps in `scripts/run.sh`.