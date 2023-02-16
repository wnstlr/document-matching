#!/bin/bash
set -e

##### I. Data collection

## 1. Collect summary-article pair information from CNN/DailyMail
python collect_data.py

## 2. Post-process and modify the dataset appropirately.
## You alreay have the modified dataset in the data/ directory 
## to be directly used to replicate the highlights from the paper. 

##### II. Run methods on the modified dataset

## 1. Generate post-hoc model explanations
python compute_explanation.py --fname hard-ans-samples
python compute_explanation.py --fname easy-ans-samples

## 2. Run BERTSum
python add_bertsum.py --fname samples

## 3. Run Cooccurrence method
python add_cooccur.py 

## 4. Run Semantic method
python add_semantic.py 

## 5. Merge all results
python merge_exps.py

##### III. Visualize the summary-article pairs and highlights for user study.

## 1. Generate html files for all questions with and without highlights 
python generate_samples.py --fname hard-ans-samples
python generate_samples.py --fname easy-ans-samples

## 2. Select a set of questions for the user study
python select_samples_for_study.py --v main