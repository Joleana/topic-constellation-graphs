CONTENTS
---------------------
 * Introduction
 * Requirements
 * Features
 * Configuration
 * Data handling
 * Model


INTRODUCTION
------------
*Topic constellations for corpus mapping and comparison*

This repository aims to serve two endpoints in the development on an R&D instrument which performs a comparative analysis between a mythological corpus and  a technological corpus, one which contains the cumulative anticipation of a 5G network:

1. *** maintain a database of a mythological corpus as seen documents for training a model and a technology corpus as unseen documents for testing ***
2. *** use a BERT-LDA model to derive topic clusters from mythological corpus and map the tech corpus to predicted topics (model source code attributed to Steve Shao https://blog.insightdatascience.com/contextual-topic-identification-4291d256a032 )***

REQUIREMENTS
------------
Install requirements.txt with Python3

Results include: 
a 2D representation of clustered topics from myth corpus '2D_viz.png' 
a 2D representation of predicted topics from tech corpus '2D_viz.png'
a number of 'wordcloud.png' files which equal the number of topics specified in main
a 'Most_Rep_MYTH.csv' which returns the most represented document from the myth corpus for each topic specified in main
a 'Dom_Topic_And_Contrib_MYTH.csv' which returns each of the documents from myth corpus and the most representative topic and its contribution for each doc
a '.file' which contains the model output in binary

DATA HANDLING
-------------

Navigate to the data/master/myth_corpus and data/master/tech_corpus respectively, and include txt files of desired data

The project aims to manage the pipeline internally; once user has included txt files in respective folders, they may run
the main.py script from using default parameters.


CONFIGURATION
-------------
from main:

fpath: csv file which contains raw text from specified corpus; (default: data/master/myth_corpus.csv)

ntopic: number of topics the user would like to see (default: 10)

method: TFIDF, LDA, BERT, LDA_BERT (default LDA_BERT)

samp-size: number of docs the user would like to re-sample (default: 100)

MODEL
-------------

LDA topic model + BERT sentence vectorization model /model.py
