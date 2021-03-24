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
This repository aims to serve two endpoints in the developmnet on an R&D instrument which charts the development of socio-cultural systems as reflected in the conversation around an anticipated 5G network:
1. *** maintain a database of a mythological corpus as seen documents for training a model and a technology corpus as unseen documents for testing ***
2. *** use a BERT-LDA model to derive topic clusters from mythological corpus and map the tech corpus to predicted topics (model source code attributed to Steve Shao https://blog.insightdatascience.com/contextual-topic-identification-4291d256a032 )***

REQUIREMENTS
------------
Install requirements.txt with Python3

FEATURES
------------

CONFIGURATION
-------------

DATA HANDLING
-------------

MODEL
-------------


user-flow

from main:
fpath: csv file which contains raw text from specified corpus; (default: data/master/myth_corpus.csv)
    index (file #) | rawText | 'layer' (myth or tech) | filename (filename.txt format)
ntopic: number of topics the user would like to see (default: 10)
    represented as cluster topics 0-N
method: TFIDF, LDA, BERT, LDA_BERT (default LDA_BERT)

