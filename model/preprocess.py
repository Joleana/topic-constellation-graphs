import re
import os
import spacy
import gensim
import pandas as pd
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from itertools import chain


###################################
#### sentence level preprocess ####
###################################


# 1. Tokenize Sentences and Clean
def preprocess_sent(sent):
    # """
    #
    # :param sent:
    # :return:
    # """
    # return sentence

###############################
#### word level preprocess ####
###############################


# 2. Tokenize Words and Clean
def preprocess_word(sentences):
    stop_words = stopwords.words('english')
    allowed_postags=['PROPN', 'NOUN', 'ADJ', 'ADV']

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(sentences, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[sentences], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [word for word in simple_preprocess(str(sentences)) if word not in stop_words]

    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en', disable=['parser', 'ner'])

    for sent in texts:
        doc = nlp(''.join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    # remove stopwords once more after lemmatization
    junk = pd.read_csv(f'{os.getcwd()}/stopwords_extended.txt', names=['common'], header=0)
    junk = [word.lower() for word in junk.common]
    for word in junk:
        stop_words.append(word.lower())
    words = [[word for word in simple_preprocess(str(doc)) if texts_out not in stop_words] for doc in texts_out]
    words = list(chain.from_iterable(words))
    # print(f"token lists: {words}")
    return words
