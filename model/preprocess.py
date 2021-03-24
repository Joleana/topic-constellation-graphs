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
    string = re.sub(r'(5g|5G)', 'five_g', sent)
    string = re.sub(r'(4g|4G)', 'four_g', string)
    string = re.sub(r'(3g|3G)', 'three_g', string)
    norm_string = re.sub(r'(2g|2G)', 'two_g', string)
    norm_sans_email = re.sub('\S*@\S*\s?', '', norm_string)  # remove emails
    norm_sans_url = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|\-)*\b', '', norm_sans_email,
                           flags=re.MULTILINE) # remove url
    " ".join([x for x in norm_sans_url.split(" ") if not x.isdigit()])
    norm_sans_newline = re.sub('\s+', ' ', norm_sans_url)  # remove newline chars
    norm_sans_quotes = re.sub("\'", "", norm_sans_newline)  # remove single quotes
    s = re.sub(r'([a-z])([A-Z])', r'\1\. \2', norm_sans_quotes)  # xThis -> xx. This
    s = s.lower() # lower case
    s = re.sub(r'&gt|&lt', ' ', s) # remove encoding format
    s = re.sub(r'([a-z])\1{2,}', r'\1', s) # letter repetition (if more than 2)
    s = re.sub(r'([\W+])\1{1,}', r'\1', s) # non-word repetition (if more than 1)
    s = re.sub(r'\W+?\.', '.', s) # xxx[?!]. -- > xxx.
    s = re.sub(r'(\.|\?|!)(\w)', r'\1 \2', s) # [.?!]xxx --> [.?!] xxx
    s = re.sub(r'\\x', r'-', s) # 'x' --> '-'
    sentence = s.strip() # remove padding spaces
    return sentence

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
