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
def preprocess_sent(sentence):
    """
    Where not digit, join sentence, lowercase and strip string of leading and trailing white spaces
    :param sent: taken from csv output from utils/text_to_df function
    :return: strings for sentence embedding in BERT model
    """
    " ".join([x for x in sentence.split(" ") if not x.isdigit()])
    sentence = sentence.lower()
    sentence = sentence.strip()  # remove padding spaces
    return sentence

###############################
#### word level preprocess ####
###############################


# 2. Tokenize Words and Clean
def preprocess_word(sentences):
    """
    Selected part-of-speech tags proper nouns, nouns, adjectives, and adverb (see Spacy docuemntation for more tags)
    go through preproccesing via Gensim library methods and build out the tokenization process for LDA model
    :param sentences: taken from preprocess_sent function
    :return: list of tokens for topic modeling in LDA model
    """
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
    junk = pd.read_csv('stopwords_extended.txt', names=['common'], header=0)
    junk = [word.lower() for word in junk.common]
    for word in junk:
        stop_words.append(word.lower())
    words = [[word for word in simple_preprocess(str(doc)) if texts_out not in stop_words] for doc in texts_out]
    words = list(chain.from_iterable(words))
    return words
