import os
import ast
from ast import literal_eval
import argparse
from model import *
from utils import *
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=Warning)


if __name__ == '__main__':
    myth_corpus_df = text_to_df(f'{os.getcwd()}/data/master/myth_corpus')
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', default=f'{os.getcwd()}/data/master/myth_corpus.csv')
    parser.add_argument('--ntopic', default=10)
    parser.add_argument('--method', default='LDA_BERT')
    parser.add_argument('--samp_size', default=50)
    args = parser.parse_args()

    data = pd.read_csv(str(args.fpath))
    data = data.fillna('')  # only the comments has NaN's

    sentences, token_lists, idx_in = preprocess(data, samp_size=int(args.samp_size), sample=True)

    # Define the topic model object
    tm = Topic_Model(k=int(args.ntopic), method=str(args.samp_size))

    # Fit the topic model by chosen method
    tm.fit(sentences, token_lists)

    # Evaluate using metrics
    with open("{}/results/saved_models/{}.file".format(os.getcwd(), tm.id), "wb") as f:
        pickle.dump(tm, f, pickle.HIGHEST_PROTOCOL)

    print('Coherence:', get_coherence(tm, token_lists, 'c_v'))
    print('Silhouette Score:', get_silhouette(tm))

    # visualize and save img
    visualize(tm)

    topic_keywords = []
    for i in range(tm.k):
        tokens = get_wordcloud(tm, token_lists, i)
        topic_keywords.append(tokens)

    #####################################################################################################

    tech_corpus_df = text_to_df(f'{os.getcwd()}/data/master/tech_corpus')
    tech_data = tech_corpus_df.fillna('')  # only the comments has NaN's

    sentencesT, token_listsT, _, titlesT = preprocess(tech_data, samp_size=int(args.samp_size), sample=False)

    tech_corpus = [tm.dictionary.doc2bow(tokens) for tokens in token_listsT]
    tech_vec = tm.vectorize(sentencesT, token_listsT)
    lbs_T = np.array(list(map(lambda x: sorted(tm.ldamodel.get_document_topics(x),
                                               key=lambda x: x[1], reverse=True)[0][0],
                              tech_corpus)))
    visualize_test(tm, lbs_T, sentencesT, token_listsT, sub_cat_labelT)