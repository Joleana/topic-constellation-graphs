from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim import corpora
import gensim
import os
import numpy as np
from Autoencoder import *
from preprocess import preprocess_word, preprocess_sent
from datetime import datetime
import pandas as pd
import logging


def preprocess(df, samp_size=None, sample=True):
    """
    Preprocess the data
    """
    if not samp_size:
        samp_size = 300
    print('Preprocessing raw texts from seen documents...')
    docs = df.rawText
    n_docs = len(docs)
    sentences = []
    token_lists = []
    titles = []
    idx_in = []
    if sample:
        samp = np.random.choice(n_docs, samp_size)
        print(f"samp looks like: {samp}")
        for i, idx in enumerate(samp):
            sentence = preprocess_sent(docs[idx])
            token_list = preprocess_word(sentence)
            if token_list:
                idx_in.append(idx)
                sentences.append(str(sentence))
                token_lists.append(token_list)
            print('{} %'.format(str(np.round((i + 1) / len(samp) * 100, 2))), end='\r')
        print('Preprocessing raw seen texts. Done!')
        return sentences, token_lists, idx_in
    else:
        print('Preprocessing raw texts from unseen documents...')
        for i, doc in enumerate(docs):
            sentence = preprocess_sent(doc)
            token_list = preprocess_word(sentence)
            if token_list:
                idx_in.append(i)
                sentences.append(str(sentence))
                token_lists.append(token_list)
                titles.append(df.title[i])
            print('{} %'.format(str(np.round((i + 1) / len(docs) * 100, 2))), end='\r')
        print('Preprocessing raw unseen texts. Done!')
        return sentences, token_lists, idx_in, titles

##################################
####    Topic Model Class     ####
##################################


# define model object
class Topic_Model:
    def __init__(self, k=10, method='TFIDF'):
        """
        :param k: number of topics
        :param method: method chosen for the topic model
        """
        if method not in {'TFIDF', 'LDA', 'BERT', 'LDA_BERT'}:
            raise Exception('Invalid method!')
        self.k = k
        self.dictionary = None
        self.corpus = None
        #         self.stopwords = None
        self.cluster_model = None
        self.ldamodel = None
        self.vec = {}
        self.gamma = 15  # parameter for reletive importance of lda
        self.method = method
        self.AE = None
        self.id = method + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def vectorize(self, sentences, token_lists, method=None):
        """
        Get vector representations from selected methods
        """
        # Default method
        if method is None:
            method = self.method

        # turn tokenized documents into a id <-> term dictionary
        self.dictionary = corpora.Dictionary(token_lists)
        # convert tokenized documents into a document-term matrix
        self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]

        if method == 'TFIDF':
            print('Getting vector representations for TF-IDF ...')
            tfidf = TfidfVectorizer()
            vec = tfidf.fit_transform(sentences)
            print('Getting vector representations for TF-IDF. Done!')
            return vec

        elif method == 'LDA':
            print('Getting vector representations for LDA ...')
            if not self.ldamodel:
                self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.k, id2word=self.dictionary,
                                                                passes=40, random_state=100, update_every=1, chunksize=2000,
                                                                alpha='symmetric', iterations=500)

            # Init output
            sent_topics_df = pd.DataFrame()

            # Get main topic in each document
            for i, row_list in enumerate(self.ldamodel[self.corpus]):
                row = row_list[0] if self.ldamodel.per_word_topics else row_list
                row = sorted(row, key=lambda x: (x[1]), reverse=True)
                # Get the Dominant topic, Perc Contribution and Keywords for each document
                for j, (topic_num, prop_topic) in enumerate(row):
                    if j == 0:  # => dominant topic
                        wp = self.ldamodel.show_topic(topic_num, topn=30)
                        topic_keywords = ", ".join([word for word, prop in wp])
                        sent_topics_df = sent_topics_df.append(
                            pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                    else:
                        break
            # Add original text to the end of the output
            contents = pd.Series(sentences)
            sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
            sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'Document']
            pd.DataFrame(sent_topics_df).to_csv(f"{os.getcwd()}/data/results/Dom_Topic_And_Contrib_MYTH.csv")

            # Most representative document for each topic
            sent_topics_sorteddf_mallet = pd.DataFrame()
            sent_topics_outdf_grpd = sent_topics_df.groupby('Dominant_Topic')
            for i, grp in sent_topics_outdf_grpd:
                sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                         grp.sort_values(['Perc_Contribution'], ascending=False).head(
                                                             1)],
                                                        axis=0)
            # Reset Index
            sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
            # Format
            sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib",  'Topic_Keywords', "Representative Text"]
            pd.DataFrame(sent_topics_sorteddf_mallet).to_csv(f"{os.getcwd()}/data/results/Most_Rep_MYTH.csv")

            def get_vec_lda(model, corpus, k):
                """
                Get the LDA vector representation (probabilistic topic assignments for all documents)
                :return: vec_lda with dimension: (n_doc * n_topic)
                """
                n_doc = len(corpus)
                vec_lda = np.zeros((n_doc, k))
                for i in range(n_doc):
                    # get the distribution for the i-th document in corpus
                    for topic, prob in model.get_document_topics(corpus[i]):
                        vec_lda[i, topic] = prob

                return vec_lda

            vec = get_vec_lda(self.ldamodel, self.corpus, self.k)
            print('Getting vector representations for LDA. Done!')
            return vec

        elif method == 'BERT':

            print('Getting vector representations for BERT ...')
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('bert-base-nli-max-tokens')
            vec = np.array(model.encode(sentences, show_progress_bar=True))
            print('Getting vector representations for BERT. Done!')
            return vec

        #         elif method == 'LDA_BERT':
        else:
            vec_lda = self.vectorize(sentences, token_lists, method='LDA')
            vec_bert = self.vectorize(sentences, token_lists, method='BERT')
            vec_ldabert = np.c_[vec_lda * self.gamma, vec_bert]
            self.vec['LDA_BERT_FULL'] = vec_ldabert
            if not self.AE:
                self.AE = Autoencoder()
                print('Fitting Autoencoder ...')
                self.AE.fit(vec_ldabert)
                print('Fitting Autoencoder Done!')
            vec = self.AE.encoder.predict(vec_ldabert)
            return vec

    def fit(self, sentences, token_lists, method=None, m_clustering=None):
        """
        Fit the topic model for selected method given the preprocessed data
        :docs: list of documents, each doc is preprocessed as tokens
        :return:
        """
        # Default method
        if method is None:
            method = self.method
        # Default clustering method
        if m_clustering is None:
            m_clustering = KMeans

        # turn tokenized documents into a id <-> term dictionary
        if not self.dictionary:
            self.dictionary = corpora.Dictionary(token_lists)
            # convert tokenized documents into a document-term matrix
            self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]

        ####################################################
        #### Getting ldamodel or vector representations ####
        ####################################################

        if method == 'LDA':
            if not self.ldamodel:
                print('Fitting LDA ...')
                self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.k, id2word=self.dictionary,
                                                                passes=40, random_state=100, update_every=1, chunksize=2000,
                                                                alpha='symmetric', iterations=500)
                print('Fitting LDA Done!')
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        else:
            print('Clustering embeddings ...')
            self.cluster_model = m_clustering(self.k)
            self.vec[method] = self.vectorize(sentences, token_lists, method)
            self.cluster_model.fit(self.vec[method])
            print('Clustering embeddings. Done!')

    def predict(self, sentences, token_lists, unseen=True):
        """
        Predict topics for new_documents
        """
        # Default as False
        unseen = unseen is not None

        if unseen:
            corpus = [self.dictionary.doc2bow(text) for text in token_lists]
            if self.method != 'LDA':
                vec = self.vectorize(sentences, token_lists)
        else:
            corpus = self.corpus
            vec = self.vec.get(self.method, None)

        if self.method == "LDA":
            lbs = np.array(list(map(lambda x: sorted(self.ldamodel.get_document_topics(x),
                                                     key=lambda x: x[1], reverse=True)[0][0],
                                    corpus)))
        else:
            lbs = self.cluster_model.predict(vec)
        return lbs
