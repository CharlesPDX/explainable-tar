# coding=utf-8

from enum import Enum, auto
import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any, Dict

import pyltr
import scipy
import numpy as np

from scipy.sparse.csr import csr_matrix

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from rank_bm25 import BM25Okapi
from nltk.stem.porter import *
import nltk

porter_stemmer = PorterStemmer()
from nltk.tokenize import word_tokenize

import torch
from sentence_transformers import SentenceTransformer

import gensim.downloader as gensim_api

from tar_framework.fuzzy_artmap import FuzzyArtMap
from tar_framework.fuzzy_artmap_gpu import FuzzyArtMapGpu
from tar_framework.utils import *


def preprocess_text(text):
    """
    1. Remove punctuation.
    2. Tokenize.
    3. Remove stopwords.
    4. Stem word.
    """
    # lowercase
    text = text.lower()
    # remove punctuation
    text = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+', ' ', text)
    # tokenize
    tokens = word_tokenize(text)
    # lowercase & filter stopwords
    filtered = [token for token in tokens if token not in ENGLISH_STOP_WORDS]
    # # stem
    stemmed = [porter_stemmer.stem(token) for token in filtered]

    return stemmed

def preprocess_without_stemming(text):
    """
    1. Remove punctuation.
    2. Tokenize.
    3. Remove stopwords.
    """
    # lowercase
    text = text.lower()

    # remove punctuation
    text = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+', ' ', text)

    # tokenize    
    try:
        tokens = word_tokenize(text)
    except LookupError as e:
        LOGGER.info(f"Tokenizer corpus not found, downloading - {e}")
        nltk.download("punkt")
        tokens = word_tokenize(text)

    # filter stopwords
    filtered = [token for token in tokens if token not in ENGLISH_STOP_WORDS]

    return filtered


def bm25_okapi_rank(complete_dids, complete_texts, query):
    tokenized_texts = [preprocess_text(doc) for doc in complete_texts]
    tokenized_query = preprocess_text(query)

    bm25 = BM25Okapi(tokenized_texts)
    scores = bm25.get_scores(tokenized_query)

    did_scores = sorted(zip(complete_dids, scores), key=lambda x: x[1], reverse=True)
    ranked_dids, ranked_scores = zip(*did_scores)

    return list(ranked_dids), list(ranked_scores)

class VectorizerType(Enum):
    tf_idf = auto()
    glove = auto()
    sbert = auto()
    word2vec = auto()


class Ranker(object):
    """
    Manager the ranking module of the TAR framework.
    """
    def __init__(self, model_type='lr', min_df=2, C=1.0, random_state=0, rho_a_bar=0.95, number_of_mapping_nodes=36):
        self.model_type = model_type
        self.random_state = random_state
        self.min_df = min_df
        self.C = C
        self.did2feature = {}
        self.name2features = {}
        self.rho_a_bar = rho_a_bar
        self.number_of_mapping_nodes = number_of_mapping_nodes
        self.glove_model = None
        self.word2vec_model = None
        self.missing_tokens = []

        if self.model_type == 'lr':
            self.model = LogisticRegression(solver='lbfgs', random_state=self.random_state, C=self.C, max_iter=10000)
        elif self.model_type == 'svm':
            self.model = SVC(probability=True, gamma='scale', random_state=self.random_state)
        elif self.model_type == 'lambdamart':
            self.model = None
        elif self.model_type == 'fam' or self.model_type == 'famg':
            self.model = None
        else:
            raise NotImplementedError

    def _dict_hash(self, dictionary: Dict[str, Any]) -> str:
        """MD5 hash of a dictionary."""
        # https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
        dhash = hashlib.md5()
        # We need to sort arguments so {'a': 1, 'b': 2} is
        # the same as {'b': 2, 'a': 1}
        encoded = json.dumps(dictionary, sort_keys=True).encode()
        dhash.update(encoded)
        return dhash.hexdigest()

    def _pickle_or_get_features(self, vectorizer_type: VectorizerType, corpus_name: str, vectorizer_params: dict, vectorizer: callable):
            if not vectorizer_params:
                vectorizer_params = {}
            pickled_corpus_file_name = f"{vectorizer_type.name}_{self._dict_hash(vectorizer_params)}_{corpus_name}.pkl"
            pickled_corpus = os.path.join(PARENT_DIR, 'data', 'pickels', pickled_corpus_file_name)
            try:
                with open(pickled_corpus, 'rb') as pickled_corpus_file:
                    stored_data = pickle.load(pickled_corpus_file)
                    features = stored_data['features']
            except FileNotFoundError as e:
                pickels_path = os.path.join(PARENT_DIR, 'data', 'pickels')
                if not Path(pickels_path).exists():
                    os.mkdir(pickels_path)
                LOGGER.info(e)
                features = vectorizer()
                with open(pickled_corpus, 'wb') as pickled_corpus_file:
                    pickle.dump({'features': features}, pickled_corpus_file, protocol=pickle.HIGHEST_PROTOCOL)
            return features

    def set_did_2_feature(self, dids, texts, corpus_texts, vectorizer_type: VectorizerType = VectorizerType.tf_idf, corpus_name=None, vectorizer_params=None):
        if vectorizer_type == VectorizerType.tf_idf:
            if not vectorizer_params:
                vectorizer_params = {'stop_words': 'english', 'min_df': int(self.min_df)}
            def tfidf_vectorize():
                tfidf_vectorizer = TfidfVectorizer(**vectorizer_params)
                tfidf_vectorizer.fit(corpus_texts)
                return tfidf_vectorizer.transform(texts)
            vectorizer = tfidf_vectorize
        elif vectorizer_type == VectorizerType.glove:
            vectorizer = lambda : self._glove_vectorize_documents(texts)
        elif vectorizer_type == VectorizerType.sbert:
            vectorizer = lambda : self.sbert_vectorize_documents(texts)
        elif vectorizer_type == VectorizerType.word2vec:
            vectorizer = lambda : self._word2vec_vectorize_documents(texts)

        if corpus_name:
            features = self._pickle_or_get_features(vectorizer_type, corpus_name, vectorizer_params, vectorizer)
        else:
            features = vectorizer()

        for did, feature in zip(dids, features):
            if vectorizer_type == VectorizerType.glove or vectorizer_type == VectorizerType.sbert or VectorizerType.word2vec:
                feature_min = feature.min()
                assert feature_min >= 0, "Negative feature value encountered"
                assert feature_min <= 1, "Feature min greater than one"
                feature_max = feature.max()
                assert feature_max >= 0, "Negative feature value encountered"
                assert feature_max <= 1, "Feature max greater than one"
            if len(feature.shape) == 1:
                self.did2feature[did] = feature[:, np.newaxis]
            else:
                self.did2feature[did] = feature
        
        if vectorizer_type == VectorizerType.glove or vectorizer_type == VectorizerType.word2vec:
            unique_missing_tokens = set(self.missing_tokens)
            LOGGER.info(f"{len(unique_missing_tokens):,} tokens not in {vectorizer_type} model")
            self.missing_tokens.clear()
        
        # TODO: figure out better shape logging
        logging.info(f'Ranker.set_feature_dict is done. - {len(self.did2feature):,} documents, {self.did2feature[dids[0]].shape} dimensions')
        return

    def sbert_vectorize_documents(self, texts):
        if torch.cuda.is_available():
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
        else:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        scalar = MinMaxScaler(feature_range=(0,1), copy=False)
        features =  model.encode(texts, convert_to_numpy=True)
        scalar.fit_transform(features)
        np.clip(features, 0.0, 1.0, out=features)
        return csr_matrix(features)

    def _word2vec_vectorize_documents(self, texts):
        if not self.word2vec_model:
            self.word2vec_model = gensim_api.load('word2vec-google-news-300')
        
        vectorized_documents = np.zeros((len(texts), 300))
        for text_index, text in enumerate(texts):
            preprocessed_tokens = preprocess_without_stemming(text)
            self.missing_tokens.extend([token for token in preprocessed_tokens if token not in self.word2vec_model])
            vectorized_tokens = np.array([self.word2vec_model[token] for token in preprocessed_tokens if token in self.word2vec_model])
            vectorized_documents[text_index] = vectorized_tokens.mean(axis=0,keepdims=True)
        

        scalar = MinMaxScaler(feature_range=(0,1), copy=False)
        scalar.fit_transform(vectorized_documents)
        return csr_matrix(vectorized_documents)

    def _glove_vectorize_documents(self, texts):
        if not self.glove_model:
            self._load_glove_model(os.path.join(os.getcwd(), 'autostop/tar_framework/glove/glove.6B.300d.txt'))

        for text in texts:
            preprocessed_tokens = preprocess_without_stemming(text)
            self.missing_tokens.extend([token for token in preprocessed_tokens if token not in self.glove_model])
            vectorized_tokens = np.array([self.glove_model[token] for token in preprocessed_tokens if token in self.glove_model])
            yield csr_matrix(vectorized_tokens.mean(axis=0,keepdims=True))


    def _load_glove_model(self, glove_file_location):
        LOGGER.info("Loading Glove Model")
        self.glove_model = {}
        glove_words = []
        glove_vectors = []
        with open(glove_file_location,'r') as glove_file:
            for line in glove_file:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                glove_words.append(word)
                glove_vectors.append(embedding)
        glove_vector_array = np.array(glove_vectors)
        LOGGER.info(f"{len(glove_words):,} words loaded")
        LOGGER.info("Scaling GloVe features")
        scalar = MinMaxScaler(feature_range=(0,1), copy=False)
        scalar.fit_transform(glove_vector_array)
        for word, vector in zip(glove_words, glove_vector_array):
            self.glove_model[word] = vector
        LOGGER.info("Scaling complete")


    def get_feature_by_did(self, dids):
        features = scipy.sparse.vstack([self.did2feature[did] for did in dids])
        return features

    def set_features_by_name(self, name, dids):
        features = scipy.sparse.vstack([self.did2feature[did] for did in dids])
        self.name2features[name] = features
        return

    def get_features_by_name(self, name):
        return self.name2features[name]

    def cache_corpus_in_model(self, document_ids):
        if self.model_type == "fam":
            if not self.model:
                number_of_features = self.did2feature[document_ids[0]].shape[1]
                self.model = FuzzyArtMap(number_of_features*2, self.number_of_mapping_nodes, rho_a_bar=self.rho_a_bar)
            corpus_features = self.get_feature_by_did(document_ids)
            document_index_mapping = {document_id: index for index, document_id in enumerate(document_ids)}
            self.model.cache_corpus(corpus_features, document_index_mapping)
        elif self.model_type == "famg":
            if not self.model:
                number_of_features = self.did2feature[document_ids[0]].shape[1]
                self.model = FuzzyArtMapGpu(number_of_features*2, self.number_of_mapping_nodes, rho_a_bar=self.rho_a_bar)
            corpus_features = self.get_feature_by_did(document_ids)
            document_index_mapping = {document_id: index for index, document_id in enumerate(document_ids)}
            self.model.cache_corpus(corpus_features, document_index_mapping)
        else:
            pass

    def remove_docs_from_cache(self, document_ids):
        if self.model_type == "fam" or self.model_type == "famg":
            self.model.remove_documents_from_cache(document_ids)
        else:
            pass

    def train(self, features, labels):
        if self.model_type == 'lambdamart':
            # retrain the model at each TAR iteration. Otherwise, the training speed will be slowed drastically.
            model = pyltr.models.LambdaMART(
                metric=pyltr.metrics.NDCG(k=10),
                n_estimators=100,
                learning_rate=0.02,
                max_features=0.5,
                query_subsample=0.5,
                max_leaf_nodes=10,
                min_samples_leaf=64,
                verbose=0,
                random_state=self.random_state)
        elif self.model_type == "fam" and not self.model:
            number_of_features = features.shape[1]
            self.model = FuzzyArtMap(number_of_features*2, self.number_of_mapping_nodes, rho_a_bar=self.rho_a_bar)
            model = self.model
        elif self.model_type == "famg" and not self.model:
            number_of_features = features.shape[1]
            self.model = FuzzyArtMapGpu(number_of_features*2, self.number_of_mapping_nodes, rho_a_bar=self.rho_a_bar)
            model = self.model
        else:
            model = self.model
        model.fit(features, labels)
        # logging.info('Ranker.train is done.')
        return

    def predict(self, features):
        probs = self.model.predict_proba(features)
        rel_class_inx = list(self.model.classes_).index(REL)
        scores = probs[:, rel_class_inx]
        return scores

    def predict_with_doc_id(self, doc_ids):
        probs = self.model.predict_proba(doc_ids)
        if probs.shape[0] != 0:
            scores = probs[:, np.r_[0:1, 2:3]]
        else:
            scores = []
        return scores