# Based on fuzzyartmap_demo.m by Rajeev Raizada, wih permission, and 
# Carpenter, G. A., Grossberg, S., Markuzon, N., Reynolds, J. H. and Rosen, D. B. (1992)
# "Fuzzy ARTMAP: A Neural Network Architecture for Incremental Supervised Learning of Analog Multidimensional Maps"
# IEEE Transactions on Neural Networks, Vol. 3, No. 5, pp. 698-713.

import math
from typing import List
import logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import numpy as np
from scipy.sparse.csr import csr_matrix
from dask.distributed import Client


# pip install "dask[complete]" 
# pip install dask distributed --upgrade

class FuzzyArtmapDistributed:
    def __init__(self, f1_size: int = 10, f2_size: int = 10, number_of_categories: int = 2, rho_a_bar = 0, scheduler_address=None):
        self.rho_a_bar = rho_a_bar  # Baseline vigilance for ARTa, in range [0,1]
        self.f2_size = f2_size
        self.f1_size = f1_size
        self.number_of_categories = number_of_categories
        self.classes_ = np.array([1])
        self.client = Client(scheduler_address)
        # self.client.upload_file(os.path.abspath(__file__))
        self.client.upload_file(__file__)
        self.worker_addresses = list(self.client.ncores().keys())
        self.workers = None
        self.training_fuzzy_artmap = LocalFuzzyArtMap(self.f1_size, self.f2_size, self.number_of_categories, self.rho_a_bar)
        self.weight_ab = self.training_fuzzy_artmap.weight_ab
        logger.info(f"Initializing {len(self.worker_addresses)} workers")
        self.distribute_actors()
        for worker in self.workers:
            worker.init(self.f1_size, self.f2_size, self.number_of_categories)
        logger.info("Initialization complete.")
    
    def distribute_actors(self):
        worker_futures = []
        for worker_address in self.worker_addresses:
            future = self.client.submit(FuzzyArtMapWorker, actors=True, workers=worker_address)
            worker_futures.append(future)
        self.workers = list(map(lambda f: f.result(), (f for f in worker_futures)))
    
    def cache_corpus(self, corpus: csr_matrix, document_index_mapping: dict):
        logger.info("Starting distributed corpus caching.")
        number_of_workers = len(self.workers)
        document_index_chunks = np.array_split(list(document_index_mapping.keys()), number_of_workers)
        caching_futures = []
        x = corpus.toarray()
        for index, corpus_chunk in enumerate(np.array_split(x, number_of_workers)):
            chunk_document_id_index = {document_id: index for index, document_id in enumerate(document_index_chunks[index])}
            caching_futures.append(self.workers[index].cache_corpus(corpus_chunk, chunk_document_id_index))
        _ = list(map(lambda f: f.result(), (f for f in caching_futures)))
        print(_)
        logger.info("Completed distributed corpus caching.")

    def remove_documents_from_cache(self, document_ids):
        for worker in self.workers:
            worker.remove_documents_from_cache(document_ids).result()
    
    def fit(self, input_vectors, class_vectors):
        number_of_new_nodes, updated_nodes, updated_a_weights, updated_ab_weights = self.training_fuzzy_artmap.fit(input_vectors, class_vectors)
        for worker in self.workers:
            worker.update_weights(number_of_new_nodes, updated_nodes, updated_a_weights, updated_ab_weights)
        self.training_fuzzy_artmap.clear_updated_nodes()

    def predict_proba(self, doc_ids: list):
        result_futures = []
        for worker in self.workers:
            result_futures.append(worker.predict_proba(doc_ids))
        distributed_results = map(lambda f: f.result(), (f for f in result_futures))
        results = []
        for result in distributed_results:
            results.extend(result)
        return np.array(results)


class LocalFuzzyArtMap:
    def __init__(self, f1_size: int = 10, f2_size: int = 10, number_of_categories: int = 2, rho_a_bar = 0):
        self.alpha = 0.001  # "Choice" parameter > 0. Set small for the conservative limit (Fuzzy AM paper, Sect.3)
        self.beta = 1  # Learning rate. Set to 1 for fast learning
        self.rho_a_bar = rho_a_bar  # Baseline vigilance for ARTa, in range [0,1]
        
        self.rho_ab = 0.95          # Map field vigilance, in [0,1]
        self.epsilon = 0.001        # Fab mismatch raises ARTa vigilance to this much above what is needed to reset ARTa
        # Initial weights in ARTa. All set to 1 Row-j, col-i entry = weight from input node i to F2 coding node j
        self.weight_a = np.ones((f2_size, f1_size), dtype=np.float32)
        
        # Row-j, col-k entry = weight from ARTa F2  node j to Map Field node k
        self.weight_ab = np.ones((f2_size, number_of_categories), dtype=np.float32)
        self.committed_nodes = set()

        self.classes_ = np.array([1])
        self.updated_nodes = set()

        self.node_increase_step = 5 # number of F2 nodes to add when required
        self.number_of_increases = 0
    
    def _resonance_search(self, input_vector: np.array, already_reset_nodes: List[int], rho_a: float, allow_category_growth = True):
        resonant_a = False
        input_vector_sum = np.sum(input_vector, axis=1)
        while not resonant_a:
            N = self.weight_a.shape[0]  # Count how many F2a nodes we have

            A_for_each_F2_node = input_vector * np.ones((N, 1), dtype=np.float32)
            A_AND_w = np.minimum(A_for_each_F2_node, self.weight_a)
            S = np.sum(A_AND_w, axis=1)
            T = S / (self.alpha + np.sum(self.weight_a, axis=1))
            T[already_reset_nodes] = np.zeros((len(already_reset_nodes), ), dtype=np.float32)
            J = np.argmax(T)
            membership_degree = S[J]/input_vector_sum           
            if membership_degree[0] >= rho_a or math.isclose(membership_degree[0], rho_a):
                resonant_a = True
            else:
                resonant_a = False
                already_reset_nodes.append(J)

            # Creating a new node if we've reset all of them
            if len(already_reset_nodes) >= N:
                if allow_category_growth:
                    self.weight_a = np.concatenate((self.weight_a, np.ones((self.node_increase_step,  self.weight_a.shape[1]), dtype=np.float32)), axis=0)
                    self.weight_ab = np.concatenate((self.weight_ab, np.ones((self.node_increase_step, self.weight_ab.shape[1]), dtype=np.float32)), axis=0)
                    self.number_of_increases += 1
                    # Give the new F2a node a w_ab entry, this new node should win
                else:
                    return -1, None

        return J, membership_degree[0]
    # @profile
    def train(self, input_vector: np.array, class_vector: np.array):
        rho_a = self.rho_a_bar # We start off with ARTa vigilance at baseline
        resonant_ab = False # Not resonating in the Fab match layer
        already_reset_nodes = [] # We haven't rest any ARTa nodes for this input pattern yet, maintain list between resonance searches of Fa
        
        while not resonant_ab:            
            J, x = self._resonance_search(input_vector, already_reset_nodes, rho_a)
            z = np.minimum(class_vector, self.weight_ab[J, np.newaxis])
            
            if np.sum(z, axis=1)/np.sum(class_vector, axis=1) >= self.rho_ab:
                resonant_ab = True
            else: 
                already_reset_nodes.append(J)
                rho_a = x + self.epsilon                
                if rho_a > 1.0:
                    rho_a = 1.0 - self.epsilon

        self.updated_nodes.add(J)
        if J in self.committed_nodes:
            beta = 0.75
        else:
            beta = self.beta

        self.weight_a[J, np.newaxis] = (beta * np.minimum(input_vector, self.weight_a[J, np.newaxis])) + ((1-beta) * self.weight_a[J, np.newaxis])
        self.weight_ab[J, np.newaxis] = (self.beta * z) + ((1-self.beta) * self.weight_ab[J, np.newaxis])
        self.committed_nodes.add(J)

    def fit(self, input_vectors, class_vectors):
        for document_index, input_vector in enumerate(input_vectors):
            self.train(FuzzyArtMapWorker.complement_encode(np.array(input_vector.toarray(), dtype=np.float32)), FuzzyArtMapWorker.complement_encode(np.array([[class_vectors[document_index]]], dtype=np.float32)))
        logger.info(f"updated nodes: {','.join([str(J) for J in self.updated_nodes])}")
        updated_nodes = list(self.updated_nodes)
        number_of_added_nodes = self.number_of_increases * self.node_increase_step
        return number_of_added_nodes, updated_nodes, self.weight_a[updated_nodes], self.weight_ab[updated_nodes]
    
    def clear_updated_nodes(self):
        self.updated_nodes.clear()
        self.number_of_increases = 0


class FuzzyArtMapWorker:
    def __init__(self):
        self.alpha = 0.001  # "Choice" parameter > 0. Set small for the conservative limit (Fuzzy AM paper, Sect.3)
        self.excluded_document_ids = set()
        self.weight_a = None
        self.weight_ab = None
        self.document_index_mapping = None

    def init(self, f1_size: int = 10, f2_size: int = 10, number_of_categories: int = 2):        
        self.weight_a = np.ones((f2_size, f1_size), dtype=np.float32) # Initial weights in ARTa. All set to 1 Row-j, col-i entry = weight from input node i to F2 coding node j
        self.weight_ab = np.ones((f2_size, number_of_categories), dtype=np.float32) # Row-j, col-k entry = weight from ARTa F2  node j to Map Field node k

    def update_weights(self, number_of_new_nodes, update_indexes, a_updates, ab_updates):
        if number_of_new_nodes > 0:
            self.weight_a = np.concatenate((self.weight_a, np.ones((number_of_new_nodes,  self.weight_a.shape[1]), dtype=np.float32)), axis=0)
            self.weight_ab = np.concatenate((self.weight_ab, np.ones((number_of_new_nodes, self.weight_ab.shape[1]), dtype=np.float32)), axis=0)
            self.S_cache = np.concatenate((self.S_cache, np.ones((self.corpus.shape[0], number_of_new_nodes), dtype=np.float32)), axis=1)
        self.weight_a[update_indexes] = a_updates
        self.weight_ab[update_indexes] = ab_updates
        self.recompute_S_cache(update_indexes)

    def cache_corpus(self, corpus: csr_matrix, document_index_mapping: dict):
        self.corpus = self.complement_encode(corpus)
        self.document_index_mapping = document_index_mapping
        print(f"aa setting mapping to: {len(self.document_index_mapping.items())}")
        self.excluded_document_ids = set()
        N = self.weight_a.shape[0]
        self.S_cache = np.zeros((self.corpus.shape[0], N), dtype=np.float32)
        self.input_sum_cache = np.sum(self.corpus, axis=1) #np.zeros((self.corpus.shape[0], 1))
        A_AND_w = np.empty(self.weight_a.shape, dtype=np.float32)

        for i in range(self.corpus.shape[0]):
            A_for_each_F2_node = self.corpus[i] * np.ones((N, 1), dtype=np.float32)
            np.minimum(A_for_each_F2_node, self.weight_a, out=A_AND_w)
            self.S_cache[i] = np.sum(A_AND_w, axis=1)

    def remove_documents_from_cache(self, document_ids):
        for document_id in document_ids:
            self.excluded_document_ids.add(document_id)

    def recompute_S_cache(self, updated_nodes):        
        N = self.weight_a.shape[0]
        A_AND_w = np.empty((len(updated_nodes),self.weight_a.shape[1]), dtype=np.float32)
        for document_id, index in self.document_index_mapping.items():
            if document_id in self.excluded_document_ids:
                continue
            A_for_each_F2_node = self.corpus[index] * np.ones((N, 1), dtype=np.float32)
            np.minimum(A_for_each_F2_node[updated_nodes], self.weight_a[updated_nodes], out=A_AND_w)
            self.S_cache[index, updated_nodes] = np.sum(A_AND_w, axis=1)

    def _cached_resonance_search(self, cached_S, input_vector_sum):
        T = cached_S / self.choice_denominator
        # Choice function vector for F2
        J = np.argmax(T) # Finding the winning node, J
        membership_degree = cached_S[J]/input_vector_sum
        return J, membership_degree

    def cached_predict(self, document_id):
        index = self.document_index_mapping[document_id]
        input_vector_sum = self.input_sum_cache[index]
        cached_S_value = self.S_cache[index]
        J, membership_degree = self._cached_resonance_search(cached_S_value, input_vector_sum)
        return self.weight_ab[J, np.newaxis], membership_degree # Fab activation vector & fuzzy membership value

    @staticmethod
    def complement_encode(original_vector: np.array) -> np.array:
        complement = 1-original_vector
        complement_encoded_value = np.concatenate((original_vector,complement), axis=1)
        return complement_encoded_value

    def predict_proba(self, doc_ids: list):
        predictions = []
        self.choice_denominator = (self.alpha + np.sum(self.weight_a, axis=1))
        for document_id in doc_ids:
            if document_id not in self.document_index_mapping:
                continue
            prediction, membership_degree = self.cached_predict(document_id)
            if prediction[0][0]:
                predictions.append((membership_degree, 0, document_id))
        
        return predictions #needs to be in shape (number_of_docs, 1)  
