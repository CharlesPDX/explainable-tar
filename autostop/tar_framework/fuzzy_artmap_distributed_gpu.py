# Based on fuzzyartmap_demo.m by Rajeev Raizada, wih permission, and 
# Carpenter, G. A., Grossberg, S., Markuzon, N., Reynolds, J. H. and Rosen, D. B. (1992)
# "Fuzzy ARTMAP: A Neural Network Architecture for Incremental Supervised Learning of Analog Multidimensional Maps"
# IEEE Transactions on Neural Networks, Vol. 3, No. 5, pp. 698-713.

import gc
import asyncio
import socket
import pickle
import io
import argparse
import math
from typing import List
import logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import numpy as np
import torch
from scipy.sparse.csr import csr_matrix

from tornado.tcpserver import TCPServer
from tornado.tcpclient import TCPClient
from tornado.iostream import StreamClosedError
import tornado.ioloop


class FuzzyArtmapGpuDistributed:
    def __init__(self, f1_size: int = 10, f2_size: int = 10, number_of_categories: int = 2, rho_a_bar = 0, scheduler_address=None, max_nodes_mode = False, use_cuda_if_available = False):
        self.rho_a_bar = rho_a_bar  # Baseline vigilance for ARTa, in range [0,1]
        self.f2_size = f2_size
        self.f1_size = f1_size
        self.number_of_categories = number_of_categories
        self.classes_ = np.array([1])
        self.max_nodes_mode = max_nodes_mode
        self.use_cuda_if_available = use_cuda_if_available
        self.client = FuzzyArtmapWorkerClient(scheduler_address)
        self.training_fuzzy_artmap = LocalFuzzyArtMapGpu(self.f1_size, self.f2_size, self.number_of_categories, self.rho_a_bar, max_nodes_mode, use_cuda_if_available)
        self.weight_ab = self.training_fuzzy_artmap.weight_ab        
    
    async def initialize_workers(self):
        logger.info("Getting workers")
        await self.client.get_workers()
        logger.info(f"Initializing {len(self.client.workers)} workers")
        params = [self.f1_size, self.f2_size, self.number_of_categories, self.rho_a_bar, self.max_nodes_mode, self.use_cuda_if_available]
        await self.client.init_workers(params)
        logger.info("Workers initialized")
    
    async def cache_corpus(self, ranker_params, document_index_mapping: dict):
        logger.info("Starting distributed corpus caching.")
        number_of_workers = len(self.client.workers)
        document_index_chunks = np.array_split(list(document_index_mapping.keys()), number_of_workers)
        await self.client.cache_corpus(ranker_params, document_index_chunks)
        logger.info("Completed distributed corpus caching.")

    async def remove_documents_from_cache(self, document_ids):
        await self.client.remove_documents_from_cache(document_ids)
    
    async def fit(self, input_vectors, class_vectors):
        self.training_fuzzy_artmap.fit(input_vectors, class_vectors)
        await self.client.fit([input_vectors, class_vectors])
        # self.training_fuzzy_artmap.clear_updated_nodes()

    async def predict_proba(self, doc_ids: list):
        results = await self.client.predict_proba(doc_ids)
        return np.array(results)


class LocalFuzzyArtMapGpu:
    def __init__(self, f1_size: int = 10, f2_size: int = 10, number_of_categories: int = 2, rho_a_bar = 0, max_nodes_mode = False, use_cuda_if_available = False):
        self.alpha = 0.001  # "Choice" parameter > 0. Set small for the conservative limit (Fuzzy AM paper, Sect.3)
        self.beta = 1  # Learning rate. Set to 1 for fast learning
        self.beta_ab = 1 #ab learning rate
        self.rho_a_bar = rho_a_bar  # Baseline vigilance for ARTa, in range [0,1]
        
        self.rho_ab = 0.95          # Map field vigilance, in [0,1]
        self.epsilon = 0.001        # Fab mismatch raises ARTa vigilance to this much above what is needed to reset ARTa

        self.max_nodes_mode = max_nodes_mode

        if use_cuda_if_available and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            if use_cuda_if_available:
                logger.warning("CUDA requested but not available, using CPU.")

        # Initial weights in ARTa. All set to 1 Row-j, col-i entry = weight from input node i to F2 coding node j
        self.weight_a = torch.ones((f2_size, f1_size), device=self.device, dtype=torch.float)
        self.input_vector_sum = f1_size / 2

        # Row-j, col-k entry = weight from ARTa F2  node j to Map Field node k
        self.weight_ab = torch.ones((f2_size, number_of_categories), device=self.device, dtype=torch.float)
        self.committed_nodes = set()

        self.classes_ = np.array([1])
        self.updated_nodes = set()

        self.node_increase_step = 5 # number of F2 nodes to add when required
        self.number_of_increases = 0
    
    def _resonance_search(self, input_vector: torch.tensor, already_reset_nodes: List[int], rho_a: float, allow_category_growth = True):
        resonant_a = False
        N, S, T = self.calculate_activation(input_vector)
        while not resonant_a:
            
            T[already_reset_nodes] = torch.zeros((len(already_reset_nodes), ), dtype=torch.float, device=self.device)
            J = np.argmax(T)
            membership_degree = S[J]/self.input_vector_sum
            if membership_degree.item() >= rho_a or math.isclose(membership_degree.item(), rho_a):
                resonant_a = True
            else:
                resonant_a = False
                already_reset_nodes.append(J.item())

            # Creating a new node if we've reset all of them
            if len(already_reset_nodes) >= N:                
                if allow_category_growth:
                    self.weight_a = torch.vstack((self.weight_a, torch.ones((self.node_increase_step,  self.weight_a.shape[1]), device=self.device, dtype=torch.float)))
                    self.weight_ab = torch.vstack((self.weight_ab, torch.ones((self.node_increase_step, self.weight_ab.shape[1]), device=self.device, dtype=torch.float)))
                    self.number_of_increases += 1
                    # Give the new F2a node a w_ab entry, this new node should win
                else:                   
                    self.rho_ab = 0
                    self.beta_ab = 0.75
                    self.rho_a_bar = 0
                    rho_a = self.rho_a_bar
                    logger.info(f"Maximum number of nodes reached, {len(already_reset_nodes)} - adjusting rho_ab to {self.rho_ab} and beta_ab to {self.beta_ab}")
                    already_reset_nodes.clear()
                
                N, S, T = self.calculate_activation(input_vector)

        return J.item(), membership_degree.item()
    
    def calculate_activation(self, input_vector):
        N = self.weight_a.shape[0]  # Count how many F2a nodes we have

        A_for_each_F2_node = input_vector * torch.ones((N, 1), device=self.device, dtype=torch.float)
            # Matrix containing a copy of A for each F2 node. 
            # was optimization for Matlab, might be different in Python

        A_AND_w = torch.minimum(A_for_each_F2_node, self.weight_a)
            # Fuzzy AND = min

        S = torch.sum(A_AND_w, 1)
            # Row vector of signals to F2 nodes

        T = S / (self.alpha + torch.sum(self.weight_a, 1))
        # Choice function vector for F2
        return N,S,T
    # @profile
    def train(self, input_vector: np.array, class_vector: np.array):
        rho_a = self.rho_a_bar # We start off with ARTa vigilance at baseline
        resonant_ab = False # Not resonating in the Fab match layer
        already_reset_nodes = [] # We haven't rest any ARTa nodes for this input pattern yet, maintain list between resonance searches of Fa
        
        class_vector = class_vector.to(self.device)
        input_vector = input_vector.to(self.device)
        
        while not resonant_ab:            
            J, x = self._resonance_search(input_vector, already_reset_nodes, rho_a, not self.max_nodes_mode)
            z = torch.minimum(class_vector, self.weight_ab[J, None])
            
            resonance = torch.sum(z, 1)/torch.sum(class_vector, 1)
            if resonance > self.rho_ab or math.isclose(resonance, self.rho_ab):
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

        self.weight_a[J, None] = (beta * torch.minimum(input_vector, self.weight_a[J, None])) + ((1-beta) * self.weight_a[J, None])
        self.weight_ab[J, None] = (self.beta_ab * z) + ((1-self.beta_ab) * self.weight_ab[J, None])
        self.committed_nodes.add(J)

    def fit(self, input_vectors, class_vectors):
        for document_index, input_vector in enumerate(input_vectors):
            self.train(FuzzyArtMapGpuWorker.complement_encode(torch.tensor(input_vector.toarray(), dtype=torch.float)), FuzzyArtMapGpuWorker.complement_encode(torch.tensor([[class_vectors[document_index]]], dtype=torch.float)))            
        logger.info(f"updated nodes: {','.join([str(J) for J in self.updated_nodes])}")
        self.updated_nodes.clear()
        self.number_of_increases = 0
    
    def clear_updated_nodes(self):
        self.updated_nodes.clear()
        self.number_of_increases = 0


class FuzzyArtMapGpuWorker:
    def __init__(self):
        self.alpha = 0.001  # "Choice" parameter > 0. Set small for the conservative limit (Fuzzy AM paper, Sect.3)
        self.excluded_document_ids = set()
        self.weight_a = None
        self.weight_ab = None
        self.document_index_mapping = None
        self.device = None
        self.input_vector_sum = None
        self.committed_nodes = set()
        self.classes_ = np.array([1])
        self.updated_nodes = set()
        self.node_increase_step = 5 # number of F2 nodes to add when required
        self.number_of_increases = 0

        self.beta = 1  # Learning rate. Set to 1 for fast learning
        self.beta_ab = 1 #ab learning rate
        
        
        self.rho_ab = 0.95          # Map field vigilance, in [0,1]
        self.epsilon = 0.001        # Fab mismatch raises ARTa vigilance to this much above what is needed to reset ARTa

        self.rho_a_bar = None  # Baseline vigilance for ARTa, in range [0,1]
        self.max_nodes_mode = None

    def init(self, f1_size: int = 10, f2_size: int = 10, number_of_categories: int = 2, rho_a_bar = 0, max_nodes_mode = False, use_cuda_if_available = False):
        self.rho_a_bar = rho_a_bar  # Baseline vigilance for ARTa, in range [0,1]
        self.max_nodes_mode = max_nodes_mode
        if use_cuda_if_available and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            if use_cuda_if_available:
                logger.warning("CUDA requested but not available, using CPU.")
        self.weight_a = torch.ones((f2_size, f1_size), device=self.device, dtype=torch.float)
        self.input_vector_sum = f1_size / 2
        self.weight_ab = torch.ones((f2_size, number_of_categories), device=self.device, dtype=torch.float)

    # def update_weights(self, number_of_new_nodes, update_indexes, a_updates, ab_updates):
    #     if number_of_new_nodes > 0:
    #         self.weight_a = torch.vstack((self.weight_a, torch.ones((number_of_new_nodes,  self.weight_a.shape[1]), device=self.device, dtype=torch.float)))
    #         self.weight_ab = torch.vstack((self.weight_ab, torch.ones((number_of_new_nodes, self.weight_ab.shape[1]), device=self.device, dtype=torch.float)))
    #         self.S_cache = torch.hstack((self.S_cache, torch.ones((self.corpus.shape[0], number_of_new_nodes), device=self.device, dtype=torch.float)))
    #     self.weight_a[update_indexes] = a_updates
    #     self.weight_ab[update_indexes] = ab_updates
    #     self.recompute_S_cache(update_indexes)
    def _resonance_search(self, input_vector: torch.tensor, already_reset_nodes: List[int], rho_a: float, allow_category_growth = True):
        resonant_a = False
        N, S, T = self.calculate_activation(input_vector)
        while not resonant_a:
            
            T[already_reset_nodes] = torch.zeros((len(already_reset_nodes), ), dtype=torch.float, device=self.device)
            J = np.argmax(T)
            membership_degree = S[J]/self.input_vector_sum
            if membership_degree.item() >= rho_a or math.isclose(membership_degree.item(), rho_a):
                resonant_a = True
            else:
                resonant_a = False
                already_reset_nodes.append(J.item())

            # Creating a new node if we've reset all of them
            if len(already_reset_nodes) >= N:                
                if allow_category_growth:
                    self.weight_a = torch.vstack((self.weight_a, torch.ones((self.node_increase_step,  self.weight_a.shape[1]), device=self.device, dtype=torch.float)))
                    self.weight_ab = torch.vstack((self.weight_ab, torch.ones((self.node_increase_step, self.weight_ab.shape[1]), device=self.device, dtype=torch.float)))
                    self.S_cache = torch.hstack((self.S_cache, torch.ones((self.corpus.shape[0], self.node_increase_step), device=self.device, dtype=torch.float)))
                    self.number_of_increases += 1
                    # Give the new F2a node a w_ab entry, this new node should win
                else:                   
                    self.rho_ab = 0
                    self.beta_ab = 0.75
                    self.rho_a_bar = 0
                    rho_a = self.rho_a_bar
                    logger.info(f"Maximum number of nodes reached, {len(already_reset_nodes)} - adjusting rho_ab to {self.rho_ab} and beta_ab to {self.beta_ab}")
                    already_reset_nodes.clear()
                
                N, S, T = self.calculate_activation(input_vector)

        return J.item(), membership_degree.item()
    
    def calculate_activation(self, input_vector):
        N = self.weight_a.shape[0]  # Count how many F2a nodes we have

        A_for_each_F2_node = input_vector * torch.ones((N, 1), device=self.device, dtype=torch.float)
            # Matrix containing a copy of A for each F2 node. 
            # was optimization for Matlab, might be different in Python

        A_AND_w = torch.minimum(A_for_each_F2_node, self.weight_a)
            # Fuzzy AND = min

        S = torch.sum(A_AND_w, 1)
            # Row vector of signals to F2 nodes

        T = S / (self.alpha + torch.sum(self.weight_a, 1))
        # Choice function vector for F2
        return N,S,T

    def train(self, input_vector: np.array, class_vector: np.array):
        rho_a = self.rho_a_bar # We start off with ARTa vigilance at baseline
        resonant_ab = False # Not resonating in the Fab match layer
        already_reset_nodes = [] # We haven't rest any ARTa nodes for this input pattern yet, maintain list between resonance searches of Fa
        
        class_vector = class_vector.to(self.device)
        input_vector = input_vector.to(self.device)
        
        while not resonant_ab:            
            J, x = self._resonance_search(input_vector, already_reset_nodes, rho_a, not self.max_nodes_mode)
            z = torch.minimum(class_vector, self.weight_ab[J, None])
            
            resonance = torch.sum(z, 1)/torch.sum(class_vector, 1)
            if resonance > self.rho_ab or math.isclose(resonance, self.rho_ab):
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

        self.weight_a[J, None] = (beta * torch.minimum(input_vector, self.weight_a[J, None])) + ((1-beta) * self.weight_a[J, None])
        self.weight_ab[J, None] = (self.beta_ab * z) + ((1-self.beta_ab) * self.weight_ab[J, None])
        self.committed_nodes.add(J)

    def fit(self, input_vectors, class_vectors):
        for document_index, input_vector in enumerate(input_vectors):
            self.train(FuzzyArtMapGpuWorker.complement_encode(torch.tensor(input_vector.toarray(), dtype=torch.float)), FuzzyArtMapGpuWorker.complement_encode(torch.tensor([[class_vectors[document_index]]], dtype=torch.float)))            
        logger.info(f"updated nodes: {','.join([str(J) for J in self.updated_nodes])}")
        self.recompute_S_cache()
        self.updated_nodes.clear()
        # updated_nodes = list(self.updated_nodes)
        # number_of_added_nodes = self.number_of_increases * self.node_increase_step
        # return number_of_added_nodes, updated_nodes

    def cache_corpus(self, corpus):
        print(f"aa setting mapping to: {len(corpus)}")
        self.corpus = FuzzyArtMapGpuWorker.complement_encode(torch.tensor(corpus, device="cpu", dtype=torch.float))
        self.excluded_document_ids = set()
        N = self.weight_a.shape[0]
        A_AND_w =  torch.empty(self.weight_a.shape, device=self.device, dtype=torch.float)
        self.S_cache = torch.zeros((self.corpus.shape[0], N), device=self.device, dtype=torch.float)
        ones = torch.ones((N, 1), device=self.device, dtype=torch.float)
        for i in range(self.corpus.shape[0]):
            A_for_each_F2_node = self.corpus[i].to(self.device) * ones
            torch.minimum(A_for_each_F2_node, self.weight_a, out=A_AND_w)
            self.S_cache[i] = torch.sum(A_AND_w, 1)
        print("worker corpus caching complete")

    def cache_doc_mapping(self, ranker_params, document_index_mapping: dict):
        from ranking import Ranker, VectorizerType
        print(f"bb setting mapping to: {len(document_index_mapping)}")
        self.document_index_mapping = document_index_mapping
        ranker = Ranker("famdg")
        ranker.set_did_2_feature(ranker_params[0], None, None, ranker_params[1], ranker_params[2], ranker_params[3])
        corpus = ranker.get_feature_by_did(document_index_mapping.keys())
        self.corpus = FuzzyArtMapGpuWorker.complement_encode(torch.tensor(corpus.toarray(), device="cpu", dtype=torch.float))
        self.excluded_document_ids = set()
        N = self.weight_a.shape[0]
        A_AND_w =  torch.empty(self.weight_a.shape, device=self.device, dtype=torch.float)
        self.S_cache = torch.zeros((self.corpus.shape[0], N), device=self.device, dtype=torch.float)
        ones = torch.ones((N, 1), device=self.device, dtype=torch.float)
        for i in range(self.corpus.shape[0]):
            A_for_each_F2_node = self.corpus[i].to(self.device) * ones
            torch.minimum(A_for_each_F2_node, self.weight_a, out=A_AND_w)
            self.S_cache[i] = torch.sum(A_AND_w, 1)
        print("worker corpus caching complete")

    def remove_documents_from_cache(self, document_ids):
        for document_id in document_ids:
            self.excluded_document_ids.add(document_id)

    def recompute_S_cache(self):
        updated_nodes = list(self.updated_nodes)
        N = self.weight_a.shape[0]
        A_AND_w =  torch.empty((len(updated_nodes),self.weight_a.shape[1]), device=self.device, dtype=torch.float)
        ones = torch.ones((N, 1), device=self.device, dtype=torch.float)
        for document_id, index in self.document_index_mapping.items():
            if document_id in self.excluded_document_ids:
                continue
            A_for_each_F2_node = (self.corpus[index].to(self.device)) * ones
            torch.minimum(A_for_each_F2_node[updated_nodes], self.weight_a[updated_nodes], out=A_AND_w)
            self.S_cache[index, updated_nodes] = torch.sum(A_AND_w, 1)

    def _cached_resonance_search(self, cached_S):
        T = cached_S / self.choice_denominator
        # Choice function vector for F2
        J = torch.argmax(T) # Finding the winning node, J
        membership_degree = cached_S[J]/self.input_vector_sum
        return J.item(), membership_degree

    def cached_predict(self, document_id):
        index = self.document_index_mapping[document_id]
        cached_S_value = self.S_cache[index]
        J, membership_degree = self._cached_resonance_search(cached_S_value)
        return self.weight_ab[J, ], membership_degree # Fab activation vector & fuzzy membership value

    @staticmethod
    def complement_encode(original_vector: np.array) -> np.array:
        complement = 1-original_vector
        complement_encoded_value = torch.hstack((original_vector,complement))
        return complement_encoded_value

    def predict_proba(self, doc_ids: list):
        predictions = []
        self.choice_denominator = (self.alpha + torch.sum(self.weight_a,  1))
        for document_id in doc_ids:
            if document_id not in self.document_index_mapping:
                continue
            prediction, membership_degree = self.cached_predict(document_id)
            if prediction[0].item():
                predictions.append((membership_degree.item(), 0, document_id))
        
        return predictions #needs to be in shape (number_of_docs, 1)  


class FuzzyArtmapWorkerServer(TCPServer):
    def __init__(self, ssl_options = None, max_buffer_size = None, read_chunk_size = None) -> None:        
        super().__init__(ssl_options, max_buffer_size, read_chunk_size)
        self.model = None
        self.end = "\n".encode("utf-8")
        self.end_mark = "|||".encode("utf-8")
        gc.disable()

    async def handle_stream(self, stream, address):
        buffer_size = 4096
        total_data = bytearray()
        while True:
            try:
                data = await stream.read_until(self.end_mark)
                print(f"{len(data)}")
                await self.handle_data(data[:-3], stream)
                # await stream.write(data)
                # data = await stream.read_bytes(buffer_size, True)                
                # total_data.extend(data)
                # if not data or len(data) < buffer_size:
                #     await self.handle_data(total_data, stream)
                #     total_data.clear()
            except StreamClosedError:
                print("connection closed")
                break
    
    async def handle_data(self, data, stream):
        logger.info(f"received header: {chr(data[0])}")
        if data[0] == 114: # "r" - remove doc ids
            doc_ids = pickle.loads(data[1:])
            self.model.remove_documents_from_cache(doc_ids)
            await stream.write(self.end)
            logger.info("remove docs completed")
        
        elif data[0] == 105: # "i" - init
            self.model = FuzzyArtMapGpuWorker()
            gc.collect()
            init_params = pickle.loads(data[1:])
            self.model.init(*init_params)
            await stream.write(self.end)
            logger.info("init completed")

        elif data[0] == 100: # "d" - cache doc mapping
            logger.info(f"{len(data)}")
            ranker_params, doc_index = pickle.loads(data[1:])
            self.model.cache_doc_mapping(ranker_params, doc_index)   
            # await stream.write(self.end)
            await stream.write(b'corpus')
            logger.info("cache doc mapping completed")

        elif data[0] == 112: # "p" - predict
            doc_ids = pickle.loads(data[1:])
            results = self.model.predict_proba(doc_ids)
            await stream.write(pickle.dumps(results)+self.end_mark)
            logger.info("predict completed")            
        
        elif data[0] == 117: # "u" - update weights
            logger.info(f"Updated weights payload size: {len(data)}")
            serialized_data = io.BytesIO(data[1:])
            # serialized_data.seek(0)
            # number_of_new_nodes, updated_nodes, updated_a_weights, updated_ab_weights = torch.load(serialized_data)
            # self.model.update_weights(number_of_new_nodes, updated_nodes, updated_a_weights, updated_ab_weights)
            params = torch.load(serialized_data)
            self.model.update_weights(params["number_of_new_nodes"], params["updated_nodes"], params["updated_a_weights"], params["updated_ab_weights"])
            await stream.write(self.end)
            logger.info("update weights completed")
        
        elif data[0] == 99: # "c" - cache corpus
            print(f"{len(data)}")
            numpy_data = io.BytesIO(data[1:-3])
            # print(f"{numpy_data.getbuffer().count()}")
            numpy_data.seek(0)
            with np.load(numpy_data, allow_pickle=True) as d:
                corpus_array = d["cache"]
            self.model.cache_corpus(corpus_array) # TODO: somehow I don't think this is blocking
            await stream.write(b'corpus')
            logger.info("cache corpus completed")

        elif data[0] == 102: # "f" - fit data
            logger.info(f"Training data size: {len(data)}")
            params = pickle.loads(data[1:])
            self.model.fit(*params)
            await stream.write(self.end)
            logger.info("training completed")
        
        else:
            print(data)

class FuzzyArtmapWorkerClient():
    def __init__(self, registrar_address) -> None:        
        self.host, self.port = registrar_address.split(":")
        self.workers = []
        self.init_header = "i".encode("utf-8")
        self.get_worker_address_header = "g".encode("utf-8")
        self.cache_corpus_header = "c".encode("utf-8")
        self.cache_doc_mapping_header = "d".encode("utf-8")
        self.remove_documents_header = "r".encode("utf-8")
        self.update_weights_header = "u".encode("utf-8")
        self.predict_header = "p".encode("utf-8")
        self.fit_header = "f".encode("utf-8")
        self.payload_seperator = "|".encode("utf-8")
        self.end_mark = b"\n"
        self.sending_end_mark = "|||".encode("utf-8")
    
    async def get_workers(self):
        client = TCPClient()
        print(f"connecting to registrar: {self.host}:{self.port}")
        stream = await client.connect(self.host, int(self.port))
        await stream.write(self.get_worker_address_header)
        response = await stream.read_until(self.end_mark)
        response = response.rstrip()
        for worker_address in response.decode("utf-8").split("|"):            
            worker_host, worker_port = worker_address.split(":")
            print(f"connecting to worker: {worker_address}")
            worker_stream = await client.connect(worker_host, int(worker_port))
            self.workers.append(worker_stream)
    
    async def init_workers(self, params):
        logger.info("init workers entered")
        futures = []
        pickled_params = pickle.dumps(params)
        for worker in self.workers:
           futures.append(worker.write(self.init_header + pickled_params + self.sending_end_mark))
        
        await asyncio.gather(*futures)
        await self.get_responses()
        logger.info("init workers completed")

    async def cache_corpus(self, ranker_params, document_index_chunks):
        logger.info("cache corpus entered")
        caching_futures = []
        for index, worker in enumerate(self.workers):
            chunk_document_id_index = {document_id: index for index, document_id in enumerate(document_index_chunks[index])}
            caching_futures.append(worker.write(self.cache_doc_mapping_header + pickle.dumps((ranker_params, chunk_document_id_index)) + self.sending_end_mark))

        for index in range(len(self.workers)):
            caching_futures.append(self.workers[index].read_until(b'corpus'))
        
        await asyncio.gather(*caching_futures)
        logger.info("cache corpus completed")
    
    async def remove_documents_from_cache(self, document_ids):
        logger.info("remove docs entered")
        futures = []
        pickled_doc_ids = pickle.dumps(document_ids)
        for worker in self.workers:
           futures.append(worker.write(self.remove_documents_header + pickled_doc_ids + self.sending_end_mark))

        await asyncio.gather(*futures)
        await self.get_responses()
        logger.info("remove docs completed")

    async def fit(self, params):
        logger.info("fit entered")
        futures = []
        pickled_params= pickle.dumps(params)
        for worker in self.workers:
           futures.append(worker.write(self.fit_header + pickled_params + self.sending_end_mark))

        await asyncio.gather(*futures)
        await self.get_responses()
        logger.info("remove docs completed")

    async def update_weights(self, params):
        logger.info("update weights entered")
        futures = []
        buffer = io.BytesIO()
        torch.save(params, buffer)
        # buffer.seek(0)
        for worker in self.workers:
           futures.append(worker.write(self.update_weights_header + buffer.getbuffer() + self.sending_end_mark))
        
        await asyncio.gather(*futures)
        await self.get_responses()
        logger.info("update weights completed")
    
    async def predict_proba(self, doc_ids):
        logger.info("predict entered")
        pickled_doc_ids = pickle.dumps(doc_ids)
        futures = []
        for worker in self.workers:
           futures.append(self.single_predict(pickled_doc_ids, worker))
        worker_results = await asyncio.gather(*futures)
        results = []
        for result in worker_results:
            results.extend(result)
        logger.info("predict completed")
        return results

    async def single_predict(self, pickled_doc_ids, worker):
        worker.write(self.predict_header + pickled_doc_ids + self.sending_end_mark)
        data = await worker.read_until(self.sending_end_mark)
        return pickle.loads(data[:-3])

    async def get_responses(self):
        response_futures = []
        for worker in self.workers:
            response_futures.append(await worker.read_until(self.end_mark))
        return response_futures

async def register_worker():
    client = TCPClient()
    host, port = args.registrar.split(":")
    print(f"connecting to registrar: {args.registrar}")
    stream = await client.connect(host, int(port))
    if args.localhost:
        data = f"r{socket.gethostbyname('localhost')}:{args.port}"
    else:
        hostname = socket.gethostname()
        data = f"r{socket.gethostbyname(hostname)}:{args.port}"
    print(f"registering worker at {data}")
    data = data.encode("utf-8")
    await stream.write(data)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-r", "--registrar", help="ip:port of the registrar server", required=True)
    arg_parser.add_argument("-l", "--localhost", help="report localhost as the worker address", action=argparse.BooleanOptionalAction)
    arg_parser.add_argument("-p", "--port", help="worker listener port, override default 48576", default="48576")
    
    args = arg_parser.parse_args()
    
    tornado.ioloop.IOLoop.current().run_sync(register_worker)
    server = FuzzyArtmapWorkerServer()
    print('Starting the server...')
    server.listen(int(args.port))
    tornado.ioloop.IOLoop.current().start()
    print('Server has shut down.')