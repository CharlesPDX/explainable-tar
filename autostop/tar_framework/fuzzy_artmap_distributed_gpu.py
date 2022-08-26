# Based on fuzzyartmap_demo.m by Rajeev Raizada, wih permission, and 
# Carpenter, G. A., Grossberg, S., Markuzon, N., Reynolds, J. H. and Rosen, D. B. (1992)
# "Fuzzy ARTMAP: A Neural Network Architecture for Incremental Supervised Learning of Analog Multidimensional Maps"
# IEEE Transactions on Neural Networks, Vol. 3, No. 5, pp. 698-713.

# import cProfile
# import pstats

import gc
import sys
import asyncio
import socket
import pickle
import argparse
import math
import struct
import traceback
from typing import List
from datetime import datetime
import logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

file_logging_handler = logging.FileHandler('fuzzy_artmap_gpu_distributed.log')
file_logging_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_logging_handler.setFormatter(file_logging_format)
logger.addHandler(file_logging_handler)


import numpy as np
import torch

from tornado.tcpserver import TCPServer
from tornado.tcpclient import TCPClient
from tornado.iostream import StreamClosedError
import tornado.ioloop

# TODO: Add option to use slow recode or not
# TODO: Add param for beta_a to use if using slow recode
class FuzzyArtmapGpuDistributed:
    def __init__(self, f1_size: int = 10, f2_size: int = 10, number_of_categories: int = 2, rho_a_bar = 0, scheduler_address=None, max_nodes = None, use_cuda_if_available = False, committed_beta = 0.75, active_learning_mode = "ranked", batch_size = 100):
        self.rho_a_bar = rho_a_bar  # Baseline vigilance for ARTa, in range [0,1]
        self.committed_beta = committed_beta
        self.active_learning_mode = active_learning_mode
        self.batch_size = batch_size
        self.f2_size = f2_size
        self.f1_size = f1_size
        self.number_of_categories = number_of_categories
        self.classes_ = np.array([1])
        self.max_nodes = max_nodes
        self.use_cuda_if_available = use_cuda_if_available
        self.client = FuzzyArtmapWorkerClient(scheduler_address)
        self.training_fuzzy_artmap = LocalFuzzyArtMapGpu(self.f1_size, self.f2_size, self.number_of_categories, self.rho_a_bar, max_nodes, use_cuda_if_available, committed_beta, active_learning_mode)
        self.weight_ab = self.training_fuzzy_artmap.weight_ab
        # self.weight_ab = torch.empty((1,1))
    
    async def initialize_workers(self):
        logger.info("Getting workers")
        await self.client.get_workers()
        logger.info(f"Initializing {len(self.client.workers)} workers")
        params = [self.f1_size, self.f2_size, self.number_of_categories, self.rho_a_bar, self.max_nodes, self.use_cuda_if_available, self.committed_beta, self.active_learning_mode, self.batch_size]
        logger.info(f"committed beta = {self.committed_beta}, active learning mode = {self.active_learning_mode}, batch size = {self.batch_size}")
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
    
    async def fit(self, input_vectors, class_vectors, doc_ids):
        io_loop = tornado.ioloop.IOLoop.current().asyncio_loop
        remote_fit = io_loop.create_task(self.client.fit([input_vectors, class_vectors, doc_ids]))
        # await asyncio.gather(remote_fit)
        local_fit = io_loop.run_in_executor(None, self.training_fuzzy_artmap.fit, input_vectors, class_vectors)
        await asyncio.gather(remote_fit, local_fit)
        self.training_fuzzy_artmap.clear_updated_nodes()

    async def predict_proba(self, doc_ids: list):
        results = await self.client.predict_proba(doc_ids)
        return np.array(results)

    def save_model(self, descriptor):
        # return ""
        return self.training_fuzzy_artmap.save_model(descriptor)
    
    def get_number_of_nodes(self):
        return self.training_fuzzy_artmap.weight_ab.shape[0]

    def get_number_of_increases(self):
        return self.training_fuzzy_artmap.number_of_increases

    def get_increase_size(self):
        return self.training_fuzzy_artmap.node_increase_step
    
    def get_committed_nodes(self):
        return ",".join([str(n) for n in self.training_fuzzy_artmap.committed_nodes])


class LocalFuzzyArtMapGpu:
    def __init__(self, f1_size: int = 10, f2_size: int = 10, number_of_categories: int = 2, rho_a_bar = 0, max_nodes = None, use_cuda_if_available = False, committed_beta = 0.75, active_learning_mode = "ranked", batch_size = 100):
        self.alpha = 0.001  # "Choice" parameter > 0. Set small for the conservative limit (Fuzzy AM paper, Sect.3)
        self.beta = 1  # Learning rate. Set to 1 for fast learning
        self.committed_beta = committed_beta
        self.active_learning_mode = active_learning_mode
        self.batch_size = batch_size
        logger.info(f"committed beta = {self.committed_beta}, active learning mode = {self.active_learning_mode}, batch size = {self.batch_size}")
        self.beta_ab = 1 #ab learning rate
        self.rho_a_bar = rho_a_bar  # Baseline vigilance for ARTa, in range [0,1]
        
        self.rho_ab = 0.95          # Map field vigilance, in [0,1]
        self.epsilon = 0.001        # Fab mismatch raises ARTa vigilance to this much above what is needed to reset ARTa

        self.max_nodes = max_nodes

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

        self.node_increase_step = 10 # number of F2 nodes to add when required
        self.number_of_increases = 0
        
        # self.A_for_each_F2_node = torch.empty(self.weight_a.shape, device=self.device, dtype=torch.float)
        self.A_and_w = torch.empty(self.weight_a.shape, device=self.device, dtype=torch.float)
        self.ones = torch.ones((f2_size, 1), device=self.device, dtype=torch.float)
        self.class_vectors = {
            0: FuzzyArtMapGpuWorker.complement_encode(torch.tensor([[0]], dtype=torch.float)),
            1: FuzzyArtMapGpuWorker.complement_encode(torch.tensor([[1]], dtype=torch.float)),
        }
    
    def _resonance_search(self, input_vector: torch.tensor, already_reset_nodes: List[int], rho_a: float):
        # self.profiler.enable()
        resonant_a = False
        N, S, T = self.calculate_activation(input_vector)
        while not resonant_a:
            T[already_reset_nodes] = torch.zeros((len(already_reset_nodes), ), dtype=torch.float, device=self.device)
            J = torch.argmax(T)
            membership_degree = S[J]/self.input_vector_sum
            if membership_degree.item() >= rho_a or math.isclose(membership_degree.item(), rho_a):
                resonant_a = True
            else:
                resonant_a = False
                already_reset_nodes.append(J.item())

            # Creating a new node if we've reset all of them
            if len(already_reset_nodes) >= N:                
                if self.max_nodes is None or self.max_nodes >= (N + self.node_increase_step):
                    self.weight_a = torch.vstack((self.weight_a, torch.ones((self.node_increase_step,  self.weight_a.shape[1]), device=self.device, dtype=torch.float)))
                    self.weight_ab = torch.vstack((self.weight_ab, torch.ones((self.node_increase_step, self.weight_ab.shape[1]), device=self.device, dtype=torch.float)))
                    self.A_and_w = torch.vstack((self.A_and_w, torch.empty((self.node_increase_step,  self.weight_a.shape[1]), device=self.device, dtype=torch.float)))
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

    def _resonance_search_vector(self, input_vector: torch.tensor, already_reset_nodes: List[int], rho_a: float):
        resonant_a = False
        N, S, T = self.calculate_activation(input_vector)
        T[already_reset_nodes] = torch.zeros((len(already_reset_nodes), ), dtype=torch.float, device=self.device)
        sorted_values, indices = torch.sort(T, stable=True, descending=True)
        all_membership_degrees = S / self.input_vector_sum
        while not resonant_a:
            for J in indices:
                if J.item() in already_reset_nodes:
                    continue

                if all_membership_degrees[J].item() >= rho_a or math.isclose(all_membership_degrees[J].item(), rho_a):
                    resonant_a = True
                    break
                else:
                    resonant_a = False
                    already_reset_nodes.append(indices[J].item())
                    T[indices[J].item()] = 0

            # Creating a new node if we've reset all of them
            if len(already_reset_nodes) >= N:                
                if self.max_nodes is None or self.max_nodes >= (N + self.node_increase_step):
                    self.weight_a = torch.vstack((self.weight_a, torch.ones((self.node_increase_step,  self.weight_a.shape[1]), device=self.device, dtype=torch.float)))
                    self.weight_ab = torch.vstack((self.weight_ab, torch.ones((self.node_increase_step, self.weight_ab.shape[1]), device=self.device, dtype=torch.float)))
                    self.A_and_w = torch.vstack((self.A_and_w, torch.empty((self.node_increase_step,  self.weight_a.shape[1]), device=self.device, dtype=torch.float)))
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
                sorted_values, indices = torch.sort(S, stable=True)
                all_membership_degrees = sorted_values / self.input_vector_sum
                T[already_reset_nodes] = torch.zeros((len(already_reset_nodes), ), dtype=torch.float, device=self.device)

        return J.item(), all_membership_degrees[J].item()

    def calculate_activation(self, input_vector):
        N = self.weight_a.shape[0]  # Count how many F2a nodes we have
        # self.ensure_A_for_each_F2_node(N)

        torch.minimum(input_vector.repeat(N,1), self.weight_a, out=self.A_and_w) # Fuzzy AND = min
        S = torch.sum(self.A_and_w, 1) # Row vector of signals to F2 nodes
        T = S / (self.alpha + torch.sum(self.weight_a, 1)) # Choice function vector for F2
        return N,S,T
    
    # @profile
    def train(self, input_vector: torch.tensor, class_vector: torch.tensor):
        rho_a = self.rho_a_bar # We start off with ARTa vigilance at baseline
        resonant_ab = False # Not resonating in the Fab match layer
        already_reset_nodes = [] # We haven't rest any ARTa nodes for this input pattern yet, maintain list between resonance searches of Fa
        
        class_vector = class_vector.to(self.device)
        input_vector = input_vector.to(self.device)
        
        while not resonant_ab:            
            # J, x = self._resonance_search(input_vector, already_reset_nodes, rho_a)
            J, x = self._resonance_search_vector(input_vector, already_reset_nodes, rho_a)
            
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
            beta = self.committed_beta
        else:
            beta = self.beta

        self.weight_a[J, None] = (beta * torch.minimum(input_vector, self.weight_a[J, None])) + ((1-beta) * self.weight_a[J, None])
        self.weight_ab[J, None] = (self.beta_ab * z) + ((1-self.beta_ab) * self.weight_ab[J, None])
        self.committed_nodes.add(J)

    def fit(self, input_vectors, class_vectors):
        for document_index, input_vector in enumerate(input_vectors):
            self.train(FuzzyArtMapGpuWorker.complement_encode(torch.tensor(input_vector.toarray(), dtype=torch.float)), self.class_vectors[class_vectors[document_index]])            
        logger.info(f"updated nodes: {','.join([str(J) for J in self.updated_nodes])}")
        self.updated_nodes.clear()
        # self.number_of_increases = 0
    
    def clear_updated_nodes(self):
        self.updated_nodes.clear()
        # self.number_of_increases = 0
    
    def save_model(self, descriptor):
        model_timestamp = datetime.now().isoformat().replace("-", "_").replace(":", "_").replace(".", "_")
        cleaned_descriptor = descriptor.replace("-", "_").replace(":", "_").replace(".", "_")
        model_path = f"models/famgd_{model_timestamp}_{cleaned_descriptor}.pt"
        torch.save((self.weight_a, self.weight_ab), model_path)
        return model_path


class FuzzyArtMapGpuWorker:
    def __init__(self, use_vector = False):
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
        self.active_learning_mode = None
        self.batch_size = None

        self.beta = 1  # Learning rate. Set to 1 for fast learning
        self.committed_beta = 0.75
        self.beta_ab = 1 #ab learning rate
        
        
        self.rho_ab = 0.95          # Map field vigilance, in [0,1]
        self.epsilon = 0.001        # Fab mismatch raises ARTa vigilance to this much above what is needed to reset ARTa

        self.rho_a_bar = None  # Baseline vigilance for ARTa, in range [0,1]
        self.max_nodes = None
        self.class_vectors = None
        # self.A_for_each_F2_node = None
        self.A_and_w = None
        self.use_vector = use_vector

    def init(self, f1_size: int = 10, f2_size: int = 10, number_of_categories: int = 2, rho_a_bar = 0, max_nodes = None, use_cuda_if_available = False, committed_beta = 0.75, active_learning_mode = "ranked", batch_size = 100):
        self.rho_a_bar = rho_a_bar  # Baseline vigilance for ARTa, in range [0,1]
        self.committed_beta = committed_beta
        self.active_learning_mode = active_learning_mode
        self.batch_size = batch_size
        self.max_nodes = max_nodes
        if use_cuda_if_available and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            if use_cuda_if_available:
                logger.warning("CUDA requested but not available, using CPU.")
        self.weight_a = torch.ones((f2_size, f1_size), device=self.device, dtype=torch.float)
        self.input_vector_sum = f1_size / 2
        self.weight_ab = torch.ones((f2_size, number_of_categories), device=self.device, dtype=torch.float)
        self.class_vectors = {
            0: FuzzyArtMapGpuWorker.complement_encode(torch.tensor([[0]], dtype=torch.float)),
            1: FuzzyArtMapGpuWorker.complement_encode(torch.tensor([[1]], dtype=torch.float)),
        }
        self.A_and_w = torch.empty(self.weight_a.shape, device=self.device, dtype=torch.float)
        logger.info(f"f1_size: {f1_size}, f2_size:{f2_size}, committed beta = {self.committed_beta}, active learning mode = {self.active_learning_mode}, batch size = {self.batch_size}")
        # self.profiler = cProfile.Profile()

    def _resonance_search(self, input_vector: torch.tensor, already_reset_nodes: List[int], rho_a: float):
        resonant_a = False
        N, S, T = self.calculate_activation(input_vector)
        while not resonant_a:
            T[already_reset_nodes] = torch.zeros((len(already_reset_nodes), ), dtype=torch.float, device=self.device)
            J = torch.argmax(T)
            membership_degree = S[J]/self.input_vector_sum
            if membership_degree.item() >= rho_a or math.isclose(membership_degree.item(), rho_a):
                resonant_a = True
            else:
                resonant_a = False
                already_reset_nodes.append(J.item())

            # Creating a new node if we've reset all of them
            if len(already_reset_nodes) >= N:                
                if self.max_nodes is None or self.max_nodes >= (N + self.node_increase_step):
                    self.weight_a = torch.vstack((self.weight_a, torch.ones((self.node_increase_step,  self.weight_a.shape[1]), device=self.device, dtype=torch.float)))
                    self.weight_ab = torch.vstack((self.weight_ab, torch.ones((self.node_increase_step, self.weight_ab.shape[1]), device=self.device, dtype=torch.float)))
                    self.S_cache = torch.hstack((self.S_cache, torch.ones((self.corpus.shape[0], self.node_increase_step), device=self.device, dtype=torch.float)))
                    self.A_and_w = torch.vstack((self.A_and_w, torch.empty((self.node_increase_step,  self.weight_a.shape[1]), device=self.device, dtype=torch.float)))
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

    def _resonance_search_vector(self, input_vector: torch.tensor, already_reset_nodes: List[int], rho_a: float):
        # self.profiler.enable()
        resonant_a = False
        N, S, T = self.calculate_activation(input_vector)
        T[already_reset_nodes] = torch.zeros((len(already_reset_nodes), ), dtype=torch.float, device=self.device)
        sorted_values, indices = torch.sort(T, stable=True, descending=True)
        all_membership_degrees = S / self.input_vector_sum
        while not resonant_a:
            for J in indices:
                if J.item() in already_reset_nodes:
                    continue

                if all_membership_degrees[J].item() >= rho_a or math.isclose(all_membership_degrees[J].item(), rho_a):
                    resonant_a = True
                    break
                else:
                    resonant_a = False
                    already_reset_nodes.append(indices[J].item())
                    T[indices[J].item()] = 0

            # Creating a new node if we've reset all of them
            if len(already_reset_nodes) >= N:                
                if self.max_nodes is None or self.max_nodes >= (N + self.node_increase_step):
                    self.weight_a = torch.vstack((self.weight_a, torch.ones((self.node_increase_step,  self.weight_a.shape[1]), device=self.device, dtype=torch.float)))
                    self.weight_ab = torch.vstack((self.weight_ab, torch.ones((self.node_increase_step, self.weight_ab.shape[1]), device=self.device, dtype=torch.float)))
                    self.S_cache = torch.hstack((self.S_cache, torch.ones((self.corpus.shape[0], self.node_increase_step), device=self.device, dtype=torch.float)))
                    self.A_and_w = torch.vstack((self.A_and_w, torch.empty((self.node_increase_step,  self.weight_a.shape[1]), device=self.device, dtype=torch.float)))
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
                sorted_values, indices = torch.sort(S, stable=True)
                all_membership_degrees = sorted_values / self.input_vector_sum
                T[already_reset_nodes] = torch.zeros((len(already_reset_nodes), ), dtype=torch.float, device=self.device)
        # self.profiler.disable()
        # stats = pstats.Stats(self.profiler).sort_stats('cumtime')
        # stats.print_stats()
        return J.item(), all_membership_degrees[J].item()

    def calculate_activation(self, input_vector):
        N = self.weight_a.shape[0]  # Count how many F2a nodes we have
        # self.ensure_A_for_each_F2_node(N)

        torch.minimum(input_vector.repeat(N,1), self.weight_a, out=self.A_and_w) # Fuzzy AND = min
        S = torch.sum(self.A_and_w, 1) # Row vector of signals to F2 nodes
        T = S / (self.alpha + torch.sum(self.weight_a, 1)) # Choice function vector for F2
        return N,S,T

    def train(self, input_vector: torch.tensor, class_vector: torch.tensor):
        rho_a = self.rho_a_bar # We start off with ARTa vigilance at baseline
        resonant_ab = False # Not resonating in the Fab match layer
        already_reset_nodes = [] # We haven't rest any ARTa nodes for this input pattern yet, maintain list between resonance searches of Fa
        
        class_vector = class_vector.to(self.device)
        input_vector = input_vector.to(self.device)
        
        while not resonant_ab:            
            # J, x = self._resonance_search(input_vector, already_reset_nodes, rho_a)
            J, x = self._resonance_search_vector(input_vector, already_reset_nodes, rho_a)
            
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
            beta = self.committed_beta
        else:
            beta = self.beta

        self.weight_a[J, None] = (beta * torch.minimum(input_vector, self.weight_a[J, None])) + ((1-beta) * self.weight_a[J, None])
        self.weight_ab[J, None] = (self.beta_ab * z) + ((1-self.beta_ab) * self.weight_ab[J, None])
        self.committed_nodes.add(J)

    def fit(self, input_vectors, class_vectors, doc_ids):
        for document_index, input_vector in enumerate(input_vectors):
            doc_id = doc_ids[document_index]
            if doc_id in self.document_index_mapping:
                self.train(self.corpus[self.document_index_mapping[doc_id]], self.class_vectors[class_vectors[document_index]])
            else:
                self.train(FuzzyArtMapGpuWorker.complement_encode(torch.tensor(input_vector.toarray(), dtype=torch.float)), self.class_vectors[class_vectors[document_index]])
        logger.info(f"updated nodes: {','.join([str(J) for J in self.updated_nodes])}")
        self.recompute_S_cache()
        logger.info("updated S cache")
        self.updated_nodes.clear()

    def cache_doc_mapping(self, ranker_params, document_index_mapping: dict):
        from ranking import Ranker
        logger.info(f"worker corpus caching {len(document_index_mapping)} documents")
        self.document_index_mapping = document_index_mapping
        ranker = Ranker("famdg")
        logger.info(f"initializing features")
        ranker.set_did_2_feature(ranker_params[0], None, None, ranker_params[1], ranker_params[2], ranker_params[3])
        logger.info(f"getting features")
        corpus = ranker.get_feature_by_did(document_index_mapping.keys())
        logger.info(f"complement encoding")
        self.corpus = FuzzyArtMapGpuWorker.complement_encode(torch.tensor(corpus.toarray(), device="cpu", dtype=torch.float))
        
        self.excluded_document_ids = set()
        N = self.weight_a.shape[0]
        logger.info(f"initializing S_cache")
        self.S_cache = torch.tensor(self.input_vector_sum, device=self.device, dtype=torch.float).repeat(self.corpus.shape[0], N)
        logger.info("worker corpus caching complete")

    def remove_documents_from_cache(self, document_ids):
        for document_id in document_ids:
            self.excluded_document_ids.add(document_id)
            self.document_index_mapping.pop(document_id, None)

    def recompute_S_cache(self):        
        updated_nodes = list(self.updated_nodes)
        N = self.weight_a.shape[0]
        if self.use_vector:
            index = list(self.document_index_mapping.values())
            # try:
            #     A_AND_w = torch.empty((len(index), len(updated_nodes), self.weight_a.shape[1]), device=self.device, dtype=torch.float)
            #     torch.minimum(self.corpus[index].unsqueeze(1).repeat(1,len(updated_nodes),1), self.weight_a[updated_nodes].unsqueeze(0).repeat(len(index),1,1), out=A_AND_w)
            #     self.S_cache.index_put_((torch.tensor(index).unsqueeze(1), torch.tensor(updated_nodes).repeat(len(index),1)), torch.sum(A_AND_w, 2))
            # except:
            chunk_size = 5
            A_AND_w = torch.empty((chunk_size, len(updated_nodes), self.weight_a.shape[1]), device=self.device, dtype=torch.float)
            expanded_weights = self.weight_a[updated_nodes].unsqueeze(0).repeat(chunk_size,1,1)
            for i in range(0, len(index), chunk_size):
                sub_indexes = index[i:i+chunk_size]
                if len(sub_indexes) < chunk_size:
                    expanded_weights = self.weight_a[updated_nodes].unsqueeze(0).repeat(len(sub_indexes),1,1)
                    A_AND_w = torch.empty((len(sub_indexes), len(updated_nodes), self.weight_a.shape[1]), device=self.device, dtype=torch.float)
                torch.minimum(self.corpus[sub_indexes].unsqueeze(1).repeat(1,len(updated_nodes),1), expanded_weights, out=A_AND_w)
                self.S_cache.index_put_((torch.tensor(sub_indexes).unsqueeze(1), torch.tensor(updated_nodes).repeat(len(sub_indexes),1)), torch.sum(A_AND_w, 2))
        else:
            A_AND_w = torch.empty((N, len(updated_nodes)), device=self.device, dtype=torch.float)
            for index in self.document_index_mapping.values():
                torch.minimum(self.corpus[index].repeat(N,1)[updated_nodes], self.weight_a[updated_nodes], out=A_AND_w)
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
    def complement_encode(original_vector: torch.tensor) -> torch.tensor:
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
                if self.active_learning_mode == "random" and len(predictions) >= self.batch_size:
                    return predictions
        
        return predictions #needs to be in shape (number_of_docs, 1)  


class FuzzyArtmapWorkerServer(TCPServer):
    def __init__(self, ssl_options = None, max_buffer_size = None, read_chunk_size = None, use_vector = False) -> None:        
        super().__init__(ssl_options, max_buffer_size, read_chunk_size)
        self.model = None
        self.end_mark = "|||".encode("utf-8")
        self.protocol_overhead = 8
        self.prediction_response_header = "r".encode("utf-8")
        self.use_vector = use_vector
        gc.disable()

    async def handle_stream(self, stream, address):
        while True:
            data_buffer = bytearray()
            try:
                data = await stream.read_until(self.end_mark)
                expected_length = struct.unpack("I", data[1:5])[0]
                actual_length = len(data) - self.protocol_overhead
                while actual_length != expected_length:
                    logger.error(f"received {actual_length} so far, expected {expected_length} - waiting on remaining data")
                    data_buffer.extend(data)
                    data = await stream.read_until(self.end_mark)
                    logger.error(f"received {len(data)} extra")
                    actual_length += len(data)
                    if actual_length == expected_length:
                        logger.error(f"expected data arrived")
                        data_buffer.extend(data)
                        data = data_buffer
                        break
                try:
                    await self.handle_data(data[:-3], stream)
                    data_buffer.clear()
                except Exception as e:
                    if sys.version_info.minor >= 10:
                        traceback_string = ''.join(traceback.format_exception(e))
                    else:
                        traceback_string = ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))
                    logger.info(f"error running {chr(data[0])} operation - {traceback_string}")
                    error_response = "e".encode("utf-8")
                    worker_id = struct.pack("I", self.worker_index)
                    error_bytes = traceback_string.encode("utf-8")
                    error_length = struct.pack("I", len(error_bytes))
                    await stream.write(error_response + worker_id + error_length + error_bytes + self.end_mark)
            except StreamClosedError:
                logger.info("connection closed")
                break
    
    async def handle_data(self, data, stream):
        logger.info(f"received header: {chr(data[0])}")
        if data[0] == 114: # "r" - remove doc ids
            expected_length = struct.unpack("I", data[1:5])[0]
            actual_length = len(data) - 5
            if actual_length != expected_length:
                logger.error(f"received {actual_length} - expected {expected_length}")
            doc_ids = pickle.loads(data[5:])
            self.model.remove_documents_from_cache(doc_ids)
            await stream.write(self.end_mark)
            logger.info("remove docs completed")
        
        elif data[0] == 105: # "i" - init
            expected_length = struct.unpack("I", data[1:5])[0]
            actual_length = len(data) - 5
            if actual_length != expected_length:
                logger.error(f"received {actual_length} - expected {expected_length}")
            self.model = FuzzyArtMapGpuWorker(self.use_vector)
            gc.collect()
            self.worker_index = struct.unpack("I", data[5:9])[0]
            logger.info(f"worker_id: {self.worker_index}")
            init_params = pickle.loads(data[9:])
            
            self.model.init(*init_params)
            await stream.write(self.end_mark)
            logger.info("init completed")

        elif data[0] == 99: # "c" - cache doc mapping/corpus
            ranker_params, doc_index = pickle.loads(data[5:])
            self.model.cache_doc_mapping(ranker_params, doc_index)   
            # await stream.write(b'corpus')
            await stream.write(self.end_mark)
            logger.info("caching completed")

        elif data[0] == 112: # "p" - predict
            doc_ids = pickle.loads(data[5:])
            results = self.model.predict_proba(doc_ids)
            pickled_results = pickle.dumps(results)
            results_length = struct.pack("I", len(pickled_results))
            await stream.write(self.prediction_response_header + results_length + pickled_results + self.end_mark)
            logger.info("predict completed")            

        elif data[0] == 102: # "f" - fit data
            params = pickle.loads(data[5:])
            self.model.fit(*params)
            await stream.write(self.end_mark)
            logger.info("training completed")
        
        else:
            print(data)

class FuzzyArtmapWorkerClient():
    def __init__(self, registrar_address) -> None:        
        self.host, self.port = registrar_address.split(":")
        self.workers = []
        self.init_header = "i".encode("utf-8")
        self.get_worker_address_header = "g".encode("utf-8")
        self.cache_header = "c".encode("utf-8")
        self.remove_documents_header = "r".encode("utf-8")
        self.predict_header = "p".encode("utf-8")
        self.fit_header = "f".encode("utf-8")
        self.payload_seperator = "|".encode("utf-8")
        self.prediction_response_header = "r".encode("utf-8")
        # self.end_mark = b"\n"
        self.end_mark = "|||".encode("utf-8")
    
    async def get_workers(self):
        client = TCPClient()
        logger.info(f"connecting to registrar: {self.host}:{self.port}")
        stream = await client.connect(self.host, int(self.port))
        await stream.write(self.get_worker_address_header)
        response = await stream.read_until(self.end_mark)        
        for worker_address in response[:-3].decode("utf-8").split(","):
            worker_host, worker_port = worker_address.split(":")
            logger.info(f"connecting to worker: {worker_address}")
            worker_stream = await client.connect(worker_host, int(worker_port))
            self.workers.append(worker_stream)
    
    async def init_workers(self, params):
        logger.info("init workers entered")
        futures = []
        pickled_params = pickle.dumps(params)
        params_length = struct.pack("I", len(pickled_params) + 4)
        for worker_index, worker in enumerate(self.workers):
           futures.append(worker.write(self.init_header + params_length + struct.pack("I", worker_index) + pickled_params + self.end_mark))
        
        await asyncio.gather(*futures)
        await self.get_responses()
        logger.info("init workers completed")

    async def cache_corpus(self, ranker_params, document_index_chunks):
        logger.info("cache corpus entered")
        caching_futures = []
        for index, worker in enumerate(self.workers):
            chunk_document_id_index = {document_id: i for i, document_id in enumerate(document_index_chunks[index])}
            pickled_index = pickle.dumps((ranker_params, chunk_document_id_index))
            index_length = struct.pack("I", len(pickled_index))
            caching_futures.append(worker.write(self.cache_header + index_length + pickled_index + self.end_mark))

        await asyncio.gather(*caching_futures)
        await self.get_responses()
        logger.info("cache corpus completed")
    
    async def remove_documents_from_cache(self, document_ids):
        futures = []
        pickled_doc_ids = pickle.dumps(document_ids)
        doc_id_length = struct.pack("I", len(pickled_doc_ids))
        for worker in self.workers:
           futures.append(worker.write(self.remove_documents_header + doc_id_length + pickled_doc_ids + self.end_mark))

        await asyncio.gather(*futures)
        await self.get_responses()

    async def fit(self, params):
        logger.info("starting remote fit")
        futures = []
        pickled_params = pickle.dumps(params)
        params_size = struct.pack("I", len(pickled_params))
        for worker_index, worker in enumerate(self.workers):
           futures.append(worker.write(self.fit_header + params_size + pickled_params + self.end_mark))
        #    logger.info(f"fit sent to {worker_index}")

        await asyncio.gather(*futures)
        await self.get_responses()
        logger.info("exiting remote fit")
    
    async def predict_proba(self, doc_ids):
        logger.info("predict entered")
        pickled_doc_ids = pickle.dumps(doc_ids)
        params_size = struct.pack("I", len(pickled_doc_ids))
        futures = []
        for worker in self.workers:
            futures.append(worker.write(self.predict_header + params_size + pickled_doc_ids + self.end_mark))
        
        await asyncio.gather(*futures)
        worker_results = await self.get_responses()
        results = []
        for result in worker_results:
            results.extend(pickle.loads(result[5:-3]))
        logger.info("predict completed")
        return results

    def check_response(self, response):
        if len(response) != 3:
            if response[0] == 114 and response[-3:] == self.end_mark:
                return

            if chr(response[0]) == "e":
                worker_id = struct.unpack("I", response[1:5])[0]
                error_stop_index = struct.unpack("I", response[5:9])[0] + 9
                error_message = response[9:error_stop_index].decode("utf-8")
                exception_message = f"worker {worker_id} returned error {error_message}"
                logger.error(exception_message)
                raise Exception(exception_message)
            else:
                raise Exception(f"unknown worker error")

    async def get_responses(self):
        response_futures = []
        for worker in self.workers:
            response_futures.append(worker.read_until(self.end_mark))
        
        responses = await asyncio.gather(*response_futures)
        results = []
        for response in responses:
            self.check_response(response)
            results.append(response)
        return results

async def register_worker():
    if args.localhost:
        data = f"r{socket.gethostbyname('localhost')}:{args.port}"
    else:        
        hostname = socket.gethostname()
        local_ip_address = socket.gethostbyname(hostname)
        if local_ip_address == "127.0.0.1":
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 1))  # connect() for UDP doesn't send packets
            local_ip_address = s.getsockname()[0]
        data = f"r{local_ip_address}:{args.port}"
    
    client = TCPClient()
    host, port = args.registrar.split(":")
    logger.info(f"connecting to registrar: {args.registrar}")
    stream = await client.connect(host, int(port))
    logger.info(f"registering worker at {data}")
    data = data.encode("utf-8")
    await stream.write(data)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-r", "--registrar", help="ip:port of the registrar server", required=True)
    arg_parser.add_argument("-l", "--localhost", help="report localhost as the worker address", action=argparse.BooleanOptionalAction)
    arg_parser.add_argument("-p", "--port", help="worker listener port, override default 48576", default="48576")
    arg_parser.add_argument("-v", "--vector", help="vectorize inference", default=False, action=argparse.BooleanOptionalAction)

    args = arg_parser.parse_args()
    
    tornado.ioloop.IOLoop.current().run_sync(register_worker)
    server = FuzzyArtmapWorkerServer(use_vector=args.vector)
    logger.info('Starting the server...')
    server.listen(int(args.port))
    tornado.ioloop.IOLoop.current().start()
    logger.info('Server has shut down.')