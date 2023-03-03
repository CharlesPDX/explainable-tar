# Based on fuzzyartmap_demo.m by Rajeev Raizada, wih permission, and 
# Carpenter, G. A., Grossberg, S., Markuzon, N., Reynolds, J. H. and Rosen, D. B. (1992)
# "Fuzzy ARTMAP: A Neural Network Architecture for Incremental Supervised Learning of Analog Multidimensional Maps"
# IEEE Transactions on Neural Networks, Vol. 3, No. 5, pp. 698-713.

import math
from typing import List

import numpy as np
import torch
from scipy.sparse.csr import csr_matrix

from tar_framework.run_utilities import LOGGER


class FuzzyArtMapGpu:
    def __init__(self, f1_size: int = 10, f2_size: int = 10, number_of_categories: int = 2, rho_a_bar = 0, max_nodes = None, use_cuda_if_available = False):
        self.alpha = 0.001  # "Choice" parameter > 0. Set small for the conservative limit (Fuzzy AM paper, Sect.3)
        self.beta = 1  # Learning rate. Set to 1 for fast learning
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
                LOGGER.warning("CUDA requested but not available, using CPU.")
        
        # Initial weights in ARTa. All set to 1 Row-j, col-i entry = weight from input node i to F2 coding node j
        self.weight_a = torch.ones((f2_size, f1_size), device=self.device, dtype=torch.float)
        self.input_vector_sum = f1_size / 2
        
        # Row-j, col-k entry = weight from ARTa F2  node j to Map Field node k
        self.weight_ab = torch.ones((f2_size, number_of_categories), device=self.device, dtype=torch.float)

        self.classes_ = np.array([1])
        self.updated_nodes = set()
        self.committed_nodes = set()

        self.node_increase_step = 5 # number of F2 nodes to add when required


    def cache_corpus(self, corpus: csr_matrix, document_index_mapping: dict):
        self.corpus = FuzzyArtMapGpu.complement_encode(torch.tensor(corpus.toarray(), device="cpu", dtype=torch.float))
        # self.corpus = self.corpus.to(self.device)
        self.document_index_mapping = document_index_mapping
        self.excluded_document_ids = set()
        N = self.weight_a.shape[0]
        A_AND_w =  torch.empty(self.weight_a.shape, device=self.device, dtype=torch.float)
        self.S_cache = torch.zeros((self.corpus.shape[0], N), device=self.device, dtype=torch.float)
        ones = torch.ones((N, 1), device=self.device, dtype=torch.float)
        for i in range(self.corpus.shape[0]):
            A_for_each_F2_node = self.corpus[i].to(self.device) * ones
            torch.minimum(A_for_each_F2_node, self.weight_a, out=A_AND_w)
            self.S_cache[i] = torch.sum(A_AND_w, 1)

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

    # @profile
    def _resonance_search(self, input_vector: torch.tensor, already_reset_nodes: List[int], rho_a: float):
        resonant_a = False
        N, S, T = self.calculate_activation(input_vector)
        while not resonant_a:            
            # Set all the reset nodes to zero
            T[already_reset_nodes] = torch.zeros((len(already_reset_nodes), ), dtype=torch.float, device=self.device)

            # Finding the winning node, J
            J = torch.argmax(T)
            # NumPy argmax function works such that J is the lowest index of max T elements, as desired. J is the winning F2 category node

            # Testing if the winning node resonates in ARTa
            membership_degree = S[J]/self.input_vector_sum
            if membership_degree.item() >= rho_a or math.isclose(membership_degree.item(), rho_a):
                resonant_a = True
                # returning from this method will return winning ARTMAPa node index (J) and weighted input vector
            else:
                # If mismatch then we reset
                resonant_a = False
                already_reset_nodes.append(J.item())
                # Record that node J has been reset already.

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # Creating a new node if we've reset all of them
            if len(already_reset_nodes) >= N:
                if self.max_nodes is None or self.max_nodes >= (N + self.node_increase_step):
                    self.weight_a = torch.vstack((self.weight_a, torch.ones((self.node_increase_step,  self.weight_a.shape[1]), device=self.device, dtype=torch.float)))
                    self.weight_ab = torch.vstack((self.weight_ab, torch.ones((self.node_increase_step, self.weight_ab.shape[1]), device=self.device, dtype=torch.float)))
                    self.S_cache = torch.hstack((self.S_cache, torch.ones((self.corpus.shape[0], self.node_increase_step), device=self.device, dtype=torch.float)))
                    # Give the new F2a node a w_ab entry, this new node should win
                else:
                    self.rho_ab = 0
                    self.beta_ab = 0.75
                    self.rho_a_bar = 0
                    rho_a = self.rho_a_bar
                    LOGGER.info(f"Maximum number of nodes reached, {len(already_reset_nodes)} - adjusting rho_ab to {self.rho_ab} and beta_ab to {self.beta_ab}")
                    already_reset_nodes.clear()
                
                N, S, T = self.calculate_activation(input_vector)
            # End of the while loop searching for ARTa resonance
            # If not resonant_a, we pick the next highest Tj and see if *that* node resonates, i.e. goto "while"
            # If resonant_a, we have found an ARTa resonance, namely node J
            # Return from method to see if we get Fab match with node J

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

    def _cached_resonance_search(self, cached_S):
        T = cached_S / self.choice_denominator
        # Choice function vector for F2
        
        J = torch.argmax(T) # Finding the winning node, J
        # NumPy argmax function works such that J is the lowest index of max T elements, as desired. J is the winning F2 category node

        membership_degree = cached_S[J]/self.input_vector_sum

        return J.item(), membership_degree

    # @profile
    def train(self, input_vector: torch.tensor, class_vector: torch.tensor):
        rho_a = self.rho_a_bar # We start off with ARTa vigilance at baseline
        resonant_ab = False # Not resonating in the Fab match layer
        already_reset_nodes = []
        # We haven't rest any ARTa nodes for this input pattern yet, maintain list between resonance searches of Fa
        class_vector = class_vector.to(self.device)
        input_vector = input_vector.to(self.device)
        while not resonant_ab:            
            J, x = self._resonance_search(input_vector, already_reset_nodes, rho_a)

            # Desired output for input number i
            z = torch.minimum(class_vector, self.weight_ab[J, None])   # Fab activation vector, z
            # (Called x_ab in Fuzzy ARTMAP paper)
            
            #Test for Fab resonance
            resonance = torch.sum(z, 1)/torch.sum(class_vector, 1)
            if resonance > self.rho_ab or math.isclose(resonance, self.rho_ab):
                resonant_ab = True
            # This will cause us to leave the while 'not resonant_ab' loop and go on to do learning.

            else: # We have an Fab mismatch
                # Increase rho_a vigilance.
                # This will cause F2a node J to get reset when we go back through the ARTa search loop again.
                # Also, *for this input*, the above-baseline vigilance will cause a finer ARTa category to win
                already_reset_nodes.append(J)
                rho_a = x + self.epsilon
                if rho_a  > 1.0:
                    rho_a = 1.0 - self.epsilon
                assert rho_a <= 1.0, f"actual rho {rho_a}"

        #### End of the while 'not resonant_ab' loop.
        #### Now we have a resonating ARTa output which gives a match at the Fab layer.
        #### So, we go on to have learning in the w_a and w_ab weights

        self.updated_nodes.add(J)
        if J in self.committed_nodes:
            beta = 0.75
        else:
            beta = self.beta
        #### Let the winning, matching node J learn
        self.weight_a[J, None] = (beta * torch.minimum(input_vector, self.weight_a[J, None])) + ((1-beta) * self.weight_a[J, None])
        # NB: x = min(A,w_J) = I and w
        
        #### Learning on F1a <--> F2a weights
        self.weight_ab[J, None] = (self.beta_ab * z) + ((1-self.beta_ab) * self.weight_ab[J, None])
        # NB: z=min(b,w_ab(J))=b and w
        self.committed_nodes.add(J)

    def predict(self, input_vector: torch.tensor):
        rho_a = 0 # set ARTa vigilance to first match
        J, membership_degree = self._resonance_search(input_vector, [], rho_a, False)
        
        # (Called x_ab in Fuzzy ARTMAP paper)
        return self.weight_ab[J, ], membership_degree # Fab activation vector & fuzzy membership value

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

    def fit(self, input_vectors, class_vectors):
        for document_index, input_vector in enumerate(input_vectors):
            self.train(FuzzyArtMapGpu.complement_encode(torch.tensor(input_vector.toarray(), dtype=torch.float)), FuzzyArtMapGpu.complement_encode(torch.tensor([[class_vectors[document_index]]], dtype=torch.float)))
        LOGGER.info(f"updated nodes: {','.join([str(J) for J in self.updated_nodes])}")
        self.recompute_S_cache()
        self.updated_nodes.clear()

    def predict_proba(self, doc_ids: list):        
        predictions = []
        self.choice_denominator = (self.alpha + torch.sum(self.weight_a,  1))
        for document_id in doc_ids:
            prediction, membership_degree = self.cached_predict(document_id)
            if prediction[0].item():
                predictions.append((membership_degree.item(), 0, document_id))
        
        return np.array(predictions) #needs to be in shape (number_of_docs, 1)  
