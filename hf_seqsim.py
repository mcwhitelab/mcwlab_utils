#!/usr/bin/env python3

import torch
import numpy as np
from statistics import NormalDist
from scipy.stats import entropy
from scipy.sparse import diags
from scipy.spatial.distance import euclidean
import random
#from hf_utils import build_index_flat

import copy
from scipy.stats import multivariate_normal
from time import time
from sklearn.preprocessing import normalize
import faiss

import pickle
import argparse

import os
import sys
import igraph
from pandas.core.common import flatten 
import pandas as pd 

from numba import njit

from collections import Counter
import matplotlib.pyplot as plt
import logging

from sklearn.metrics.pairwise import cosine_similarity



def get_seqsim_args():
    """
    Parse command-line arguments for the sequence similarity analysis.

    Returns:
        argparse.Namespace: An object containing all the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-e", "--emb", dest="embedding_paths", type=str, nargs='+', required=False,
                        help="Path(s) to embeddings, stored in .pkl files as values in dictionaries")

    parser.add_argument("-ek", "--emb_key", dest = "embedding_key", type = str, required = False, choices = ["sequence_embeddings", "sequence_activations"],
                        help="In the embedding file dictionary, what is the key to access the embedding array?")
    parser.add_argument("-es", "--emb_sigmas_key", dest = "embedding_sigmas_key", type = str, required = False,
                        help="In the embedding file dictionary, what is the key to access the embedding array sigmas? Only for doing meansig strategy")


    parser.add_argument("-o", "--outfile", dest = "out_path", type = str, required = True,
                        help="Path to outfile")


    parser.add_argument("-dx", "--index_means", dest = "index_path", required = False,
                        help="Prebuilt index of means i.e. sequence embeddings")
    parser.add_argument("-dxs", "--index_sigmas", dest = "index_path_sigmas", required = False,
                        help="Prebuilt index of sigmas (standard deviations) from when sequence embeddings were calculated")
    parser.add_argument("-dxn", "--index_names", dest = "index_names_file", required = False,
                        help="Prebuilt index names, One protein name per line, in order added to index")

    parser.add_argument("-ss", "--strategy", dest = "strat", type = str, required = False, default = "mean", choices = ['mean','meansig'],
                        help="Whether to search with cosine similarity of mean only, or follow by comparison of gaussians")

    parser.add_argument("-k", "--knn", dest = "k",  type = int, required = False,
                        help="Limit edges to k nearest neighbors")

    parser.add_argument("-m", "--model", dest = "model_name",  type = str, required = True,
                        help="Model name or path to local model")

    parser.add_argument("-p", "--pca_plot", dest = "pca_plot",  action = "store_true", required = False, 
                        help="If flagged, output 2D pca plot of amino acid clusters")

    parser.add_argument("-t", "--threshold", dest="threshold", type=float, default=0.0,
                        help="Similarity threshold. Only output similarities above this value.")

    args = parser.parse_args()

    return(args)



def graph_from_simindex(index, similarities):  
    edges = []
    weights = []
    for i in range(len(index)):
       for j in range(len(index[i])):
          weight = similarities[i,j]
          # Slightly negative weights can arise
          if weight < 0:
             weight = 0.001
          edge = (i, index[i, j])
          edges.append(edge)
          weights.append(weight)
    print("edge preview", edges[0:15])
    G = igraph.Graph.TupleList(edges=edges, directed=True) # Prevent target from being placed first in edges
    G.es['weight'] = weights

    return(G)


def seq_index_search(sequence_array, k_select, sequence_index):
    """
    Perform k-nearest neighbor search using FAISS index.

    Args:
        sequence_array (np.array): 2D array of sequence embeddings, shape (n_sequences, embedding_dim).
        k_select (int): Number of nearest neighbors to retrieve for each sequence.
        sequence_index (faiss.Index): Precomputed FAISS index (CPU or GPU).

    Returns:
        tuple: (similarities, indices)
            similarities (np.array): 2D array of similarity scores, shape (n_sequences, k_select).
                Each row corresponds to a query sequence and contains the similarity scores
                of its k nearest neighbors, sorted from most similar to least similar.
            indices (np.array): 2D array of indices, shape (n_sequences, k_select).
                Each row corresponds to a query sequence and contains the indices of
                its k nearest neighbors in the original sequence_array, sorted from
                most similar to least similar. The index values range from 0 to n_sequences-1.

    Note:
        - The first column of the indices array often contains the index of the query sequence itself
          (as each sequence is most similar to itself), unless explicitly excluded in the search.
        - The similarity scores are typically cosine similarities, where higher values indicate
          greater similarity (closer to 1 for identical sequences, closer to 0 for dissimilar sequences).

    Example:
        If you have 1000 sequences and you're finding the 5 nearest neighbors for each (k_select = 5),
        your `indices` array might look something like this:

        [[  0   42  567  123  789]
         [  1  456  789   23   42]
         [  2  789  456  123   42]
         ...
         [999  456  123   42  789]]

        Here:
        - The first row shows that for sequence 0, its nearest neighbors are sequences 0 (itself), 42, 567, 123, and 789.
        - The second row shows that for sequence 1, its nearest neighbors are sequences 1 (itself), 456, 789, 23, and 42.
        - And so on...

        This structure allows you to quickly look up which sequences are most similar to any given sequence in your dataset.
    """
    similarities, indices = sequence_index.search(sequence_array, k = k_select)
    return similarities, indices

def get_seqsims(sequence_array, k, sequence_index):
    """
    Calculate k-nearest neighbors using the provided sequence array and index.
    
    Args:
        sequence_array (np.array): Array of sequence embeddings.
        k (int): Number of nearest neighbors to find.
        sequence_index (faiss.Index): Pre-computed FAISS index for similarity search.
    
    Returns:
        tuple: (G, similarities, indices)
            G (igraph.Graph): Graph representation of sequence similarities.
            similarities (np.array): 2D array of similarity scores.
            indices (np.array): 2D array of indices of similar sequences.
    """
    print("k", k)

    print("Searching index")
    start_time = time()
    similarities, indices = seq_index_search(sequence_array, k, sequence_index)
    end_time = time()
    print("Index searched in {} seconds".format(end_time - start_time))
    
    start_time = time()
    G = graph_from_simindex(indices, similarities)
    end_time = time()
    print("Index converted to edges in {} seconds".format(end_time - start_time))
    
    return G, similarities, indices


def kl_gauss(m1, m2, s1, s2):
    """
    Calculate the Kullback-Leibler divergence between two normal distributions.

    Args:
        m1 (float): Mean of the first distribution.
        m2 (float): Mean of the second distribution.
        s1 (float): Standard deviation of the first distribution.
        s2 (float): Standard deviation of the second distribution.

    Returns:
        float: The KL divergence from the first distribution to the second.
    """
    kl = np.log(s2/s1) + (s1**2 + (m1-m2)**2)/(2*s2**2) - 1/2
    return(kl)

def kl_mvn(m0, S0, m1, S1):
    """
    Calculate the Kullback-Leibler (KL) divergence between two multivariate normal distributions.

    This function computes the KL divergence from the first distribution (m0, S0) to the second distribution (m1, S1).
    The KL divergence is a measure of how one probability distribution differs from a second, reference probability distribution.

    The formula used is:
    KL((m0, S0) || (m1, S1)) = 0.5 * (tr(S1^{-1} S0) + (m1 - m0)^T S1^{-1} (m1 - m0) - k + ln(det(S1)/det(S0)))

    Where:
    - tr() is the trace of a matrix
    - det() is the determinant of a matrix
    - k is the dimension of the distributions

    Args:
        m0 (np.array): Mean vector of the first distribution.
        S0 (np.array): Covariance matrix of the first distribution.
        m1 (np.array): Mean vector of the second distribution.
        S1 (np.array): Covariance matrix of the second distribution.

    Returns:
        float: The KL divergence from the first distribution to the second.

    Note:
        This implementation assumes that the covariance matrices are positive definite.
        If they are not, the function may raise numerical errors.
    """
    N = m0.shape[0]  # Dimensionality of the distributions
    
    # Calculate inverse of S1
    iS1 = np.linalg.inv(S1)
    
    # Calculate difference between means
    diff = m1 - m0

    # Compute the three terms of the KL divergence
    tr_term   = np.trace(np.dot(iS1, S0))  # Trace term
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0))  # Determinant term
    quad_term = np.dot(np.dot(diff.T, iS1), diff)  # Quadratic term

    # Combine terms and return the KL divergence
    return 0.5 * (tr_term + det_term + quad_term - N)


def get_ovl(m1, m2, s1, s2):
   """
   Calculate the overlap between two normal distributions.

   Args:
       m1 (float): Mean of the first distribution.
       m2 (float): Mean of the second distribution.
       s1 (float): Standard deviation of the first distribution.
       s2 (float): Standard deviation of the second distribution.

   Returns:
       float: The overlap between the two distributions.
   """
   ovl = NormalDist(mu=m1, sigma=s1).overlap(NormalDist(mu=m2, sigma=s2)) 
   return(ovl)

def get_w2(m1, m2, s1, s2):
    """
    Calculate the Wasserstein-2 (W2) distance between two normal distributions.

    Args:
        m1 (float): Mean of the first distribution.
        m2 (float): Mean of the second distribution.
        s1 (float): Standard deviation of the first distribution.
        s2 (float): Standard deviation of the second distribution.

    Returns:
        float: The W2 distance between the two distributions.
    """
    w2 = np.sqrt((m1 - m2)**2 + (s1 - s2)**2)
    return(w2) 

@njit
def numba_w2(m1, m2, s1, s2, w2):
    """
    Calculate W2 distance using numba for optimization.

    Args:
        m1 (np.array): Mean of the first distribution.
        m2 (np.array): Mean of the second distribution.
        s1 (np.array): Standard deviation of the first distribution.
        s2 (np.array): Standard deviation of the second distribution.
        w2 (np.array): Pre-allocated array to store results.

    Returns:
        np.array: Array of W2 distances.
    """
    for i in range(m1.shape[0]):
        w2[int(i)] = np.sqrt((m1[i] - m2[i])**2 + (s1[i] - s2[i])**2)
 
    return w2

@njit
def numba_ovl(m1, m2, s1, s2, o):
    """
    Calculate the overlap between multiple pairs of normal distributions using Numba for optimization.

    Args:
        m1 (np.array): Array of means for the first set of distributions.
        m2 (np.array): Array of means for the second set of distributions.
        s1 (np.array): Array of standard deviations for the first set of distributions.
        s2 (np.array): Array of standard deviations for the second set of distributions.
        o (np.array): Pre-allocated array to store the results.

    Returns:
        np.array: Array of overlap values for each pair of distributions.
    """
    for i in range(m1.shape[0]):
          o[int(i)] = NormalDist(mu=m1[i], sigma=s1[i]).overlap(NormalDist(mu=m2[i], sigma=s2[i])) 
    return(o) 
    

def load_embeddings(embedding_paths, embedding_key):
    """
    Load embeddings from multiple pickle files.

    Args:
        embedding_paths (list): List of paths to pickle files containing embeddings.

    Returns:
        tuple: (combined_embeddings, combined_seq_names)
    """
    combined_embeddings = []
    combined_seq_names = []
    for path in embedding_paths:
        with open(path, "rb") as f:
            embedding_dict = pickle.load(f)
            combined_embeddings.append(embedding_dict[embedding_key])
        
        # Load sequence names for each file
        seqnames_path = path + '.seqnames'
        with open(seqnames_path, 'r') as f:
            combined_seq_names.extend([line.strip() for line in f])
    
    return np.vstack(combined_embeddings), combined_seq_names

def load_precomputed_index(precomputed_index_path, precomputed_index_seqnames_path, precomputed_index_sigmas_path=None, use_gpu=False):
    """
    Load precomputed FAISS indices and sequence names.

    Args:
        precomputed_index_path (str): Path to the precomputed FAISS index.
        precomputed_index_seqnames_path (str): Path to the file containing sequence names.
        precomputed_index_sigmas_path (str, optional): Path to the precomputed sigma index.
        use_gpu (bool): Whether to use GPU for index operations.

    Returns:
        tuple: (sequence_index, sequence_sigma_index, index_names)
            sequence_index (faiss.Index): FAISS index for sequence embeddings.
            sequence_sigma_index (faiss.Index or None): FAISS index for sigma values, if provided.
            index_names (dict): Dictionary mapping indices to sequence names.
    """
    if not precomputed_index_seqnames_path:
        print("Provide file of index names in order added to index")
        exit(1)
    
    with open(precomputed_index_seqnames_path, "r") as infile:
        df = pd.read_csv(infile, header=None)
        df.columns = ['prot', 'idx']
        index_names = dict(zip(df.idx, df.prot))
    
    if use_gpu:
        res = faiss.StandardGpuResources()
        sequence_index = faiss.index_cpu_to_gpu(res, 0, faiss.read_index(precomputed_index_path))
        sequence_sigma_index = None
        if precomputed_index_sigmas_path:
            sequence_sigma_index = faiss.index_cpu_to_gpu(res, 0, faiss.read_index(precomputed_index_sigmas_path))
    else:
        sequence_index = faiss.read_index(precomputed_index_path)
        sequence_sigma_index = None
        if precomputed_index_sigmas_path:
            sequence_sigma_index = faiss.read_index(precomputed_index_sigmas_path)
    
    return sequence_index, sequence_sigma_index, index_names

def process_similarities(seq_names, index_names, similarities, indices, threshold = 0.0, sequence_index=None, sequence_sigma_index=None, sigma_array=None):
    """
    Process similarities between protein sequences based on pre-computed similarities and indices.

    This function takes the results of a k-nearest neighbors search and processes
    the similarities between protein sequences. It uses a two-step approach for efficiency:
    1. First, it processes the fast similarity calculations for all pairs.
    2. Then, only for the "meansig" strategy, it performs the more computationally 
       expensive mean and sigma overlap calculations.

    Args:
        seq_names (list): List of source protein sequence names.
        index_names (dict): Dictionary mapping indices to target protein sequence names.
        similarities (np.array): 2D array of similarities from k-nearest neighbors search.
        indices (np.array): 2D array of indices from k-nearest neighbors search.
        sequence_index (faiss.Index, optional): FAISS index for mean embeddings. Required for "meansig" strategy.
        sequence_sigma_index (faiss.Index, optional): FAISS index for sigma embeddings. Required for "meansig" strategy.
        sigma_array (np.array, optional): Array of sigma values for source sequences. Required for "meansig" strategy.

    Returns:
        pd.DataFrame: DataFrame containing processed similarities with columns:
            - source: Name of the source protein sequence
            - target: Name of the target protein sequence
            - similarity: Similarity measure between source and target
            If "meansig" strategy is used, additional columns are included:
            - w2_mean: W2 distance-based similarity measure
            - w2_mean_neg_e: Exponential of negative W2 distance
            - w2_mean_neg_e_1_10: Scaled exponential of negative W2 distance

    Note:
        - This function assumes that the similarities array contains actual
          similarity values. Higher values indicate greater similarity.
        - The mean and sigma overlap calculations (for "meansig" strategy) are computationally 
          expensive. By performing these calculations only on the pre-filtered nearest 
          neighbors, we significantly reduce the overall computation time.
    """
    results = []
    for i, neighbors in enumerate(indices):
        for j, neighbor in enumerate(neighbors):
            if i == neighbor:
                continue  # Skip self-comparisons
            
            similarity = similarities[i][j]
            
            # Only process similarities above the threshold
            if similarity > threshold:
                source = seq_names[i]
                target = index_names[neighbor]
                
                result = {
                    'source': source,
                    'target': target,
                    'similarity': similarity
                }
                
                # Additional calculations for "meansig" strategy
                if all([sequence_index, sequence_sigma_index, sigma_array]):
                    # Reconstruct mean embeddings for source and target
                    source_mean = sequence_index.reconstruct(i)
                    target_mean = sequence_index.reconstruct(int(neighbor))
                    
                    # Get sigma values for source and target
                    source_sigma = sigma_array[i]
                    target_sigma = sequence_sigma_index.reconstruct(int(neighbor))
                    
                    # Prepare empty array for W2 distance calculation
                    nb_w2_vect = np.empty(source_mean.shape[0], dtype=np.float32)
                    
                    # Calculate W2 distance using numba-optimized function
                    nb_w2_vect = numba_w2(source_mean, target_mean, source_sigma, target_sigma, nb_w2_vect)
                    
                    # Calculate mean W2 distance
                    mean_w2 = np.mean(nb_w2_vect)
                    
                    # Calculate various similarity measures based on W2 distance
                    result.update({
                        'w2_mean': 1/(1 + mean_w2),  # Inverse of W2 distance
                        'w2_mean_neg_e': np.exp(-mean_w2),  # Exponential of negative W2 distance
                        'w2_mean_neg_e_1_10': np.exp(-mean_w2/10)  # Scaled exponential of negative W2 distance
                    })
                
                results.append(result)

    return pd.DataFrame(results)

def load_seq_names(embedding_path):
    """
    Load sequence names from a .seqnames file.
    
    Args:
        embedding_path (str): Path to the embedding .pkl file
    
    Returns:
        list: List of sequence names
    """
    seqnames_path = embedding_path+ '.seqnames'
    with open(seqnames_path, 'r') as f:
        return [line.strip() for line in f]

def main():
    args = get_seqsim_args()
    
    # Check for GPU availability
    use_gpu = faiss.get_num_gpus() > 0
    
    if use_gpu:
        print("GPU detected. Using GPU for computations.")
        res = faiss.StandardGpuResources()  # GPU resource object
        config = faiss.GpuIndexFlatConfig()
        config.device = 0  # GPU device to use
    else:
        print("No GPU detected. Using CPU for computations.")

    # Load embeddings from multiple files
    sequence_array, seq_names = load_embeddings(args.embedding_paths, args.embedding_key)
    
    # Convert embeddings to numpy array and normalize
    sequence_array = np.array(sequence_array).astype(np.float32)
    sequence_array = normalize(sequence_array, axis=1, norm='l2')

    print(sequence_array.shape)
    print(seq_names)

    # Ensure the number of sequences matches the number of embeddings
    if len(seq_names) != sequence_array.shape[0]:
        raise ValueError("Number of sequence names ({}) does not match number of embeddings ({})".format(len(seq_names), sequence_array.shape[0]))

    if args.index_path:
        # Load pre-computed index if provided
        sequence_index, sequence_sigma_index, index_names = load_precomputed_index(
            args.index_path, 
            args.index_names_file, 
            args.index_path_sigmas if args.strat == "meansig" else None,
            use_gpu
        )
    else:
        # Build sequence index if not provided
        if use_gpu:
            # Move sequence_array to GPU
            sequence_array = faiss.float32_to_gpu(res, sequence_array)
            sequence_index = faiss.GpuIndexFlatIP(res, sequence_array.shape[1], config)
        else:
            sequence_index = faiss.IndexFlatIP(sequence_array.shape[1])
        sequence_index.add(sequence_array)
        
        sequence_sigma_index = None
        index_names = seq_names  # Use seq_names as index_names when not using precomputed index
        
        if args.strat == "meansig":
            # Build sigma index for "meansig" strategy
            sigma_array = np.array(embedding_dict[args.embedding_sigmas_key]).astype(np.float32)
            if use_gpu:
                # Move sigma_array to GPU
                sigma_array = faiss.float32_to_gpu(res, sigma_array)
                sequence_sigma_index = faiss.GpuIndexFlatL2(res, sigma_array.shape[1], config)
            else:
                sequence_sigma_index = faiss.IndexFlatL2(sigma_array.shape[1])
            sequence_sigma_index.add(sigma_array)

    # Set k to number of sequences if not provided
    k = args.k if args.k else sequence_array.shape[0]
    
    # Compute sequence similarities and build graph
    G, similarities, indices = get_seqsims(sequence_array, k=k, sequence_index=sequence_index)
    
    sigma_array = None
    if args.strat == "meansig":
        # Load sigma array for "meansig" strategy
        sigma_array = np.array(embedding_dict[args.embedding_sigmas_key]).astype(np.float32)
        if use_gpu:
            sigma_array = faiss.float32_to_gpu(res, sigma_array)
    
    # Process similarities, using "meansig" strategy if specified
    if args.strat == "meansig":
        results_df = process_similarities(seq_names, index_names, similarities, indices, 
                                          threshold=args.threshold,
                                          sequence_index=sequence_index, 
                                          sequence_sigma_index=sequence_sigma_index, 
                                          sigma_array=sigma_array)
    else:
        results_df = process_similarities(seq_names, index_names, similarities, indices,
                                          threshold=args.threshold)
    
    # Write results to TSV file without header
    results_df.to_csv(args.out_path, sep='\t', index=False, header=False, float_format='%.5f')

    print(f"Results written to {args.out_path}")

if __name__ == '__main__':
    main()