#!/usr/bin/env python3
"""
This is an example script format for the mcwlab


Style notes:

    See PEP 8 standards:
           https://llego.dev/posts/writing-clean-pep-8-compliant-code-better-collaboration/

    Indent with 4 spaces (as opposed to tabs or mix of tabs and space)
    For readability, try to be liberal with spaces
        x = 5, not x=5
        output = sum(1, 2, 3), not output=sum(1,2,3)     
        y = process(this, that), not y=process(this,that)

    Name functions by what they do, ex. process_input(), calculate_mean()

    Add comments frequently
    Add descriptions (aka docstrings) to functions
       - Not necessarily immediately, but as the code matures, add them in

"""




# Section for importing packages and functions
# This goes above the first function
import argparse
from hf_embed import parse_fasta


def get_args():
    """
    Parse command-line arguments

    Place as the first function the top of the script, so it's easy to see immediately what arguments a script takes

    Additionally, using argparse allows you to see what arguments a script takes from the command line using:

    python scriptname.py -h

    Included below are some different argument examples
       -A required string
       -An optional list of strings
       -A required string limited to specific options
       -An optional integer 
       -An optional float with a default value
       -A boolean flag (has the value True if the argument is used (ex, python scriptname.py --pca_plot)

    Usage:
    
        args = get_args()
        print(args.destname)
        ex. args.embedding_key, or args.model 
  
    Notes:
        The idea is to avoid controlling the script's behavior by making changes to it with each run
        So if you need to run a script on a particular filename, this should be added as an argument
            and not changed manually in the script each time

            so instead of
            filename="myfile.txt"
            do
            filename = args.filename

    Returns:
        argparse.Namespace: An object containing all the parsed arguments.
        
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-m", "--model", dest = "model_name",  type = str, required = True,
                        help="Model name or path to local model")

    parser.add_argument("-e", "--emb", dest="embedding_paths", type=str, nargs='+', required=False,
                        help="Path(s) to embeddings, stored in .pkl files as values in dictionaries")

    parser.add_argument("-ek", "--emb_key", dest = "embedding_key", type = str, required = True, choices = ["sequence_embeddings", "sequence_activations"],
                        help="In the embedding file dictionary, what is the key to access the embedding array?")

    parser.add_argument("-k", "--knn", dest = "k",  type = int, required = False,
                        help="Limit edges to k nearest neighbors")

    parser.add_argument("-t", "--threshold", dest="threshold", type=float, default=0.0,
                        help="Similarity threshold. Only output similarities above this value.")

    parser.add_argument("-p", "--pca_plot", dest = "pca_plot",  action = "store_true", required = False, 
                        help="If flagged, output 2D pca plot of amino acid clusters")

    args = parser.parse_args()

    return(args)



def function_example(sequence_array, sequence_index, k_select = 10):
    """
    Function descriptions don't need to start fully formatted like this, but as
    you proceed with the code, start filling out these sections.
    
    Many functions won't need such extensive description

    # Here, describe what the function does
    A function to showcase the formatting for a function 

    # Here, describe the arguments that the function takes
    Args:
        sequence_array (np.array): 2D array of sequence embeddings, shape (n_sequences, embedding_dim).
        sequence_index (faiss.Index): Precomputed FAISS index (CPU or GPU).
        k_select (int): Number of nearest neighbors to retrieve for each sequence (Default 10)

    # Here, what the function returns
    Returns:
        tuple: (similarities, indices)
            similarities (np.array): 2D array of similarity scores, shape (n_sequences, k_select).
                Each row corresponds to a query sequence and contains the similarity scores
                of its k nearest neighbors, sorted from most similar to least similar.
            indices (np.array): 2D array of indices, shape (n_sequences, k_select).
                Each row corresponds to a query sequence and contains the indices of
                its k nearest neighbors in the original sequence_array, sorted from
                most similar to least similar. The index values range from 0 to n_sequences-1.

    # Here, any notes
    Note:
        - The first column of the indices array often contains the index of the query sequence itself
          (as each sequence is most similar to itself), unless explicitly excluded in the search.
        - The similarity scores are cosine similarities, where higher values indicate
          greater similarity (closer to 1 for similar sequences, closer to 0 for dissimilar sequences).

    # If necessary, examples of anything potentially confusing
    Example:
        If you have 1000 sequences and you're finding the 5 nearest neighbors for each (k_select = 5),
        your indices array might look something like this:

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

def load_seq_names(embedding_path):
    """
    This is an example of a function that doesn't require much description
    Load sequence names from a .seqnames file.
    
    Args:
        embedding_path (str): Path to the embedding .pkl file
    
    Returns:
        list: List of sequence names

    Note:
        Each embedding path x.pkl file comes with another file x.pkl.seqnames that contains one-per-line the sequence names contained in the embedding

    """
    seqnames_path = embedding_path+ '.seqnames'
    with open(seqnames_path, 'r') as f:
        return [line.strip() for line in f]



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


def main():
    '''
    The main function acts as an entry point for the script

    Using a main function allows you to control what gets executed when the script is run directly versus when it's imported as a module in another script. 
     This is done with the following condition (see below this function at the bottom of the script):
     if __name__ == "__main__":
          main()
    When the script is run directly, __name__ is set to "__main__", triggering the main function. 
    When imported, it prevents the main function from running automatically.
    So if you import a function from here to use in another script (ex. from mcwlab_example_script import get_overlap)
       everything in the main function won't run. 

    '''   

 
    # Get the command line arguments
    # As a style choice, I like to explicitly rename each arguments, ex. k = args.k
    args = get_seqsim_args()
    k = args.k
    threshold = args.threshold 
    out_path = args.out_path


    # Use a function
    ovl = get_ovl(1, 2, 0.3, 0.5)

    #Use a function and save output (_df indicates that this is a pandas dataframe
    results_df = process_similarities(seq_names, index_names, similarities, indices,
                                          threshold=threshold)
    
    # Save any results
    # Ex. write results to TSV file 
    results_df.to_csv(out_path, sep='\t', index=False, float_format='%.5f')

    print(f"Results written to {args.out_path}")

# The main function acts as the entry point for the program. When you run a Python script, the interpreter executes the code in that file. By convention, using a main function helps organize the execution flow.

if __name__ == '__main__':
    main()
