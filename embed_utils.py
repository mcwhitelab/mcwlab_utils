import argparse
from Bio import SeqIO

import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset

def get_embed_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest = "model_path", type = str, required = True,
                        help="Model directory Ex. /path/to/model_dir")
    parser.add_argument("-f", "--fasta", dest = "fasta_path", type = str, required = True,
                        help="path to a fasta of protein sequences")
    parser.add_argument("-o", "--outpickle", dest = "pkl_out", type = str, required = False,
                        help="Optional: output .pkl filename to save embeddings in")
    parser.add_argument("-ss", "--strategy", dest = "strat", type = str, nargs="+", required = False, 
                        default = ["meansig"], choices = ['swe', 'mean', 'meansig'],
                        help="Embedding strategies to use. Can specify multiple: mean, meansig, swe. Default: meansig")
    parser.add_argument("-s", "--get_sequence_embeddings", dest = "get_sequence_embeddings", action = "store_true",
                        help="Flag: Whether to get sequence embeddings")
    parser.add_argument("-a", "--get_aa_embeddings", dest = "get_aa_embeddings", action = "store_true",
                        help="Flag: Whether to get amino-acid embeddings")

    parser.add_argument("-sa", "--get_sequence_activations", dest = "get_sequence_activations", action = "store_true",
                        help="Flag: Whether to get sequence activations")
    parser.add_argument("-aa", "--get_aa_activations", dest = "get_aa_activations", action = "store_true",
                        help="Flag: Whether to get amino-acid activations")
    parser.add_argument("-p", "--padding", dest = "padding", type = int, default = 0,
                        help="Add if using unaligned sequence fragments (to reduce first and last character effects). Add n X's to start and end of sequencesPotentially not needed for sets of complete sequences or domains that start at the same character, default: 0")
    parser.add_argument("-t", "--truncate", dest = "truncate", type = int, required = False,
                        help= "Optional: Truncate all sequences to this length")
    parser.add_argument("-ad", "--aa_target_dim", dest = "aa_target_dim", type = int, required = False,
                        help= "Optional: Run a new PCA on all amino acid embeddings with target n dimensions prior to saving")
    parser.add_argument("-am", "--aa_pcamatrix_pkl", dest = "aa_pcamatrix_pkl", type = str, required = False,
                        help= "Optional: Use a pretrained PCA matrix to reduce dimensions of amino acid embeddings (pickle file with objects pcamatrix and bias")
    parser.add_argument("-sd", "--sequence_target_dim", dest = "sequence_target_dim", type = int, required = False,
                        help= "Optional: Run a new PCA on all sequence embeddings with target n dimensions prior to saving")
    parser.add_argument("-sm", "--sequence_pcamatrix_pkl", dest = "sequence_pcamatrix_pkl", type = str, required = False,
                        help= "Optional: Use a pretrained PCA matrix to reduce dimensions of amino acid embeddings (pickle file with objects pcamatrix and bias")
    parser.add_argument("-l", "--layers", dest = "layers", nargs="+", type=int, required = False, default = [-1],
                        help="Which layers to use for embeddings, default: -1 (last layer). Use 'all' for all layers.")
    parser.add_argument("--all_layers", dest = "all_layers", action = "store_true",
                        help="Use all available layers for embeddings")
    parser.add_argument("-co", "--cpu_only", dest = "cpu_only",  action = "store_true",
                        help="If --cpu_only flag is included, will run on cpu even if gpu available")
    parser.add_argument("-b", "--batch_size", dest = "batch_size", type = int, default = 1,
                        help="Batch size for processing sequences. Default: 1")
    args = parser.parse_args()
    
    return(args)



def parse_fasta_for_embed(fasta_path, truncate = None, padding = 0, minlength = 1):
    '''
    Load a fasta of protein sequences and
    add a space between each amino acid in sequence (needed to compute embeddings)
    Takes:
        str: Path of the fasta file
        truncate (int): Length to truncate all sequences to (based on model's max length)
        padding (int): Optional padding to add to each sequence
        minlength (int): Minimum sequence length to include
    Returns:
        [ids], [sequences], [sequences with spaces and any padding]
    '''
    sequences = []
    sequences_spaced = []
    ids = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = record.seq

        if truncate:
            if len(seq) > truncate:
                print(f"Warning: Truncating sequence {record.id} from length {len(seq)} to {truncate}")
            seq = seq[0:truncate]

        if len(seq) < minlength:
            print(f"Skipping sequence {record.id} with length {len(seq)} < {minlength}")
            continue

        sequences.append(seq)
        if padding > 0:
            pad_string = "X" * padding
            seq = f"{pad_string}{seq}{pad_string}"

        seq_spaced = " ".join(seq)
        ids.append(record.id)
        sequences_spaced.append(seq_spaced)

    if sequences:
        print(f"Loaded {len(sequences)} sequences")
        print(f"Length range: {min(len(s) for s in sequences)} - {max(len(s) for s in sequences)}")
    else:
        print("Warning: No sequences loaded!")

    return(ids, sequences, sequences_spaced)

def set_device(model, config_attrs, cpu_only): 
    # Determine if half precision is effectively used (based on model state passed in)
    # Note: The 'half' parameter passed here is less critical now,
    # as the model's precision is determined when loaded.
    # We might still use it for SWE model loading later.
    model_is_half = next(model.parameters()).dtype == torch.float16
    print(f"Model received is in {'half' if model_is_half else 'full'} precision.")

    model_type = config_attrs["model_type"] # Use config_attrs passed in
    print("Using pre-loaded {} model".format(model_type))

    aa_shapes = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    device_ids =list(range(0, torch.cuda.device_count()))
    print("device_ids", device_ids)
    model = model.eval()

    # Send model to the correct device if not already there
    # Check if model is already on the target device
    current_device = next(model.parameters()).device
    if current_device != device:
         if torch.cuda.device_count() > 1 and not cpu_only:
             print("Let's use", torch.cuda.device_count(), "GPUs!")
             # Check if model is already DataParallel
             if not isinstance(model, nn.DataParallel):
                  model = nn.DataParallel(model, device_ids=device_ids).to(device) # Send to CUDA
             else:
                  print("Model already wrapped in DataParallel.")
                  model = model.to(device) # Ensure it's on the primary CUDA device if multi-GPU
         else:
             if cpu_only:
                 print("Embedding on cpu, even though gpu available")
                 model = model.to('cpu')
             else:
                  print(f"Moving model to {device}")
                  model = model.to(device)
    else:
        print(f"Model already on device: {current_device}")

    return device


def write_pkl(pkl_out, fasta_path, model_path, model_config_attrs, half_precision_effective, all_layers, layers_arg, strat, padding, truncate_len, embedding_dict):
    print(f"Saving results to {pkl_out}...")
    try:
        with open(pkl_out, "wb") as fOut:
           pickle.dump(embedding_dict, fOut, protocol=pickle.HIGHEST_PROTOCOL)

        pkl_log = "{}.description".format(pkl_out)
        with open(pkl_log, "w") as pOut:
            pOut.write(f"Embeddings generated from: {fasta_path}\n")
            pOut.write(f"Using model: {model_path}\n")
            pOut.write(f"Model type: {model_config_attrs.get('model_type', 'N/A')}\n") # Use .get safely
            pOut.write(f"Effective precision: {'half' if half_precision_effective else 'full'}\n")
            pOut.write(f"Layers used: {'All' if all_layers else layers_arg}\n")
            pOut.write(f"Strategies used: {strat}\n")
            pOut.write(f"Padding: {padding}\n")
            pOut.write(f"Truncation length: {truncate_len if truncate_len else 'None'}\n") # Handle None case
            pOut.write("-" * 20 + "\n")
            pOut.write("Output objects and dimensions:\n")

            # Add shapes safely using .get() on embedding_dict
            for key in ['aa_activations', 'sequence_activations', 'sequence_embeddings', 'sequence_embeddings_sigma', 'sequence_embeddings_swe', 'aa_embeddings']:
                data = embedding_dict.get(key)
                if data is not None:
                    try:
                         # Check if it's numpy array or tensor and print shape
                         if isinstance(data, np.ndarray):
                             shape_str = str(data.shape)
                         elif isinstance(data, torch.Tensor):
                             shape_str = str(data.shape)
                         else:
                             shape_str = f"(Type: {type(data)})"
                         pOut.write(f"  {key}: {shape_str}\n")
                    except AttributeError:
                        pOut.write(f"  {key}: (Error getting shape)\n")
                # else: key not present


            pOut.write("-" * 20 + "\n")
            pOut.write(f"Contains {len(ids)} sequences:\n")
            seq_file = "{}.seqnames".format(pkl_out)
            with open(seq_file, "w") as pOut2:
                for x in ids:
                  pOut2.write("{}\n".format(x))
            pOut.write(f"Full list of sequence IDs written to: {seq_file}\n")

        print(f"Output saved to {pkl_out}")
        print(f"Description saved to {pkl_log}")

    except Exception as e:
        print(f"Error saving output pickle/description: {e}")

def determine_layers(config_attrs, all_layers, ):
    num_total_layers = config_attrs["num_layers"]
    if num_total_layers is None:
         print("Error: num_layers is None. Cannot proceed.")
         return {}

    layers = [] # Defined so name is in scope for return
    # If all_layers is True, use all available layers
    if all_layers:
        layers = list(range(num_total_layers))
        print(f"Using all {num_total_layers} layers for embeddings: {layers}")
    elif layers == [-1]:
        layers = [num_total_layers - 1]
        print(f"Using last layer ({layers[0]}) for embeddings.")
    elif layers is not None:
        # Adjust negative layer indices relative to num_total_layers
        layers = [l if l >= 0 else num_total_layers + l for l in layers]
        
        # Check if any requested layers exceed the available layers
        if any(l >= num_total_layers for l in layers):
            print(f"Warning: Requested layers {layers} exceed available layers (0 to {num_total_layers-1})")
            print(f"Falling back to using all {num_total_layers} available layers")
            layers = list(range(num_total_layers))
        else:
            print(f"Using specified layers: {layers}")
    elif layers is None:
         layers = [num_total_layers - 1]
         print(f"Defaulting to last layer ({layers[0]}) for embeddings.")
    return layers


class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
