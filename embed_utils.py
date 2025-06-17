import argparse
from Bio import SeqIO

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

def set_device(model, config_attrs): 
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
