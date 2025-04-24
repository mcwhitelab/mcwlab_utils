from transformers import AutoTokenizer, AutoModel, AutoConfig, T5Tokenizer, T5EncoderModel, AutoModelForMaskedLM, EsmTokenizer

from pca_embeddings import control_pca, load_pcamatrix, apply_pca

from transformers.models.t5.modeling_t5 import T5LayerFF
import torch
import torch.nn as nn
from Bio import SeqIO
import pickle
import argparse
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gc
import time
from model.architectures import SWE_Pooling


np.random.seed(42)

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


def get_model_config_attributes(model_path):
    """
    Get model-specific configuration attributes
    
    Args:
        model_path: Path to the model
        
    Returns:
        dict containing:
            - max_sequence_length: Maximum sequence length the model can handle
            - num_layers: Number of layers in the model
            - hidden_size: Size of hidden layers
            - ff_size: Size of feedforward layers
            - model_type: Type of model (t5, esm, bert, protst)
    """
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = model_config.model_type
    
    # Handle nested config for protst (has both a protein and a text component)
    if model_type == "protst" and hasattr(model_config, 'protein_config'):
        protein_config = model_config.protein_config
    else:
        protein_config = model_config # Use the main config if not protst or no protein_config

    print(dir(protein_config)) # Print attributes of the config being used

    # Get max sequence length: Prioritize config attribute, fallback to dictionary
    if hasattr(protein_config, 'max_position_embeddings'):
        max_sequence_length = protein_config.max_position_embeddings
        print(f"Using max_position_embeddings ({max_sequence_length}) from config for max sequence length")
    else:
        max_lengths = {
            "t5": 512,
            "bert": 1024,  # ProtBERT
            "ESMplusplus": 2048,
            "esm": 2048 if "esm2" in model_path.lower() else 1024,  # ESM-2 vs ESM-1
            "protst": 1024  # ProST-ESM1b - Fallback value
        }
        max_sequence_length = max_lengths.get(model_type)
        if max_sequence_length:
            print(f"Using fallback max length ({max_sequence_length}) for model type {model_type}")
        else:
            print(f"Warning: Could not determine max sequence length for model type {model_type} from config or fallback dict.")


    print("model_type", model_type)

    # Get number of layers
    if hasattr(protein_config, 'num_hidden_layers'):
        num_layers = protein_config.num_hidden_layers
    elif hasattr(protein_config, 'num_layers'):
        num_layers = protein_config.num_layers
    else:
        print(f"Warning: Could not determine number of layers for model type {model_type}")
        num_layers = None 

    # Get hidden size
    if hasattr(protein_config, 'hidden_size'):
        hidden_size = protein_config.hidden_size
    elif hasattr(protein_config, 'd_model'):
        hidden_size = protein_config.d_model
    else:
        print(f"Warning: Could not determine hidden size for model type {model_type}")
        hidden_size = None # Or a sensible default/error handling

    # Get feedforward size
    if hasattr(protein_config, 'intermediate_size'):
        ff_size = protein_config.intermediate_size
    elif hasattr(protein_config, 'd_ff'):
        ff_size = protein_config.d_ff
    elif model_type in ["ESMplusplus"]: # Specific models known to potentially lack this
         ff_size = 0 # Keep default 0 if not found for these types
         print(f"Setting ff_size to 0 for model type {model_type} as attribute not found.")
    else:
        print(f"Warning: Could not determine feedforward size for model type {model_type}")
        ff_size = None # Or a sensible default/error handling

    return {
        "max_sequence_length": max_sequence_length,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "ff_size": ff_size,
        "model_type": model_type
    }




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



def retrieve_aa_embeddings(model_output, model_type, layers=None, padding=0, seqlens=None):
    '''
    Get the amino acid embeddings for each sequences
    '''
    # Get all hidden states
    hidden_states = model_output.hidden_states
    
    # If no layers specified, use the final layer
    if layers is None:
        aa_embeddings = hidden_states[-1]
    else:
        # Concatenate specified hidden states into long vector
        aa_embeddings = torch.cat(tuple([hidden_states[i] for i in layers]), dim=-1)
    

    # First trim the embeddings
    if model_type in ["bert", "esm", "ESMplusplus", "protst"]:
        front_trim = 1 + padding
        end_trim = 1 + padding
    elif model_type == "t5" or model_type == "gpt2":
        front_trim = 0 + padding
        end_trim = 1 + padding
    else:
        print("Model type required to extract aas. Currently supported bert, t5, esm")
        return(0)

    aa_embeddings = aa_embeddings[:,front_trim:-end_trim,:]
    
    # Create attention mask directly using aa_embeddings dimensions
    attention_mask = torch.zeros(aa_embeddings.shape[:2], device=aa_embeddings.device)
    # Only process the actual number of sequences in this batch
    for i in range(min(len(seqlens), aa_embeddings.shape[0])):
        attention_mask[i, :seqlens[i]] = 1
    


    return aa_embeddings, attention_mask, aa_embeddings.shape


def load_model(model_path, output_hidden_states = True, output_attentions = False, half = False, return_config = False):
    '''
    Takes path to huggingface model directory
    Returns the model and the tokenizer, and optionally the model config
    '''
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = model_config.model_type
    print("This is a {} model".format(model_type))

    # Use get_model_config_attributes to get model attributes
    config_attrs = get_model_config_attributes(model_path)
    
    # Print all attributes
    print("\nModel Configuration:")
    print(model_path)
    print("-" * 50)
    for key, value in config_attrs.items():
        print(f"{key:.<30} {value}")
    print("-" * 50 + "\n")
    
    print("load tokenizer")
    print("load_model:model_path", model_path)
    if model_type == "t5":
        print("T5 model")
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        print("tokenizer loaded")
        model = T5EncoderModel.from_pretrained(model_path, 
                       output_hidden_states=output_hidden_states, 
                       output_attentions = output_attentions)
        print("model_loaded")
    elif model_type == "ESMplusplus":


            model =  AutoModelForMaskedLM.from_pretrained(model_path, 
                   
                       output_attentions = output_attentions,
                       trust_remote_code=True)
            tokenizer = model.tokenizer
    elif model_type == "protst":
        # For ProtST models, use ESM tokenizer since it's based on ESM
        full_model = AutoModel.from_pretrained(model_path, 
                   output_hidden_states=output_hidden_states,
                   output_attentions=output_attentions,
                   trust_remote_code=True)
        tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        model = full_model.protein_model
        print("Using ESM tokenizer for ProtST model")
    else:
        print("Automodel", model_path, model_type)
        model = AutoModel.from_pretrained(model_path, 
                       output_hidden_states=True, 
                       output_attentions = output_attentions,
                       trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if model_type == "gpt2":
         # This needs to be checked before using
         tokenizer.pad_token = tokenizer.eos_token
    if half == True:
        model.half() # Put model in half precision mode for faster embedding

    if return_config == True:
       return(model, tokenizer, model_config)
    else:
       return(model, tokenizer)





class Collate:
    def __init__(self, tokenizer, max_length, model_type=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type

    def __call__(self, batch):
        if self.model_type == "protst":
            # Special handling for ProtST tokenization
            encodings = []
            for seq in batch:
                # Remove spaces since ProtST expects raw sequence
                seq = seq.replace(" ", "")
                encoding = self.tokenizer(seq, 
                                        add_special_tokens=True,
                                        max_length=self.max_length,
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors='pt')
                encodings.append(encoding)
            
            # Combine all encodings into a batch
            return {
                'input_ids': torch.cat([enc['input_ids'] for enc in encodings]),
                'attention_mask': torch.cat([enc['attention_mask'] for enc in encodings])
            }
        else:
            encoding = self.tokenizer.batch_encode_plus(
                list(batch),
                return_tensors='pt',
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=True
            )
            return encoding

class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



def get_embeddings(seqs, model_path, config_attrs, seqlens, get_sequence_embeddings = True, get_aa_embeddings = True, get_sequence_activations = False, get_aa_activations = False, padding = 0, aa_pcamatrix_pkl = None, sequence_pcamatrix_pkl = None, layers = None, all_layers = False, strat=["meansig"], cpu_only = False, half = False, batch_size = 1):
    '''
    Encode sequences with a transformer model

    Takes:
       model_path (str): Path to a particular transformer model
                         ex. "prot_bert_bfd"
       config_attrs (dict): Dictionary containing model configuration attributes
                            (max_sequence_length, num_layers, hidden_size, ff_size, model_type)
       sequences (list): List of sequences with a space between each amino acid.
                         ex ["M E T", "S E Q"]
 
   '''
    print("CUDA available?", torch.cuda.is_available())

    # half precision doesn't work on CPU
    if not torch.cuda.is_available():
        half = False
    else:
        # Keep half precision as passed, or default based on availability
        pass # half is already set by the argument or default logic

    if get_aa_embeddings == True or get_sequence_embeddings == True:
         output_hidden_states = True
    else:
         output_hidden_states = False

    model, tokenizer, _ = load_model(model_path, output_hidden_states = output_hidden_states, return_config = False, half = half) # Don't need config again
    model_type = config_attrs["model_type"] # Use config_attrs
    print("This is a {} model".format(model_type))
    print("Model loaded")
    aa_shapes = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    device_ids =list(range(0, torch.cuda.device_count()))
    print("device_ids", device_ids)
    model = model.eval()

    hooked_activations = []
    def hook_seq(module, input, output):

        # Important to convert to float32 before summing, otherwise overflow errors from values < -65504
        fxn = "max"
        if fxn == "sum":
            hooked_activations.append(torch.sum(output.float(), dim = 1).cpu()) # Sum scales with sequence length obviously
        elif fxn == "max":

            max_values, _ = torch.max(output.float(), dim=1)  # Extracting max values
            hooked_activations.append(max_values.cpu())

        elif fxn == "min":

            max_values, _ = torch.min(output.float(), dim=1)  # Extracting min values
            hooked_activations.append(max_values.cpu())

    # Set up sequence_activations only hook if not getting aa_activations
    if get_sequence_activations == True:
        if  model_type == "t5":
            for i, block in enumerate(model.encoder.block):
               # Then loop through components (attn, ff) of the layer
               for x in block.layer:
                   if isinstance(x, T5LayerFF) and (layers is None or i in layers):
                       x.DenseReluDense.wi.register_forward_hook(hook_seq)
                       print(f"Hook registered for layer {i}")
            print("sequence hooks registered")

    if torch.cuda.device_count() > 1 and  cpu_only == False:
       print("Let's use", torch.cuda.device_count(), "GPUs!")
       model = nn.DataParallel(model, device_ids=device_ids).cuda()

    else:
       if cpu_only == True:
           half = False
           print("Embedding on cpu, even though gpu available")

       model = model.to(device)

    batch_size = batch_size
    # Use config_attrs directly
    max_length = config_attrs["max_sequence_length"]

    collate = Collate(tokenizer=tokenizer, max_length=max_length, model_type=model_type)

    data_loader = DataLoader(dataset=ListDataset(seqs),
                      batch_size=batch_size,
                      shuffle=False,
                      collate_fn=collate,
                      pin_memory=False)
    start = time.time()

    # Need to concatenate output of each chunk
    sequence_array_list = []
    sequence_array_swe_list = []
    sequence_sigma_array_list = []
    aa_array_list = []

    if sequence_pcamatrix_pkl:
          seq_pcamatrix, seq_bias = load_pcamatrix(sequence_pcamatrix_pkl)

    if aa_pcamatrix_pkl:
          aa_pcamatrix, aa_bias = load_pcamatrix(aa_pcamatrix_pkl)


    # Determine number of layers in the model using config_attrs
    num_total_layers = config_attrs["num_layers"]

    # If all_layers is True, use all available layers
    if all_layers:
        layers = list(range(num_total_layers))
        print(f"Using all {num_total_layers} layers for embeddings")

    # Use hidden_size from config_attrs
    hidden_size = config_attrs["hidden_size"]

    if "swe" in strat:
        torch.manual_seed(42)

        swe_model = SWE_Pooling(d_in = hidden_size * num_layers_to_use,
                               num_slices = hidden_size * num_layers_to_use,
                               num_ref_points=100,
                               freeze_swe=True)

        # Move model to device and convert to half precision if needed
        swe_model = swe_model.to(device)
        #if half:
        #   swe_model.half()


    count = 0
    with torch.inference_mode():
        for data in data_loader:
            batch_size = data['input_ids'].shape[0]
            batch_seqlens = seqlens[count:count+batch_size]

            # Run model

            if model_type == "ESMplusplus":
                inputs = data.to(device)
                model_output = model(**inputs, output_hidden_states=True)
            elif model_type == "protst":
                # For ProtST, we only need input_ids and attention_mask for protein embeddings
                protein_outputs = model(
                    data['input_ids'].to(device),
                    attention_mask=data['attention_mask'].to(device),
                    return_dict=True
                )
                # print("protein_outputs", protein_outputs)
                # print("protein_outputs type:", type(protein_outputs))

                # Get protein_feature for embeddings
                hidden_states = protein_outputs.protein_feature
                if hidden_states is None:
                    print("Warning: Could not find protein_feature in model output")
                    print("Available attributes:", dir(protein_outputs))
                    hidden_states = None
                else:
                    # ProtST outputs hidden states directly in a tuple
                     hidden_states = (hidden_states,)

                # Get last_hidden_state for embeddings
                model_output = type('ProtSTOutput', (), {
                    'hidden_states': hidden_states
                })
                # print("Debug - model output type:", type(model_output))
            else:
                inputs = data.to(device)
                model_output = model(**inputs)

            # Ensure model_output.hidden_states is not None before proceeding
            if model_output.hidden_states is None:
                 print(f"Warning: hidden_states not found in model output for batch starting at index {count}. Skipping batch.")
                 count += batch_size
                 continue


            aa_embeddings_tensor, attention_mask, aa_shape = retrieve_aa_embeddings(
                model_output,
                model_type=model_type,
                layers=layers,
                padding=padding,
                seqlens=batch_seqlens
            )

            aa_embeddings = aa_embeddings_tensor.to('cpu')
            attention_mask = attention_mask.to('cpu')
            aa_embeddings = np.array(aa_embeddings)
            attention_mask = np.array(attention_mask)

            if get_sequence_embeddings == True:
                # Compute masked mean
                # Expand attention mask to match embedding dimensions
                mask_expanded = attention_mask[..., None]  # Shape: [batch_size, seq_length, 1]
                # Mask out padding tokens and compute mean only over real tokens
                masked_embeddings = aa_embeddings * mask_expanded
                sequence_embeddings = masked_embeddings.sum(axis=1) / attention_mask.sum(axis=1, keepdims=True)
                sequence_array_list.append(sequence_embeddings)

                if "meansig" in strat:
                    # Similarly mask the std calculation
                    mean_expanded = sequence_embeddings[:, None, :]  # Shape: [batch_size, 1, hidden_size]
                    squared_diff = ((aa_embeddings - mean_expanded) * mask_expanded) ** 2
                    variance = squared_diff.sum(axis=1) / attention_mask.sum(axis=1, keepdims=True)
                    sequence_embeddings_sigma = np.sqrt(variance)
                    sequence_sigma_array_list.append(sequence_embeddings_sigma)

            if "swe" in strat:
                  # Convert to float32 for SWE_Pooling
                  aa_embeddings_tensor = aa_embeddings_tensor.float()
                  sequence_embeddings_swe = swe_model(aa_embeddings_tensor)
                  sequence_embeddings_swe = sequence_embeddings_swe.cpu().numpy()
                  sequence_array_swe_list.append(sequence_embeddings_swe)

            if sequence_pcamatrix_pkl:
                sequence_embeddings = apply_pca(sequence_embeddings, seq_pcamatrix, seq_bias)


            if aa_pcamatrix_pkl:
                aa_embeddings = np.apply_along_axis(apply_pca, 2, aa_embeddings, aa_pcamatrix, aa_bias)

            # Trim each down to just its sequence length
            if get_aa_embeddings == True:
                    # If not using ragged arrays, must pad to same dim as longest sequence
                    # print(maxlen - (aa_embeddings.shape[1] - 1))
                     #if padding:
                    #dim2 = maxlen - (aa_embeddings.shape[1])
                    #npad = ((0,0), (0, dim2), (0,0))
                    #aa_embeddings = np.pad(aa_embeddings, npad)
                    aa_array_list.append(aa_embeddings)

            count += batch_size

    end = time.time()
    print("Total time to embed = {}".format(end - start))

    # Collect results
    embedding_dict = {}

    # Get number of neurons in feedforward layer based on model type using config_attrs
    numneurons = config_attrs["ff_size"] # Use config_attrs

    if get_sequence_activations == True:
        stacked = np.stack([x for x in hooked_activations])
        # Need numseqs and numlayers for reshaping
        numseqs = len(seqs) # Get the total number of sequences
        num_layers_used = len(layers) if layers is not None else num_total_layers
        # go from (numlayers, numseqs, numneurons) to (numseqs, numlayers * numneurons)
        # Check dimensions before reshaping
        expected_elements = num_layers_used * numseqs * numneurons
        actual_elements = stacked.size
        print(f"Debug: stacked shape {stacked.shape}, numseqs {numseqs}, num_layers_used {num_layers_used}, numneurons {numneurons}")
        print(f"Debug: expected elements {expected_elements}, actual elements {actual_elements}")

        if actual_elements == expected_elements and numneurons > 0: # Ensure numneurons is valid
           sequence_activations = stacked.reshape(numseqs, num_layers_used * numneurons)
           embedding_dict['sequence_activations'] = sequence_activations
        elif numneurons == 0 and model_type not in ["protst", "ESMplusplus"]: # Only warn if ff_size is expected
            print(f"Warning: Feedforward size (numneurons) is 0 for model type {model_type}. Cannot reshape sequence activations.")
        elif actual_elements != expected_elements:
            print(f"Warning: Mismatch in activation dimensions. Expected {expected_elements} elements, got {actual_elements}. Cannot reshape sequence activations.")

    # Move this outside the sequence_activations check
    if get_sequence_embeddings == True:
        if sequence_array_list:  # Check if we have any embeddings
            embedding_dict['sequence_embeddings'] = np.concatenate(sequence_array_list)
            if "meansig" in strat:
                if sequence_sigma_array_list: # Check if sigma was computed
                    embedding_dict['sequence_embeddings_sigma'] = np.concatenate(sequence_sigma_array_list)
            if "swe" in strat:
                if sequence_array_swe_list: # Check if swe was computed
                    embedding_dict['sequence_embeddings_swe'] = np.concatenate(sequence_array_swe_list)

    if get_aa_embeddings == True:
        if aa_array_list:  # Check if we have any embeddings
            embedding_dict['aa_embeddings'] = np.concatenate(aa_array_list)

    print("Complete")
    return(embedding_dict)



if __name__ == "__main__":
    args = get_embed_args()

    # Unpack all args at the start
    model_path = args.model_path
    fasta_path = args.fasta_path
    pkl_out = args.pkl_out
    get_sequence_embeddings = args.get_sequence_embeddings
    get_aa_embeddings = args.get_aa_embeddings
    get_sequence_activations = args.get_sequence_activations
    get_aa_activations = args.get_aa_activations
    truncate = args.truncate
    layers = args.layers
    padding = args.padding
    cpu_only = args.cpu_only
    strat = args.strat
    aa_pcamatrix_pkl = args.aa_pcamatrix_pkl
    sequence_pcamatrix_pkl = args.sequence_pcamatrix_pkl
    aa_target_dim = args.aa_target_dim
    sequence_target_dim = args.sequence_target_dim
    batch_size = args.batch_size
    all_layers = args.all_layers

    if get_sequence_embeddings == False:
         if get_aa_embeddings == False:
             if get_sequence_activations == False:
                 if get_aa_activations == False:
                     print("Must add --get_sequence_embeddings and/or --get_aa_embeddings and/or --get_sequence_activations and/or --get_aa_activations, otherwise nothing to compute")
                     exit(1)

    # Get model configuration once
    model_config_attrs = get_model_config_attributes(model_path)

    # Set truncation length if not specified
    if not truncate:
        truncate = model_config_attrs["max_sequence_length"] # Use fetched attribute
        if truncate:
            print(f"Setting maximum sequence length to {truncate} based on model type {model_config_attrs['model_type']}")
        else:
            print("Warning: Could not determine max sequence length, sequences will not be truncated")

    ids, sequences, sequences_spaced = parse_fasta_for_embed(fasta_path=fasta_path,
                                                           truncate=truncate,
                                                           padding=padding)

    print("First sequences")
    seqlens = [len(x) for x in sequences]

    # If all_layers is True, override the layers parameter
    # layers will be set correctly inside get_embeddings based on all_layers flag and config_attrs
    if all_layers:
        layers_arg = None # Pass None, get_embeddings will handle it
    else:
        layers_arg = layers # Pass the user specified layers


    embedding_dict = get_embeddings(sequences_spaced,
                                    model_path,
                                    config_attrs=model_config_attrs, # Pass the config dict
                                    seqlens=seqlens,
                                    get_sequence_embeddings=get_sequence_embeddings,
                                    get_aa_embeddings=get_aa_embeddings,
                                    get_sequence_activations=get_sequence_activations,
                                    get_aa_activations=get_aa_activations,
                                    padding=padding,
                                    layers=layers_arg, # Use the potentially modified layers
                                    all_layers=all_layers,
                                    aa_pcamatrix_pkl=aa_pcamatrix_pkl,
                                    sequence_pcamatrix_pkl=sequence_pcamatrix_pkl,
                                    strat=strat,
                                    cpu_only=cpu_only,
                                    batch_size=batch_size)

    # Reduce sequence dimension with a new pca transform
    if sequence_target_dim:
       pkl_pca_out = "{}.sequence.{}dim.pcamatrix.pkl".format(fasta_path, sequence_target_dim)
       embedding_dict['sequence_embeddings'] = control_pca(embedding_dict,
                                                'sequence_embeddings',
                                                pkl_pca_out=pkl_pca_out,
                                                target_dim=sequence_target_dim,
                                                max_train_sample_size=None)

    # Reduce aa dimension with a new pca transform
    if aa_target_dim:
       pkl_pca_out = "{}.aa.{}dim.pcamatrix.pkl".format(fasta_path, aa_target_dim)
       embedding_dict['aa_embeddings'] = control_pca(embedding_dict,
                                                'aa_embeddings',
                                                pkl_pca_out=pkl_pca_out,
                                                target_dim=aa_target_dim,
                                                max_train_sample_size=None)

    # Store sequences & embeddings on disk
    if pkl_out:
        with open(pkl_out, "wb") as fOut:
           pickle.dump(embedding_dict, fOut, protocol=pickle.HIGHEST_PROTOCOL)

        pkl_log = "{}.description".format(pkl_out)
        with open(pkl_log, "w") as pOut:
            if get_aa_activations and 'aa_activations' in embedding_dict: # Check if key exists
               pOut.write("Object {} dimensions: {}\n".format('aa_activations', embedding_dict['aa_activations'].shape))

            if get_sequence_activations and 'sequence_activations' in embedding_dict: # Check if key exists
               pOut.write("Object {} dimensions: {}\n".format('sequence_activations', embedding_dict['sequence_activations'].shape))

            if get_sequence_embeddings and 'sequence_embeddings' in embedding_dict: # Check if key exists
               pOut.write("Object {} dimensions: {}\n".format('sequence_embeddings', embedding_dict['sequence_embeddings'].shape))
               if "meansig" in strat and 'sequence_embeddings_sigma' in embedding_dict: # Check if key exists
                    pOut.write("Object {} dimensions: {}\n".format('sequence_embeddings_sigma', embedding_dict['sequence_embeddings_sigma'].shape))
               if "swe" in strat and 'sequence_embeddings_swe' in embedding_dict: # Check if key exists
                    pOut.write("Object {} dimensions: {}\n".format('sequence_embeddings_swe', embedding_dict['sequence_embeddings_swe'].shape))

            if get_aa_embeddings and 'aa_embeddings' in embedding_dict: # Check if key exists
                pOut.write("Object {} dimensions: {}\n".format('aa_embeddings', embedding_dict['aa_embeddings'].shape))

            pOut.write("Contains sequences:\n")
            for x in ids:
              pOut.write("{}\n".format(x))

            seq_file = "{}.seqnames".format(pkl_out)
            with open(seq_file, "w") as pOut2:
                for x in ids:
                  pOut2.write("{}\n".format(x))


