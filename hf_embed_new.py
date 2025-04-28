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
            - max_sequence_length: Maximum sequence length the model can handle (for tokenizer max_length)
            - num_layers: Number of layers in the model
            - hidden_size: Size of hidden layers
            - ff_size: Size of feedforward layers
            - model_type: Type of model (t5, esm, bert, protst)
    """
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = model_config.model_type
    
    # Handle nested config for protst
    if model_type == "protst" and hasattr(model_config, 'protein_config'):
        protein_config = model_config.protein_config
    else:
        protein_config = model_config

    print(dir(protein_config))

    # --- Get max sequence length: Prioritize hardcoded dictionary, fallback to config ---
    max_lengths = {
        "t5": 512,
        "bert": 1024,  # ProtBERT
        "ESMplusplus": 2048,
        "esm": 2048 if "esm2" in model_path.lower() else 1024,  # ESM-2 vs ESM-1
        "protst": 1024  # Use 1024 like the archive script
    }
    max_sequence_length = max_lengths.get(model_type)

    if max_sequence_length:
        print(f"Using dictionary max length ({max_sequence_length}) for model type {model_type}")
    elif hasattr(protein_config, 'max_position_embeddings'):
        # Fallback to max_position_embeddings only if type not in dict
        max_sequence_length = protein_config.max_position_embeddings - 2
        print(f"Warning: Model type {model_type} not in dictionary. Using max_position_embeddings ({max_sequence_length}) -2 from config.")
        print(f"If both CLS and SEP aren't used in the model, will cause CUDA errors")
    else:
        print(f"Warning: Could not determine max sequence length for model type {model_type} from config or fallback dict.")
        max_sequence_length = None # Ensure it's None if undetermined


    print("model_type", model_type)

    # Get number of layers (using protein_config or model_config)
    if hasattr(protein_config, 'num_hidden_layers'):
        num_layers = protein_config.num_hidden_layers
    elif hasattr(protein_config, 'num_layers'):
        num_layers = protein_config.num_layers
    else:
        print(f"Warning: Could not determine number of layers for model type {model_type}")
        num_layers = None

    # Get hidden size
    if model_type == "protst":
        hidden_size = 512
    elif hasattr(protein_config, 'hidden_size'):
        hidden_size = protein_config.hidden_size
    elif hasattr(protein_config, 'd_model'):
        hidden_size = protein_config.d_model
    else:
        print(f"Warning: Could not determine hidden size for model type {model_type}")
        hidden_size = None

    # Get feedforward size
    if hasattr(protein_config, 'intermediate_size'):
        ff_size = protein_config.intermediate_size
    elif hasattr(protein_config, 'd_ff'):
        ff_size = protein_config.d_ff
    elif model_type in ["ESMplusplus", "protst"]: # ProtST ff_size is handled by its protein_config, ESM++ might lack it
         if not hasattr(protein_config, 'intermediate_size') and not hasattr(protein_config, 'd_ff'):
              ff_size = 0 # Keep default 0 if not found for these types
              print(f"Setting ff_size to 0 for model type {model_type} as standard attributes not found.")
         else: # If attributes were found above, ff_size is already set
             pass
    else: # Other model types
         if not hasattr(protein_config, 'intermediate_size') and not hasattr(protein_config, 'd_ff'):
            print(f"Warning: Could not determine feedforward size for model type {model_type}")
            ff_size = None

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
    print("hidden_states", hidden_states)
    # Handle different model types
    if model_type == "protst":
        # For ProtST, hidden_states is already the last layer's tensor
        # Layer selection isn't applicable as we only have access to the final layer
        aa_embeddings = hidden_states
    else:
        # For other models, hidden_states is a tuple of tensors (one per layer)
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


def load_model(model_path, output_hidden_states = True, output_attentions = False, half = False):
    '''
    Takes path to huggingface model directory.
    Loads the model and tokenizer.
    Gets and returns model configuration attributes.
    Returns the model, the tokenizer, and the config_attrs dictionary.
    '''
    # Get config attributes first
    config_attrs = get_model_config_attributes(model_path)
    model_type = config_attrs["model_type"]

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
        # For ProtST models
        print("Loading full ProtST model...")
        full_model = AutoModel.from_pretrained(model_path,
                   output_hidden_states=output_hidden_states,
                   output_attentions=output_attentions,
                   trust_remote_code=True)
        
        # --- REVERT to always using standard ESM tokenizer for ProtST ---
        print(f"Using standard ESM tokenizer (facebook/esm2_t33_650M_UR50D) for ProtST.")
        tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        # --- End of revert ---

        model = full_model.protein_model # Use only the protein part
        
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

    # Apply half precision if requested and possible
    if half and torch.cuda.is_available():
        try:
            model.half() # Put model in half precision mode for faster embedding
            print("Model loaded in half precision.")
        except Exception as e:
            print(f"Warning: Could not load model in half precision: {e}")
            half = False # Revert flag if half precision fails
    elif half and not torch.cuda.is_available():
        print("Warning: Half precision requested but CUDA not available. Loading in full precision.")
        half = False

    # Return model, tokenizer, and the fetched config attributes
    return model, tokenizer, config_attrs





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



def get_embeddings(model, tokenizer, config_attrs, seqs, seqlens, get_sequence_embeddings = True, get_aa_embeddings = True, get_sequence_activations = False, get_aa_activations = False, padding = 0, aa_pcamatrix_pkl = None, sequence_pcamatrix_pkl = None, layers = None, all_layers = False, strat=["meansig"], cpu_only = False, half = False, batch_size = 1):
    '''
    Encode sequences with a pre-loaded transformer model

    Takes:
       model (torch.nn.Module): The pre-loaded transformer model.
       tokenizer: The pre-loaded tokenizer.
       config_attrs (dict): Dictionary containing model configuration attributes
                            (max_sequence_length, num_layers, hidden_size, ff_size, model_type).
       seqs (list): List of sequences with a space between each amino acid.
                         ex ["M E T", "S E Q"]
       seqlens (list): List of original sequence lengths (before padding).
       ... other args ...
   '''
    print("CUDA available?", torch.cuda.is_available())

    # Determine if half precision is effectively used (based on model state passed in)
    # Note: The 'half' parameter passed here is less critical now,
    # as the model's precision is determined when loaded.
    # We might still use it for SWE model loading later.
    model_is_half = next(model.parameters()).dtype == torch.float16
    print(f"Model received is in {'half' if model_is_half else 'full'} precision.")


    # No need to load model/tokenizer here - they are passed in.
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
        # Hook registration needs the actual model object (which might be wrapped)
        target_model = model.module if isinstance(model, nn.DataParallel) else model
        if model_type == "t5":
             # Ensure we're accessing the encoder correctly, even if model is T5EncoderModel directly
             encoder_module = target_model if isinstance(target_model, T5EncoderModel) else getattr(target_model, 'encoder', None)
             if encoder_module and hasattr(encoder_module, 'block'):
                 for i, block in enumerate(encoder_module.block):
                    # Then loop through components (attn, ff) of the layer
                    for x in block.layer:
                        if isinstance(x, T5LayerFF) and (layers is None or i in layers):
                            x.DenseReluDense.wi.register_forward_hook(hook_seq)
                            print(f"Hook registered for T5 layer {i}")
                 print("T5 sequence hooks registered")
             else:
                 print("Warning: Could not find encoder blocks to register hooks for T5.")
        # Add hook registration logic for other model types if needed


    batch_size = batch_size
    # Use config_attrs directly
    max_length = config_attrs["max_sequence_length"]

    # Check if max_length is None and handle appropriately
    if max_length is None:
        print("Warning: max_sequence_length is None. Attempting to proceed without it, but padding/truncation might be unpredictable.")
        # Optionally set a default or raise an error
        # max_length = 1024 # Example default


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
    if num_total_layers is None:
         print("Error: num_layers is None. Cannot proceed.")
         return {}


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


    # Use hidden_size from config_attrs
    hidden_size = config_attrs.get("hidden_size") # Use .get for safety
    if hidden_size is None:
         print("Error: hidden_size is None in config_attrs. Cannot proceed.")
         return {} # Handle error

    # Setup SWE model if needed
    swe_model = None
    if "swe" in strat:
        torch.manual_seed(42)
        # For ProtST, we only have the last layer regardless of layers parameter
        if model_type == "protst":
            num_layers_to_use = 1
            swe_d_in = hidden_size
        else:
            # For other models, use the number of specified layers
            num_layers_to_use = len(layers)
            if num_layers_to_use == 0:
                print("Error: Cannot initialize SWE model with 0 layers selected.")
                return {}
            swe_d_in = hidden_size * num_layers_to_use
            
        print(f"Initializing SWE Pooling with d_in={swe_d_in} (hidden_size={hidden_size}, num_layers_used={num_layers_to_use})")

        swe_model = SWE_Pooling(d_in = swe_d_in,
                               num_slices = swe_d_in, # Typically num_slices matches d_in
                               num_ref_points=100,
                               freeze_swe=True)

        # Move SWE model to device
        swe_model = swe_model.to(device)

    count = 0
    output_hs_needed = get_aa_embeddings or get_sequence_embeddings # Check if hidden states are needed at all

    # Main embedding loop
    with torch.inference_mode():
        for i, data in enumerate(data_loader): # Add enumerate for batch index
            batch_size_actual = data['input_ids'].shape[0] # Use actual batch size
            batch_seqlens = seqlens[count:count+batch_size_actual]

            # Run model
            inputs = {k: v.to(device) for k, v in data.items()}

            # Adapt model call based on type and expected output
            try:
                if model_type == "protst":
                     # ProtST requires only input_ids and attention_mask for the protein_model part
                     protein_outputs = model(
                         input_ids=inputs['input_ids'],
                         attention_mask=inputs['attention_mask'],
                         return_dict=True,
                         output_hidden_states=True # Pass flag here too
                     )
                    
                     hidden_states_output = protein_outputs.residue_feature
                

                     model_output = type('ModelOutput', (), {'hidden_states': hidden_states_output})

                else: # General case for T5EncoderModel, AutoModel, etc.
                     output_hs = get_aa_embeddings or get_sequence_embeddings
                     model_output = model(**inputs, output_hidden_states=output_hs)
                     print(model_output) 

                # Ensure model_output.hidden_states is not None before proceeding
                if not hasattr(model_output, 'hidden_states') or model_output.hidden_states is None:
                     print(f"Warning: hidden_states not found in model output for batch starting at index {count}. Skipping batch.")
                     count += batch_size_actual
                     continue


                aa_embeddings_tensor, attention_mask, aa_shape = retrieve_aa_embeddings(
                    model_output,
                    model_type=model_type,
                    layers=layers, # Pass processed layer list
                    padding=padding,
                    seqlens=batch_seqlens
                )
                
                # Check if retrieve_aa_embeddings returned successfully
                if isinstance(aa_embeddings_tensor, int) and aa_embeddings_tensor == 0:
                    print(f"Error retrieving embeddings for batch starting at index {count}. Skipping batch.")
                    count += batch_size_actual
                    continue


                aa_embeddings = aa_embeddings_tensor.to('cpu')
                attention_mask = attention_mask.to('cpu')
                aa_embeddings = np.array(aa_embeddings)
                attention_mask = np.array(attention_mask)

                if get_sequence_embeddings == True:
                                            # Compute masked mean
                    # Expand attention mask to match embedding dimensions
                    mask_expanded = attention_mask[..., None]  # Shape: [batch_size, seq_length, 1]
                    # Mask out padding tokens and compute mean only over real tokens
                    # Add epsilon to avoid division by zero if a sequence has zero length after masking
                    sum_mask = attention_mask.sum(axis=1, keepdims=True)
                    masked_embeddings = aa_embeddings * mask_expanded
                    if model_type == "protst":

                        sequence_embeddings = np.array(protein_outputs.protein_feature.to("cpu"))
                    else:

                        sequence_embeddings = masked_embeddings.sum(axis=1) / (sum_mask + 1e-9) # Add epsilon
                    sequence_array_list.append(sequence_embeddings)

                    if "meansig" in strat:
                        # Similarly mask the std calculation
                        mean_expanded = sequence_embeddings[:, None, :]  # Shape: [batch_size, 1, hidden_size]
                        squared_diff = ((aa_embeddings - mean_expanded) * mask_expanded) ** 2
                        variance = squared_diff.sum(axis=1) / (sum_mask + 1e-9) # Add epsilon
                        sequence_embeddings_sigma = np.sqrt(variance)
                        sequence_sigma_array_list.append(sequence_embeddings_sigma)
              
                # --- SWE handling reverted fully to archive logic ---
                if "swe" in strat and swe_model is not None:
                    # Explicitly convert input tensor to float32 like in archive
                    aa_embeddings_tensor_float = aa_embeddings_tensor.float()
                    print("aa_embeddings_tensor_float", aa_embeddings_tensor_float.shape)

                    sequence_embeddings_swe = swe_model(aa_embeddings_tensor_float)
                    
                    sequence_array_swe_list.append(sequence_embeddings_swe.cpu().numpy())
                # --- End reverted SWE handling ---

                # Apply PCA if requested (after calculations)
                if sequence_pcamatrix_pkl:
                    if get_sequence_embeddings and sequence_embeddings is not None:
                        sequence_embeddings = apply_pca(sequence_embeddings, seq_pcamatrix, seq_bias)

                if aa_pcamatrix_pkl:
                    if aa_embeddings is not None and aa_embeddings.size > 0:
                       aa_embeddings = np.apply_along_axis(apply_pca, 2, aa_embeddings, aa_pcamatrix, aa_bias)

                # Append AA embeddings if requested
                if get_aa_embeddings == True and aa_embeddings is not None:
                        aa_array_list.append(aa_embeddings)

                count += batch_size_actual # Increment by actual batch size processed

            except Exception as e:
                print(f"Error processing hidden states for batch starting at {count}: {e}")
                count += batch_size_actual
                continue # Skip batch

        end = time.time()
        print("Total time to embed = {}".format(end - start))

    # Collect results
    embedding_dict = {}

    # Get number of neurons in feedforward layer based on model type using config_attrs
    numneurons = config_attrs["ff_size"] # Use config_attrs
    if numneurons is None:
         print("Warning: ff_size (numneurons) is None. Activations cannot be processed correctly.")
         # Handle appropriately, maybe skip activation processing


    if get_sequence_activations == True:
        # Ensure hooked_activations is not empty and numneurons is valid
        if hooked_activations and numneurons is not None and numneurons > 0 :
            try:
                stacked = np.stack([x.numpy() for x in hooked_activations]) # Convert tensors in list to numpy before stacking
                # Need numseqs and numlayers for reshaping
                numseqs = len(seqs) # Get the total number of sequences
                # Determine the number of layers hooks were registered for
                # This is tricky if hooks weren't registered for all layers specified in 'layers' list
                # Assuming hook was registered for each layer in the 'layers' list if applicable (e.g., for T5)
                num_layers_hooked = len(layers) if model_type == "t5" and layers is not None else 0 # Adjust based on actual hooking logic

                # Reshape: The shape of hooked_activations might be complex depending on hook implementation
                # Assuming hook_seq appends tensors of shape [batch_size, numneurons]
                # And they are concatenated across batches correctly.
                # Let's try concatenating first, then reshaping if necessary.

                concatenated_activations = np.concatenate([act for act in hooked_activations], axis=0)

                # Expected shape: (total_seqs, num_layers_hooked * numneurons) or similar
                print(f"Debug: Concatenated activations shape: {concatenated_activations.shape}")
                # Example reshape (needs verification based on hook logic):
                # If each hook saves activations per layer: (num_hooks, num_seqs, num_neurons)
                # We might need a different stacking/reshaping approach

                # Placeholder: Assign concatenated directly if reshape logic is uncertain
                embedding_dict['sequence_activations'] = concatenated_activations


            except Exception as e:
                 print(f"Error processing sequence activations: {e}")
                 print(f"Hooked activations list length: {len(hooked_activations)}")
                 if hooked_activations:
                     print(f"Shape of first hooked activation tensor: {hooked_activations[0].shape}")


        elif numneurons == 0:
            print(f"Warning: Feedforward size (numneurons) is 0 for model type {model_type}. Cannot retrieve/reshape sequence activations.")
        elif not hooked_activations:
             print("Warning: get_sequence_activations was True, but no activations were hooked.")


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

    # Unpack necessary args
    model_path = args.model_path
    fasta_path = args.fasta_path
    pkl_out = args.pkl_out
    get_sequence_embeddings = args.get_sequence_embeddings
    get_aa_embeddings = args.get_aa_embeddings
    get_sequence_activations = args.get_sequence_activations
    get_aa_activations = args.get_aa_activations
    truncate_arg = args.truncate # Keep original arg name
    layers_arg = args.layers # Keep original arg name
    padding = args.padding
    cpu_only = args.cpu_only
    strat = args.strat
    aa_pcamatrix_pkl = args.aa_pcamatrix_pkl
    sequence_pcamatrix_pkl = args.sequence_pcamatrix_pkl
    aa_target_dim = args.aa_target_dim
    sequence_target_dim = args.sequence_target_dim
    batch_size = args.batch_size
    all_layers = args.all_layers


    if not any([get_sequence_embeddings, get_aa_embeddings, get_sequence_activations, get_aa_activations]):
         print("Must specify at least one output type: --get_sequence_embeddings, --get_aa_embeddings, --get_sequence_activations, or --get_aa_activations.")
         exit(1)

    # Initialize variables to ensure they exist in scope, even if loading fails
    model = None
    tokenizer = None
    model_config_attrs = None
    half_precision_requested = False # What the user asked for (implicitly or explicitly)
    half_precision_effective = False # What actually happened

    # Determine if half precision should be attempted
    if not cpu_only and torch.cuda.is_available():
        half_precision_requested = True
        print("CUDA available. Attempting to load model in half precision.")
    elif cpu_only:
        print("CPU only mode selected. Model will be loaded in full precision.")
    else: # Not cpu_only but CUDA not available
        print("CUDA not available. Model will be loaded in full precision.")


    # Load model and get config attributes *once* upfront
    print(f"Loading model from: {model_path}")
    try:
        # Pass output_hidden_states based on whether any embedding type is requested
        output_hs_needed_for_load = get_sequence_embeddings or get_aa_embeddings
        model, tokenizer, model_config_attrs = load_model(
            model_path,
            output_hidden_states=output_hs_needed_for_load,
            output_attentions=False, # Assuming attentions are not needed based on args
            half=half_precision_requested # Request half if applicable
        )
        # Check the actual precision of the loaded model
        half_precision_effective = next(model.parameters()).dtype == torch.float16
        print(f"Model, tokenizer, and config loaded. Effective precision: {'half' if half_precision_effective else 'full'}")

    except Exception as e:
        print(f"Fatal Error: Failed to load model, tokenizer, or config from {model_path}.")
        print(f"Error details: {e}")
        exit(1) # Ensure exit if loading fails

    # --- Code below here only runs if load_model succeeded and variables are assigned ---

    # Double-check that essential components were loaded
    if model is None or tokenizer is None or model_config_attrs is None:
        print("Fatal Error: Model, tokenizer, or config attributes were not loaded correctly after load_model call. Exiting.")
        exit(1)


    # Set truncation length based on config or argument override
    truncate_len = truncate_arg # Use the value from args if provided
    if truncate_len is None: # If not provided via args (it defaults to None)
        truncate_len = model_config_attrs.get("max_sequence_length") # Use .get for safety
        if truncate_len:
            print(f"Setting maximum sequence length for truncation to {truncate_len} based on model config {model_config_attrs.get('model_type', 'N/A')}")
        else:
            print("Warning: Could not determine max sequence length from config and none provided via --truncate. Sequences will not be truncated by default.")
    elif truncate_len <= 0:
         print("Truncate length must be positive. Disabling truncation.")
         truncate_len = None # Disable truncation if user provides non-positive value
    else:
        print(f"Using truncation length provided via --truncate: {truncate_len}")
        # Optional: Warn if user truncate value exceeds model max length
        config_max_len = model_config_attrs.get("max_sequence_length")
        if config_max_len and truncate_len > config_max_len:
            print(f"Warning: User-specified truncation length ({truncate_len}) exceeds model's reported max length ({config_max_len}).")


    # Parse FASTA using the determined truncation length
    print(f"Parsing FASTA file: {fasta_path} with truncation={truncate_len}, padding={padding}")
    ids, sequences, sequences_spaced = parse_fasta_for_embed(fasta_path=fasta_path,
                                                           truncate=truncate_len,
                                                           padding=padding)

    if not sequences:
        print("Error: No valid sequences loaded from FASTA file after filtering/truncation. Exiting.")
        exit(1)

    print(f"Sequences parsed. Number of sequences: {len(ids)}")
    seqlens = [len(s) for s in sequences] # Get original lengths *after* truncation but *before* spacing/padding


    # Call the modified get_embeddings function
    print("Starting embedding generation...")
    # Provide the first 5 arguments positionally
    embedding_dict = get_embeddings(
        model=model,                       # Pass loaded model
        tokenizer=tokenizer,               # Pass loaded tokenizer
        config_attrs=model_config_attrs,   # Pass config dict
        seqs=sequences_spaced,             # Pass sequences with spaces
        seqlens=seqlens,                   # Pass original lengths
        # Keyword arguments for the rest
        get_sequence_embeddings=get_sequence_embeddings,
        get_aa_embeddings=get_aa_embeddings,
        get_sequence_activations=get_sequence_activations,
        get_aa_activations=get_aa_activations,
        padding=padding,
        layers=layers_arg, # CORRECTED: Use layers_arg here
        all_layers=all_layers, # Pass all_layers flag
        aa_pcamatrix_pkl=aa_pcamatrix_pkl,
        sequence_pcamatrix_pkl=sequence_pcamatrix_pkl,
        strat=strat,
        cpu_only=cpu_only, # Pass CPU flag
        half=half_precision_effective, # Pass effective half precision status
        batch_size=batch_size
    )


    # Post-processing (PCA)
    if embedding_dict: # Check if embeddings were generated
        if sequence_target_dim and 'sequence_embeddings' in embedding_dict:
           pkl_pca_out = "{}.sequence.{}dim.pcamatrix.pkl".format(fasta_path, sequence_target_dim)
           print(f"Applying PCA to sequence embeddings (target dim: {sequence_target_dim})...")
           embedding_dict['sequence_embeddings'] = control_pca(embedding_dict,
                                                    'sequence_embeddings',
                                                    pkl_pca_out=pkl_pca_out,
                                                    target_dim=sequence_target_dim,
                                                    max_train_sample_size=None) # Add sample size limit?

        if aa_target_dim and 'aa_embeddings' in embedding_dict:
           pkl_pca_out = "{}.aa.{}dim.pcamatrix.pkl".format(fasta_path, aa_target_dim)
           print(f"Applying PCA to amino acid embeddings (target dim: {aa_target_dim})...")
           # Note: control_pca for AA might need adjustments if input is 3D (batch, seq, features)
           # Assuming control_pca can handle or needs reshaped input
           embedding_dict['aa_embeddings'] = control_pca(embedding_dict,
                                                    'aa_embeddings',
                                                    pkl_pca_out=pkl_pca_out,
                                                    target_dim=aa_target_dim,
                                                    max_train_sample_size=500000) # Limit sample size for AA PCA
    else:
        print("Warning: Embedding dictionary is empty after get_embeddings call. Skipping PCA and output.")


    # Store sequences & embeddings on disk
    if pkl_out and embedding_dict:
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

    elif not pkl_out:
        print("No output pickle file specified (--outpickle). Results will not be saved.")
    elif not embedding_dict:
         print("Embedding dictionary is empty, nothing to save.")

    print("Script finished.")


