import traceback
from progen3.modeling import ProGen3ForCausalLM # For Model and Tokenizer
from progen3.config import ProGen3Config # For Config
from progen3.batch_preparer import ProGen3BatchPreparer

from pca_embeddings import control_pca, load_pcamatrix, apply_pca

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
from embed_utils import get_embed_args, parse_fasta_for_embed, set_device, write_pkl, ListDataset


def load_model(model_path, output_hidden_states = True, output_attentions = False, half = False):
    '''
    Loads a Progen3 model and batch preparer
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

    print("load_model:model_path", model_path)

    model = ProGen3ForCausalLM.from_pretrained(model_path) # Find real parameters for function

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

    # Return model, batch preparer, and the fetched config attributes
    return model, config_attrs


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
            - model_type: Type of model (Should be progen3)
    """
    model_config = ProGen3Config.from_pretrained(model_path)
    model_type = model_config.model_type
    protein_config = model_config

    #print(dir(protein_config))

    max_sequence_length = protein_config.max_position_embeddings # Should be 65536

    print("model_type", model_type)

    # Get number of layers (using protein_config or model_config)
    if hasattr(protein_config, 'num_hidden_layers'):
        num_layers = protein_config.num_hidden_layers
    elif hasattr(protein_config, 'num_layers'):
        num_layers = protein_config.num_layers
    elif hasattr(protein_config, 'n_layer'):
        num_layers = protein_config.n_layer
    else:
        print(f"Warning: Could not determine number of layers for model type {model_type}")
        num_layers = None

    # Get hidden size
    if hasattr(protein_config, 'hidden_size'):
        hidden_size = protein_config.hidden_size
    else:
        print(f"Warning: Could not determine hidden size for model type {model_type}")
        hidden_size = None

    # Get feedforward size
    if hasattr(protein_config, 'intermediate_size'):
        ff_size = protein_config.intermediate_size
    else: # Other model types
            print(f"Warning: Could not determine feedforward size for model type {model_type}")
            ff_size = None

    return {
        "max_sequence_length": max_sequence_length,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "ff_size": ff_size,
        "model_type": model_type
    }


def retrieve_aa_embeddings(model_output, model_type, layers=None, padding=0, seqlens=None):
    '''
    Get the amino acid embeddings for each sequences
    '''
    # Get all hidden states
    hidden_states = model_output.hidden_states
    print("hidden_states", hidden_states)
    # For other models, hidden_states is a tuple of tensors (one per layer)
    if layers is None:
        aa_embeddings = hidden_states[-1]
    else:
        # Concatenate specified hidden states into long vector
        aa_embeddings = torch.cat(tuple([hidden_states[i] for i in layers]), dim=-1)
    
    # Trim embeddings - ProGen3 uses bos and eos
    front_trim = 1 + padding
    end_trim = 1 + padding

    aa_embeddings = aa_embeddings[:,front_trim:-end_trim,:]
    
    # Create attention mask directly using aa_embeddings dimensions
    attention_mask = torch.zeros(aa_embeddings.shape[:2], device=aa_embeddings.device)
    # Only process the actual number of sequences in this batch
    for i in range(min(len(seqlens), aa_embeddings.shape[0])):
        attention_mask[i, :seqlens[i]] = 1
 
    return aa_embeddings, attention_mask, aa_embeddings.shape

def get_embeddings(model, config_attrs, seqs, seqlens, get_sequence_embeddings = True, get_aa_embeddings = True, get_sequence_activations = False, get_aa_activations = False, padding = 0, aa_pcamatrix_pkl = None, sequence_pcamatrix_pkl = None, layers = None, all_layers = False, strat=["meansig"], cpu_only = False, half = False, batch_size = 1):
    batch_preparer = ProGen3BatchPreparer()

    model = model.eval()
    model_type = config_attrs["model_type"]
    device = set_device(model, config_attrs, cpu_only)
    max_length = config_attrs["max_sequence_length"]

    # Check if max_length is None and handle appropriately
    if max_length is None:
        print("Warning: max_sequence_length is None. Attempting to proceed without it, but padding/truncation might be unpredictable.")

    # Prep data loader
    prepared_seqs = [] # Must be a list of dictionaries for batch_preparer
    prepared_seqs.append(batch_preparer.get_batch_kwargs(seqs))
    print("Prepared Seqs", prepared_seqs)
    data_loader = DataLoader(dataset=ListDataset(prepared_seqs),
                      batch_size=batch_size,
                      shuffle=False,
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
            print(f"Data at {i}: {data}")
            batch_size_actual = data['input_ids'].shape[0] # Use actual batch size
            batch_seqlens = seqlens[count:count+batch_size_actual]

            # Run model
            inputs = {k: v.to(device) for k, v in data.items()}
            print("Inputs:", inputs)

            # Adapt model call based on type and expected output
            try:
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
                traceback.print_exc()
                count += batch_size_actual
                continue # Skip batch

        end = time.time()
        print("Total time to embed = {}".format(end - start))

    # Collect results
    embedding_dict = {}

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
    
    # Unpack args
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
        model, model_config_attrs = load_model(
            model_path,
            output_hidden_states=output_hs_needed_for_load,
            output_attentions=False,
            half=half_precision_requested
        )
        # Check the actual precision of the loaded model
        half_precision_effective = next(model.parameters()).dtype == torch.float16
        print(f"Model, batch preparer, and config loaded. Effective precision: {'half' if half_precision_effective else 'full'}")
    except Exception as e:
        print(f"Fatal Error: Failed to load model, batch preparer, or config from {model_path}.")
        print(f"Error details: {e}")
        exit(1) # Ensure exit if loading fails

    # Double-check that model and batch preparer were loaded
    if model is None or model_config_attrs is None:
        print("Fatal Error: Model or config attributes were not loaded correctly after load_model call. Exiting.")
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
    

    # Check if a ProGen3 Model was loaded
    if model_config_attrs["model_type"] != "progen3":
        print(f'This script only supports ProGen3 models. Please supply a ProGen3 model. Exiting.')
        exit(1)


    # Parse FASTA 
    ids, sequences, sequences_spaced = parse_fasta_for_embed(fasta_path=fasta_path,
                                                             truncate=truncate_len) #add remaining parameters later? TODO
    print("Sequences after parse_fasta:", sequences_spaced)  

    if not sequences:
        print("Error: No valid sequences loaded from FASTA file after filtering/truncation. Exiting.")
        exit(1)

    print(f"Sequences parsed. Number of sequences: {len(ids)}")
    seqlens = [len(s) for s in sequences] # Get original lengths *after* truncation but *before* spacing/padding

    #Get embeddings (the big one, possibly) TODO assign parameters required by get_embeddings
    embedding_dict = get_embeddings(
        model=model,                       # Pass loaded model
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

    #Post-processing (PCA) 

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

    #Store embeddings
    if pkl_out and embedding_dict:
       write_pkl(pkl_out=pkl_out,
                 fasta_path=fasta_path,
                 model_path=model_path,
                 model_config_attrs=model_config_attrs,
                 half_precision_effective=half_precision_effective,
                 all_layers=all_layers,
                 layers_arg=layers_arg,
                 strat=strat,
                 padding=padding,
                 truncate_len=truncate_len,
                 embedding_dict=embedding_dict)
    elif not pkl_out:
        print("No output pickle file specified (--outpickle). Results will not be saved.")
    elif not embedding_dict:
         print("Embedding dictionary is empty, nothing to save.")

    print("Script finished.")

