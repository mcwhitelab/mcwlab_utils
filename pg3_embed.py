from progen3.modeling import ProGen3ForCausalLM # For Model and Tokenizer
from progen3.config import ProGen3Config # For Config

from pca_embeddings import control_pca, load_pcamatrix, apply_pca

import torch
import torch.nn as nn
import numpy as np
import time
from embed_utils import get_embed_args, parse_fasta_for_embed, set_device 


def load_model(model_path, output_hidden_states = True, output_attentions = False, half = False):
    '''
    Loads a Progen3 model and tokenizer
    '''
    model = ProGen3ForCausalLM.from_pretrained(model_path) # Find real parameters for function
    tokenizer = ProGen3ForCausalLM(model_path)
    return model, tokenizer


def get_embeddings(model, tokenizer, config_attrs, seqs, seqlens, get_sequence_embeddings = True, get_aa_embeddings = True, get_sequence_activations = False, get_aa_activations = False, padding = 0, aa_pcamatrix_pkl = None, sequence_pcamatrix_pkl = None, layers = None, all_layers = False, strat=["meansig"], cpu_only = False, half = False, batch_size = 1):
    model_type = config_attrs["model_type"]
    set_device(model, config_attrs)
    
    # Set up sequence_activations
    if get_sequence_activations == True:
        # Hook registration needs the actual model object (which might be wrapped)
        target_model = model.module if isinstance(model, nn.DataParallel) else model

    max_length = config_attrs["max_sequence_length"]

    # Check if max_length is None and handle appropriately
    if max_length is None:
        print("Warning: max_sequence_length is None. Attempting to proceed without it, but padding/truncation might be unpredictable.")

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
            batch_size_actual = data['input_ids'].shape[0] # Use actual batch size
            batch_seqlens = seqlens[count:count+batch_size_actual]

            # Run model
            inputs = {k: v.to(device) for k, v in data.items()}

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
    
    model_path = args.model_path
    fasta_path = args.fasta_path
    
    #Initialize variables
    model = None
    tokenizer = None
    model_config_attrs = None

    #Load model 
    model, tokenizer = load_model()

    # Double-check that model and tokenizer were loaded
    if model is None or tokenizer is None:
        print("Fatal Error: Model, tokenizer, or config attributes were not loaded correctly after load_model call. Exiting.")
        exit(1)

    #Parse FASTA 
    ids, sequences, sequences_spaed = parse_fasta_for_embed(fasta_path=fasta_path) #add remaining parameters later? TODO

    if not sequences:
        print("Error: No valid sequences loaded from FASTA file after filtering/truncation. Exiting.")
        exit(1)

    #Get embeddings (the big one, possibly) TODO assign parameters required by get_embeddings
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

    #Post-processing (PCA) TODO

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

    #Store embeddings TODO
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

