from transformers import AutoTokenizer, AutoModel, AutoConfig, T5Tokenizer, T5EncoderModel

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

'''

## Note, this script is ready for reorganization into smaller functions

Get pickle of embeddings for a fasta of protein sequences
with a huggingface transformer model

Can return aa-level, sequence-level, or both

Optional to do PCA on embeddings prior to saving, or use pre-trained PCA matrix on them. 

#### Default pickle shapes
pickle['aa_embeddings']:  (numseqs x longest seqlength x (1024 * numlayers)
pickle['sequence_embeddings']: (numseqs x 1024)


### --use_ragged_arrays
Amino acid embeddings are by default save in a numpy array of dimensions 
   pickle['aa_embeddings']:  (numseqs x longest seqlength x (1024 * numlayers)

The package `awkward` allows saving of arrays of different lengths. 
If all sequences are around the same lengths, there's not much different
However, one long sequence can greatly increase file sizes.
For a set of 50 ~300aa sequence + one 5000aa sequence, there's a tenfold difference in file size. 

3.9G test_np.pkl
369M test_awkward.pkl

### PCA
Another route to smaller file size is training a PCA transform to reduced dimensionality.
It can either be applied to sequence or amino acid embeddings. 

Previously trained PCA matrices can be used as well.



#### Example command
 python transformer_infrastructure/hf_embed.py -m /scratch/gpfs/cmcwhite/prot_bert_bfd/ -f tester.fasta -o test.pkl

#### To load a pre-computed embedding:

 with open("embeddings.pkl", "rb") as f:
     cache_data = pickle.load(f)
     sequence_embeddings = cache_data['sequence_embeddings']
     aa_embeddings = cache_data['aa_embeddings']


#### extra_padding argument
Adding 5 X's to the beginning and end of each sequence seems to improve embeddings
I'd be interested in feedback with this parameter set to True or False

#### To download a huggingface model locally:

from transformers import AutoModel, AutoTokenizer

sourcename = "Rostlab/prot_bert_bfd"
modelname = "prot_bert_bfd"
outdir = "/scratch/gpfs/cmcwhite/hfmodels/" + modelname

tokenizer = AutoTokenizer.from_pretrained(sourcename)
tokenizer.save_pretrained(outdir)
model = AutoModel.from_pretrained(sourcename)
model.save_pretrained(outdir)

#### Minimal anaconda environment
conda create --name hf-transformers -c conda-forge -c pytorch transformers pytorch::pytorch numpy biopython

Claire D. McWhite
7/8/20
'''

def get_embed_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest = "model_path", type = str, required = True,
                        help="Model directory Ex. /path/to/model_dir")
    parser.add_argument("-f", "--fasta", dest = "fasta_path", type = str, required = True,
                        help="path to a fasta of protein sequences")
    parser.add_argument("-o", "--outpickle", dest = "pkl_out", type = str, required = False,
                        help="Optional: output .pkl filename to save embeddings in")
    parser.add_argument("-ss", "--strategy", dest = "strat", type = str, required = False, default = "meansig", choices = ['mean','meansig'],
                        help="For sequences, get embeddings of mean, or two embeddings, one mean, one sigma. Choice of mean or meansig. Default: meansig")
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
    parser.add_argument("-r", "--use_ragged_arrays", dest = "ragged_arrays", action = "store_true", required = False,
                        help= "Optional: Use package 'awkward' to save ragged arrays fo amino acid embeddings (not implemented)")
    parser.add_argument("-l", "--layers", dest = "layers", nargs="+", type=int, required = False,
                        help="Additionally exclude outlier sequences from final alignment")
    parser.add_argument("-hds", "--heads", dest = "heads", required = False,
                        help="Additionally exclude outlier sequences from final alignment")
    parser.add_argument("-co", "--cpu_only", dest = "cpu_only",  action = "store_true",
                        help="If --cpu_only flag is included, will run on cpu even if gpu available")
    args = parser.parse_args()
    
    return(args)


# sequence_lols = parse_fasta(args.fasta_path, fasta_tbl, True, maxlength)

def parse_fasta_for_embed(fasta_path, truncate = None, padding = 0, minlength = 1):

   ''' 
   Load a fasta of protein sequences and
     add a space between each amino acid in sequence (needed to compute embeddings)
   Takes: 
       str: Path of the fasta file
       truncate (int): Optional length to truncate all sequences to
       extra_padding (bool): Optional to add padding to each sequence to avoid start/end of sequence effects
                             Useful for when all sequences don't start/end at "same" amino acid. 
 
   Returns: 
       [ids], [sequences], [sequences with spaces and any extra padding] 
   '''
   sequences = []
   sequences_spaced = []
   ids = []
   for record in SeqIO.parse(fasta_path, "fasta"):

       seq = record.seq

       if truncate:
          print("truncating to {}".format(truncate))
          seq = seq[0:truncate]

       if len(seq) < minlength:
           continue

       sequences.append(seq)
       if padding > 0: 
           pad_string = "X" * padding
           #seq = "XXXXX{}XXXXX".format(seq)
           seq = "{}{}{}".format(pad_string, seq, pad_string)

       seq_spaced =  " ".join(seq)
       #if extra_padding == True:
       #   seq = seq[5:]
       #   seq = seq[:-5]

       # 5 X's seems to be a good amount of neutral padding
       #if extra_padding == True:
       #     padding_aa = " X" * 5
       #     padding_left = padding_aa.strip(" ")
    
            # To do: Figure out why embedding are better with removed space between last X and first AA?
       #     seq_spaced = padding_left + seq_spaced  
       #     seq_spaced = seq_spaced + padding_aa
  
       ids.append(record.id)
       sequences_spaced.append(seq_spaced)
   return(ids, sequences, sequences_spaced)


def mean_pooling(model_output, attention_mask):
    '''
    Mean Pooling - Take attention mask into account for correct averaging

    This function is from sentence_transformers
    #https://www.sbert.net/examples/applications/computing-embeddings/README.html
    '''
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    print("token_embeddings_size", token_embeddings.size())
    #print("token_embeddings_dtype", token_embeddings.dtype))
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def headnames_to_index(headnames, heads_per_layer = 16):
      '''
      Convert layer0_head0 to index (output of split_index) 
      '''
      heads = []
      for headname in headnames:
          #print(headname)
          layer_tmp = headname.split("_")[0]
          head_tmp = headname.split("_")[1]
          layer = int(layer_tmp.replace("layer", ""))
          head= int(head_tmp.replace("head", ""))
          head_index  = (heads_per_layer * layer) + head   
          heads.append(head_index)
      return(heads)


def split_hidden_states(hidden_states, head_len = 64, aa_dim = 0 ):
    '''
    Splits a concatenation of all the layers into heads
    '''
    
    head_ids = []
    numaas = hidden_states.shape[aa_dim]
    #seqlen = hidden_states.shape[1]
    num_heads = hidden_states.shape[aa_dim + 1]/head_len
    num_layers = hidden_states.shape[aa_dim + 1]/1024
    heads_per_layer = int(1024/head_len)
    #print('num_heads', num_heads)
    #print('num_layers', num_layers)
    for layer in range(0,int(num_layers)):
        for head in range(0,int(heads_per_layer)):
           head_ids.append('layer{}_head{}'.format(layer, head))
    if aa_dim ==0: 
       hstates_heads = hidden_states.reshape(numaas, -1, head_len)
       #print(hstates_heads.shape)
       hstates_split = np.split(hstates_heads, num_heads, axis = 1)
       #print(hstates_split[0].shape)
       hstates_split = [x.reshape(numaas,  head_len) for x in hstates_split]
       #print(hstates_split[0].shape)
    if aa_dim == 1:
       num_seqs = hidden_states.shape[0]
       hstates_split = hidden_states.reshape(num_seqs, numaas, -1, head_len)
       #print(hstates_split.shape)
       #hstates_split = np.split(hstates_heads, num_heads, axis = (aa_dim + 1))
       #print(hstates_split[0].shape)
       #hstates_split = [x.reshape(num_seqs, numaas,  head_len) for x in hstates_split]
       #print(hstates_split[0].shape)
      

    return(hstates_split, head_ids)

def retrieve_aa_embeddings(model_output, model_type, layers = None, padding = 0, heads = None):
    '''
    Get the amino acid embeddings for each sequences
    Pool layers by concatenating selection of layers OR selection of heads
    Return shape: (numseqs, length of longest sequence, 1024*numlayers)
    If adding padding, it's usually 5. 

    Takes: 
       model_output: From sequence encoding
       layers (list of ints): By default, pool final four layers of model
       heads (list of specific heads): format head1_layer1, head2_layer2, etc. indexed from 1
       padding (int): If padding was added, remove embeddings corresponding to extra padding before returning 

    Return shape (numseqs x longest seqlength x (1024 * numlayers)

    Note: If the output shape of this function is [len(seqs), 3, x], make sure there are spaces between each amino acid
    The "3" corresponds to CLS,seq,END 
     '''
    
    # Get all hidden states
    hidden_states = model_output.hidden_states
    # Concatenate hidden states into long vector

    # Either layers or heads
    if layers is not None:
        aa_embeddings = torch.cat(tuple([hidden_states[i] for i in layers]), dim=-1)
        #print(aa_embeddings)
        print(aa_embeddings.shape)


    if heads is not None: 
        #print("selecting heads", heads)
        head_indices = headnames_to_index(heads, heads_per_layer = 16)
        #print(len(hidden_states))
        aa_embeddings = torch.cat(tuple([hidden_states[i] for i in list(range(-31, 0))]), dim = -1)
        #print("full concatenation", aa_embeddings.shape)
         
        # Tensor must be copied to host cpu
        aa_embeddings_split, head_ids = split_hidden_states(np.array(aa_embeddings.cpu()), head_len = 64, aa_dim = 1)
        #print("split embedding_shape", aa_embeddings_split.shape)
        aa_embeddings_sel = np.take(aa_embeddings_split, head_indices, axis = 2)
        #print(aa_embeddings_sel.shape)
        aa_embeddings = aa_embeddings_sel.reshape(aa_embeddings_sel.shape[0], aa_embeddings_sel.shape[1], len(head_indices)*aa_embeddings_sel.shape[3])
        #print(aa_embeddings.shape) 
        aa_embeddings = torch.from_numpy(aa_embeddings)
        #aa_embeddings = torch.cat(tuple([aa_embeddings_split[i] for i in head_indices]), dim = -1)

    if model_type == "bert":
      front_trim = 1 + padding
      end_trim = 1 + padding
    elif model_type == "t5" or model_type == "gpt2":
      front_trim = 0 + padding
      end_trim = 1 + padding
    else:
       print("Model type required to extract aas. Currently supported bert and t5")
       return(0)

    aa_embeddings = aa_embeddings[:,front_trim:-end_trim,:]

    return(aa_embeddings, aa_embeddings.shape)


def load_model(model_path, output_hidden_states = True, output_attentions = False, half = False, return_config = False):
    '''
    Takes path to huggingface model directory
    Returns the model and the tokenizer, and optionally the model config
    output_hidden_states
    output_attentions=False
    
    '''
    model_config = AutoConfig.from_pretrained(model_path)
    model_type = model_config.model_type
    print("This is a {} model".format(model_type))

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
    else:
        print("Automodel")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, 
                       output_hidden_states=output_hidden_states, 
                       output_attentions = output_attentions)

    if model_type == "gpt2":
         # This needs to be checked before using
         tokenizer.pad_token = tokenizer.eos_token
    if half == True:
        model.half() # Put model in half precision mode for faster embedding

    print(model)
    print(model_config)

    if return_config == True:
       return(model, tokenizer, model_config)
    else:
       return(model, tokenizer)

# ? 
class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        encoding = self.tokenizer.batch_encode_plus(
            list(batch),
            return_tensors  = 'pt', 
            padding = True
            
        )
        return(encoding)

# ?
# Start trying to reduce these
class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


"""

T5DenseActDense(
  (wi): Linear(in_features=512, out_features=2048, bias=False)
  (wo): Linear(in_features=2048, out_features=512, bias=False)
  (dropout): Dropout(p=0.1, inplace=False)
  (act): ReLU()

"""



def get_embeddings(seqs, model_path, seqlens, get_sequence_embeddings = True, get_aa_embeddings = True, get_sequence_activations = False, get_aa_activations = False, padding = 0, ragged_arrays = False, aa_pcamatrix_pkl = None, sequence_pcamatrix_pkl = None, heads = None, layers = None, strat="meansig", cpu_only = False, half = False):
    '''
    Encode sequences with a transformer model

    Takes:
       model_path (str): Path to a particular transformer model
                         ex. "prot_bert_bfd"
       sequences (list): List of sequences with a space between each amino acid.  
                         ex ["M E T", "S E Q"]
 
   '''
    if ragged_arrays == True:
       ak.numba.register()
    print("CUDA available?", torch.cuda.is_available())

    # half precision doesn't work on CPU 
    if not torch.cuda.is_available():
        half = False
    else:
        half = True
   

    if get_aa_embeddings == True or get_sequence_embeddings == True:
         output_hidden_states = True
    else:
         output_hidden_states = False
    model, tokenizer, model_config = load_model(model_path, output_hidden_states = output_hidden_states, return_config = True, half = half)
    model_type = model_config.model_type
    print("This is a {} model".format(model_type))
    print("Model loaded")
    aa_shapes = [] 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device) 
    device_ids =list(range(0, torch.cuda.device_count()))
    print("device_ids", device_ids)
 
   
    hooked_activations = []
    # If needed, this can be turned in a nested dictionary
    # Saving all aa activations across layers = lots of space
    # If not needed, only save sequence activations. i.e. sum across aa activations
    def hook_aa(module, input, output):
        # The hooked values are appended to the 'hooked_activations' list.
        #output = torch.where(output == float('-inf'), torch.zeros_like(output), output)
        #print(output[:,:,:3])
        #hooked_activations.append(output[:,:,:3].cpu())  # Just first 3 neurons for testing
        hooked_activations.append(output.cpu())
        # Each output is a list of activations per token. then passing through each layer. 

        # Each layer of t5-small (6) is 2048 long (goes from 512 -> 2048 -> 512)
        # First loop through layers (ex. 6 layers)

    # If only doing sequence activations, use this one
    def hook_seq(module, input, output):
       
        #print("min", torch.min(output))
        # The hooked values are appended to the 'hooked_activations' list.
        #output = torch.where(output == float('-inf'), torch.zeros_like(output), output)

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

   

        # Each layer of t5-small (6) is 2048 long (goes from 512 -> 2048 -> 512)
        # First loop through layers (ex. 6 layers)

    # Set up aa_actiation hook
    if get_aa_activations == True:
        if  model_type == "t5":
            for i, block in enumerate(model.encoder.block):
               print("+++++ ", i, layers)# Then loop through components (attn, ff) of the layer
               if layers is None or i in layers:
                    block.layer[-1].DenseReluDense.wi.register_forward_hook(hook_aa) 
                    print("hooking layer: ", i)

            #    for x in block.layer:
            #        if isinstance(x, T5LayerFF) :
            #            x.DenseReluDense.wi.register_forward_hook(hook_aa)
            print("amino acid hooks registered")
  
        # Bert not tested yet
        # if model_type == "bert":
        #      for x in model.encoder.layer:
        #           if isinstance(x, BertIntermediate):
        #              x.dense.wi.register_forward_hook(hook_seq)

    # Set up sequence_activations only hook if not getting aa_activations
    if get_sequence_activations == True and get_aa_activations == False:
        if  model_type == "t5":
            for block in model.encoder.block:
               # Then loop through components (attn, ff) of the layer
               for x in block.layer:
                   if isinstance(x, T5LayerFF) :
                       x.DenseReluDense.wi.register_forward_hook(hook_seq)
            print("sequence hooks registered")
    
        if model_type == "bert":
             for x in model.encoder.layer:
                  if isinstance(x, BertIntermediate):
                     x.dense.wi.register_forward_hook(hook_seq)
                   
 
    if torch.cuda.device_count() > 1 and  cpu_only == False:
       print("Let's use", torch.cuda.device_count(), "GPUs!")
       model = nn.DataParallel(model, device_ids=device_ids).cuda()
   
    else:
       if cpu_only == True:
           half = False 
           print("Embedding on cpu, even though gpu available")

       model = model.to(device)

    batch_size = 1
    collate = Collate(tokenizer=tokenizer)

    data_loader = DataLoader(dataset=ListDataset(seqs),
                      batch_size=batch_size,
                      shuffle=False,
                      collate_fn=collate,
                      pin_memory=False)
    start = time.time()

    # Need to concatenate output of each chunk
    sequence_array_list = []
    sequence_sigma_array_list = []
    aa_array_list = []

    if sequence_pcamatrix_pkl:
          seq_pcamatrix, seq_bias = load_pcamatrix(sequence_pcamatrix_pkl)

    if aa_pcamatrix_pkl:
          aa_pcamatrix, aa_bias = load_pcamatrix(aa_pcamatrix_pkl)



    # Using awkward arrays for amino acids because sequence lengths are variable
    # If 10,000 long sequence was used, all sequences would be padded to 10,000
    # Awkward arrays allow concatenating ragged arrays
    count = 0
    maxlen = max(seqlens)
    print("padding", padding)
    print('maxlen', maxlen)
    get_sequence_embeddings_final_layer_only = False    
    numseqs = len(seqs)
    with torch.no_grad():

         # For each chunk of data
        counter = 0
        for data in data_loader:
            #seq_time = time.time()
            #print(count * batch_size, numseqs)
            input = data.to(device)
            # DataParallel model splits data to the different devices and gathers back
            # nvidia-smi shows 4 active devices (when there are 4 GPUs)
            #embed_time = time.time()
            model_output = model(**input)
             
            counter = counter + 1
            if counter % 100 == 0:
                   print("{} sequences run".format(counter))
                   torch.cuda.empty_cache()
                   gc.collect()
                   if get_sequence_embeddings == False and get_aa_embeddings == False:
                        del model_output  # Not needed, stored in hooked_activations
                        del data 

            if get_aa_embeddings == True or get_sequence_embeddings == True: 
                aa_time = time.time()
                aa_embeddings, aa_shape = retrieve_aa_embeddings(model_output, model_type = model_type, layers = layers, heads = heads, padding = padding)
                aa_embeddings = aa_embeddings.to('cpu')
                aa_embeddings = np.array(aa_embeddings)
                
                if get_sequence_embeddings == True:
                     # Get sentence embeddings by averaging aa embeddings
                     # Note that this includes padding characters in the mean
                     # Is there any overflow error coming from here?
                     sequence_embeddings = np.mean(aa_embeddings, axis = 1)

                     if strat == "meansig":
                          sequence_embeddings_sigma = np.std(aa_embeddings, axis = 1)
                          sequence_sigma_array_list.append(sequence_embeddings_sigma)


                     if sequence_pcamatrix_pkl:
                        sequence_embeddings = apply_pca(sequence_embeddings, seq_pcamatrix, seq_bias)
   

                     sequence_array_list.append(sequence_embeddings)


                if aa_pcamatrix_pkl:
                    aa_embeddings = np.apply_along_axis(apply_pca, 2, aa_embeddings, aa_pcamatrix, aa_bias)

                # Trim each down to just its sequence length
                if ragged_arrays == True:
                    aa_embed_ak_intermediate_list = []
                    for j in range(len(aa_embeddings)):
                         seqindex = (batch_size * count) + j
                         aa_embed_trunc = aa_embeddings[j][:seqlens[seqindex], :]
                    
                         aa_embed_ak = ak.Array(aa_embed_trunc)
                         aa_embed_ak_intermediate_list.append(aa_embed_ak)   

 
                    aa_array_list.append(np.concatenate(aa_embed_ak_intermediate))
               
                else:
                    if get_aa_embeddings == True:
                        # If not using ragged arrays, must pad to same dim as longest sequence
                        # print(maxlen - (aa_embeddings.shape[1] - 1))
                         #if padding:
                        dim2 = maxlen - (aa_embeddings.shape[1])
                        npad = ((0,0), (0, dim2), (0,0))
                        aa_embeddings = np.pad(aa_embeddings, npad)
                        aa_array_list.append(aa_embeddings)

            count = count + 1
        end = time.time() 
        print("Total time to embed = {}".format(end - start))
  
 
        # Collect hooked activations 
        lengths = np.array(seqlens)
        embedding_dict = {}
        if layers:                 #______________________________________________________
            numlayers = len(layers)
        else:
            numlayers = model_config.num_layers
        numneurons =  model_config.d_ff
 
        if get_aa_activations == True: 
                # Need to use hook_aa here
 
                # This list is 24 (num layers) * num seqs
                # Squeeze removes extra dimension (1, x, x)
                hooked_activations = [x.squeeze(0) for x in hooked_activations]


                # Break into individual lists for each sequence
                new_lists = [hooked_activations[i:i+numlayers] for i in range(0, len(hooked_activations), numlayers)]

                # Combine each sequence's activations into one array
                stacked_lists = [np.stack(x) for x in new_lists]

                # Combine all the activations from different layers into one dimension
                reshaped = [np.concatenate(x, axis = 1) for x  in stacked_lists]

                
                # Do the sentence one first, because doesnt require padding. 
                # This is related to sequence length...what about the mean?
                if get_sequence_activations == True:
                    sequence_activations_list = [np.max(x.astype(np.float32), axis=0) for x in reshaped]
                    sequence_activations = np.stack(sequence_activations_list)
                    contains_inf = np.any(np.isinf(sequence_activations) & (sequence_activations < 0))
                    embedding_dict['sequence_activations'] = sequence_activations


                padded_activations = [np.pad(x, [(0, maxlen + 1 - x.shape[0] ), (0, 0)]) for x in reshaped]
                print("padded activations done")
                #print("--------> ", padded_activations)
                # stacked = np.stack(padded_activations)
                embedding_dict['aa_activations'] = padded_activations #stacked

        elif get_sequence_activations == True:
                stacked = np.stack([x for x in hooked_activations])
                # go from (numlayers, numseqs, numneurons) to (numseqs, numlayers * numneurons)
                sequence_activations = stacked.reshape(numseqs, numlayers * numneurons)  #np.transpose(stacked, (1, 0, 2)).reshape(2, -1)  
                embedding_dict['sequence_activations'] = sequence_activations
               


        if get_sequence_embeddings == True:
            embedding_dict['sequence_embeddings'] = np.concatenate(sequence_array_list)
            if strat == "meansig":
                embedding_dict['sequence_embeddings_sigma'] = np.concatenate(sequence_sigma_array_list)

        if get_aa_embeddings == True:

            embedding_dict['aa_embeddings'] = np.concatenate(aa_array_list)

        print("Complete")
        return(embedding_dict)

    

if __name__ == "__main__":

    args = get_embed_args()



   
    if args.get_sequence_embeddings == False:
         if args.get_aa_embeddings == False:
             if args.get_sequence_activations == False:
                 if args.get_aa_activations == False:    # <-----------------
                     print("Must add --get_sequence_embeddings and/or --get_aa_embeddings and/or --get_sequence_activations and/or --get_aa_activations, otherwise nothing to compute")
                     exit(1)
    ids, sequences, sequences_spaced = parse_fasta_for_embed(fasta_path = args.fasta_path, 
                                                             truncate = args.truncate, 
                                                             padding = args.padding)

    print("First sequences")
    seqlens = [len(x) for x in sequences]
    
    layers = args.layers
    heads = args.heads
    padding = args.padding 
    cpu_only = args.cpu_only
    if heads is not None:
       with open(heads, "r") as f:
         headnames = f.readlines()
         print(headnames)
         headnames = [x.replace("\n", "") for x in headnames]

         print(headnames)
    else:
       headnames = None


    embedding_dict = get_embeddings(sequences_spaced, 
                                    args.model_path, 
                                    get_sequence_embeddings = args.get_sequence_embeddings, 
                                    get_aa_embeddings = args.get_aa_embeddings, 
                                    get_sequence_activations = args.get_sequence_activations,
                                    get_aa_activations = args.get_aa_activations,
                                    padding = padding, 
                                    seqlens = seqlens,
                                    layers = layers,
                                    heads = headnames,
                                    ragged_arrays = args.ragged_arrays,
                                    aa_pcamatrix_pkl = args.aa_pcamatrix_pkl, 
                                    sequence_pcamatrix_pkl = args.sequence_pcamatrix_pkl,
                                    strat = args.strat,
                                    cpu_only = cpu_only)
    # Reduce sequence dimension with a new pca transform 

    if args.sequence_target_dim:

       pkl_pca_out = "{}.sequence.{}dim.pcamatrix.pkl".format(args.fasta_path, args.sequence_target_dim)

       embedding_dict['sequence_embeddings'] =  control_pca(embedding_dict, 

                                                'sequence_embeddings', 

                                                pkl_pca_out = pkl_pca_out, 

                                                target_dim = args.sequence_target_dim, 

                                                max_train_sample_size = None)



    # Reduce aa dimension with a new pca transform 

    if args.aa_target_dim:

       pkl_pca_out = "{}.aa.{}dim.pcamatrix.pkl".format(args.fasta_path, args.aa_target_dim)

       embedding_dict['aa_embeddings'] =  control_pca(embedding_dict, 

                                                'aa_embeddings', 

                                                pkl_pca_out = pkl_pca_out, 

                                                target_dim = args.aa_target_dim, 

                                                max_train_sample_size = None)



             

    #Store sequences & embeddings on disk

    if args.pkl_out:



        with open(args.pkl_out, "wb") as fOut:

           pickle.dump(embedding_dict, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    

        pkl_log = "{}.description".format(args.pkl_out)

        with open(pkl_log, "w") as pOut: 
            if args.get_aa_activations == True:
               pOut.write("Object {} dimensions: {}\n".format('aa_activations', embedding_dict['aa_activations'].shape))

            if args.get_sequence_activations == True:
               pOut.write("Object {} dimensions: {}\n".format('sequence_activations', embedding_dict['sequence_activations'].shape))
  
            if args.get_sequence_embeddings == True:
               pOut.write("Object {} dimensions: {}\n".format('sequence_embeddings', embedding_dict['sequence_embeddings'].shape))


            if args.get_aa_embeddings == True:    
                if args.ragged_arrays == True:
                    pOut.write("Object {} dimensions: {}\n".format('aa_embeddings', embedding_dict['aa_embeddings'].type))
                    pOut.write("aa_embeddings are an `awkward` arrays with dimensions\n")   
                    pOut.write("aa_embeddings are an `awkward` arrays with dimensions\n")   
            
                    pOut.write("{}".format(ak.num(embedding_dict['aa_embeddings'], axis=0)))
                    pOut.write("{}".format(ak.num(embedding_dict['aa_embeddings'], axis=1)))
                # Else it's a square numpy array
                else:
                    pOut.write("Object {} dimensions: {}\n".format('aa_embeddings', embedding_dict['aa_embeddings'].shape))
               
    
            pOut.write("Contains sequences:\n")
            for x in ids:
              pOut.write("{}\n".format(x))
    
            seq_file = "{}.seqnames".format(args.pkl_out)
            with open(seq_file, "w") as pOut2:
                for x in ids:
                  pOut2.write("{}\n".format(x))

    
    
def embed_sequences(model_path, sequences, extra_padding,  pkl_out):
    '''
    
    Get a pkl of embeddings for a list of sequences using a particular model
    Embeddings will have shape xx

    Takes:
       model_path (str): Path to a particular transformer model
                         ex. "prot_bert_bfd"
       sequences (list): List of sequences with a space between each acids.  
                         ex ["M E T", "S E Q"]
       pkl_out (str)   : Filename of output pickle of embeddings
 
    '''
    print("Create word embedding model")
    word_embedding_model = models.Transformer(model_path)

    # Default pooling strategy
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    print("SentenceTransformer model created")
    
    #pool = model.start_multi_process_pool()

    # Compute the embeddings using the multi-process pool
    # about 1.5 hours to this step with 4 GPU and 1.4 million sequences 
    print("Computing embeddings")
    print("Prior max sequence length", model.max_seq_length)
    model.max_seq_length = 1024
    print("New max sequence length", model.max_seq_length)

    embeddings = model.encode(sequences, output_value = 'token_embeddings')
    #print(e)

    #embeddings = model.encode_multi_process(sequences, pool, output_value = 'token_embeddings')

    print("Embeddings computed. Shape:", embeddings.shape)

    #Optional: Stop the proccesses in the pool
    #model.stop_multi_process_pool(pool)

    return(embeddings)    


