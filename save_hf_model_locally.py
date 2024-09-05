
import os

os.environ['HF_HOME'] = '/scratch/gpfs/cmcwhite/.cache/'


#############
#from transformers import T5Tokenizer, T5Model
#tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
#model = T5Model.from_pretrained("Rostlab/prot_t5_xl_uniref50", cache_dir="/scratch/gpfs/cmcwhite/cache/")                                               
#tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", cache_dir="/scratch/gpfs/cmcwhite/cache/")
#tokenizer.save_pretrained('/scratch/gpfs/cmcwhite/prot_t5_xl_uniref50')
#model.save_pretrained('/scratch/gpfs/cmcwhite/prot_t5_xl_uniref50')     



#############
#from transformers import AutoTokenizer, EsmForProteinFolding
#tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
#model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
#tokenizer.save_pretrained('/scratch/gpfs/cmcwhite/esmfold_v1')
#model.save_pretrained('/scratch/gpfs/cmcwhite/esmfold_v1')     


############
#from transformers import AutoModelForSequenceClassification, BertTokenizer
#sourcename = "Rostlab/prot_t5_xxl_bfd"
#modelname = "prot_t5_xxl_bfd"
#outdir = "/scratch/gpfs/cmcwhite/hfmodels/" + modelname
#print(sourcename, modelname, outdir)
#tokenizer = BertTokenizer.from_pretrained(sourcename)
#tokenizer.save_pretrained(outdir)
#model = AutoModelForSequenceClassification.from_pretrained(sourcename)
#model.save_pretrained(outdir)

#########
#from transformers import AutoTokenizer, AutoModelForCausalLM
#tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
#model = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2")
#tokenizer.save_pretrained('/scratch/gpfs/cmcwhite/protGPT2')
#model.save_pretrained('/scratch/gpfs/cmcwhite/protGPT2')     



##############
#from transformers import AutoTokenizer, AutoModelForMaskedLM
#tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
#model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t36_3B_UR50D")
#tokenizer.save_pretrained('/scratch/gpfs/cmcwhite/esm2_t36_3B_UR50D')
#model.save_pretrained('/scratch/gpfs/cmcwhite/esm2_t36_3B_UR50D')     

#########
#tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t48_15B_UR50D")
#model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t48_15B_UR50D")
#tokenizer.save_pretrained('/scratch/gpfs/cmcwhite/esm2_t48_15B_UR50D')
#model.save_pretrained('/scratch/gpfs/cmcwhite/esm2_t48_15B_UR50D')     

############
#from transformers import AutoTokenizer, AutoModelForTokenClassification
#tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd_ss3")
#tokenizer.save_pretrained('/scratch/gpfs/cmcwhite/hfmodels/prot_bert_bfd_ss3')
#model = AutoModelForTokenClassification.from_pretrained("Rostlab/prot_bert_bfd_ss3")
#model.save_pretrained("/scratch/gpfs/cmcwhite/hfmodels/prot_bert_bfd_ss3")


##############
#tokenizer = AlbertTokenizer.from_pretrained("Rostlab/prot_t5_xxl_bfd")
#tokenizer.save_pretrained('/scratch/gpfs/cmcwhite/hfmodels/prot_t5_xxl_bfd')
#model = AutoModel.from_pretrained("Rostlab/prot_t5_xxl_bfd")
#model.save_pretrained('/scratch/gpfs/cmcwhite/hfmodels/prot_t5_xxl_bfd')

##############
##from transformers import AutoModelForSequenceClassification, BertTokenizer
#tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_t5_xxl_bfd')
#tokenizer.save_pretrained('/scratch/gpfs/cmcwhite/hfmodels/prot_t5_xxl_bfd')
#model = AutoModelForSequenceClassification.from_pretrained('Rostlab/prot_t5_xxl_bfd')
##model.save_pretrained('/scratch/gpfs/cmcwhite/hfmodels/prot_t5_xxl_bfd')
