import torch

# Path or parameters for data
DATASET_NAME = 'ncduy/mt-en-vi'
SP_DIR = 'trained_tokenizer'
SRC_DIR = 'src'
TRG_DIR = 'trg'
SRC_RAW_DATA_NAME = 'raw_data.src'
TRG_RAW_DATA_NAME = 'raw_data.trg'
TRAIN_NAME = 'train.txt'
VALID_NAME = 'valid.txt'
TEST_NAME = 'test.txt'

# Parameters for sentencepiece tokenizer
pad_id = 3
bos_id = 1
eos_id = 2
unk_id = 0  
src_model_prefix = 'english_toknizer_spm'
trg_model_prefix = 'vietnamese_toknizer_spm'
sp_vocab_size = 16000
character_coverage = 1.0
model_type = 'unigram'

# Parameters for Transformer & training
num_workers = 4
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
learning_rate = 1e-4 
batch_size = 32
max_len = 5000
seq_len = 256 #max sequence length
num_heads = 8 # of attention heads
num_layers = 6 
d_model = 512 #embedding size
d_ff = 2048 #feedforward inner-layer dimension
d_k = d_model // num_heads 
drop_out_rate = 0.1
num_epochs = 1
beam_size = 3
ckpt_dir = 'saved_model'
eval_step = 50
USE_ROPE = True
#aha