from tqdm import tqdm
from constants import *
from custom_data import *
from transformer import *
from data_structure import *
from torch import nn
from accelerate import Accelerator
from custom_data import get_dataloader

import torch
import sys, os
import numpy as np
import argparse
import datetime
import copy
import heapq
import sentencepiece as spm

class Manager():    
    def __init__(self, is_train=True, ckpt_name=None, dataset_name=None):
        
        # 1. INITIALIZE ACCELERATOR
        # This automatically detects if you have 1 GPU, 4 GPUs, or TPUs.
        # It replaces "device = torch.device('cuda')"
        self.accelerator = Accelerator()
        self.device = self.accelerator.device 
        
        # Only print logs on the main process (GPU 0)
        # Otherwise, if you have 4 GPUs, you will see "Loading..." printed 4 times.
        if self.accelerator.is_main_process:
            print(f"Distributed Training Enabled. Device: {self.device}")
            print("Loading SentencePiece models...")

        self.src_sp = spm.SentencePieceProcessor()
        self.src_sp.Load(f"{SP_DIR}/{src_model_prefix}.model")
        self.trg_sp = spm.SentencePieceProcessor()
        self.trg_sp.Load(f"{SP_DIR}/{trg_model_prefix}.model")
        
        self.src_vocab_size = self.src_sp.GetPieceSize()
        self.trg_vocab_size = self.trg_sp.GetPieceSize()
        self.pad_id = 3 # Ensure this matches your tokenizer training

        # 2. MODEL SETUP
        if self.accelerator.is_main_process:
            print("Loading Transformer model...")
            
        self.model = Transformer(
            src_vocab_size=self.src_vocab_size, 
            trg_vocab_size=self.trg_vocab_size,
        )
        # Note: No need for .to(device) here. Accelerator handles it in .prepare()

        self.optim = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        self.best_loss = sys.float_info.max

        # 3. CHECKPOINT LOADING
        if ckpt_name:
            ckpt_path = f"{ckpt_dir}/{ckpt_name}"
            if os.path.exists(ckpt_path):
                # Map location is crucial for distributed loading
                checkpoint = torch.load(ckpt_path, map_location='cpu') 
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optim.load_state_dict(checkpoint['optim_state_dict'])
                self.best_loss = checkpoint.get('loss', sys.float_info.max)
                if self.accelerator.is_main_process:
                    print(f"Loaded checkpoint: {ckpt_name}")
        else:
            # CASE C: No checkpoint name provided -> Init from scratch
            if self.accelerator.is_main_process:
                print("No checkpoint provided. Initializing new parameters.")
            for param in self.model.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

        # 4. PREPARE DATALOADERS
        if is_train:
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
            
            # Get the standard PyTorch dataloader
            raw_loader = get_dataloader(dataset_name, self.src_sp, self.trg_sp)
            
            # 5. THE MAGIC LINE: ACCELERATOR.PREPARE
            # This wraps the model in DDP, moves it to GPU, and 
            # splits the dataloader across GPUs automatically.
            self.model, self.optim, self.train_loader = self.accelerator.prepare(
                self.model, self.optim, raw_loader
            )

    def train(self):
        if self.accelerator.is_main_process:
            print("Training starts.")
            
        num_epochs = self.config.get('epochs', 10)
        
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            train_losses = []
            
            # Disable tqdm on non-main processes to avoid messy output
            progress_bar = tqdm(
                self.train_loader, 
                desc=f"Epoch {epoch}", 
                disable=not self.accelerator.is_main_process
            )

            for batch in progress_bar:
                # 6. DATA PLACEMENT
                # Accelerator handles .to(device) automatically!
                # You just unpack.
                src_padded, tgt_in_padded, tgt_out_padded, e_mask, d_mask = batch
                
                # Forward pass
                output = self.model(src_padded, tgt_in_padded, e_mask, d_mask) 
                
                output_flat = output.view(-1, self.trg_vocab_size)
                target_flat = tgt_out_padded.view(-1)

                loss = self.criterion(output_flat, target_flat)

                # 7. BACKWARD PASS
                # Replace loss.backward() with this:
                self.accelerator.backward(loss)
                
                # Clip gradients (using accelerator method)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optim.step()
                self.optim.zero_grad()

                train_losses.append(loss.item())

            # 8. LOGGING & SAVING (Main Process Only)
            # We wait for all GPUs to finish the epoch before saving
            self.accelerator.wait_for_everyone()
            
            # Calculate mean loss across all GPUs for accurate reporting
            mean_loss = np.mean(train_losses)
            
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch} | Loss: {mean_loss:.4f}")

                if mean_loss < self.best_loss:
                    self.best_loss = mean_loss
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    
                    # 9. UNWRAP MODEL BEFORE SAVING
                    # DDP wraps model in 'module.', we need to unwrap it 
                    # so we can load it easily later on a CPU or single GPU.
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    
                    torch.save({
                        'model_state_dict': unwrapped_model.state_dict(),
                        'optim_state_dict': self.optim.state_dict(),
                        'loss': self.best_loss
                    }, f"{ckpt_dir}/best_ckpt.tar")
                    print("Checkpoint Saved.")
        
    def validation(self, dataset_name = None):
        print("Validation processing...")
        self.model.eval()
        
        valid_losses = []
        start_time = datetime.datetime.now()

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.valid_loader)):
                src_input, trg_input, trg_output = batch
                src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)

                e_mask, d_mask = self.make_mask(src_input, trg_input)

                output = self.model(src_input, trg_input, e_mask, d_mask) # (B, L, vocab_size)

                trg_output_shape = trg_output.shape
                loss = self.criterion(
                    output.view(-1, sp_vocab_size),
                    trg_output.view(trg_output_shape[0] * trg_output_shape[1])
                )

                valid_losses.append(loss.item())

                del src_input, trg_input, trg_output, e_mask, d_mask, output
                torch.cuda.empty_cache()

        end_time = datetime.datetime.now()
        validation_time = end_time - start_time
        seconds = validation_time.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        
        mean_valid_loss = np.mean(valid_losses)
        
        return mean_valid_loss, f"{hours}hrs {minutes}mins {seconds}secs"

    def inference(self, input_sentence, method):
        self.model.eval()
        
        # 1. Encode Input
        input_ids = self.src_sp.EncodeAsIds(input_sentence)
        src_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(self.device) # (1, L)
        
        # 2. Create Mask
        e_mask = (src_tensor != self.pad_id).unsqueeze(1).unsqueeze(2)
        start_time = datetime.datetime.now()
        
        print(f"Translating: '{input_sentence}'...")

        with torch.no_grad():
            # --- CORRECTION HERE ---
            # 1. Embed
            src_emb = self.model.src_embedding(src_tensor)
            
            # 2. Add Position Info (CRITICAL!)
            src_emb = self.model.positional_encoder(src_emb)
            
            # 3. Pass to Encoder
            e_output = self.model.encoder(src_emb, e_mask) 
            # -----------------------

            if method == 'greedy':
                result = self.greedy_search(e_output, e_mask)
            elif method == 'beam':
                result = self.beam_search(e_output, e_mask)

        end_time = datetime.datetime.now()

        total_inference_time = end_time - start_time
        seconds = total_inference_time.seconds
        minutes = seconds // 60
        seconds = seconds % 60

        print(f"Input: {input_sentence}")
        print(f"Result: {result}")
        print(f"Inference finished! || Total inference time: {minutes}mins {seconds}secs")
        
    def greedy_search(self, e_output, e_mask):
        last_words = torch.LongTensor([pad_id] * seq_len).to(device) # (L)
        last_words[0] = bos_id # (L)
        cur_len = 1

        for i in range(seq_len):
            d_mask = (last_words.unsqueeze(0) != pad_id).unsqueeze(1).to(device) # (1, 1, L)
            nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)  # (1, L, L)
            nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
            d_mask = d_mask & nopeak_mask  # (1, L, L) padding false

            trg_embedded = self.model.trg_embedding(last_words.unsqueeze(0))
            trg_positional_encoded = self.model.positional_encoder(trg_embedded)
            decoder_output = self.model.decoder(
                trg_positional_encoded,
                e_output,
                e_mask,
                d_mask
            ) # (1, L, d_model)

            output = self.model.softmax(
                self.model.output_linear(decoder_output)
            ) # (1, L, trg_vocab_size)

            output = torch.argmax(output, dim=-1) # (1, L)
            last_word_id = output[0][i].item()
            
            if i < seq_len-1:
                last_words[i+1] = last_word_id
                cur_len += 1
            
            if last_word_id == eos_id:
                break

        if last_words[-1].item() == pad_id:
            decoded_output = last_words[1:cur_len].tolist()
        else:
            decoded_output = last_words[1:].tolist()
        decoded_output = self.trg_sp.decode_ids(decoded_output)
        
        return decoded_output
    
    def beam_search(self, e_output, e_mask):
        cur_queue = PriorityQueue()
        for k in range(beam_size):
            cur_queue.put(BeamNode(bos_id, -0.0, [bos_id]))
        
        finished_count = 0
        
        for pos in range(seq_len):
            new_queue = PriorityQueue()
            for k in range(beam_size):
                node = cur_queue.get()
                if node.is_finished:
                    new_queue.put(node)
                else:
                    trg_input = torch.LongTensor(node.decoded + [pad_id] * (seq_len - len(node.decoded))).to(device) # (L)
                    d_mask = (trg_input.unsqueeze(0) != pad_id).unsqueeze(1).to(device) # (1, 1, L)
                    nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)
                    nopeak_mask = torch.tril(nopeak_mask) # (1, L, L) to triangular shape
                    d_mask = d_mask & nopeak_mask # (1, L, L) padding false
                    
                    trg_embedded = self.model.trg_embedding(trg_input.unsqueeze(0))
                    trg_positional_encoded = self.model.positional_encoder(trg_embedded)
                    decoder_output = self.model.decoder(
                        trg_positional_encoded,
                        e_output,
                        e_mask,
                        d_mask
                    ) # (1, L, d_model)

                    output = self.model.softmax(
                        self.model.output_linear(decoder_output)
                    ) # (1, L, trg_vocab_size)
                    
                    output = torch.topk(output[0][pos], dim=-1, k=beam_size)
                    last_word_ids = output.indices.tolist() # (k)
                    last_word_prob = output.values.tolist() # (k)
                    
                    for i, idx in enumerate(last_word_ids):
                        new_node = BeamNode(idx, -(-node.prob + last_word_prob[i]), node.decoded + [idx])
                        if idx == eos_id:
                            new_node.prob = new_node.prob / float(len(new_node.decoded))
                            new_node.is_finished = True
                            finished_count += 1
                        new_queue.put(new_node)
            
            cur_queue = copy.deepcopy(new_queue)
            
            if finished_count == beam_size:
                break
        
        decoded_output = cur_queue.get().decoded
        
        if decoded_output[-1] == eos_id:
            decoded_output = decoded_output[1:-1]
        else:
            decoded_output = decoded_output[1:]
            
        return self.trg_sp.decode_ids(decoded_output)
        

    def make_mask(self, src_input, trg_input):
        e_mask = (src_input != pad_id).unsqueeze(1)  # (B, 1, L)
        d_mask = (trg_input != pad_id).unsqueeze(1)  # (B, 1, L)

        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask).to(device)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (B, L, L) padding false

        return e_mask, d_mask


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, help="train or inference?")
    parser.add_argument('--ckpt_name', required=False, help="best checkpoint file")
    parser.add_argument('--input', type=str, required=False, help="input sentence when inferencing")
    parser.add_argument('--decode', type=str, required=True, default="greedy", help="greedy or beam?")
    parser.add_argument('--dataset_name', type=str, required=False, help="path to config file")

    args = parser.parse_args()

    if args.mode == 'train':
        if args.ckpt_name is not None:
            manager = Manager(is_train=True, ckpt_name=args.ckpt_name)
        else:
            manager = Manager(is_train=True)

        manager.train(dataset_name=args.dataset_name)
    elif args.mode == 'inference':
        assert args.ckpt_name is not None, "Please specify the model file name you want to use."
        assert args.input is not None, "Please specify the input sentence to translate."
        assert args.decode == 'greedy' or args.decode =='beam', "Please specify correct decoding method, either 'greedy' or 'beam'."
       
        manager = Manager(is_train=False, ckpt_name=args.ckpt_name)
        manager.inference(args.input, args.decode)

    else:
        print("Please specify mode argument either with 'train' or 'inference'.")