from tqdm import tqdm
from constants import *
from custom_data import *
from transformer import *
from data_structure import *
from torch import nn
from accelerate import Accelerator
from custom_data import get_dataloader

import wandb
import torch
import sys, os
import numpy as np
import argparse
import datetime
import copy
import heapq
import sentencepiece as spm

class Manager():    
    def __init__(self, is_train=True, ckpt_name=None):
        
        # 1. INITIALIZE ACCELERATOR
        # This automatically detects if you have 1 GPU, 4 GPUs, or TPUs.
        # It replaces "device = torch.device('cuda')"
        self.accelerator = Accelerator(
            mixed_precision="fp16",
            log_with="wandb"  # <--- NEW
        )
        self.config = {
            'lr' : learning_rate,
            'epochs': num_epochs,
            'batch_size': batch_size,  # Very important to track
            'd_model': d_model,  # Model size
            'n_layers': num_layers,  # Depth
            'dropout': drop_out_rate  # Regularization
        }
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
        self.dataset_name = DATASET_NAME
        #hi
        
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
            self.accelerator.init_trackers(
                project_name = "nmt_project",
                config = self.config,
            )
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
            
            # Get the standard PyTorch dataloader
            train_loader = get_dataloader(self.dataset_name, self.src_sp, self.trg_sp, split = 'train[:500000]')
            valid_loader = get_dataloader(self.dataset_name, self.src_sp, self.trg_sp, split = 'validation')
            
            # 5. THE MAGIC LINE: ACCELERATOR.PREPARE
            # This wraps the model in DDP, moves it to GPU, and 
            # splits the dataloader across GPUs automatically.
            self.model, self.optim, self.train_loader, self.valid_loader = self.accelerator.prepare(
                self.model, self.optim, train_loader, valid_loader
            )

    def train(self):
        if self.accelerator.is_main_process:
            print("Training starts.")

        num_epochs = self.config.get('epochs', 10)

        # 1. Initialize Global Step
        global_step = 0
        VALIDATION_FREQ = eval_step  # Validate every 500 steps

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            train_losses = []

            # Disable tqdm on non-main GPUs
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=not self.accelerator.is_main_process)

            for batch in progress_bar:
                # --- FORWARD PASS ---
                src_padded, tgt_in_padded, tgt_out_padded = batch
                e_mask, d_mask = self.create_mask(src_padded, tgt_in_padded)

                output = self.model(src_padded, tgt_in_padded, e_mask, d_mask)

                output_flat = output.view(-1, self.trg_vocab_size)
                target_flat = tgt_out_padded.view(-1)
                loss = self.criterion(output_flat, target_flat)

                # --- BACKWARD PASS ---
                self.accelerator.backward(loss)
                self.optim.step()
                self.optim.zero_grad()

                # --- LOGGING ---
                # Gather loss for accurate logging
                current_loss = self.accelerator.gather(loss).mean().item()
                train_losses.append(current_loss)

                # Increment Step
                global_step += 1
                progress_bar.set_postfix(loss=current_loss, step=global_step)

                # Log training loss to WandB every 10 steps (optional, to keep graphs smooth)
                if global_step % 10 == 0:
                    self.accelerator.log({
                        "train_loss": current_loss,
                        "epoch": epoch,
                        "learning_rate": self.optim.param_groups[0]['lr']
                    }, step=global_step)

                # ============================================================
                # VALIDATION CHECK (EVERY "eval_step" STEPS)
                # ============================================================
                if global_step % VALIDATION_FREQ == 0:

                    # A. Run Validation
                    # (This function puts model in .eval() mode)
                    val_loss, val_time = self.validation()

                    if self.accelerator.is_main_process:
                        print(f"\n[Step {global_step}] Val Loss: {val_loss:.4f} | Time: {val_time}")

                        # B. Log to WandB
                        self.accelerator.log({"val_loss": val_loss}, step=global_step)

                        # C. Save Best Checkpoint
                        if val_loss < self.best_loss:
                            self.best_loss = val_loss
                            print(f"New Best Val Loss! Saving checkpoint...")

                            unwrapped_model = self.accelerator.unwrap_model(self.model)

                            if not os.path.exists(ckpt_dir):
                                os.makedirs(ckpt_dir)

                            torch.save({
                                'model_state_dict': unwrapped_model.state_dict(),
                                'optim_state_dict': self.optim.state_dict(),
                                'loss': self.best_loss,
                                'step': global_step
                            }, f"{ckpt_dir}/best_ckpt.tar")

                    # D. CRITICAL: SWITCH BACK TO TRAIN MODE
                    # If you forget this, training stops working correctly
                    self.model.train()

        self.accelerator.end_training()

    def validation(self):
        # Only print on main process
        if self.accelerator.is_main_process:
            print("Validation processing...")

        self.model.eval()
        valid_losses = []
        start_time = datetime.datetime.now()

        with torch.no_grad():
            # Disable tqdm on non-main GPUs to keep output clean
            for batch in tqdm(self.valid_loader, desc="Validating", disable=not self.accelerator.is_main_process):
                # 1. Unpack & Move to Device
                # (If valid_loader wasn't prepared by accelerator, we move manually)
                src_padded, tgt_in_padded, tgt_out_padded = batch
                src_padded = src_padded.to(self.device)
                tgt_in_padded = tgt_in_padded.to(self.device)
                tgt_out_padded = tgt_out_padded.to(self.device)

                # 2. Create Masks (The new 4D function)
                e_mask, d_mask = self.create_mask(src_padded, tgt_in_padded)

                # 3. Forward
                output = self.model(src_padded, tgt_in_padded, e_mask, d_mask)

                # 4. Compute Loss
                output_flat = output.view(-1, self.trg_vocab_size)
                target_flat = tgt_out_padded.view(-1)
                loss = self.criterion(output_flat, target_flat)

                # 5. Gather Loss from all GPUs (Crucial for Distributed Training)
                # If we don't gather, we only see the loss from GPU 0
                avg_loss = self.accelerator.gather(loss).mean().item()
                valid_losses.append(avg_loss)

        # Time Calculation
        end_time = datetime.datetime.now()
        validation_time = end_time - start_time

        # Format Time string
        seconds = validation_time.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        time_str = f"{hours}h {minutes}m {seconds}s"

        mean_valid_loss = np.mean(valid_losses)

        return mean_valid_loss, time_str

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
        # 1. ANCHOR DEVICE (Crucial for Accelerate)
        device = e_output.device

        # 2. Use Log-Probabilities (NLL)
        # PriorityQueue in Python pops the LOWEST value first.
        # Since LogProbs are negative (0 is best, -inf is worst),
        # we usually minimize Negative Log Likelihood (Positive numbers).
        # Score 0.0 is perfect. Score 100.0 is bad.
        cur_queue = PriorityQueue()

        # Start with [BOS]
        # (Score, Sequence) -> We store Score first for sorting
        cur_queue.put(BeamNode(bos_id, 0.0, [bos_id]))

        finished_nodes = []

        # Max Length Loop
        for pos in range(100):  # Don't rely on global seq_len, set a reasonable inference limit
            new_queue = PriorityQueue()

            # If queue is empty (all beams finished), break
            if cur_queue.empty():
                break

            # Process the top K beams
            # Note: A true vectorized beam search processes all K in one batch.
            # This loop version is slower but easier to understand.
            for k in range(min(beam_size, cur_queue.qsize())):
                node = cur_queue.get()

                if node.is_finished:
                    new_queue.put(node)
                    continue

                # --- DYNAMIC TENSOR CREATION ---
                # 1. Create Tensor on the correct device (No Padding needed for inference!)
                trg_input = torch.LongTensor([node.decoded]).to(device)  # Shape (1, Curr_Len)

                # 2. Create Masks Dynamically
                # We use the helper logic directly here for speed
                trg_len = trg_input.size(1)
                d_pad_mask = (trg_input != self.pad_id).unsqueeze(1)  # (1, 1, L)
                nopeak_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()
                d_mask = d_pad_mask & nopeak_mask.unsqueeze(0)  # (1, 1, L, L)

                # 3. Model Forward
                # Embed & PE
                trg_emb = self.model.trg_embedding(trg_input)
                trg_emb = self.model.positional_encoder(trg_emb)

                # Decoder
                decoder_output = self.model.decoder(
                    trg_emb,
                    e_output,
                    e_mask,
                    d_mask
                )

                # 4. Get Log Softmax (CRITICAL CHANGE)
                # We take the last token output
                logits = self.model.output_linear(decoder_output[:, -1, :])  # (1, Vocab)
                log_probs = self.model.log_softmax(logits)  # (1, Vocab)

                # 5. Top K
                # We want the highest log_probs (closest to 0)
                # torch.topk returns values (largest first).
                topk_output = torch.topk(log_probs, k=beam_size, dim=-1)

                last_word_ids = topk_output.indices[0].tolist()
                last_word_log_probs = topk_output.values[0].tolist()

                for i, idx in enumerate(last_word_ids):
                    # NLL Score = Previous Score + (-1 * current_log_prob)
                    # We minimize the score.
                    score_increment = -last_word_log_probs[i]
                    new_score = node.prob + score_increment

                    new_decoded = node.decoded + [idx]
                    new_node = BeamNode(idx, new_score, new_decoded)

                    if idx == eos_id:
                        # Length Penalty (Normalize score by length)
                        # Otherwise short sentences are always preferred
                        new_node.prob = new_node.prob / len(new_decoded)
                        new_node.is_finished = True
                        finished_nodes.append(new_node)
                    else:
                        new_queue.put(new_node)

            # Keep only top Beam Size nodes for next round
            # PriorityQueue doesn't slice, so we manually transfer
            cur_queue = PriorityQueue()
            for _ in range(beam_size):
                if new_queue.empty(): break
                cur_queue.put(new_queue.get())

            # If we found enough finished sentences, we can stop early
            if len(finished_nodes) >= beam_size:
                break

        # Select best
        if len(finished_nodes) > 0:
            # Sort by score (lowest NLL is best)
            finished_nodes.sort(key=lambda x: x.prob)
            best_node = finished_nodes[0]
        else:
            best_node = cur_queue.get()  # Get current best incomplete

        decoded_output = best_node.decoded

        # Strip special tokens
        if decoded_output[0] == bos_id: decoded_output = decoded_output[1:]
        if decoded_output and decoded_output[-1] == eos_id: decoded_output = decoded_output[:-1]

        return self.trg_sp.DecodeIds(decoded_output)

    def create_mask(self, src, tgt):
        # src: (Batch, Src_Len)
        # tgt: (Batch, Tgt_Len)

        # 1. Source Padding Mask
        # Use src.device as the anchor!
        src_mask = (src != self.pad_id).unsqueeze(1).unsqueeze(2).to(src.device)

        # 2. Target Padding Mask
        tgt_pad_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2).to(tgt.device)

        # 3. Target No-Peak (Causal) Mask
        tgt_len = tgt.size(1)
        # Create on the same device as tgt
        nopeak_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()

        # 4. Combine
        tgt_mask = tgt_pad_mask & nopeak_mask.unsqueeze(0)

        return src_mask, tgt_mask


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

        manager.train()
    elif args.mode == 'inference':
        assert args.ckpt_name is not None, "Please specify the model file name you want to use."
        assert args.input is not None, "Please specify the input sentence to translate."
        assert args.decode == 'greedy' or args.decode =='beam', "Please specify correct decoding method, either 'greedy' or 'beam'."
       
        manager = Manager(is_train=False, ckpt_name=args.ckpt_name)
        manager.inference(args.input, args.decode)

    else:
        print("Please specify mode argument either with 'train' or 'inference'.")