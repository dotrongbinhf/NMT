from tqdm import tqdm
import math

import constants
from constants import *
from custom_data import *
from transformer import *
from data_structure import *
from torch import nn
from accelerate import Accelerator
from custom_data import get_dataloader
from transformers import get_scheduler

import wandb
import torch
import sys, os
import numpy as np
import argparse
import datetime
import copy
import heapq
import sentencepiece as spm
import sacrebleu


#Accelerate: thu vien cua huggingface dung` de train tren multiple gpus
#Loss: CrossEntropyLoss
#Optimizer: Adam
#500000 pair of data
#epoch: 5
#global_step: train tren bao nhieu buoc
#best: bao nhieu loss
#Bleu: t gui sau
#Thuat toan de inference: BeamSearch, GreedySearch
#Mo hinh: Dung transformer
#Preprocess: Dung sentencepiece de tao vocab
#Ablation study: RMSnorm, RoPE
#Preprocess: Unigram; Vocab-size: 32000
#Training phase: 2xT4 GPU + Thoi gian train/epoch
class Manager():    
    def __init__(self, is_train=True, ckpt_name=None):
        
        # 1. INITIALIZE ACCELERATOR
        # This automatically detects if you have 1 GPU, 4 GPUs, or TPUs.
        # It replaces "device = torch.device('cuda')"
        self.accelerator = Accelerator(
            mixed_precision="fp16",
            log_with="wandb"  # <--- NEW
        )

        self.max_len = max_len
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

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),  # Adjusted for NMT stability
            eps=1e-9,  # Prevents division by zero errors
            weight_decay=0.01  # Regularization (Prevents overfitting)
        )
        self.best_loss = sys.float_info.max

        # 3. CHECKPOINT LOADING
        if ckpt_name:
            if os.path.exists(ckpt_name):
                ckpt_path = ckpt_name
            else:
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)

            # --- LOADING LOGIC ---
            if os.path.exists(ckpt_path):
                # Map location 'cpu' is safest to avoid GPU OOM on load
                checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

                incompatible = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                self.optim.load_state_dict(checkpoint['optim_state_dict'])
                self.best_loss = checkpoint.get('loss', sys.float_info.max)

                if self.accelerator.is_main_process:
                    print(f"Loaded checkpoint from: {ckpt_path}")
                    print("Missing keys:", incompatible.missing_keys)
                    print("Unexpected keys:", incompatible.unexpected_keys)
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
            train_loader = get_dataloader(self.dataset_name, self.src_sp, self.trg_sp, split = 'train')
            valid_loader = get_dataloader(self.dataset_name, self.src_sp, self.trg_sp, split = 'validation')

            num_update_steps_per_epoch = len(train_loader)
            max_train_steps = int(constants.num_epochs) * num_update_steps_per_epoch

            # Create the Scheduler
            self.lr_scheduler = get_scheduler(
                name="cosine",  # "cosine" decay is generally SOTA for NMT
                optimizer=self.optim,
                num_warmup_steps=4000,  # Warmup for the first 4000 steps
                num_training_steps=max_train_steps
            )
            # 5. THE MAGIC LINE: ACCELERATOR.PREPARE
            # This wraps the model in DDP, moves it to GPU, and 
            # splits the dataloader across GPUs automatically.
            self.model, self.optim, self.train_loader, self.valid_loader, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optim, train_loader, valid_loader, self.lr_scheduler
            )

    def train(self):
        if self.accelerator.is_main_process:
            print("Training starts.")

        my_num_epochs = constants.num_epochs

        # 1. Initialize Global Step
        global_step = 0
        VALIDATION_FREQ = eval_step  # Validate every 500 steps

        for epoch in range(1, my_num_epochs + 1):
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
                self.lr_scheduler.step()
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

    def inference(self, input_sentence, method='beam', verbose=False):
        self.model.eval()
        my_device = next(self.model.parameters()).device

        input_ids = self.src_sp.EncodeAsIds(input_sentence)
        max_length = getattr(self, 'max_len', 256)
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]

        src_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(my_device)

        e_mask = (src_tensor != pad_id).unsqueeze(1).unsqueeze(2)

        with torch.no_grad():
            src_emb = self.model.src_embedding(src_tensor) * math.sqrt(d_model)

            # Apply PE only if model has it (USE_ROPE=False)
            if hasattr(self.model, "positional_encoding") and self.model.positional_encoding is not None:
                src_emb = self.model.positional_encoding(src_emb)

            e_output = self.model.encoder(src_emb, e_mask)

            if method == 'greedy':
                result = self.greedy_search(e_output, e_mask)
            elif method == 'beam':
                result = self.beam_search(e_output, e_mask, self.accelerator)

        return result


    # Add this inside your Manager class
    def evaluate_bleu(self, test_loader, beam_size=beam_size):
        print(f"Starting BLEU evaluation with Beam Size {beam_size}...")
        self.model.eval()

        predictions = []
        references = []

        # 1. Disable Gradients to save memory
        with torch.no_grad():
            # We iterate one by one (or batch if beam search supports it)
            # Since beam search is single-sample, we loop.
            for i, batch in tqdm(enumerate(test_loader), desc="Translating", total=len(test_loader)):

                # Unpack - we only need source for inference
                src_padded, tgt_in_padded, tgt_out_padded = batch
                # We iterate through the batch (because beam_search handles 1 item at a time)
                # Note: This is slow but simple. Optimized beam search does batching.
                for j in range(src_padded.size(0)):

                    src_ids = src_padded[j].tolist()
                    # Filter out pad_id
                    src_ids = [x for x in src_ids if x != pad_id and x != eos_id]


                    src_text = self.src_sp.DecodeIds(src_ids)

                    ref_ids = tgt_out_padded[j].tolist()
                    ref_ids = [x for x in ref_ids if x != -100 and x != pad_id and x != eos_id]
                    ref_text = self.trg_sp.DecodeIds(ref_ids)

            #hehe

                    # Note: We need to ensure inference() returns the STRING, not print it.
                    pred_text = self.inference(src_text, method='beam', verbose=False)
                    predictions.append(pred_text)
                    #a
                    references.append(ref_text)

                    if i < 3 and j == 0:
                        if self.accelerator.is_main_process:
                            print(f"\nSrc: {src_text}")
                            print(f"Ref: {ref_text}")
                            print(f"Pred: {pred_text}")


        # 3. Calculate BLEU
        # SacreBLEU expects references as a list of lists (for multiple refs per sentence)
        bleu = sacrebleu.corpus_bleu(predictions, [references])

        print(f"\n---------------------------------")
        print(f"BLEU Score: {bleu.score}")
        print(f"---------------------------------")

        return bleu.score
        
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

    def beam_search(self, e_output, e_mask, accelerator=None):
        # 1. Setup & Constants
        if accelerator:
            my_device = next(self.model.parameters()).device
            model_engine = accelerator.unwrap_model(self.model)
        else:
            my_device = e_output.device
            model_engine = self.model

        # beam_size = 4  # Or self.beam_size
        max_len = 100
        batch_size = e_output.size(0)  # Should be 1 for inference

        # 2. Prepare Encoder Output for Beam Batching
        # We must expand the encoder output to match beam size
        # Shape: (Beam_Size, Src_Len, Dim)
        e_output = e_output.repeat(beam_size, 1, 1)
        e_mask = e_mask.repeat(beam_size, 1, 1, 1)

        # 3. Initialize Loop Variables
        # Current indices: [Beam_Size, 1] -> start with BOS
        cur_seq = torch.full((beam_size, 1), bos_id, dtype=torch.long, device=my_device)

        # Scores: [Beam_Size] -> First beam is 0, others are -inf to force selection of first beam initially
        cur_scores = torch.zeros(beam_size, device=my_device)
        cur_scores[1:] = -1e9

        # Keep track of finished sequences
        finished_seqs = []
        finished_scores = []

        # 4. Vectorized Loop
        for pos in range(max_len):

            # --- PREPARE INPUT ---
            # No loop here! We feed [Beam_Size, Seq_Len] directly.
            trg_input = cur_seq

            # Standard mask creation (Vectorized)
            trg_len = trg_input.size(1)
            d_pad_mask = (trg_input != self.pad_id).unsqueeze(1).unsqueeze(2)
            nopeak_mask = torch.tril(torch.ones((trg_len, trg_len), device=my_device)).bool()
            d_mask = d_pad_mask & nopeak_mask.unsqueeze(0)

            # --- FORWARD PASS (Run 4 beams at once) ---
            # Note: If you implement KV Caching, this changes significantly.
            # For now, we assume standard full-forward pass.
            with torch.amp.autocast('cuda', enabled=True):  # Enable FP16 for speed
                trg_emb = model_engine.trg_embedding(trg_input)

                trg_emb = trg_emb * math.sqrt(d_model)
                # trg_emb = model_engine.positional_encoding(trg_emb)
                decoder_output = model_engine.decoder(trg_emb, e_output, e_mask, d_mask)
                logits = model_engine.output_linear(decoder_output[:, -1, :])
                log_probs = torch.log_softmax(logits, dim=-1)  # (Beam, Vocab)

            # --- CALCULATE SCORES ---
            # Add current scores to log_probs
            # Shape: (Beam, Vocab)
            next_scores = cur_scores.unsqueeze(1) + log_probs

            # --- RANKING & SELECTION ---
            # We flatten the matrix to find the top K the best tokens across ALL beams
            # Shape: (Beam * Vocab)
            next_scores_flat = next_scores.view(-1)

            if pos == 0:
                # On the first step, we only look at the first beam (since all others are duplicates)
                # otherwise we get the same 4 words.
                next_scores_flat = next_scores[0]  # (Vocab)

            # Get top K best scores and their indices
            topk_scores, topk_indices = torch.topk(next_scores_flat, beam_size, dim=0)

            # Convert flat indices back to (Beam_Index, Word_Index)
            vocab_size = log_probs.size(-1)
            beam_indices = topk_indices.div(vocab_size, rounding_mode='floor')  # Which beam did it come from?
            word_indices = topk_indices % vocab_size  # Which word is it?

            # --- BUILD NEXT STEP ---
            new_seqs = []
            new_scores = []

            num_active = 0

            for i in range(beam_size):
                b_idx = beam_indices[i]  # Index of the parent beam
                w_idx = word_indices[i]  # The new word
                score = topk_scores[i]

                # Reconstruct the sequence: Parent Sequence + New Word
                # Note: We must clone to avoid reference issues
                seq = torch.cat([cur_seq[b_idx], w_idx.unsqueeze(0)])

                if w_idx.item() == eos_id:
                    # Finished!
                    # Length Penalty
                    penalty_score = score / len(seq)
                    finished_seqs.append(seq)
                    finished_scores.append(penalty_score)
                else:
                    new_seqs.append(seq)
                    new_scores.append(score)
                    num_active += 1

            # Check if we have enough finished sequences
            if len(finished_seqs) >= beam_size:
                break

            # Pad new_seqs if we lost some beams to EOS (to keep batch size constant)
            # This keeps the tensor shape valid for the next model run
            while len(new_seqs) < beam_size:
                new_seqs.append(new_seqs[0])  # Duplicate the best one (it won't matter, it effectively forks)
                new_scores.append(torch.tensor(-1e9, device = my_device))  # Give it bad score so it dies next round

            # Stack back into tensors
            cur_seq = torch.stack(new_seqs)  # (Beam, Len)
            cur_scores = torch.stack(new_scores)  # (Beam)

        # 5. Final Selection
        if len(finished_seqs) == 0:
            best_seq = cur_seq[0]
        else:
            # Sort by best score (Higher is better for LogProbs, but check your sign!)
            # Since log_probs are negative, closer to 0 is better.
            # Using sorted(reverse=True) because -0.5 > -10.0
            sorted_finished = sorted(zip(finished_seqs, finished_scores), key=lambda x: x[1], reverse=True)
            best_seq = sorted_finished[0][0]

        # Clean up (remove BOS/EOS)
        best_seq = best_seq.tolist()
        if best_seq[0] == bos_id: best_seq = best_seq[1:]
        if best_seq[-1] == eos_id: best_seq = best_seq[:-1]

        return self.trg_sp.DecodeIds(best_seq)

    def create_mask(self, src, tgt):
        # src: (Batch, Src_Len)
        # tgt: (Batch, Tgt_Len)

        # 1. Source Padding Mask
        # Use src.device as the anchor!
        src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2).to(src.device)

        # 2. Target Padding Mask
        tgt_pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2).to(tgt.device)

        # 3. Target No-Peak (Causal) Mask
        tgt_len = tgt.size(1)
        # Create on the same device as tgt
        nopeak_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()

        # 4. Combine
        tgt_mask = tgt_pad_mask & nopeak_mask.unsqueeze(0)

        return src_mask, tgt_mask


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, help="train or inference?")
    parser.add_argument('--ckpt_name', required=False, help="best checkpoint file")
    parser.add_argument('--input', type=str, required=False, help="input sentence when inferencing")
    parser.add_argument('--decode', type=str, required=True, default="greedy", help="greedy or beam?")
    parser.add_argument('--dataset_name', type=str, required=False, help="path to config file")
    parser.add_argument('--num_epochs', type=int, default=constants.num_epochs, required=False, help="path to config file")

    # ADD THIS BACK
    parser.add_argument('--use_rope', type=str2bool, default=constants.USE_ROPE, required=False, help="use RoPE (true/false)")

    args = parser.parse_args()

    # APPLY FLAG
    constants.USE_ROPE = args.use_rope
    print(f"USE_ROPE = {constants.USE_ROPE}")

    constants.num_epochs = args.num_epochs
    print(f"Number of train epoch: {constants.num_epochs}")

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
    elif args.mode == 'evaluate':
        # Load the best checkpoint
        assert args.ckpt_name is not None, "Provide a checkpoint!"
        manager = Manager(is_train=False, ckpt_name=args.ckpt_name)

        # We need a loader. Let's use validation or a dedicated test set
        test_loader = get_dataloader(
            dataset_name=DATASET_NAME,
            src_sp=manager.src_sp,
            trg_sp=manager.trg_sp,
            split='train[:10]',  # Or 'validation' if test doesn't exist
            workers = 0,
            my_batch_size = 1
        )

        manager.evaluate_bleu(test_loader, beam_size=beam_size)

    else:
        print("Please specify mode argument either with 'train' or 'inference'.")
