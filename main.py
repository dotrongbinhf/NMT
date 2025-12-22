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
            train_loader = get_dataloader(self.dataset_name, self.src_sp, self.trg_sp, split = 'train[:500000]')
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

                    # if i < 3 and j == 0:
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
    
    def length_penalty(self, length, alpha=0.6):
        return ((5 + length) ** alpha) / ((5 + 1) ** alpha)


    def has_repeat_ngram(self, seq, n):
        if len(seq) < n * 2:
            return False
        seen = set()
        for i in range(len(seq) - n + 1):
            ng = tuple(seq[i:i+n])
            if ng in seen:
                return True
            seen.add(ng)
        return False


    @torch.no_grad()
    def beam_search(self, e_output, e_mask, beam_size=4):
        device = e_output.device
        model = self.model
        model.eval()

        max_len = 100
        alpha = 0.6
        repetition_penalty = 1.2
        no_repeat_ngram_size = 3

        bos_id = self.bos_id
        eos_id = self.eos_id
        pad_id = self.pad_id
        d_model = model.d_model

        # ===== FIX 1: repeat encoder output ĐÚNG CHUẨN =====
        # e_output: (1, src_len, d_model)
        e_output = e_output.repeat(beam_size, 1, 1)
        # e_mask: (1, 1, 1, src_len)
        e_mask = e_mask.repeat(beam_size, 1, 1, 1)

        # ===== Beam states =====
        sequences = torch.full(
            (beam_size, 1),
            bos_id,
            dtype=torch.long,
            device=device
        )

        scores = torch.zeros(beam_size, device=device)
        scores[1:] = -1e9  # chỉ beam đầu tiên active ở step 0

        finished = []

        for step in range(max_len):

            trg_len = sequences.size(1)

            # ===== FIX 2: decoder mask chuẩn Transformer =====
            pad_mask = (sequences != pad_id).unsqueeze(1).unsqueeze(2)
            causal_mask = torch.tril(
                torch.ones((trg_len, trg_len), device=device)
            ).bool()
            d_mask = pad_mask & causal_mask.unsqueeze(0)

            # ===== Decoder forward =====
            trg_emb = model.trg_embedding(sequences) * math.sqrt(d_model)
            if model.positional_encoding is not None:
                trg_emb = model.positional_encoding(trg_emb)

            dec_out = model.decoder(trg_emb, e_output, e_mask, d_mask)
            logits = model.output_linear(dec_out[:, -1])
            log_probs = torch.log_softmax(logits, dim=-1)

            # ===== FIX 3: repetition penalty ĐÚNG =====
            for i in range(beam_size):
                for tok in set(sequences[i].tolist()):
                    log_probs[i, tok] /= repetition_penalty

            # ===== Beam expansion =====
            total_scores = scores.unsqueeze(1) + log_probs

            if step == 0:
                total_scores = total_scores[0:1]  # chỉ mở rộng beam đầu

            flat_scores = total_scores.view(-1)
            topk_scores, topk_ids = torch.topk(flat_scores, beam_size)

            vocab_size = log_probs.size(-1)
            beam_ids = topk_ids // vocab_size
            token_ids = topk_ids % vocab_size

            new_sequences = []
            new_scores = []

            for i in range(beam_size):
                parent = beam_ids[i]
                token = token_ids[i]
                score = topk_scores[i]

                seq = torch.cat([sequences[parent], token.view(1)])

                # ===== FIX 4: EOS xử lý chuẩn =====
                if token.item() == eos_id:
                    lp = self.length_penalty(len(seq), alpha)
                    finished.append((seq, score / lp))
                else:
                    if self.has_repeat_ngram(seq.tolist(), no_repeat_ngram_size):
                        continue
                    new_sequences.append(seq)
                    new_scores.append(score)

            # ===== Early stopping =====
            if len(finished) >= beam_size:
                break

            if len(new_sequences) == 0:
                break

            # ===== FIX 5: pad beam an toàn =====
            while len(new_sequences) < beam_size:
                new_sequences.append(new_sequences[0])
                new_scores.append(new_scores[0])

            sequences = torch.stack(new_sequences)
            scores = torch.stack(new_scores)

        # ===== FIX 6: chọn beam tốt nhất =====
        if len(finished) == 0:
            best = sequences[0]
        else:
            best = max(finished, key=lambda x: x[1])[0]

        output = best.tolist()
        if output and output[0] == bos_id:
            output = output[1:]
        if eos_id in output:
            output = output[:output.index(eos_id)]

        return self.trg_sp.DecodeIds(output)

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
            split='test[:20]',  # Or 'validation' if test doesn't exist
            workers = 0,
            my_batch_size = 1
        )

        manager.evaluate_bleu(test_loader, beam_size=beam_size)

    else:
        print("Please specify mode argument either with 'train' or 'inference'.")
