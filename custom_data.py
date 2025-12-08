from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from constants import *

import torch
import sentencepiece as spm
import numpy as np
from datasets import load_dataset , Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

def get_dataloader(dataset_name, src_sp, trg_sp, batch_size = batch_size, split = 'train[:500000'):
    print(f"Loading {dataset_name}...")
    dataset = load_dataset(dataset_name, split=split)

    # --- TOKENIZER FUNCTION ---
    def tokenize_pair(examples):
        src_batch = []
        tgt_input_batch = []
        tgt_label_batch = []

        # Handle Hugging Face structure
        if 'translation' in examples:
            iter_data = examples['translation']
            src_texts = [x['en'] for x in iter_data]
            tgt_texts = [x['vi'] for x in iter_data]
        else:
            src_texts = examples['en']
            tgt_texts = examples['vi']

        max_len = seq_len
        # Tokenize Source
        for text in src_texts:
            # Source: [IDs] + [EOS]
            encoded = src_sp.EncodeAsIds(text)
            if len(encoded) > max_len - 1:  # -1 for EOS
                encoded = encoded[:max_len - 1]
            src_batch.append(encoded + [eos_id])

        # Tokenize Target
        for text in tgt_texts:
            encoded = trg_sp.EncodeAsIds(text)
            # Input: [BOS] + [IDs]
            tgt_input_batch.append([bos_id] + encoded)
            # Label: [IDs] + [EOS]
            if len(encoded) > max_len - 2:  # -2 for BOS/EOS
                encoded = encoded[:max_len - 2]
            tgt_label_batch.append(encoded + [eos_id])

        return {
            "src_ids": src_batch,
            "tgt_input_ids": tgt_input_batch,
            "tgt_labels": tgt_label_batch
        }

    # --- MAP & FORMAT ---
    print("Tokenizing...")
    tokenized_dataset = dataset.map(
        tokenize_pair, 
        batched=True, 
        batch_size=1000, 
        remove_columns=dataset.column_names 
    )
    tokenized_dataset.set_format(type=None, columns=['src_ids', 'tgt_input_ids', 'tgt_labels'])

    # --- LOADER ---
    dataloader = DataLoader(
        tokenized_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers = num_workers,
        pin_memory=True
    )

    return dataloader


def collate_fn(batch):
    src_batch = [b['src_ids'] for b in batch]
    tgt_in_batch = [b['tgt_input_ids'] for b in batch]
    tgt_out_batch = [b['tgt_labels'] for b in batch]

    # Just Pad. Don't touch masks here.
    # Note: Use your constants! pad_id=3
    src_padded = pad_sequence([torch.tensor(x) for x in src_batch], batch_first=True, padding_value=3)
    tgt_in_padded = pad_sequence([torch.tensor(x) for x in tgt_in_batch], batch_first=True, padding_value=3)
    tgt_out_padded = pad_sequence([torch.tensor(x) for x in tgt_out_batch], batch_first=True, padding_value=-100)

    return src_padded, tgt_in_padded, tgt_out_padded