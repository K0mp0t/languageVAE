import os
import glob
from queue import Queue

import pyarrow.parquet as pq

import torch
from torch.utils.data import Dataset

from tokenizers import Tokenizer


class DDPStreamingParquetDataLoader(object):
    def __init__(self, tokenizer, data_path, seq_len, mask_ratio=0.0, tail_padding_ratio=0.0, batch_size=32, split='train', 
        buffer_size=16384, tokenizer_batch_size=512, ddp_rank=0, ddp_world_size=1, resume_from_step=0, max_tail_padding_len=0):

        assert split in ("train", "validation", "all")
        self.split = split

        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        self.tail_padding_ratio = tail_padding_ratio
        self.max_tail_padding_len = max_tail_padding_len
        self.batch_size = batch_size

        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size

        self.resume_from_step = resume_from_step

        self.tokenizer = tokenizer
        self.tokenizer_batch_size = tokenizer_batch_size
        self.sep_token_id = None

        for sep_token in ['[SEP]', '<|endoftext|>']:
            sep_token_id = self.tokenizer.token_to_id(sep_token)
            if sep_token_id is not None:
                self.sep_token_id = sep_token_id
                break

        self.pad_token_id = None

        for pad_token in ['<|pad|>', '[PAD]']:
            pad_token_id = self.tokenizer.token_to_id(pad_token)
            if pad_token_id is not None:
                self.pad_token_id = pad_token_id
                break

        self.mask_token_id = None

        for mask_token in ['<|mask|>', '[MASK]']:
            mask_token_id = self.tokenizer.token_to_id(mask_token)
            if mask_token_id is not None:
                self.mask_token_id = mask_token_id
                break

        assert self.sep_token_id is not None
        assert self.pad_token_id is not None
        assert self.mask_token_id is not None

        data_filepaths = glob.glob(os.path.join(data_path, '*.parquet'))
        if split == 'train':
            data_filepaths = data_filepaths[:-1]
        elif split == 'validation':
            data_filepaths = data_filepaths[-1:]
        self.data_files = [pq.ParquetFile(fp, memory_map=True) for fp in data_filepaths]

        self.pf_idx = 0
        self.rg_idx = 0

        self.buffer = Queue(maxsize=buffer_size)

        if self.resume_from_step > 0:
            self.put_rg_into_buffer(0, 0)
            approx_rg_size = self.buffer.qsize()
            self.buffer.empty()

            current_idx = 0
            if self.ddp_rank == 0:
                print(f'Skipping pfs and rgs, target idx: {self.resume_from_step}, {approx_rg_size}')

            for pf_idx in range(len(self.data_files)):
                # pf_size = self.data_files[pf_idx].scan_contents() // self.ddp_world_size
                pf_size = self.data_files[pf_idx].num_row_groups * approx_rg_size

                if self.resume_from_step - pf_size >= current_idx:
                    current_idx += pf_size
                    if self.ddp_rank == 0:
                        print(f'Skipping file #{pf_idx}, {current_idx}')
                    continue

                self.pf_idx = pf_idx

                self.rg_idx = (self.resume_from_step - current_idx) // approx_rg_size
                break

            if self.ddp_rank == 0:
                print(f'Resuming from file #{self.pf_idx} row group #{self.rg_idx}')

        
        if buffer_size // batch_size < 8:
            print(f'WARNING! buffer of size {buffer_size} is too small for batch size of {batch_size}')


        # prefetch one rg on init so if something is wrong it will crash here
        self.put_rg_into_buffer(self.pf_idx, self.rg_idx)
        self.rg_idx += 1

        if self.rg_idx >= self.data_files[self.pf_idx].num_row_groups:
            self.rg_idx = 0
            self.pf_idx += 1

        if self.pf_idx >= len(self.data_files):
            self.pf_idx = 0


    def put_rg_into_buffer(self, pf_idx, rg_idx):
        rg_texts = self.data_files[pf_idx].read_row_group(rg_idx, columns=['text'])
        
        if rg_texts.num_rows // self.ddp_world_size < self.batch_size and self.ddp_rank == 0:
            print(f'WARNING! group_rows={rg_texts.num_rows} is too small for world_size={self.ddp_world_size}')
            
        rg_texts_indices = list(range(self.ddp_rank, rg_texts.num_rows, self.ddp_world_size))
        rg_texts = rg_texts.take(rg_texts_indices).column(0).to_pylist()
        
        rg_texts_tokenized = [self.tokenizer.encode_batch(rg_texts[i: i+self.tokenizer_batch_size]) for i in range(0, len(rg_texts), self.tokenizer_batch_size)]
        
        rg_texts_tokenized = [ee.ids + [self.sep_token_id] for e in rg_texts_tokenized for ee in e]
        rg_texts_tokenized = [ee for e in rg_texts_tokenized for ee in e]

        rg_texts_tokenized = [rg_texts_tokenized[i: i+self.seq_len] for i in range(0, len(rg_texts_tokenized), self.seq_len)]
        rg_texts_tokenized[-1] = rg_texts_tokenized[-1] + [self.pad_token_id] * (self.seq_len - len(rg_texts_tokenized[-1]))

        for b in rg_texts_tokenized:
            self.buffer.put(b)
    

    def __iter__(self):
        self.pf_idx = 0
        self.rg_idx = 0

        return self


    def __next__(self):
        batch = list()

        if self.buffer.qsize() < self.batch_size:
            self.put_rg_into_buffer(self.pf_idx, self.rg_idx)

            self.rg_idx += 1

            if self.rg_idx >= self.data_files[self.pf_idx].num_row_groups:
                self.rg_idx = 0
                self.pf_idx += 1

            if self.pf_idx >= len(self.data_files):
                self.pf_idx = 0
        
        for _ in range(self.batch_size):
            batch.append(self.buffer.get())

        batch = torch.as_tensor(batch, dtype=torch.long)
        masked_batch = batch.clone()
        src_key_padding_mask = batch == self.pad_token_id

        if self.mask_ratio > 0:
            if self.mask_token_id is not None:
                mask = torch.rand(*masked_batch.shape) < self.mask_ratio
                # prevent masking padding tokens
                mask = torch.logical_and(mask, torch.logical_not(src_key_padding_mask))
                masked_batch[mask] = self.mask_token_id

            else:
                print(f'[WARNING] dataset mask ratio is set to a positive number, but mask_token_id is not set')

        src_key_padding_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(1).expand(-1, -1, src_key_padding_mask.shape[-1], -1)

        return masked_batch, src_key_padding_mask, batch
