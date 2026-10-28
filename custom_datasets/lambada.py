from torch.utils.data import Dataset
from tqdm.contrib.concurrent import process_map, thread_map
from torch.nn import functional as F
import torch

from multiprocessing.pool import ThreadPool


def slice_element_fn(tokenizer, element, seq_len):
    return torch.split(torch.as_tensor(tokenizer.encode(element['text'], add_special_tokens=False).ids, dtype=torch.long), seq_len)


class LambadaDataset(Dataset):
    def __init__(self, hf_dataset, seq_len, tokenizer, mask_ratio=0.0):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.mask_ratio = mask_ratio
        self.mask_token_id = tokenizer.token_to_id('[MASK]')
        
        self.pad_token_id = None

        pad_tokens_to_try = ['<|pad|>', '[PAD]']
        for pad_token in pad_tokens_to_try:
            pad_token_id = tokenizer.token_to_id(pad_token)
            if pad_token_id is not None:
                self.pad_token_id = pad_token_id

        if self.pad_token_id is None:
            raise ValueError('Could not estimate pad token id')

        # self.samples = [slice_element_fn(s) for s in tqdm(hf_dataset)]
        # self.samples = thread_map(lambda s: slice_element_fn(tokenizer, s, seq_len), hf_dataset, max_workers=8, chunksize=32)
        with ThreadPool(processes=8) as pool:
            self.samples = pool.map(lambda s: slice_element_fn(tokenizer, s, seq_len), hf_dataset, chunksize=32)
        self.samples = [s for ss in self.samples for s in ss]
        self.samples = [F.pad(s, (0, self.seq_len-s.shape[0]), value=self.pad_token_id) for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        masked_sample = sample.clone()
        attention_mask = sample == self.pad_token_id

        if self.mask_ratio > 0:
            if self.mask_token_id is not None:
                mask = torch.rand(*masked_sample.shape) < self.mask_ratio
                masked_sample[mask] = self.mask_token_id

            else:
                print(f'[WARNING] dataset mask ratio is set to a positive number, but mask_token_id is not set')

        return masked_sample, attention_mask, sample