from torch.utils.data import Dataset
from tqdm.contrib.concurrent import process_map, thread_map
from torch.nn import functional as F
import torch
import os
import cv2
from tqdm.auto import tqdm
import numpy as np


class BLIPDataset(Dataset):
    def __init__(self, dataset_dir, seq_len, tokenizer):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.mask_token_id = tokenizer.token_to_id('[MASK]')
        self.pad_token_id = tokenizer.token_to_id('[PAD]')

        images = filter(lambda x: x.endswith('.jpg'), os.listdir(dataset_dir))
        texts = filter(lambda x: x.endswith('.txt'), os.listdir(dataset_dir))

        images = map(lambda x: x.replace('.jpg', ''), images)
        texts = map(lambda x: x.replace('.txt', ''), texts)

        self.samples = list(set(images) & set(texts))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]

        img = cv2.imread(os.path.join(self.dataset_dir, sample_name+'.jpg')).astype(np.float32) / 127.5 - 1
        img = torch.as_tensor(img).permute(2, 0, 1)
        
        with open(os.path.join(self.dataset_dir, sample_name+'.txt')) as f:
            text = f.read()

        token_ids = torch.as_tensor(self.tokenizer.encode(text, add_special_tokens=False).ids, dtype=torch.long)[:self.seq_len]
        token_ids = F.pad(token_ids, (self.pad_token_id, self.seq_len - token_ids.shape[0]))

        attention_mask = token_ids == self.pad_token_id

        return img, token_ids, attention_mask