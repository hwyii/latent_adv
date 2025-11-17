import torch
from transformers import AutoTokenizer

class Collator:
    def __init__(self, model_name='EleutherAI/pythia-410m', max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def __call__(self, batch):
        texts = [item['text'] for item in batch]
        labels = [item['label'] for item in batch]

        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        encodings['labels'] = torch.tensor(labels, dtype=torch.long)
        return encodings
    
# from enron_spam_hf import AdvDataset
# ds = AdvDataset("train")
# print(ds[0])
