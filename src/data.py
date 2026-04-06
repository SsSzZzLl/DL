import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class UnifiedFakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Linguistic Metadata (Member 1 Fusion)
        self.meta_features = self._extract_linguistic_metadata()
        # Causal Bias Targets (Member 2 GRL)
        self.bias_labels = self._extract_causal_bias_labels()
        
    def _extract_linguistic_metadata(self):
        features = []
        for text in self.texts:
            words = text.split()
            upper_ratio = sum(1 for w in words if w.isupper()) / max(len(words), 1)
            excl_count = text.count('!') / 10.0
            length_norm = len(text) / 5000.0
            features.append([upper_ratio, excl_count, length_norm])
        return torch.tensor(features, dtype=torch.float)
        
    def _extract_causal_bias_labels(self):
        # Fake news shortcut bias: Predict if 'reuters' or 'washington' is in text
        return [1 if "reuters" in t.lower() or "washington" in t.lower() else 0 for t in self.texts]
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "meta_features": self.meta_features[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "bias_labels": torch.tensor(self.bias_labels[idx], dtype=torch.long)
        }

def get_unified_dataloaders(cfg):
    print("📦 Bootstrapping Mega-Dataset with Fusion Vectors & Bias Indicators...")
    ds = load_dataset(cfg.hf_dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    
    train_ds = UnifiedFakeNewsDataset(ds["train"]["text"][:cfg.train_limit], ds["train"]["label"][:cfg.train_limit], tokenizer, cfg.max_length)
    val_ds = UnifiedFakeNewsDataset(ds["validation"]["text"][:cfg.test_limit], ds["validation"]["label"][:cfg.test_limit], tokenizer, cfg.max_length)
    test_ds = UnifiedFakeNewsDataset(ds["test"]["text"][:cfg.test_limit], ds["test"]["label"][:cfg.test_limit], tokenizer, cfg.max_length)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)
    
    return train_loader, val_loader, test_loader, tokenizer
