import torch
import torch.nn as nn
import random
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.config import DeepFakeConfig
from src.data import get_unified_dataloaders
from src.models import DeepFakeNewsNet

class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}
    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}

def evaluate(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            meta = batch.get("meta_features", None)
            if meta is not None:
                meta = meta.to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=ids, attention_mask=mask, meta_features=meta)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    return acc, p, r, f1

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_pipeline():
    print("="*70)
    print("🔥 BOOTSTRAPPING UNIFIED DEEPFAKENEWSNET PIPELINE (v3.0) 🔥")
    print("="*70)
    
    cfg = DeepFakeConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)
    
    train_loader, val_loader, test_loader, tokenizer = get_unified_dataloaders(cfg)
    
    if cfg.model_mode == "lstm":
        print("\n🔧 M0: Forging Hand-Rolled Baseline (CustomTextLSTM)...")
        from src.models import CustomTextLSTM
        model = CustomTextLSTM(vocab_size=tokenizer.vocab_size).to(device)
    else:
        print("\n🔨 M1~M3: Forging DeepFakeNewsNet (RoBERTa + MLP Fusion + GRL)...")
        model = DeepFakeNewsNet(cfg.model_name, cfg.meta_dim).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_task = nn.CrossEntropyLoss()
    loss_bias = nn.CrossEntropyLoss()
    
    fgm = FGM(model) if cfg.use_fgm and cfg.model_mode != "lstm" else None
    
    print("\n🚀 IGNITING MULTI-LOSS TENSOR ENGINE WITH FGM LOCK")
    for epoch in range(cfg.epochs):
        model.train()
        total_loss, total_bias = 0, 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for i, batch in enumerate(pbar):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            meta = batch["meta_features"].to(device)
            labels = batch["labels"].to(device)
            bias_labels = batch["bias_labels"].to(device)
            
            # --- 1. Main Forward Pass (Fusion + GRL) ---
            outputs = model(input_ids=ids, attention_mask=mask, meta_features=meta)
            
            l_main = loss_task(outputs.logits, labels)
            l_bias = loss_bias(outputs.bias_logits, bias_labels)
            loss = l_main + cfg.lambda_bias * l_bias
            loss.backward()
            
            # --- 2. Adversarial Defense Pass (FGM) ---
            if fgm is not None:
                fgm.attack(epsilon=cfg.fgm_epsilon)
                outputs_adv = model(input_ids=ids, attention_mask=mask, meta_features=meta)
                loss_adv = loss_task(outputs_adv.logits, labels) + cfg.lambda_bias * loss_bias(outputs_adv.bias_logits, bias_labels)
                loss_adv.backward()
                fgm.restore()
            
            if (i+1) % cfg.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            total_loss += l_main.item()
            total_bias += l_bias.item()
            pbar.set_postfix({"Base Loss": f"{total_loss/(i+1):.4f}", "Bias Loss": f"{total_bias/(i+1):.4f}"})
            
    print("\n🔬 COMMENCING EVALUATION PROTOCOL ON TEST SET...")
    acc, p, r, f1 = evaluate(model, test_loader, device)
    
    print("="*50)
    print("📊 FINAL MISSION REPORT")
    print(f"Architecture: {cfg.model_mode}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"Macro F1:  {f1:.4f}")
    print("="*50)
    
    import json
    metrics = {"Accuracy": acc, "Precision": p, "Recall": r, "Macro F1": f1}
    with open(os.path.join(cfg.out_dir, f"metrics_{cfg.model_mode}.json"), "w") as f:
        json.dump(metrics, f)
        
    weight_path = os.path.join(cfg.out_dir, f"model_{cfg.model_mode}.pth")
    torch.save(model.state_dict(), weight_path)
    print(f"\n🏁 UNIFIED TRAINING EXECUTION COMPLETE. Weights sealed in {weight_path}")

if __name__ == "__main__":
    run_pipeline()
