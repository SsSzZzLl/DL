import torch
import torch.nn as nn

class UnifiedOutput:
    def __init__(self, logits, bias_logits):
        self.logits = logits
        self.bias_logits = bias_logits

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)

class CustomTextLSTM(nn.Module):
    """The Weak Baseline: Purely Hand-Rolled Bi-LSTM without World Knowledge"""
    def __init__(self, vocab_size: int, embed_dim=128, hidden_dim=256, num_labels=2):
        super().__init__()
        # Matches the expected naming convention for FGM hooking
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=1) 
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        # We output dual logits so it seamlessly fits the multi-branch train loop without crashing!
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)
        self.bias_classifier = nn.Linear(hidden_dim * 2, 2)
        
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        x = self.word_embeddings(input_ids)
        lstm_out, (hn, cn) = self.lstm(x)
        hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        logits = self.classifier(hidden)
        # Dummy bias logits block just to fulfill the API hook
        bias_logits = self.bias_classifier(hidden)
        
        return UnifiedOutput(logits, bias_logits)

class DeepFakeNewsNet(nn.Module):
    """The Ultimate Monolithic Architecture: Fusion + GRL Debiasing"""
    def __init__(self, model_name: str, meta_dim: int, num_labels: int = 2):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.encoder.config.hidden_size
        
        # 1. Main Path (Late Semantic Fusion)
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_size + meta_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )
        
        # 2. Adversarial Causal Branch (GRL)
        self.grl = GRL(alpha=1.0)
        self.bias_classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        
    def forward(self, input_ids=None, attention_mask=None, meta_features=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]
            
        fused_vector = torch.cat([pooled, meta_features], dim=1)
        logits = self.fusion_head(fused_vector)
        
        # GRL Adversarial Branch (We reverse pure semantics, not fused metadata!)
        reversed_features = self.grl(pooled)
        bias_logits = self.bias_classifier(reversed_features)
        
        return UnifiedOutput(logits, bias_logits)
