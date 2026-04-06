from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

@dataclass
class DeepFakeConfig:
    seed: int = 42
    device: str = "cuda"
    
    # Dataset
    hf_dataset_name: str = "GonzaloA/fake_news"
    max_length: int = 128
    
    # Core modeling
    model_name: str = "roberta-base"
    model_mode: str = "lstm" # Toggle: "deepfake" (Monster) or "lstm" (Weak Baseline)
    meta_dim: int = 3
    lambda_bias: float = 0.5
    
    # Training Loop
    epochs: int = 3
    batch_size: int = 4
    grad_accum_steps: int = 4
    lr: float = 2e-5
    
    # FGM Constraints
    use_fgm: bool = True
    fgm_epsilon: float = 1.0
    
    # Scaled Environment Test Limits
    train_limit: int = 2000 # Option A: 4000 samples to run the DeepFakeNewsNet within ~15 mins
    test_limit: int = 1000
    
    out_dir: str = "./outputs/unified_run"
