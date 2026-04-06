import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train import run_pipeline

if __name__ == "__main__":
    '''
    DeepFakeNewsNet Single-Command Initialization Vector.
    Executes: Multi-Modal Linguistic Fusion + GRL Debiasing + FGM Stress Testing
    '''
    run_pipeline()
