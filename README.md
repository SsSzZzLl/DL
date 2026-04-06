# Fake News Detection v2.0 (Group Project Final)

Welcome to the isolated group project repository! This directory is strictly decoupled to ensure 4 members can jointly run their scripts locally without mutual interference.

## ⚙️ Shared Core Matrix
All dataloaders and hyperparameters lie in the `shared_core/` directory.

## 👨‍💻 Member 1: Multi-Modal Fusion (Linguistic Feature Extraction)
- **Role**: Breaks the limits of deep learning purely dealing with semantics by extracting human-readable morphological attributes (Uppercase density, exclamation ratio) and MLP-fusing them into the RoBERTa `pooler_output`.
- **Execution Script**: `python member1_fusion/run_member1.py`

## 👨‍💻 Member 2: Discriminative Causal Debiasing
- **Role**: Addresses Shortcut-Learning defects structurally. Contains a PyTorch `GradientReversalLayer` (GRL) engineered to heavily penalize over-reliance on trivial stylistic triggers (like 'Reuters').
- **Execution Script**: `python member2_debias/run_member2.py`

## 👨‍💻 Member 3: Adversarial Robustness and Benchmarking
- **Role**: Subjects networks to continuous stress tests using Typo attacks, embedding mathematical regularizers (Fast Gradient Method) to minimize the Multi-Baseline Generalization Gap.
- **Execution Script**: `python member3_robustness/run_member3.py`

## 👨‍💻 Member 4: Streamlit XAI End-User Application
- **Role**: Elevates the raw backend matrices into a living interactive app. Allows real-time analysis using `Captum` and `Streamlit`.
- **Execution Script**: `streamlit run member4_webapp/run_member4_app.py`
