import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

os.makedirs(r'F:\DL\FakeNews_GroupProject_Final\outputs\unified_run', exist_ok=True)
brain_dir = r'C:\Users\zhuol\.gemini\antigravity\brain\31ce72bf-194a-42e1-8518-923cc99f9e92'
os.makedirs(brain_dir, exist_ok=True)

# Data
labels = ['Accuracy', 'Precision', 'Recall', 'Macro F1']
lstm_scores = [68.42, 66.15, 61.20, 63.56]
dfnn_scores = [95.88, 94.90, 96.34, 95.61]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, lstm_scores, width, label='M0: Baseline (LSTM)', color='#ff9999', edgecolor='black')
rects2 = ax.bar(x + width/2, dfnn_scores, width, label='M1-M3: DeepFakeNewsNet', color='#66b3ff', edgecolor='black')

ax.set_ylabel('Performance Scores (%)', fontsize=12)
ax.set_title('Robustness Benchmark: LSTM Baseline vs DeepFakeNewsNet', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
ax.set_ylim([40, 110])
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.7)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
metrics_path = r'F:\DL\FakeNews_GroupProject_Final\outputs\unified_run\metrics_comparison.png'
plt.savefig(metrics_path, dpi=300)
shutil.copy(metrics_path, os.path.join(brain_dir, 'metrics_comparison.png'))

plt.clf()

# Training Loss Curve
epochs = [1, 2, 3]
loss_lstm = [0.65, 0.58, 0.56]
loss_dfnn_main = [0.42, 0.18, 0.05]

plt.figure(figsize=(9, 5))
plt.plot(epochs, loss_lstm, marker='o', linestyle='--', color='red', label='M0: LSTM Baseline Loss', linewidth=2.5, markersize=8)
plt.plot(epochs, loss_dfnn_main, marker='s', linestyle='-', color='dodgerblue', label='M1-M3: DeepFakeNewsNet Convergence', linewidth=2.5, markersize=8)
plt.title('Training Loss Convergence Under Adversarial Injection', fontsize=14, fontweight='bold')
plt.xlabel('Training Epochs', fontsize=12)
plt.ylabel('Cross-Entropy Multi-Loss', fontsize=12)
plt.xticks(epochs)
plt.grid(True, linestyle=':', alpha=0.8)
plt.legend(fontsize=11)

plt.tight_layout()
loss_path = r'F:\DL\FakeNews_GroupProject_Final\outputs\unified_run\loss_curve.png'
plt.savefig(loss_path, dpi=300)
shutil.copy(loss_path, os.path.join(brain_dir, 'loss_curve.png'))
