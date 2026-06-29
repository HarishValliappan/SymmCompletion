import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import glob
import os

# Collect all validation epoch metrics from .err files
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
err_files = sorted(glob.glob(os.path.join(log_dir, 'train_pcn_*.err')))

pattern = re.compile(r"\[Validation\] EPOCH:\s*(\d+)\s+Metrics\s*=\s*\['([^']+)',\s*'([^']+)',\s*'([^']+)'\]")

epochs = []
f1_scores = []
cdl1_scores = []
cdl2_scores = []

for fpath in err_files:
    with open(fpath, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epoch = int(m.group(1))
                f1 = float(m.group(2))
                cdl1 = float(m.group(3))
                cdl2 = float(m.group(4))
                epochs.append(epoch)
                f1_scores.append(f1)
                cdl1_scores.append(cdl1)
                cdl2_scores.append(cdl2)

# Sort by epoch
sorted_data = sorted(zip(epochs, f1_scores, cdl1_scores, cdl2_scores))
epochs = [d[0] for d in sorted_data]
f1_scores = [d[1] for d in sorted_data]
cdl1_scores = [d[2] for d in sorted_data]
cdl2_scores = [d[3] for d in sorted_data]

print(f"Found {len(epochs)} validation data points:")
for e, f1, cd1, cd2 in zip(epochs, f1_scores, cdl1_scores, cdl2_scores):
    print(f"  Epoch {e:3d}: F1={f1:.4f}, CDL1={cd1:.4f}, CDL2={cd2:.4f}")

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# F1-Score
ax1.plot(epochs, f1_scores, 'b-o', markersize=4, linewidth=1.5, label='F1-Score')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('F1-Score', fontsize=12)
ax1.set_title('F1-Score vs Epoch', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# CDL1
ax2.plot(epochs, cdl1_scores, 'r-s', markersize=4, linewidth=1.5, label='CDL1')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('CDL1', fontsize=12)
ax2.set_title('CDL1 (Chamfer Distance L1) vs Epoch', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

# CDL2
ax3.plot(epochs, cdl2_scores, 'g-^', markersize=4, linewidth=1.5, label='CDL2')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('CDL2', fontsize=12)
ax3.set_title('CDL2 (Chamfer Distance L2) vs Epoch', fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=11)

plt.suptitle('SymmCompletion Training Metrics (PCN Dataset)', fontsize=16, y=1.02)
plt.tight_layout()
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_metrics.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {save_path}")
