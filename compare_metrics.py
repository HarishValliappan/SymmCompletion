import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load CSV data
def load_csv(path):
    epochs, f1, cdl1 = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            e = int(row['Epoch'])
            if e % 10 == 0:
                epochs.append(e)
                f1.append(float(row['F1_Score']))
                cdl1.append(float(row['CDL1']))
    return epochs, f1, cdl1

symm3d_tri_path = os.path.join(script_dir, 'validation_metrics.csv')
symm3d_path = os.path.join(script_dir, '..', '..', 'Symm3d', 'SymmCompletion', 'validation_metrics.csv')

tri_epochs, tri_f1, tri_cdl1 = load_csv(symm3d_tri_path)
s3d_epochs, s3d_f1, s3d_cdl1 = load_csv(symm3d_path)

# Plot 1: F1 Score comparison
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(s3d_epochs, s3d_f1, 'b-o', markersize=6, linewidth=2, label='Symm3d')
ax1.plot(tri_epochs, tri_f1, 'r-s', markersize=6, linewidth=2, label='Symm3dTri')
ax1.set_xlabel('Epoch', fontsize=13)
ax1.set_ylabel('F1 Score', fontsize=13)
ax1.set_title('F1 Score Comparison (Symm3d vs Symm3dTri)', fontsize=15)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(tri_epochs)
ax1.tick_params(axis='x', rotation=45)
plt.tight_layout()
save1 = os.path.join(script_dir, 'compare_f1_score.png')
fig1.savefig(save1, dpi=150, bbox_inches='tight')
print(f"Saved: {save1}")

# Plot 2: CDL1 comparison
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(s3d_epochs, s3d_cdl1, 'b-o', markersize=6, linewidth=2, label='Symm3d')
ax2.plot(tri_epochs, tri_cdl1, 'r-s', markersize=6, linewidth=2, label='Symm3dTri')
ax2.set_xlabel('Epoch', fontsize=13)
ax2.set_ylabel('CDL1', fontsize=13)
ax2.set_title('CDL1 Comparison (Symm3d vs Symm3dTri)', fontsize=15)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(tri_epochs)
ax2.tick_params(axis='x', rotation=45)
plt.tight_layout()
save2 = os.path.join(script_dir, 'compare_cdl1.png')
fig2.savefig(save2, dpi=150, bbox_inches='tight')
print(f"Saved: {save2}")
