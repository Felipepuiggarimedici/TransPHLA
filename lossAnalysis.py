import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch

# --- Consistent font sizes across all figures ---
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# --- Seeds for reproducibility ---
seed = 6
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# --- Colours ---
colours = ['#9BC995', "#083D77", '#9A031E', '#C4B7CB', '#FC7753']

# --- Paths ---
folder = "model/new"
number = ""
output_dir = os.path.join(folder, number)
losses_path = os.path.join(output_dir, 'avg_losses.npy')
losses = np.load(losses_path)

# --- Remove trailing zeros if early stopping ended training ---
losses = losses[losses > 0]

# --- Early stopping parameters ---
patienceFull = 10
last_epoch = len(losses)
early_stop_start = last_epoch - patienceFull

# --- Find minimum loss ---
min_epoch = np.argmin(losses) + 1  
min_loss = losses[min_epoch - 1]

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 5))  # wide style for training curves

ax.plot(
    range(1, len(losses) + 1),
    losses,
    marker='o',
    markersize=4,
    linewidth=2,
    color=colours[1],
    label='Training Loss'
)

# Shade the last 'patienceFull' epochs
ax.axvspan(early_stop_start + 1, last_epoch, color=colours[4], alpha=0.2, label='Early Stopping Window')

# Vertical line where early stopping triggered
ax.axvline(x=last_epoch, color=colours[4], linestyle='--', linewidth=1.5, label='Training Stopped')

# Highlight minimum loss
ax.scatter(min_epoch, min_loss, color=colours[0], s=120, zorder=5,
           label=f'Min Loss (Epoch {min_epoch}, Loss: {min_loss:.4f})')

ax.set_title("Cross Entropy Loss per Epoch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross Entropy Loss")
ax.legend(loc='upper right')

# --- Save + show ---
plot_path = os.path.join(output_dir, "loss_curve.png")
plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.03)
plt.show()

print(f"Loss curve saved to: {plot_path}")