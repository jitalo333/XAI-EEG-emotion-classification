import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if x.ndim == 4:
            x = x.squeeze(1)
        return self.model(x)

def save_plot(save_path, filename):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, filename))
    plt.close()

def heatmap_plot(args, heatmap_data, save_path, pred_class, bands_labels):
    plt.figure(figsize=(10, 11))
    sns.heatmap(
        heatmap_data,
        xticklabels=bands_labels,
        yticklabels=args['channels'],
        cmap='turbo',
        cbar_kws={'label': 'Attribution (SmoothGrad)'},
        vmin=0, vmax=1
    )
    plt.title(f"SmoothGrad - {args['model_type']}\nSubject {args['subject_id']} | Sample {args['sample_idx']} | Class {pred_class}")
    plt.xlabel("Frequency Bands")
    plt.ylabel("EEG Channels")
    plt.tight_layout()
    save_plot(save_path, f"smoothgrad_heatmap_sub{args['subject_id']}_idx{args['sample_idx']}.png")

def channel_importance_plot(args, heatmap_data, save_path, pred_class):
    mean_importance = heatmap_data.mean(axis=1)
    channels = args['channels']
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(channels)), mean_importance, color='steelblue')
    plt.xticks(range(len(channels)), channels, rotation=45, ha='right')
    plt.ylabel("Mean Importance")
    plt.title(f"Channel Importance (SmoothGrad)\nSubject {args['subject_id']} | Class {pred_class}")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_plot(save_path, f"smoothgrad_channels_sub{args['subject_id']}_idx{args['sample_idx']}.png")

def band_importance_plot(args, heatmap_data, save_path, pred_class, bands_labels):
    mean_importance = heatmap_data.mean(axis=0)
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(bands_labels)), mean_importance, color='coral', edgecolor='black', linewidth=1.5)
    plt.xticks(range(len(bands_labels)), bands_labels)
    plt.ylabel("Mean Importance")
    plt.title(f"Band Importance (SmoothGrad)\nSubject {args['subject_id']} | Class {pred_class}")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_plot(save_path, f"smoothgrad_bands_sub{args['subject_id']}_idx{args['sample_idx']}.png")