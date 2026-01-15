import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Adjusting dimensions from (Batch, 1, Channels, Bands) to (Batch, Channels, Bands) if needed
        if x.ndim == 4:
            x = x.squeeze(1)
        return self.model(x)
    

def heatmap_plot(args, heatmap_data, save_path, pred_class, bands_labels):
    plt.figure(figsize=(10, 11))
    sns.heatmap(
        heatmap_data,
        xticklabels=bands_labels,
        yticklabels=args['channels'],
        cmap='turbo',
        cbar_kws={'label': 'Attribution'},
        vmin=0, vmax=1
    )
    plt.title(f"Saliency Map - {args['model_type']}\nSubject {args['subject_id']} | Sample {[args['sample_idx']]} | Class {pred_class}")
    plt.xlabel("Frequency Bands")
    plt.ylabel("EEG Channels")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"heatmap_sub{args['subject_id']}_sample_{args['sample_idx']}_{args['model_type']}.png"))
    plt.close()

def channel_importance_plot(args, heatmap_data, save_path, pred_class):
    model_type = args['model_type']
    channels_names = args['channels']
    subject_id = args['subject_id']
    sample_idx = args['sample_idx']
    
    mean_importance = heatmap_data.mean(axis=1)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(channels_names)), mean_importance, color='steelblue')
    plt.xticks(range(len(channels_names)), channels_names, rotation=45, ha='right')
    plt.ylabel("Mean Importance")
    plt.xlabel("EEG Channels")
    plt.title(f"Importance by Channel - {model_type}\nSubject {subject_id} | Sample {[sample_idx]} | Class {pred_class}")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"channels_sub{subject_id}_sample{sample_idx}_{model_type}.png"))
    plt.close()
    
def band_importance_plot(args, heatmap_data, save_path, pred_class, bands_labels):
    model_type = args['model_type']
    subject_id = args['subject_id']
    sample_idx = args['sample_idx']
    
    mean_band_importance = heatmap_data.mean(axis=0)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(bands_labels)), mean_band_importance, color='coral', edgecolor='black', linewidth=2)
    plt.xticks(range(len(bands_labels)), bands_labels)
    plt.ylabel("Mean Importance")
    plt.xlabel("Frequency Bands")
    plt.title(f"Importance by Band - {model_type}\nSubject {subject_id} | Sample {[sample_idx]} | Class {pred_class}")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"bands_sub{subject_id}_sample{sample_idx}_{model_type}.png"))
    plt.close()