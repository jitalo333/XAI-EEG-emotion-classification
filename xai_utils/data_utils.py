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

def get_plot_params(args, pred_class):
    is_global = args.get('is_global', False)
    
    if is_global:
        prefix = "avg_" # prefix for global files
        count = args.get('n_samples', '?')
        title_context = f"Global Avg (n={count})"
        file_suffix = "global"
    else:
        prefix = ""
        idx = args.get('sample_idx', 0)
        title_context = f"Sample {idx}"
        file_suffix = f"sample_{idx}"
    
    # detect name method. If not given, XAI_method is used
    method_name = args.get('method_name', 'XAI_Method')
    
    return prefix, title_context, file_suffix, method_name


def normalize_for_plot(data, mode="minmax", eps=1e-8):
    data = data.astype(np.float32)

    if mode == "minmax":
        dmin, dmax = data.min(), data.max()
        return (data - dmin) / (dmax - dmin + eps)

    elif mode == "percentile":
        p1, p99 = np.percentile(data, [1, 99])
        data = np.clip(data, p1, p99)
        return (data - p1) / (p99 - p1 + eps)

    else:
        return data

def get_plot_data(heatmap_data, args):
    attr_scale = args.get('attr_scale', 'normalized')
    
    print(
        "Before plot:",
        heatmap_data.min(),
        heatmap_data.max()
    )

    if attr_scale == 'normalized':
        return heatmap_data

    plot_data = normalize_for_plot(heatmap_data, mode='minmax')
    print(
        "After plot norm:",
        plot_data.min(),
        plot_data.max()
    )
    return plot_data


def heatmap_plot(args, heatmap_data, save_path, pred_class, bands_labels):
    prefix, title_context, file_suffix, method_name = get_plot_params(args, pred_class)

    plot_data = get_plot_data(heatmap_data, args)

    plt.figure(figsize=(10, 11))
    sns.heatmap(
        plot_data,
        xticklabels=bands_labels,
        yticklabels=args['channels'],
        cmap='turbo',
        cbar_kws={'label': f'Attribution ({method_name})'},
        vmin=0, vmax=1
    )

    plt.title(
        f"{method_name} Map - {args['model_type']}\n"
        f"Subject {args['subject_id']} | {title_context} | Class {pred_class}"
    )
    plt.xlabel("Frequency Bands")
    plt.ylabel("EEG Channels")
    plt.tight_layout()

    filename = f"{prefix}heatmap_{method_name}_sub{args['subject_id']}_{file_suffix}_{args['model_type']}.png"
    plt.savefig(os.path.join(save_path, filename))
    plt.close()

def channel_importance_plot(args, heatmap_data, save_path, pred_class):
    prefix, title_context, file_suffix, method_name = get_plot_params(args, pred_class)
    channels_names = args['channels']

    plot_data = get_plot_data(heatmap_data, args)
    mean_importance = plot_data.mean(axis=1)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(channels_names)), mean_importance)
    plt.xticks(range(len(channels_names)), channels_names, rotation=45, ha='right')
    plt.ylabel("Mean Importance")
    plt.xlabel("EEG Channels")

    plt.title(
        f"Channel Importance ({method_name}) - {args['model_type']}\n"
        f"Subject {args['subject_id']} | {title_context} | Class {pred_class}"
    )
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    filename = f"{prefix}channels_{method_name}_sub{args['subject_id']}_{file_suffix}_{args['model_type']}.png"
    plt.savefig(os.path.join(save_path, filename))
    plt.close()
    
def band_importance_plot(args, heatmap_data, save_path, pred_class, bands_labels):
    prefix, title_context, file_suffix, method_name = get_plot_params(args, pred_class)

    plot_data = get_plot_data(heatmap_data, args)
    mean_band_importance = plot_data.mean(axis=0)

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(bands_labels)), mean_band_importance, edgecolor='black', linewidth=1.5)
    plt.xticks(range(len(bands_labels)), bands_labels)
    plt.ylabel("Mean Importance")
    plt.xlabel("Frequency Bands")

    plt.title(
        f"Band Importance ({method_name}) - {args['model_type']}\n"
        f"Subject {args['subject_id']} | {title_context} | Class {pred_class}"
    )
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    filename = f"{prefix}bands_{method_name}_sub{args['subject_id']}_{file_suffix}_{args['model_type']}.png"
    plt.savefig(os.path.join(save_path, filename))
    plt.close()
