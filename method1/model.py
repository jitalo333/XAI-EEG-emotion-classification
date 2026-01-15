from .utils import ModelWrapper
from .utils import heatmap_plot, channel_importance_plot, band_importance_plot

import torch
import os
import numpy as np

# vainilla gradient based
def saliency_map(model, save_path, verbose, args):
    device = next(model.parameters()).device

    wrapped_model = ModelWrapper(model).to(device)
    wrapped_model.eval()

    input_data = args['input_data']
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True) 

    # Forward Pass
    #with torch.no_grad():
    logits = wrapped_model(input_tensor)
    pred_class = logits.argmax(dim=1).item()

    # Backward Pass
    wrapped_model.zero_grad()
    logits[0, pred_class].backward()

    # Extract Gradients
    gradients = torch.abs(input_tensor.grad).squeeze().cpu().numpy()
    
    # normalization (0-1)
    heatmap_data = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)

    # Labels for bands
    bands_labels = ['Delta', 'Theta', 'Alpha','Beta', 'Gamma']

    heatmap_plot(args, heatmap_data, save_path, pred_class, bands_labels)
    channel_importance_plot(args, heatmap_data, save_path, pred_class)
    band_importance_plot(args, heatmap_data, save_path, pred_class, bands_labels)
    
    mean_importance = heatmap_data.mean(axis=1)
    mean_band_importance = heatmap_data.mean(axis=0)
    channels_names = args['channels']
    
    if verbose:
        print(f"\n--- SALIENCY STATISTICS ({args['model_type']}) ---")
        print(f"Range: [{heatmap_data.min():.4f}, {heatmap_data.max():.4f}]")
        print(f"Global Mean: {heatmap_data.mean():.4f}")
        print(f"Global Std: {heatmap_data.std():.4f}")

        print(f"\n--- TOP 5 CHANNELS ---")
        top_channels = np.argsort(mean_importance)[-5:][::-1]
        for ch in top_channels:
            print(f"  {channels_names[ch]}: {mean_importance[ch]:.4f}")

        print(f"\n--- BAND IMPORTANCE ---")
        for i, band in enumerate(bands_labels):
            clean_band = band.replace('\n', ' ')
            print(f"  {clean_band}: {mean_band_importance[i]:.4f}")