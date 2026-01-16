from .utils import ModelWrapper
from .utils import heatmap_plot, channel_importance_plot, band_importance_plot

import torch
import numpy as np

#computes importance as abs(input x gradient)
def input_x_gradient(model, save_path, verbose, args):
    device = next(model.parameters()).device
    model_type = args['model_type']

    wrapped_model = ModelWrapper(model).to(device)
    wrapped_model.eval()

    input_data = args['input_data']
    
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    input_tensor.requires_grad_(True)

    # Forward Pass
    logits = wrapped_model(input_tensor)
    pred_class = logits.argmax(dim=1).item()
    
    # Backward Pass
    wrapped_model.zero_grad()
    logits[0, pred_class].backward()

    # Extract Gradients and Input
    gradients = input_tensor.grad.squeeze().cpu().numpy() # (32, 5)
    original_signal = input_tensor.squeeze().detach().cpu().numpy() # (32, 5)

    # Compute Input * Gradient. abs() because we care about the magnitude of contribution
    saliency_map = np.abs(gradients * original_signal)

    # Normalization (Min-Max scaling to 0-1)
    if saliency_map.max() > 0:
        heatmap_data = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    else:
        heatmap_data = saliency_map

    # Visualization
    bands_labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    heatmap_plot(args, heatmap_data, save_path, pred_class, bands_labels)
    channel_importance_plot(args, heatmap_data, save_path, pred_class)
    band_importance_plot(args, heatmap_data, save_path, pred_class, bands_labels)

    
    #statistics 
    if verbose:
        print(f"\n--- INPUT x GRADIENT STATISTICS ({model_type}) ---")
        print(f"Class Predicted: {pred_class}")
        print(f"Range: [{heatmap_data.min():.4f}, {heatmap_data.max():.4f}]")
        print(f"Global Mean: {heatmap_data.mean():.4f}")
        
        # Top Channels
        mean_ch_importance = heatmap_data.mean(axis=1)
        top_channels_idx = np.argsort(mean_ch_importance)[-5:][::-1]
        print(f"\n--- TOP 5 CHANNELS ---")
        for idx in top_channels_idx:
            print(f"  {args['channels'][idx]}: {mean_ch_importance[idx]:.4f}")
        
        # Band Importance
        mean_band_importance = heatmap_data.mean(axis=0)
        print(f"\n--- BAND IMPORTANCE ---")
        for i, band in enumerate(bands_labels):
            print(f"  {band}: {mean_band_importance[i]:.4f}")