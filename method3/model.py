from .utils import ModelWrapper
from .utils import heatmap_plot, channel_importance_plot, band_importance_plot
import torch
import numpy as np

#integrated gradients implementation for eeg 
def integrated_gradients(model, save_path, verbose, args):
    device = next(model.parameters()).device
    steps = args.get('steps', 50)
    
    wrapped_model = ModelWrapper(model).to(device)
    wrapped_model.eval()

    input_data = args['input_data']
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device) # (1, 32, 5)
    
    #default baseline is 0
    baseline = args.get('baseline')
    if baseline is None:
        baseline = torch.zeros_like(input_tensor).to(device)
    else:
        baseline = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0).to(device)

    # Predict Class
    with torch.no_grad():
        logits = wrapped_model(input_tensor)
        pred_class = logits.argmax(dim=1).item()

    # Interpolation and Gradient Calculation
    # scaled inputs: baseline + (i/steps) * (input - baseline)
    alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1, 1)
    delta = input_tensor - baseline
    interpolated_inputs = baseline + alphas * delta
    interpolated_inputs.requires_grad_(True)

    # Forward pass
    logits = wrapped_model(interpolated_inputs)
    
    wrapped_model.zero_grad()
    target_score = logits[:, pred_class].sum()
    target_score.backward()

    #  Averaging and Attribution
    avg_grads = interpolated_inputs.grad.mean(dim=0, keepdim=True)
    attributions = delta * avg_grads
    
    # Convert to numpy and remove batch dim
    attributions = attributions.squeeze(0).detach().cpu().numpy()

    # 6. Normalization (Abs + Min-Max)
    attributions = np.abs(attributions)
    if attributions.max() > 0:
        heatmap_data = (attributions - attributions.min()) / (attributions.max() - attributions.min() + 1e-8)
    else:
        heatmap_data = attributions

    # Visualization
    bands_labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    heatmap_plot(args, heatmap_data, save_path, pred_class, bands_labels)
    channel_importance_plot(args, heatmap_data, save_path, pred_class)
    band_importance_plot(args, heatmap_data, save_path, pred_class, bands_labels)

    if verbose:
        print(f"\n--- INTEGRATED GRADIENTS STATISTICS ({args['model_type']}) ---")
        print(f"Class Predicted: {pred_class} | Steps: {steps}")
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