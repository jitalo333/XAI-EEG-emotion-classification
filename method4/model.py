from .utils import ModelWrapper, heatmap_plot, channel_importance_plot, band_importance_plot
import torch
import numpy as np

#shows which regions produce more consistent changes in the output due to small perturbations in the input
def smooth_grad(model, save_path, verbose, args):
    """
    SmoothGrad implementation for EEG. 
    Averages gradients of an input with added Gaussian noise.
    """
    device = next(model.parameters()).device
    stdev_spread = args.get('stdev_spread', 0.15)
    n_samples = args.get('n_samples', 50)
    magnitude = args.get('magnitude', True)
    
    wrapped_model = ModelWrapper(model).to(device)
    wrapped_model.eval()

    input_data = args['input_data']
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    #std of noise based on signal range
    stdev = stdev_spread * (input_tensor.max() - input_tensor.min()).item()

    #base prediction
    with torch.no_grad():
        logits_base = wrapped_model(input_tensor)
        pred_class = logits_base.argmax(dim=1).item()

    total_gradients = torch.zeros_like(input_tensor)

    for i in range(n_samples):
        #add noise
        noise = torch.randn_like(input_tensor) * stdev
        x_noisy = (input_tensor + noise).detach().requires_grad_(True)
        
        # Forward pass
        logits = wrapped_model(x_noisy)
        
        # Backward pass
        wrapped_model.zero_grad()
        score = logits[0, pred_class]
        score.backward()
        
        grad = x_noisy.grad
        if magnitude:
            total_gradients += (grad * grad)
        else:
            total_gradients += grad.abs()

    #average and Normalize
    avg_gradients = total_gradients / n_samples
    heatmap_data = avg_gradients.squeeze(0).detach().cpu().numpy()
    
    # Min-Max Scaling (0 a 1)
    if heatmap_data.max() > 0:
        heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-8)

    # Visualization
    bands_labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    heatmap_plot(args, heatmap_data, save_path, pred_class, bands_labels)
    channel_importance_plot(args, heatmap_data, save_path, pred_class)
    band_importance_plot(args, heatmap_data, save_path, pred_class, bands_labels)

    if verbose:
        print(f"\n--- SMOOTHGRAD STATISTICS ({args['model_type']}) ---")
        print(f"Samples: {n_samples} | Noise Spread: {stdev_spread} | Class Predicted: {pred_class}")
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

    return heatmap_data