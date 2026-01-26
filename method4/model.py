from xai_utils.data_utils import ModelWrapper
from xai_utils.data_utils import heatmap_plot, channel_importance_plot, band_importance_plot, get_plot_data

import torch
import numpy as np
from tqdm import tqdm

def smooth_grad(model, save_path, verbose, args):
    """
    Main entry point for SmoothGrad.
    Dispatches to local or global implementation based on args['is_global'].
    """
    args['method_name'] = 'SmoothGrad'
    
    if args.get('is_global', False):
        return _smooth_grad_global(model, save_path, verbose, args)
    else:
        return _smooth_grad_local(model, save_path, verbose, args)

#normalized local method
def _smooth_grad_local(model, save_path, verbose, args):
    device = next(model.parameters()).device
    wrapped_model = ModelWrapper(model).to(device)
    wrapped_model.eval()

    input_data = args['input_data']
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Base prediction
    with torch.no_grad():
        logits_base = wrapped_model(input_tensor)
        pred_class = logits_base.argmax(dim=1).item()

    # Calculate SmoothGrad for this single sample
    attribution = _compute_smooth_grad_logic(wrapped_model, input_tensor, pred_class, device, args)
    
    # Normalization
    heatmap_data = _normalize(attribution)

    # Visualization and Stats
    _generate_plots_and_stats(heatmap_data, args, save_path, pred_class, verbose)
    
    return heatmap_data

def _smooth_grad_global(model, save_path, verbose, args):
    """
    Global implementation: Averages SmoothGrad attributions 
    over correctly classified test samples.
    """
    device = next(model.parameters()).device
    wrapped_model = ModelWrapper(model).to(device)
    wrapped_model.eval()
    
    X_test = args['X_test']
    y_test = args['y_test']
    target_class = args.get('target_class', None)
    
    accumulated_attributions = None
    count = 0
    
    print(f"Starting Global SmoothGrad for Subject {args['subject_id']}...")
    
    for i in tqdm(range(len(X_test))):
        if target_class is not None and y_test[i] != target_class:
            continue

        input_tensor = torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(0).to(device)

        # 1. Prediction check
        with torch.no_grad():
            logits = wrapped_model(input_tensor)
            pred_idx = logits.argmax(dim=1).item()
            
        if pred_idx == y_test[i]:
            # 2. Compute SmoothGrad for this sample
            attrs = _compute_smooth_grad_logic(wrapped_model, input_tensor, pred_idx, device, args)
            
            # 3. Accumulate
            if accumulated_attributions is None:
                accumulated_attributions = np.zeros_like(attrs)
            
            accumulated_attributions += attrs
            count += 1
            
    if count == 0:
        return None
    
    if verbose:
        print(f"\nAveraged over {count} samples.")

    # Average and Normalize
    avg_attributions = accumulated_attributions / count
    args['n_samples_global'] = count # Rename to avoid conflict with SmoothGrad internal samples
    heatmap_data = avg_attributions #avg_contributtions is not normalized
    args['attr_scale'] = 'raw'#with this parameter, the heatmap_data will be normalized only for plotting
    
    label_plot = target_class if target_class is not None else "All_Correct"
    _generate_plots_and_stats(heatmap_data, args, save_path, label_plot, verbose)
    
    return heatmap_data

def _compute_smooth_grad_logic(model, input_tensor, target_class, device, args):
    """
    Helper function to perform SmoothGrad noise-averaging math.
    """
    stdev_spread = args.get('stdev_spread', 0.15)
    n_samples = args.get('n_samples', 50)
    magnitude = args.get('magnitude', True)
    
    # Std of noise based on signal range
    stdev = stdev_spread * (input_tensor.max() - input_tensor.min()).item()
    total_gradients = torch.zeros_like(input_tensor)

    for i in range(n_samples):
        # Add noise
        noise = torch.randn_like(input_tensor) * stdev
        x_noisy = (input_tensor + noise).detach().requires_grad_(True)
        
        # Forward pass
        logits = model(x_noisy)
        
        # Backward pass
        model.zero_grad()
        score = logits[0, target_class]
        score.backward()
        
        grad = x_noisy.grad
        if magnitude:
            # Using square to emphasize high variance regions
            total_gradients += (grad * grad)
        else:
            total_gradients += grad.abs()

    # Average and convert to numpy
    avg_gradients = total_gradients / n_samples
    return avg_gradients.squeeze(0).detach().cpu().numpy()

def _normalize(data):
    if data.max() > 0:
        return (data - data.min()) / (data.max() - data.min() + 1e-8)
    return data

def _generate_plots_and_stats(heatmap_data, args, save_path, label, verbose):
    bands_labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    plot_data = get_plot_data(heatmap_data, args)
    
    heatmap_plot(args, plot_data, save_path, label, bands_labels)
    channel_importance_plot(args, plot_data, save_path, label)
    band_importance_plot(args, plot_data, save_path, label, bands_labels)

    if verbose:
        mode = 'Global' if args.get('is_global') else 'Local'
        print(f"\n--- {args['method_name']} STATISTICS ({args['model_type']} - {mode}) ---")
        print(f"Target/Pred Class: {label}")
        print(f"Noise Samples per Input: {args.get('n_samples', 50)}")
        print(f"Global Mean Attribution: {heatmap_data.mean():.4f}")
        
        mean_importance = heatmap_data.mean(axis=1)
        mean_band_importance = heatmap_data.mean(axis=0)
        channels_names = args['channels']
        
        print(f"\n--- TOP 5 CHANNELS ---")
        top_channels = np.argsort(mean_importance)[-5:][::-1]
        for ch in top_channels:
            print(f"  {channels_names[ch]}: {mean_importance[ch]:.4f}")
            
         # Band Importance
        print(f"\n--- BAND IMPORTANCE ---")
        for i, band in enumerate(bands_labels):
            print(f"  {band}: {mean_band_importance[i]:.4f}")