from xai_utils.data_utils import ModelWrapper
from xai_utils.data_utils import heatmap_plot, channel_importance_plot, band_importance_plot, get_plot_data

import torch
import numpy as np
from tqdm import tqdm

def input_x_gradient(model, save_path, verbose, args):
    """
    Main entry point for Input x Gradient.
    Dispatches to local or global implementation based on args['is_global'].
    """
    # Define method name for dynamic plot titles and filenames
    args['method_name'] = 'Input_x_Gradient'
    
    if args.get('is_global', False):
        return _input_x_grad_global(model, save_path, verbose, args)
    else:
        return _input_x_grad_local(model, save_path, verbose, args)

def _input_x_grad_local(model, save_path, verbose, args):
    device = next(model.parameters()).device
    wrapped_model = ModelWrapper(model).to(device)
    wrapped_model.eval()

    input_data = args['input_data']
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Predict Class and calculate attribution
    attribution, pred_class = _compute_input_x_grad_logic(wrapped_model, input_tensor)
    
    # Normalization (Min-Max)
    heatmap_data = _normalize(attribution)

    # Visualization and Stats
    _generate_plots_and_stats(heatmap_data, args, save_path, pred_class, verbose)
    
    return heatmap_data

def _input_x_grad_global(model, save_path, verbose, args):
    """
    Global implementation: Averages Input x Gradient attributions 
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
    
    for i in tqdm(range(len(X_test))):
        # Filter by class if target_class is specified
        if target_class is not None and y_test[i] != target_class:
            continue

        input_tensor = torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(0).to(device)

        # 1. Prediction (check if model is correct)
        with torch.no_grad():
            logits = wrapped_model(input_tensor)
            pred_idx = logits.argmax(dim=1).item()
            
        if pred_idx == y_test[i]:
            # 2. Compute attribution
            attrs, _ = _compute_input_x_grad_logic(wrapped_model, input_tensor)
            
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
    args['n_samples'] = count
    heatmap_data = avg_attributions
    args['attr_scale'] = 'raw'
    
    label_plot = target_class if target_class is not None else "All_Correct"
        
    _generate_plots_and_stats(heatmap_data, args, save_path, label_plot, verbose)
    
    return heatmap_data

def _compute_input_x_grad_logic(model, input_tensor):
    """
    Helper function to perform Input * Gradient math.
    """
    input_tensor.requires_grad_(True)
    logits = model(input_tensor)
    pred_class = logits.argmax(dim=1).item()
    
    model.zero_grad()
    logits[0, pred_class].backward()

    # Calculation: Input * Gradient
    # We use abs() to see the magnitude of the contribution regardless of sign
    gradients = input_tensor.grad.squeeze().cpu().numpy()
    original_signal = input_tensor.squeeze().detach().cpu().numpy()
    attribution = np.abs(gradients * original_signal)
    
    return attribution, pred_class

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
        print(f"Global Mean: {heatmap_data.mean():.4f}")
        
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