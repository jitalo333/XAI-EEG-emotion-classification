from xai_utils.data_utils import ModelWrapper
from xai_utils.data_utils import heatmap_plot, channel_importance_plot, band_importance_plot

import torch
import numpy as np
from tqdm import tqdm

def integrated_gradients(model, save_path, verbose, args):
    """
    Main entry point. Dispatches to local or global implementation based on args.
    """
    # Set method name for the dynamic plotting utils
    args['method_name'] = 'Integrated Gradients'
    
    if args.get('is_global', False):
        return _ig_global(model, save_path, verbose, args)
    else:
        return _ig_local(model, save_path, verbose, args)

def _ig_local(model, save_path, verbose, args):
    device = next(model.parameters()).device
    steps = args.get('steps', 50)
    
    wrapped_model = ModelWrapper(model).to(device)
    wrapped_model.eval()

    input_data = args['input_data']
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device) # (1, 32, 5)
    
    # Predict Class
    with torch.no_grad():
        logits = wrapped_model(input_tensor)
        pred_class = logits.argmax(dim=1).item()

    # Calculate IG
    attributions = _compute_ig_for_sample(wrapped_model, input_tensor, pred_class, steps, device, args.get('baseline'))

    # Normalization (Abs + Min-Max). 
    # This shows how important a certain channel was for the class prediction, 
    # whether it was positive or negative contribution (np.abs)
    attributions = np.abs(attributions)
    heatmap_data = _normalize(attributions)
        
    # Visualization and Stats
    _generate_plots_and_stats(heatmap_data, args, save_path, pred_class, verbose)
    
    return heatmap_data

def _ig_global(model, save_path, verbose, args):
    """
    Global implementation. Averages IG attributions over correctly classified test samples.
    """
    device = next(model.parameters()).device
    steps = args.get('steps', 50)
    
    wrapped_model = ModelWrapper(model).to(device)
    wrapped_model.eval()
    
    X_test = args['X_test']
    y_test = args['y_test']
    target_class = args.get('target_class', None)
    
    accumulated_attributions = None
    count = 0
    
    
    # Iterate over test set
    for i in tqdm(range(len(X_test))):
        # Filter by target class if requested
        if target_class is not None and y_test[i] != target_class:
            continue

        input_data = X_test[i]
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)

        # Check Prediction (only explain correct predictions)
        with torch.no_grad():
            logits = wrapped_model(input_tensor)
            pred_idx = logits.argmax(dim=1).item()
            
        if pred_idx == y_test[i]:
            # Compute IG for this sample
            # We use the predicted class (which is also the true label)
            attrs = _compute_ig_for_sample(wrapped_model, input_tensor, pred_idx, steps, device, args.get('baseline'))
            
            # Accumulate raw attributions
            if accumulated_attributions is None:
                accumulated_attributions = np.zeros_like(attrs)
            
            accumulated_attributions += attrs
            count += 1
            
    if count == 0:
        return None

    if verbose:
        print(f"count of accumulated gradients: {count}")
        
    # 6. Average
    avg_attributions = accumulated_attributions / count
    args['n_samples'] = count # For plot titles
    
    # 7. Normalize (Abs + Min-Max)
    avg_attributions = np.abs(avg_attributions)
    heatmap_data = _normalize(avg_attributions)
    
    # 8. Plotting
    label_for_plot = target_class if target_class is not None else "All_Correct"
    print(f"Averaged over {count} samples.")
    _generate_plots_and_stats(heatmap_data, args, save_path, label_for_plot, verbose)
    
    return heatmap_data

def _compute_ig_for_sample(model, input_tensor, target_class, steps, device, baseline_val=None):
    """
    Helper function to perform the math of IG on a single tensor.
    Returns numpy array of attributions (unnormalized).
    """
    # default baseline is 0
    if baseline_val is None:
        baseline = torch.zeros_like(input_tensor).to(device)
    else:
        baseline = torch.tensor(baseline_val, dtype=torch.float32).unsqueeze(0).to(device)

    # Interpolation: scaled inputs = baseline + (i/steps) * (input - baseline)
    alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1, 1)
    delta = input_tensor - baseline
    interpolated_inputs = baseline + alphas * delta
    interpolated_inputs.requires_grad_(True)

    # Forward pass on batch of interpolated inputs
    logits = model(interpolated_inputs)
    
    # Gradients w.r.t target class
    model.zero_grad()
    target_score = logits[:, target_class].sum()
    target_score.backward()

    # Averaging gradients
    avg_grads = interpolated_inputs.grad.mean(dim=0, keepdim=True)
    
    # Attribution: (input - baseline) * avg_gradients
    attributions = delta * avg_grads
    
    # Remove batch dim and convert to numpy
    return attributions.squeeze(0).detach().cpu().numpy()

def _normalize(data):
    if data.max() > 0:
        return (data - data.min()) / (data.max() - data.min() + 1e-8)
    return data

def _generate_plots_and_stats(heatmap_data, args, save_path, label, verbose):
    bands_labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    heatmap_plot(args, heatmap_data, save_path, label, bands_labels)
    channel_importance_plot(args, heatmap_data, save_path, label)
    band_importance_plot(args, heatmap_data, save_path, label, bands_labels)

    if verbose:
        mode = 'Global' if args.get('is_global') else 'Local'
        print(f"\n--- INTEGRATED GRADIENTS STATISTICS ({args['model_type']} - {mode}) ---")
        print(f"Class: {label}")
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