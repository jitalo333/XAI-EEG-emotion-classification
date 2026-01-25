from utils.utils import ModelWrapper
from utils.utils import heatmap_plot, channel_importance_plot, band_importance_plot

import torch
import numpy as np
from tqdm import tqdm

#main function. It decides which method executes
def saliency_map(model, save_path, verbose, args):
    if args.get('is_global', False):
        return _saliency_global(model, save_path, verbose, args)
    else:
        return _saliency_local(model, save_path, verbose, args)

def _saliency_local(model, save_path, verbose, args):
    device = next(model.parameters()).device
    wrapped_model = ModelWrapper(model).to(device)
    wrapped_model.eval()

    input_data = args['input_data']
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True) 

    # Forward
    logits = wrapped_model(input_tensor)
    pred_class = logits.argmax(dim=1).item()

    # Backward
    wrapped_model.zero_grad()
    logits[0, pred_class].backward()

    # Extract Gradients (Vanilla)
    gradients = torch.abs(input_tensor.grad).squeeze().cpu().numpy()
    
    # Normalization (0-1)
    heatmap_data = _normalize(gradients)

    # Plotting
    _generate_plots_and_stats(heatmap_data, args, save_path, pred_class, verbose)
    
    return heatmap_data


#global method averages all gradients from the samples correctly classified. It requires X_test and y_test
#optionally, 'target_class' parameter can be given to explain a particular class.
def _saliency_global(model, save_path, verbose, args):
    device = next(model.parameters()).device
    wrapped_model = ModelWrapper(model).to(device)
    wrapped_model.eval()
    
    X_test = args['X_test']
    y_test = args['y_test']
    target_class = args.get('target_class', None) # If None, it averages all correct classes
    
    accumulated_gradients = None
    count = 0
    
    
    for i in tqdm(range(len(X_test))):
        #Class filter if given target_class
        if target_class is not None and y_test[i] != target_class:
            continue
            
        input_data = X_test[i]
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = wrapped_model(input_tensor)
            pred_idx = logits.argmax(dim=1).item()
        
        #only look at correct predictions
        if pred_idx == y_test[i]:
            input_tensor.requires_grad_(True)
            logits = wrapped_model(input_tensor)
            
            wrapped_model.zero_grad()
            logits[0, pred_idx].backward()
            
            grad = torch.abs(input_tensor.grad).squeeze().cpu().numpy()
            
            if accumulated_gradients is None:
                accumulated_gradients = np.zeros_like(grad)
            
            accumulated_gradients += grad
            count += 1
            
    if count == 0:
        return None
        
    # average
    avg_gradients = accumulated_gradients / count
    args['n_samples'] = count
    
    heatmap_data = _normalize(avg_gradients)
    
    class_label = target_class if target_class is not None else "All_Correct"
    
    if verbose:
        print(f"Averaged over {count} samples.")
    _generate_plots_and_stats(heatmap_data, args, save_path, class_label, verbose)
    
    return heatmap_data

def _normalize(data):
    return (data - data.min()) / (data.max() - data.min() + 1e-8)

def _generate_plots_and_stats(heatmap_data, args, save_path, label, verbose):
    bands_labels = ['Delta', 'Theta', 'Alpha','Beta', 'Gamma']
    
    args['method_name'] = 'SaliencyMaps'
    
    heatmap_plot(args, heatmap_data, save_path, label, bands_labels)
    channel_importance_plot(args, heatmap_data, save_path, label)
    band_importance_plot(args, heatmap_data, save_path, label, bands_labels)
    
    if verbose:
        print(f"\n--- SALIENCY STATISTICS ({args['model_type']} - {'Global' if args.get('is_global') else 'Local'}) ---")
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