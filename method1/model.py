# ------SHAP-----
import shap
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

def method1(model, savepath, verbose, args):
    """
    Implementation of SHAP for EEG models (DGCNN/CDCN).
    """
    # Extract data from args
    X_background = args.get('background_data')
    X_to_explain = args.get('eval_data')
    clase_idx = args.get('class_idx', 0)
    nsamples = args.get('nsamples', 100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Automatic dimension detection
    n_channels = X_background.shape[1]
    n_bands = X_background.shape[2]
    
    if verbose:
        print(f"SHAP: Processing {n_channels} channels and {n_bands} bands.")

    # Flatten for KernelExplainer
    X_bg_flat = X_background.reshape(X_background.shape[0], -1)
    X_exp_flat = X_to_explain.reshape(X_to_explain.shape[0], -1)

    # Bridge function to reconstruct 3D tensor inside the explainer
    def predict_fn(x_flat):
        x_3d = x_flat.reshape(x_flat.shape[0], n_channels, n_bands)
        x_tensor = torch.from_numpy(x_3d).float().to(device)
        with torch.no_grad():
            output = model(x_tensor)
        return output.cpu().numpy()

    # SHAP execution
    explainer = shap.KernelExplainer(predict_fn, X_bg_flat)
    shap_values = explainer.shap_values(X_exp_flat, nsamples=nsamples)

    # Result selection (Handle multiclass list)
    if isinstance(shap_values, list):
        sv_matrix = shap_values[clase_idx]
    else:
        sv_matrix = shap_values

    # Saving results
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # Plotting
    plt.figure()
    shap.summary_plot(sv_matrix, X_exp_flat, plot_type="bar", show=False)
    plt.savefig(os.path.join(savepath, f'shap_summary_class_{clase_idx}.png'))
    plt.close()

    # Data dump
    joblib.dump(shap_values, os.path.join(savepath, 'shap_values.pkl'))
    
    if verbose:
        print(f"SHAP results saved to {savepath}")