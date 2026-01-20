#---------LIME------------
import lime
import lime.lime_tabular
import torch
import numpy as np
import os
import joblib

# Standard electrode lists
CHANNELS_32 = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'T7', 'C3', 'CZ', 'C4', 'T8',
    'TP7', 'CP3', 'CPZ', 'CP4', 'TP8', 'P7', 'P3', 'PZ', 'P4', 'P8',
    'O1', 'O2'
]

CHANNELS_62 = [
    'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'IZ'
]

def method2(model, savepath, verbose, args):
    """
    Universal LIME implementation for EEG.
    """
    # 1. Extract arguments from the dictionary
    X_train = args.get('train_data')        # Required for LIME distribution
    x_instance = args.get('eval_instance')  # The single sample to explain
    class_names = args.get('class_names', ['Negative', 'Neutral', 'Positive'])
    num_features = args.get('num_features', 15)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 2. Dynamic Channel and Band Detection
    n_channels = X_train.shape[1]
    n_bands = X_train.shape[2]
    
    if n_channels == 32:
        channel_names = CHANNELS_32
    elif n_channels == 62:
        channel_names = CHANNELS_62
    else:
        channel_names = [f"Ch{i+1}" for i in range(n_channels)]

    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    feature_names = [f"{c}_{b}" for c in channel_names for b in bands[:n_bands]]

    # 3. Predict Proba Wrapper (The Bridge)
    def predict_proba_proxy(np_array_flat):
        # Reconstruct 3D shape: (Batch, Channels, Bands)
        samples = np_array_flat.reshape(np_array_flat.shape[0], n_channels, n_bands)
        inputs = torch.tensor(samples, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            # LIME requires probabilities, so we apply Softmax
            probs = torch.nn.functional.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    # 4. LIME Training Data Flattening
    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    # 5. Explainer Initialization
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_flat,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )

    # 6. Run Explanation
    instance_flat = x_instance.flatten()
    exp = explainer.explain_instance(
        instance_flat,
        predict_proba_proxy,
        num_features=num_features
    )

    # 7. Saving Results
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    exp.save_to_file(os.path.join(savepath, 'lime_report.html'))
    
    if verbose:
        print(f"LIME analysis complete. Results saved in {savepath}")