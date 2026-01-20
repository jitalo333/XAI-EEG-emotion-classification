# Ejemplo de uso (esto iría en tu main o notebook)
import numpy as np
import shap
# En lugar de: from method1.model import XAI_module
from model import XAI_module

# --- PASO 1: Definir X_test (Aquí es donde lo vinculas) ---
# Si ya bajaste el archivo de Drive a una carpeta llamada 'data'
try:
    ruta_seed = 'data/1_20131027.mat' 
    mat_data = sio.loadmat(ruta_seed)
    # Suponiendo que la variable en el .mat se llama 'de_LDS1'
    # Ajusta el nombre según lo que viste en Colab
    X_test = mat_data['de_LDS1'] 
except:
    print("No se encontró el archivo, usando datos aleatorios para probar")
    X_test = np.random.rand(150, 32, 5)

# 1. Definir argumentos
my_args = {
    'background': X_test[:20],       # Tus datos de fondo (primeras 20 muestras)
    'test': X_test[100:103],         # Tus datos a explicar
    'class_idx': 2,                  # La emoción que quieres explicar
    'n_electrodos': 32,
    'n_bandas': 5,
    'nsamples': 100                  # Iteraciones de SHAP
}

# 2. Instanciar la clase
# Agregamos "../" para que suba una carpeta y encuentre 'models'
xai_tool = XAI_module(
    model_path='../models/mi_modelo_entrenado.pth', 
    savepath='../resultados_xai'
)

# 3. Ejecutar el método
valores_shap = xai_tool.method1(my_args)