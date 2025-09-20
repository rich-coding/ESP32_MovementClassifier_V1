import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal.windows import hamming, hann, blackman, bartlett
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
import tracemalloc
import os
import random

# Importación clave para la paralelización
from multiprocessing import Pool, cpu_count
import itertools

# --- FIJACIÓN DE SEMILLAS PARA REPRODUCIBILIDAD ---
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

######################################################################
# FUNCIONES.
######################################################################

def measure_performance(func, *args, **kwargs):
    """
    Mide y muestra el tiempo de ejecución y el uso de memoria de una función.
    """
    start_time = time.time()
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    execution_time = time.time() - start_time
    tracemalloc.stop()
    print("S=========")
    print(f"Tiempo de ejecución de {func.__name__}: {execution_time:.4f} segundos")
    print(f"Uso de memoria de {func.__name__}: {current / 10**6:.4f} MB; Pico de memoria: {peak / 10**6:.4f} MB")
    print("E=========")
    print(" ")
    return result

def load_data(filename):
    """
    Carga los datos desde un archivo CSV.
    """
    return pd.read_csv(filename)

def apply_filter(data, filter_type='none', window_size=20):
    """
    Aplica un filtro a los datos.
    """
    if filter_type is None or filter_type == 'none':
        return data
    if filter_type == 'moving_average':
        filtered = data.rolling(window=window_size).mean()
    elif filter_type == 'exponential':
        filtered = data.ewm(span=window_size).mean()
    elif filter_type == 'savitzky_golay':
        filtered = savgol_filter(data, window_length=window_size, polyorder=2, axis=0)
    else:
        raise ValueError("Tipo de filtro desconocido. Use 'moving_average', 'exponential', 'savitzky_golay', o 'none'.")
    # Rellenar los valores NaN iniciales causados por el filtro
    return pd.DataFrame(filtered, columns=data.columns).fillna(method='bfill')

def normalize_data(data, method='min_max'):
    """
    Normaliza los datos usando el método especificado.
    """
    columns = data.columns
    if method == 'min_max':
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
    elif method == 'z_score':
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
    elif method is None:
        return data
    else:
        raise ValueError("Método de normalización desconocido. Use 'min_max', 'z_score' o None.")
    return pd.DataFrame(scaled_data, columns=columns)

def segment_data_by_category(data, window_size=20, overlap=True):
    """
    Segmenta los datos por categoría en ventanas.
    """
    segments = []
    labels = []
    step = window_size // 2 if overlap else window_size
    grouped = data.groupby('category')
    for category, group in grouped:
        # Asegurarse de que el grupo no contenga la columna 'category' antes de segmentar
        features = group.drop('category', axis=1)
        for start in range(0, len(group) - window_size + 1, step):
            segments.append(features.iloc[start:start + window_size].values)
            labels.append(category)
    return np.array(segments), np.array(labels)

def apply_window(segment, window_type='hamming'):
    """
    Aplica una función de ventana a un segmento de datos.
    """
    num_samples = segment.shape[0]
    if window_type is None or window_type == 'none':
        return segment
    if window_type == 'hamming':
        window = hamming(num_samples)
    elif window_type == 'hanning':
        window = hann(num_samples)
    elif window_type == 'rectangular':
        window = np.ones(num_samples)
    elif window_type == 'blackman':
        window = blackman(num_samples)
    elif window_type == 'bartlett':
        window = bartlett(num_samples)
    else:
        raise ValueError("Tipo de ventana desconocido. Use 'hamming', 'hanning', 'rectangular', 'blackman', 'bartlett', o 'none'.")
    return segment * window[:, np.newaxis]

def train_lstm_model(X_train, y_train, X_val, y_val, lstm1=7, lstm2=4, epochs=10, batch_size=64):
    """
    Construye y entrena un modelo LSTM.
    """
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(lstm1, return_sequences=True),
        LSTM(lstm2),
        Dense(y_train.shape[1], activation='softmax')
    ])
    # Usar una métrica de F1-Score compatible con TensorFlow
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.F1Score(average='macro'), 'accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1)
    return model, history

def plot_metrics(history, artefact_name):
    """
    Grafica y guarda las métricas de precisión y pérdida del modelo.
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Crear directorio si no existe
    os.makedirs('artefacts/metrics', exist_ok=True)
    plt.savefig(f'artefacts/metrics/metrics_{artefact_name}.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, artefact_name):
    """
    Grafica y guarda la matriz de confusión.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Crear directorio si no existe
    os.makedirs('artefacts/confusion_matrix', exist_ok=True)
    plt.savefig(f'artefacts/confusion_matrix/cm_{artefact_name}.png')
    plt.close()
    
def save_model_and_weights(model, artefact_name):
    """
    Guarda el modelo completo y sus pesos.
    """
    os.makedirs('artefacts/models', exist_ok=True)
    os.makedirs('artefacts/weights', exist_ok=True)
    model.save(f'artefacts/models/model_{artefact_name}.keras')
    model.save_weights(f'artefacts/weights/weights_{artefact_name}.weights.h5')

def save_model_parameters_txt(model, artefact_name):
    """
    Guarda los parámetros (pesos) del modelo en un archivo de texto.
    """
    os.makedirs('artefacts/params', exist_ok=True)
    with open(f'artefacts/params/params_{artefact_name}.txt', 'w') as f:
        for i, layer_weights in enumerate(model.get_weights()):
            #f.write(f"Parámetros de la capa {i}:\n")
            f.write(str(layer_weights.tolist()))
            #f.write("\n\n")

def abreviar_parametros(lstm1, lstm2, filtro, norm, superposicion, ventana):
    """
    Crea un nombre abreviado para un conjunto de hiperparámetros.
    """
    filtro_map = {'moving_average': 'ma', 'exponential': 'exp', 'savitzky_golay': 'sg', None: 'N', 'none': 'N'}
    norm_map = {'min_max': 'mm', 'z_score': 'zs', None: 'N'}
    super_map = {True: 'T', False: 'F'}
    ventana_map = {'hamming': 'ham', 'hanning': 'han', 'rectangular': 'rec', 'blackman': 'blk', 'bartlett': 'bart', None: 'N', 'none': 'N'}
    
    filtro_abbr = filtro_map.get(filtro, str(filtro))
    norm_abbr = norm_map.get(norm, str(norm))
    super_abbr = super_map.get(superposicion, str(superposicion))
    ventana_abbr = ventana_map.get(ventana, str(ventana))
    
    return f"l1_{lstm1}_l2_{lstm2}_f_{filtro_abbr}_n_{norm_abbr}_s_{super_abbr}_v_{ventana_abbr}"

######################################################################
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO.
######################################################################

def main(filename, filtro='savitzky_golay', normalizacion='min_max', superposicion=True, tipo_ventana='hanning', lstm1=7, lstm2=4):
    """
    Función principal que ejecuta el pipeline de preprocesamiento y entrenamiento.
    """
    artefact_name = abreviar_parametros(lstm1, lstm2, filtro, normalizacion, superposicion, tipo_ventana)
    
    data = load_data(filename)
    features = data.drop('category', axis=1)
    labels_col = data['category']
    
    print(f"Aplicando filtro: {filtro}")
    filtered_data = measure_performance(apply_filter, features, filtro, 40)
    
    print(f"Aplicando normalización: {normalizacion}")
    normalized_data = measure_performance(normalize_data, filtered_data, normalizacion)
    
    processed_data_with_labels = pd.concat([normalized_data, labels_col], axis=1)
    
    print(f"Aplicando segmentación (superposición={superposicion})")
    segments, labels = measure_performance(segment_data_by_category, processed_data_with_labels, window_size=40, overlap=superposicion)
    
    print("Número total de segmentos:", len(segments))
    
    print(f"Aplicando ventana de {tipo_ventana}")
    windowed_segments = measure_performance(np.array, [apply_window(seg, tipo_ventana) for seg in segments])
    
    X_train, X_val, y_train, y_val = train_test_split(windowed_segments, labels, test_size=0.35, random_state=42, stratify=labels)
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    
    y_train_categorical = to_categorical(y_train_encoded)
    y_val_categorical = to_categorical(y_val_encoded)
    
    model, history = measure_performance(train_lstm_model, X_train, y_train_categorical, X_val, y_val_categorical, lstm1=lstm1, lstm2=lstm2, epochs=120, batch_size=32)
    
    # Guardado condicional de artefactos (modelos, gráficos, etc.)
    threshold_accuracy = 0.95
    if max(history.history['val_accuracy']) > threshold_accuracy:
        print(f"Umbral de accuracy > {threshold_accuracy} alcanzado. Guardando artefactos para {artefact_name}...")
        plot_metrics(history, artefact_name)
        
        y_pred = np.argmax(model.predict(X_val), axis=1)
        plot_confusion_matrix(y_val_encoded, y_pred, label_encoder.classes_, artefact_name)
        
        save_model_and_weights(model, artefact_name)
        save_model_parameters_txt(model, artefact_name)
    else:
        print(f"Umbral de accuracy > {threshold_accuracy} NO alcanzado. No se guardarán artefactos para {artefact_name}.")

    return history, artefact_name

######################################################################
# FUNCIÓN "TRABAJADORA" PARA LA PARALELIZACIÓN
######################################################################

def run_experiment(params):
    """
    Esta función envuelve la lógica de un experimento para ser ejecutada
    en un proceso separado.
    """
    # Desempaquetar los hiperparámetros
    l1, l2, filtro, norm, superposicion, ventana = params
    filename = 'data/datos_mpu6050_out.csv' # Asegurarse que el path es accesible

    print("-" * 50)
    print(f"INICIANDO Proceso para: l1={l1}, l2={l2}, f={filtro}, n={norm}, v={ventana}")
    print("-" * 50)
    
    # Silenciar la salida de TensorFlow para este proceso para no sobrecargar la consola
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    try:
        # Ejecutar la lógica principal del entrenamiento
        history, artefact_name = main(filename, filtro=filtro, normalizacion=norm, superposicion=superposicion, tipo_ventana=ventana, lstm1=l1, lstm2=l2)
        
        # Analizar los resultados del historial
        val_accuracies = history.history.get('val_accuracy', [0])
        max_val_acc = np.max(val_accuracies)

        threshold_accuracy = 0.95
        if max_val_acc > threshold_accuracy:
            best_epoch_index = np.argmax(val_accuracies)
            best_epoch_number = best_epoch_index + 1
            
            f1_score_key = [key for key in history.history.keys() if 'f1_score' in key and 'val' not in key][0]
            val_f1_score_key = [key for key in history.history.keys() if 'f1_score' in key and 'val' in key][0]
            
            best_metrics = {
                'model_name': artefact_name,
                'best_epoch': best_epoch_number,
                'val_accuracy': max_val_acc,
                'train_accuracy': history.history['accuracy'][best_epoch_index],
                'val_f1_score': history.history[val_f1_score_key][best_epoch_index],
                'train_f1_score': history.history[f1_score_key][best_epoch_index],
                'val_loss': history.history['val_loss'][best_epoch_index],
                'train_loss': history.history['loss'][best_epoch_index]
            }
            print(f"PROCESO FINALIZADO con ÉXITO para {artefact_name}. val_acc: {max_val_acc:.4f}")
            return best_metrics
    except Exception as e:
        print(f"ERROR en el proceso para la combinación {params}: {e}")
    
    # Si el modelo no supera el umbral o falla, devuelve None
    print(f"PROCESO FINALIZADO sin resultado para {abreviar_parametros(l1,l2,filtro,norm,superposicion,ventana)}")
    return None


######################################################################
# EJECUCIÓN DEL EXPERIMENTO PARALELIZADO.
######################################################################

if __name__ == "__main__":
    
    # --- Línea para iniciar el conteo del tiempo total ---
    total_start_time = time.time()

    # 1. Definir el espacio de búsqueda de hiperparámetros
    param_grid = {
        'l1': [3, 4, 5, 6, 7, 8, 9, 10],
        'l2': [3, 4, 5, 6, 7, 8, 9, 10],
        'filtro': #['moving_average', 'exponential', 'savitzky_golay'],
                   ['moving_average', 'exponential'],
        'norm': ['z_score'],#['min_max', 'z_score'],
        'superposicion': [True],
        'ventana': ['rectangular']#['hamming', 'hanning', 'rectangular', 'blackman', 'bartlett']
    }
    
    # 2. Crear una lista con todas las combinaciones posibles
    keys, values = zip(*param_grid.items())
    hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # Convertir a tuplas para que la función starmap las pueda desempaquetar
    params_for_pool = [tuple(combo.values()) for combo in hyperparameter_combinations]

    # 3. Configurar y ejecutar el Pool de procesos
    # Usar cpu_count() - 1 para dejar un núcleo libre para el sistema operativo
    num_processes = cpu_count() - 1
    print(f"Iniciando búsqueda de hiperparámetros con {len(params_for_pool)} combinaciones en {num_processes} procesos paralelos...")
    
    with Pool(processes=num_processes) as pool:
        # Usar starmap para pasar los argumentos desempaquetados a run_experiment
        # El pool distribuirá el trabajo y bloqueará hasta que todos terminen
        results = pool.starmap(run_experiment, [(p,) for p in params_for_pool])

    # --- Línea para detener el conteo del tiempo total ---
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time

    # 4. Filtrar los resultados que no fueron exitosos (devolvieron None)
    best_results = [res for res in results if res is not None]

    # 5. Guardar el resumen final
    if best_results:
        summary_df = pd.DataFrame(best_results)
        summary_df = summary_df.sort_values(by='val_accuracy', ascending=False)
        
        os.makedirs('artefacts', exist_ok=True)
        summary_df.to_csv('artefacts/best_models_summary.csv', index=False)
        
        print("\n" + "="*80)
        print(f"Entrenamiento completado. Se encontraron {len(best_results)} modelos que superaron el umbral.")
        print("El resumen de los mejores modelos se ha guardado en 'artefacts/best_models_summary.csv'")
        # --- Línea para imprimir el tiempo total ---
        print(f"Tiempo total de ejecución del programa: {total_execution_time:.4f} segundos")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("Entrenamiento completado. Ningún modelo superó el umbral de 0.95 en val_accuracy.")
        # --- Línea para imprimir el tiempo total ---
        print(f"Tiempo total de ejecución del programa: {total_execution_time:.4f} segundos")
        print("="*80)