import pandas as pd
import matplotlib.pyplot as plt
import os

def visualizar_datos():
    """
    Lee datos de un archivo CSV y genera gráficos de la aceleración y la rotación.
    """
    # Verificar si el archivo existe
    if not os.path.exists("data/datos_sensor.csv"):
        print("Error: No se encontró el archivo 'data/datos_sensor.csv'.")
        print("Por favor, ejecuta el script 'guardar_datos_serial.py' primero para generar los datos.")
        return

    # Leer el archivo CSV usando pandas
    try:
        df = pd.read_csv("data/datos_sensor.csv")
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return

    # Crear una figura con dos subgráficos (uno encima del otro)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Gráfico para la Aceleración
    ax1.plot(df.index, df['A_X_ms2'], label='A_X (m/s²)')
    ax1.plot(df.index, df['A_Y_ms2'], label='A_Y (m/s²)')
    ax1.plot(df.index, df['A_Z_ms2'], label='A_Z (m/s²)')
    ax1.set_title('Datos del Acelerómetro')
    ax1.set_ylabel('Aceleración (m/s²)')
    ax1.legend()
    ax1.grid(True)

    # Gráfico para el Giroscopio
    ax2.plot(df.index, df['G_X_rads'], label='G_X (rad/s)')
    ax2.plot(df.index, df['G_Y_rads'], label='G_Y (rad/s)')
    ax2.plot(df.index, df['G_Z_rads'], label='G_Z (rad/s)')
    ax2.set_title('Datos del Giroscopio')
    ax2.set_xlabel('Muestras')
    ax2.set_ylabel('Velocidad Angular (rad/s)')
    ax2.legend()
    ax2.grid(True)

    # Ajustar el diseño para evitar que los títulos se superpongan
    plt.tight_layout()

    # Mostrar los gráficos
    plt.show()

# Ejecutar la función para visualizar los datos
if __name__ == "__main__":
    visualizar_datos()
