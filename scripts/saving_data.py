import serial
import time
import csv
import os

def guardar_datos_serial(puerto='COM5', baudrate=115200, timeout=1, num_datos=100):
    """
    Captura datos de un sensor (ej. MPU6050) desde un puerto serial y los 
    guarda en un archivo CSV, etiquetándolos con una categoría de movimiento.

    Se espera que el dispositivo envíe 7 valores numéricos en una sola línea,
    separados por comas. Por ejemplo: "ax,ay,az,gx,gy,gz,temp".

    Args:
        puerto (str): El puerto serial al que está conectado el dispositivo (ej. 'COM3' en Windows).
        baudrate (int): La velocidad de comunicación del puerto serial.
        timeout (float): El tiempo de espera en segundos para leer del puerto serial.
        num_datos (int): El número de conjuntos de datos a capturar por categoría.
    """
    try:
        categoria_input = input("Introduce la categoría del movimiento (0-4): ")
        categoria = int(categoria_input)
        if categoria not in range(5):
            print("Error: La categoría debe ser un número entero entre 0 y 4.")
            return
    except ValueError:
        print("Error: Entrada no válida. Por favor, introduce un número entero.")
        return

    # Crear carpeta 'data' si no existe
    carpeta_data = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(carpeta_data, exist_ok=True)

    nombre_archivo = f"datos_mpu6050_class_{categoria}.csv"
    ruta_archivo = os.path.join(carpeta_data, nombre_archivo)
    file_exists = os.path.isfile(ruta_archivo)

    try:
        with serial.Serial(puerto, baudrate, timeout=timeout) as ser:
            time.sleep(2)
            print(f"Conectado al puerto {puerto}. Presiona el botón de reset en tu ESP32 si no recibes datos.")
            ser.reset_input_buffer()

            with open(ruta_archivo, "a", newline='') as file:
                writer = csv.writer(file)

                if not file_exists:
                    writer.writerow(["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "temp_c", "category"])

                print(f"\nRecolectando {num_datos} muestras para  categoría {categoria}. Mantén el movimiento...")
                
                datos_recolectados = 0
                while datos_recolectados < num_datos:
                    if ser.in_waiting > 0:
                        try:
                            linea_datos = ser.readline().decode('utf-8').strip()

                            if not linea_datos:
                                continue

                            parts = linea_datos.split(',')
                            if len(parts) == 7:
                                datos = [float(p) for p in parts]
                                datos.append(categoria)
                                
                                writer.writerow(datos)
                                print(f"Muestra {datos_recolectados + 1}/{num_datos}: {datos}")
                                datos_recolectados += 1
                            else:
                                print(f"Línea ignorada (formato incorrecto, {len(parts)} valores): {linea_datos}")

                        except (ValueError, UnicodeDecodeError) as e:
                            print(f"Error procesando línea. Ignorando. Detalle: {e}")
                        except Exception as e:
                            print(f"Ocurrió un error inesperado: {e}")
                            break

    except serial.SerialException as e:
        print(f"Error crítico: No se pudo abrir o leer el puerto serial '{puerto}'. Verifica la conexión. Detalle: {e}")
        return
    
    print(f"\nRecolección completada. {datos_recolectados} muestras guardadas en '{ruta_archivo}'.")

if __name__ == "__main__":
    puerto_esp32 = 'COM5' 
    muestras_a_tomar = 1000
    
    guardar_datos_serial(puerto=puerto_esp32, num_datos=muestras_a_tomar)