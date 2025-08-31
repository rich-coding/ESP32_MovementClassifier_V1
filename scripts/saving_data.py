import serial
import time
import csv

def guardar_datos_serial(puerto='COM5', baudrate=115200, timeout=1, num_datos=100):
    """
    Captura datos del puerto serial y los guarda en un archivo CSV.

    Args:
        puerto (str): El puerto serial al que está conectado el dispositivo (e.g., '/dev/ttyUSB0' en Linux, 'COM3' en Windows).
        baudrate (int): La velocidad de comunicación del puerto serial.
        timeout (int): El tiempo de espera para leer del puerto serial en segundos.
        num_datos (int): El número de conjuntos de datos a capturar.
    """
    try:
        # Inicializar la conexión serial
        ser = serial.Serial(puerto, baudrate, timeout=timeout)
        time.sleep(2)  # Esperar a que la conexión serial se establezca
        print(f"Conectado al puerto {puerto}. Comenzando a recibir datos...")

    except serial.SerialException as e:
        print(f"Error al abrir el puerto serial: {e}")
        return

    # Abrir el archivo para escribir los datos en formato CSV
    with open("data\datos_sensor.csv", "w", newline='') as file:
        writer = csv.writer(file)

        # Escribir el encabezado del archivo CSV
        writer.writerow(["A_X_ms2", "A_Y_ms2", "A_Z_ms2", "G_X_rads", "G_Y_rads", "G_Z_rads"])
        
        # Descartar las primeras líneas de inicialización o vacías
        print("Esperando la primera línea de datos válida...")
        while True:
            try:
                linea = ser.readline().decode('utf-8').strip()
                if linea.startswith("A (m/s2):"):
                    print("Primera línea de datos recibida. Comenzando la captura.")
                    break
            except Exception as e:
                # Omitir errores en las líneas iniciales que no son datos
                pass

        try:
            # Leer y guardar el número de conjuntos de datos especificado
            datos_guardados = 0
            while datos_guardados < num_datos:
                # Leer una línea del puerto serial
                linea_a = ser.readline().decode('utf-8').strip()

                # Si la línea es de aceleración, leemos la siguiente para giroscopio
                if linea_a.startswith("A (m/s2):"):
                    linea_g = ser.readline().decode('utf-8').strip()

                    if linea_g.startswith("G (rad/s):"):
                        # Extraer los valores numéricos
                        valores_a = linea_a.replace("A (m/s2): ", "").split(', ')
                        valores_g = linea_g.replace("G (rad/s): ", "").split(', ')

                        # Asegurarse de que ambas listas tienen 3 valores
                        if len(valores_a) == 3 and len(valores_g) == 3:
                            # Convertir los valores a números (flotantes) y unirlos
                            datos = [float(val) for val in valores_a + valores_g]
                            
                            # Escribir los datos en el archivo CSV
                            writer.writerow(datos)
                            print(f"Datos guardados: {datos}")
                            datos_guardados += 1
                        else:
                            print(f"Líneas de datos con formato incorrecto. Ignorando este ciclo.")
                    else:
                        print(f"Línea de giroscopio inesperada después de acelerómetro: '{linea_g}'")
                else:
                    print(f"Línea inesperada descartada: '{linea_a}'")


        except Exception as e:
            print(f"Error al leer del puerto serial o al procesar datos: {e}")

        finally:
            ser.close()
            print("Conexión serial cerrada.")
            print("Datos guardados en 'data/datos_sensor.csv'.")

# Ejecutar la función cuando el script se ejecute directamente
if __name__ == "__main__":
    guardar_datos_serial(puerto='COM5', num_datos=200)
