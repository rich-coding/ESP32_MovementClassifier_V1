import serial as sr
import time
import csv

def guardar_datos_serial(puerto='COM5', baudrate=115200, timeout=1, num_datos=100):
    """
    Captura datos del puerto serial y los guarda en un archivo CSV.

    Args:
        puerto (str): El puerto serial al que está conectado el dispositivo (e.g., '/dev/ttyUSB0' en Linux, 'COM3' en Windows).
        baudrate (int): La velocidad de comunicación del puerto serial.
        timeout (int): El tiempo de espera para leer del puerto serial en segundos.
        num_datos (int): El número de líneas de datos a capturar.
    """
    try:
        # Inicializar la conexión serial
        ser = sr.Serial(puerto, baudrate, timeout=timeout)
        time.sleep(2)  # Esperar a que la conexión serial se establezca
        print(f"Conectado al puerto {puerto}. Comenzando a recibir datos...")

    except sr.SerialException as e:
        print(f"Error al abrir el puerto serial: {e}")
        return

    # Abrir el archivo para escribir los datos en formato CSV
    with open("datos_sensor.csv", "w", newline='') as file:
        writer = csv.writer(file)

        # Escribir el encabezado del archivo CSV
        writer.writerow(["A_X_ms2", "A_Y_ms2", "A_Z_ms2", "G_X_rads", "G_Y_rads", "G_Z_rads"])

        try:
            # Leer y guardar el número de líneas de datos especificado
            for i in range(num_datos):
                # Leer una línea del puerto serial y decodificarla
                linea = ser.readline().decode('utf-8').strip()

                if linea:
                    # Dividir la línea en valores usando ',' como delimitador
                    # Asegurarse de que el formato de salida del Arduino sea: A_X,A_Y,A_Z,G_X,G_Y,G_Z
                    valores = linea.split(',')
                    
                    # Validar que se hayan recibido 6 valores
                    if len(valores) == 6:
                        # Convertir los valores a números (flotantes)
                        datos = [float(val) for val in valores]
                        
                        # Escribir los datos en el archivo CSV
                        writer.writerow(datos)
                        print(f"Datos guardados: {datos}")
                    else:
                        print(f"Línea de datos incompleta o incorrecta: '{linea}'")

        except Exception as e:
            print(f"Error al leer del puerto serial o al procesar datos: {e}")

        finally:
            ser.close()
            print("Conexión serial cerrada.")
            print("Datos guardados en 'datos_sensor.csv'.")

# Ejecutar la función cuando el script se ejecute directamente
if __name__ == "__main__":
    # Cambia el puerto serial si es necesario (e.g., 'COM3' en Windows)
    guardar_datos_serial(puerto='COM5', num_datos=200)
