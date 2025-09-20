import pandas as pd
import os

def merge_csv_files(input_folder="data", output_file="datos_mpu6050_out.csv"):
    """
    Combina todos los archivos CSV de una carpeta en un solo archivo.
    """
    # Se crea una lista para guardar todos los DataFrames
    all_dataframes = []

    # Lista de los 5 movimientos que esperamos
    movimientos_esperados = range(5)
    
    # Se itera sobre los archivos en la carpeta de entrada
    for i in movimientos_esperados:
        file_path = os.path.join(input_folder, f"datos_mpu6050_class_{i}.csv")
        if os.path.exists(file_path):
            print(f"Leyendo el archivo: {file_path}")
            df = pd.read_csv(file_path)
            all_dataframes.append(df)
        else:
            print(f"ADVERTENCIA: No se encontró el archivo {file_path}. ¿Has recolectado datos para todos los movimientos?")

    # Concatenar todos los DataFrames en uno solo
    if all_dataframes:
        merged_dataframe = pd.concat(all_dataframes, ignore_index=True)
        # Guardar el DataFrame combinado en un nuevo archivo CSV
        merged_dataframe.to_csv(f".\data\{output_file}", index=False)
        print(f"\n¡Archivos combinados exitosamente! El resultado se guardó en: {output_file}")
        print(f"Total de registros combinados: {len(merged_dataframe)}")
    else:
        print("\nNo se encontraron archivos para combinar. Por favor, asegúrate de que los archivos estén en la carpeta 'data'.")

if __name__ == "__main__":
    merge_csv_files()