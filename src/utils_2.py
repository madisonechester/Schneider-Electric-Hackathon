import pandas as pd
import os



def get_generation_files(country, data_folder):
    file_names = os.listdir(data_folder)
    file_paths = [os.path.join(data_folder, file_name) for file_name in file_names if file_name.startswith(f'gen_{country}')]
    return file_paths


def get_load_file(country, data_folder):
    file_names = os.listdir(data_folder)
    for file in file_names:
        if file.startswith(f'load_{country}'):
            return os.path.join(data_folder, file)


import pandas as pd
def get_hours_in_year():
    
    # Especifica el rango de fechas desde el primer día de 2022 hasta el último
    start_date = "2022-01-01 00:00:00"
    end_date = "2022-12-31 23:00:00"
    # Crea un rango de fechas cada hora
    start_time_range = pd.date_range(start=start_date, end=end_date, freq='H')
    # Convierte el rango de fechas en un array de strings en el formato deseado
    start_time_array = start_time_range.strftime('%Y-%m-%d %H:%M:%S%z').to_numpy()


    # Especifica el rango de fechas desde el primer día de 2022 hasta el último
    start_date = "2022-01-01 01:00:00"
    end_date = "2023-01-01 00:00:00"
    # Crea un rango de fechas cada hora
    end_time_range = pd.date_range(start=start_date, end=end_date, freq='H')
    # Convierte el rango de fechas en un array de strings en el formato deseado
    end_time_array = end_time_range.strftime('%Y-%m-%d %H:%M:%S%z').to_numpy()
    
    return start_time_array, end_time_array
