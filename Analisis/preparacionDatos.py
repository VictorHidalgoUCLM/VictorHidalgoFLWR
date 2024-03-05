import pandas as pd
import os

scenarios = ["Escenario 1", "Escenario 2", "Escenario 3"]

for scenario in scenarios:
    directorio_logs = f"/home/usuario/Escritorio/{scenario}/logs"
    directorio_metrics = f"/home/usuario/Escritorio/{scenario}/metrics"

    if scenario == "Escenario 1" or scenario == "Escenario 3":
        dispositivos = ['raspberrypi1', 'raspberrypi4', 'raspberrypi6']
    else:
        dispositivos = ['raspberrypi1', 'raspberrypi4', 'raspberrypi3']

    df_logs = pd.DataFrame()

    # Preprocesamiento de los datos
    for i, (archivo) in enumerate(os.listdir(directorio_logs)):
        # Lee el archivo CSV
        df_logs_raw = pd.read_csv(f"{directorio_logs}/{archivo}")
        df_logs_raw['Ejecucion'] = i+1

        df_logs = pd.concat([df_logs, df_logs_raw], ignore_index=True)

    df_logs.to_csv(f"{directorio_logs}/summ.csv", index=False)

    # Bucle que almacena todas las ejecuciones en un mismo dataframe 
    # y lo guarda en un .csv para analizarlo después
    for dispositivo in dispositivos:
        df_total = pd.DataFrame()
        list_raw_metrics = []
        directorio_metrics_disp = directorio_metrics + "/" + f"{dispositivo}"

        # Preprocesamiento de los datos
        for i, (metric_log) in enumerate(os.listdir(directorio_metrics_disp)):
            df_metrics = pd.DataFrame()

            if i == 0:
                df_fixed = pd.read_csv(f"{directorio_metrics_disp}/{metric_log}", usecols=[7, 8, 9], nrows=0)
                list_fixed = df_fixed.columns.to_list()

                cpu_total = int(list_fixed[0])
                ram_total = int(list_fixed[1])
                swap_total = int(list_fixed[2])

            # Lee el archivo CSV
            df_raw_metrics = pd.read_csv(f"{directorio_metrics_disp}/{metric_log}", usecols=[0, 1, 2, 3, 4, 5])

            # Preprocesamos los datos y los almacenamos en df_metrics
            df_metrics['Time_stamp(s)'] = df_raw_metrics['Time_stamp(s)'].diff().dropna().astype(pd.Int64Dtype())
            df_metrics['Net_transmit(B/s)'] = (df_raw_metrics['Net_transmit(B)'].diff().dropna() / df_metrics['Time_stamp(s)']).astype(pd.Float64Dtype()).clip(lower=0)
            df_metrics['Net_receive(B/s)'] = (df_raw_metrics['Net_receive(B)'].diff().dropna() / df_metrics['Time_stamp(s)']).astype(pd.Float64Dtype()).clip(lower=0)
            df_metrics['CPU_usage(%)'] = (((df_raw_metrics['CPU_time(s)'].diff().dropna() / df_metrics['Time_stamp(s)']) / cpu_total) * 100).astype(pd.Float64Dtype()).clip(lower=0, upper=100)
            df_metrics['RAM_usage(MB)'] = (df_raw_metrics['RAM_usage(B)'] / 1024 / 1024).astype(pd.Float64Dtype())
            df_metrics['RAM_usage(%)'] = (df_raw_metrics['RAM_usage(B)'] / ram_total * 100).astype(pd.Float64Dtype()).clip(lower=0, upper=100)
            df_metrics['Swap_usage(MB)'] = ((swap_total - df_raw_metrics['Swap_free(B)']) / 1024 / 1024).astype(pd.Float64Dtype())
            df_metrics['Swap_usage(%)'] = ((1 - df_raw_metrics['Swap_free(B)'] / swap_total) * 100).astype(pd.Float64Dtype()).clip(lower=0, upper=100)
            df_metrics['Ejecucion'] = i+1

            df_total = pd.concat([df_total, df_metrics], ignore_index=True)

        df_total.to_csv(f"{directorio_metrics_disp}/summ.csv", index=False)