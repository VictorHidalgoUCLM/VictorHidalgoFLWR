import pandas as pd
import os

from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import f_oneway
from scipy.stats import kruskal

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import matplotlib.ticker as ticker

plt.rcParams["font.family"] = "serif"

def hex_to_rgb(hex_color):
    # Eliminar el caracter '#' si está presente
    hex_color = hex_color.lstrip('#')
    # Convertir los valores hexadecimales a valores RGB
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    # Convertir los valores RGB a código hexadecimal
    return '#{:02x}{:02x}{:02x}'.format(*rgb_color)

def darken_color(hex_color, factor=0.8):
    # Convertir el color hexadecimal a RGB
    r, g, b = hex_to_rgb(hex_color)
    # Oscurecer los valores de RGB
    r_nuevo = int(r * factor)
    g_nuevo = int(g * factor)
    b_nuevo = int(b * factor)
    # Asegurarse de que los valores estén en el rango válido (0-255)
    r_nuevo = max(0, min(255, r_nuevo))
    g_nuevo = max(0, min(255, g_nuevo))
    b_nuevo = max(0, min(255, b_nuevo))
    # Convertir los nuevos valores RGB a código hexadecimal
    return rgb_to_hex((r_nuevo, g_nuevo, b_nuevo))

def format_yticks(y, pos):
    return f'{y:.2f}'


metricas = ['accuracy', 'loss', 'precision', 'recall', 'f1_score']
disp_color = ['tomato', 'greenyellow', 'aqua', 'violet']

scenarios = ['FedAvg']

for scenario in scenarios:
    if scenario == 'Escenario 1' or scenario == 'Escenario 3':
        dispositivos = ['Global', 'raspberrypi1_ev', 'raspberry4_ev', 'raspberry6_ev']
        disp_name = ['Modelo agregado', 'Cliente 1 (RP4)', 'Cliente 2 (RP4)', 'Cliente 3 (RP4)']

    elif scenario == 'Escenario 2':
        dispositivos = ['Global', 'raspberrypi1_ev', 'raspberry4_ev', 'raspberry3_ev']
        disp_name = ['Modelo agregado', 'Cliente 1 (RP4)', 'Cliente 2 (RP4)', 'Cliente 4 (RP3)']
    else:
        dispositivos = ['Global', 'raspberrypi1_ev', 'raspberry4_ev', 'raspberry3_ev']
        disp_name = ['Modelo agregado', 'Cliente 1 (RP4)', 'Cliente 2 (RP4)', 'Cliente 3 (RP3)']

    directorio = f"/home/usuario/Escritorio/results/{scenario}/logs"

    directorio_figuras = f"/home/usuario/Escritorio/results/{scenario}/graph"
    if not os.path.exists(directorio_figuras):
        os.makedirs(directorio_figuras)

    df_logs = pd.read_csv(f"{directorio}/summ.csv")

    disp_darkened = [darken_color(mcolors.to_hex(color)) for color in disp_color]

    # Crear subplots para cada combinación de métrica y dispositivo
    fig, axs = plt.subplots(nrows=len(metricas), ncols=len(dispositivos), figsize=(18, 9), sharex=True, layout='constrained')
    fig2, axs2 = plt.subplots(nrows=len(metricas), ncols=len(dispositivos), figsize=(18, 9), sharex=True, layout='constrained')

    # Diccionario para almacenar los resultados de las pruebas
    resultados = {'Metrica': [], 'Dispositivo': [], 'Shapiro': [], 'Levene': [], 'ANOVA': [], 'Kruskal': []}

    for i, metrica in enumerate(metricas):
        if i%2 == 0:
            actual_colors = disp_darkened
        else:
            actual_colors = disp_color

        for j, dispositivo in enumerate(dispositivos):
            ax = axs[i][j]
            ax2 = axs2[i][j]

            col_name = f"{dispositivo}_{metrica}"

            lista = []

            for _, datos_grupo in df_logs.groupby('Ejecucion'):
                # Test shapiro-wilk -> Normalidad de los datos
                estadistico, p_valor = shapiro(datos_grupo[col_name])
                if p_valor > 0.05:
                    resultado_shapiro = "Pasa"
                else:
                    resultado_shapiro = "No pasa"
                    
                lista.append(datos_grupo[col_name])

            sns.histplot(df_logs.loc[df_logs['Ejecucion']==1][col_name], ax=ax2, color=actual_colors[j])

            if i == 0:
                ax2.set_title(f'{disp_name[j]}', fontsize=16, pad=16)
            else:
                ax2.set_xlabel('Valores')

            if j == 0:
                ax2.set_ylabel(metrica, fontsize=16)
            else:
                ax2.set_ylabel('', fontsize=16)

            # Test levene para comprobar homocedasticidad de los datos -> varianzas
            estadistico, p_valor = levene(*lista)
            if p_valor > 0.05:
                resultado_levene = "Pasa"

                
                resultado_ANOVA = f_oneway(*lista)

                if resultado_ANOVA.pvalue < 0.05:
                    resultado_ANOVA = "No pasa"
                else:
                    resultado_ANOVA = "Pasa"

            else:
                resultado_levene = "No pasa"
                resultado_ANOVA = "----"

            try:
                if resultado_shapiro == "No pasa" or resultado_levene == "No pasa":
                    estadistico, p_valor = kruskal(*lista)

                    if p_valor > 0.05:
                        resultado_kruskal = "Pasa"
                    else:
                        resultado_kruskal = "No pasa"

                else:
                    resultado_kruskal = "----"
                    
            except ValueError as e:
                resultado_kruskal = "No pasa"
                print("Error al realizar la prueba de Kruskal-Wallis:", e)
                pass

            # Almacenar los resultados en el diccionario
            resultados['Metrica'].append(metrica)
            resultados['Dispositivo'].append(dispositivo)
            resultados['Shapiro'].append(resultado_shapiro)
            resultados['Levene'].append(resultado_levene)
            resultados['ANOVA'].append(resultado_ANOVA)
            resultados['Kruskal'].append(resultado_kruskal)

            sns.boxplot(x='Ejecucion', y=col_name, data=df_logs, ax=ax, color=actual_colors[j])
            ax.set_ylim(0, df_logs[col_name].max() * 1.1)

            if i == 0:
                ax.set_title(f'{disp_name[j]}', fontsize=16, pad=16)
            else:
                ax.set_xlabel('Ejecución')

            if j == 0:
                ax.set_ylabel(metrica, fontsize=16)
            else:
                ax.set_ylabel('', fontsize=16)

    df_ejecucion_1 = df_logs.loc[df_logs['Ejecucion']==1].drop(columns=['Ejecucion'])

    max_Global_accuracy = df_ejecucion_1["Global_accuracy"].max()
    min_Global_loss = df_ejecucion_1["Global_loss"].min()

    for i, metrica in enumerate(metricas):
        for j, dispositivo in enumerate(dispositivos):
            ax = axs[i][j]
            ax2 = axs2[i][j]

            if i%2 == 0:
                max_accuracy = df_ejecucion_1[dispositivo+"_"+metrica].max()

                yticks = list(ax.get_yticks()[:-2]) + [max_accuracy]

            else:
                min_loss = df_ejecucion_1[dispositivo+"_"+metrica].min()

                yticks = [min_loss] + list(ax.get_yticks()[2:])
                
            ax.set_yticks(yticks)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_yticks))

    fig.savefig(f'{directorio_figuras}/boxandwhispers.png', dpi=150)
    fig2.savefig(f'{directorio_figuras}/histogram.png', dpi=150)

    # Crear un DataFrame a partir del diccionario
    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.set_index('Dispositivo').loc[dispositivos].reset_index()
    df_resultados['Metrica'] = df_resultados['Dispositivo'].str.replace('_ev', '') + '_' + df_resultados['Metrica']
    df_resultados.drop(columns=['Dispositivo'], inplace=True)


    """Creación de la tabla que resume los test estadísticos"""
    # Crear la figura y el eje
    fig, ax = plt.subplots(figsize=(8, 6))

    # Ocultar los ejes
    ax.axis('off')

    colors = ["#65e830", "#e8be30", "#e84c30", "#add8e6"]
    darkened_colors = [darken_color(color) for color in colors]
    grey_color = darken_color("#ffffff")

    # Crear la tabla
    tabla = ax.table(cellText=df_resultados.values,
                    colLabels=df_resultados.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.12 for _ in df_resultados.columns])

    tabla.auto_set_font_size(False)
    tabla.set_fontsize(12)
    tabla.scale(1.5, 1.5)  # Escalar la tabla
    tabla.auto_set_column_width(0)

    # Iterar sobre las celdas y aplicar el color según las condiciones
    for i in range(len(df_resultados)):  # Iterar sobre filas
        if i%2 == 0:
            actual_colors = colors
        else:
            actual_colors = darkened_colors

        for j in range(len(df_resultados.columns)):  # Iterar sobre columnas
            cell = tabla.get_celld()[(i + 1, j)]  # +1 para saltar la fila de encabezados
            text = cell.get_text().get_text()
            if df_resultados.columns[j] == 'Shapiro':
                if text == 'Pasa':
                    cell.set_facecolor(actual_colors[0])
                elif text == 'No pasa':
                    cell.set_facecolor(actual_colors[1])
            else:
                if text == 'Pasa':
                    cell.set_facecolor(actual_colors[0])
                elif text == 'No pasa':
                    cell.set_facecolor(actual_colors[2])
                elif text == '----':
                    cell.set_facecolor(actual_colors[3])

    # Iterar sobre las cabeceras de fila y aplicar el color
    for i in range(len(df_resultados.index)):  # Iterar sobre las etiquetas de fila
        cell = tabla[(i + 1, 0)]  # Índices de la fila i y última columna
        if i%2 != 0:
            cell.set_facecolor(grey_color)  # Cambia '#your_color' con el color deseado

    plt.savefig(f'{directorio_figuras}/tabla.png', dpi=150)


    """Creación de gráfica que muestra accuracy y loss de la ejecución 1"""
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 4), sharex=True)

    sns.lineplot(x=list(range(1, len(df_ejecucion_1)+1)), y='Global_accuracy', data=df_ejecucion_1, ax=ax1, label='Global accuracy', color='limegreen')
    ax1.axhline(y=max_Global_accuracy, color='black', linestyle='--', linewidth=0.7, alpha=0.7)

    yticks = list(ax1.get_yticks()[:-2]) + [max_Global_accuracy]
    ax1.set_yticks(yticks)

    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0,1)
    ax1.set_xlim(1,60)
    ax1.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))


    sns.lineplot(x=list(range(1, len(df_ejecucion_1)+1)), y='Global_loss', data=df_ejecucion_1, ax=ax2, label='Global loss', color='cornflowerblue')
    ax2.axhline(y=min_Global_loss, color='black', linestyle='--', linewidth=0.7, alpha=0.7)

    yticks = [min_Global_loss] + list(ax2.get_yticks()[2:])
    ax2.set_yticks(yticks)

    ax2.set_xlabel('Ronda de Federated Learning')
    ax2.set_ylabel('Loss')
    ax2.set_ylim(0, df_ejecucion_1['Global_loss'].max() + 0.1)
    ax2.set_xlim(1,60)

    plt.tight_layout()
    plt.savefig(f'{directorio_figuras}/model.png', dpi=150)