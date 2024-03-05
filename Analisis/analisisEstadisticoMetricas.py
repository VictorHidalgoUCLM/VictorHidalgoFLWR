import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import kruskal
from scipy.stats import f_oneway

import statsmodels.api as sm
from statsmodels.formula.api import ols

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

scenarios = ['Escenario 1', 'Escenario 2', 'Escenario 3']

for scenario in scenarios:
    if scenario == 'Escenario 1' or scenario == 'Escenario 3':
        dispositivos = ['raspberrypi1', 'raspberrypi4', 'raspberrypi6']
        disp_name = ['Rasbperry Pi 1', 'Raspberry Pi 4', 'Raspberry Pi 6']

    else:
        dispositivos = ['raspberrypi1', 'raspberrypi4', 'raspberrypi3']
        disp_name = ['Rasbperry Pi 1', 'Raspberry Pi 4', 'Raspberry Pi 3']

    metricas = ['Net_transmit(B/s)', 'Net_receive(B/s)', 'CPU_usage(%)', 'RAM_usage(%)', 'Swap_usage(%)']

    # Diccionario para almacenar los resultados de las pruebas
    resultados = {'Metrica': [], 'Dispositivo': [], 'Shapiro': [], 'Levene': [], 'ANOVA': [], 'Kruskal': []}

    disp_color = ['tomato', 'greenyellow', 'aqua']
    disp_darkened = [darken_color(mcolors.to_hex(color)) for color in disp_color]

    directorio_figuras = f"/home/usuario/Escritorio/{scenario}/graphMetricas"
    if not os.path.exists(directorio_figuras):
        os.makedirs(directorio_figuras)

    nan = [0] * len(metricas)

    for j, dispositivo in enumerate(dispositivos):
        directorio = f"/home/usuario/Escritorio/{scenario}/metrics/{dispositivo}"

        # Preprocesamiento de los datos
        for metric_log in os.listdir(directorio):
            if not metric_log.endswith("summ.csv"):
                continue

            # Lee el archivo CSV
            df_summ = pd.read_csv(f"{directorio}/{metric_log}")

            # Analizamos cada una de las métricas que queremos
            for i, (metrica) in enumerate(metricas):    
                list_metrics = []
                list_anova = []
                list_zeros = []
                list_unos = []

                for _, datos_grupo in df_summ.groupby('Ejecucion'):
                    if datos_grupo[metrica].nunique() == 1:
                        continue

                    # Test shapiro-wilk -> Normalidad de los datos
                    estadistico, p_valor = shapiro(datos_grupo[metrica])
                    
                    if p_valor < 0.05:
                        resultado_shapiro = "No pasa"
                    else:
                        resultado_shapiro = "Pasa"

                    list_metrics.append(datos_grupo[metrica])

                if len(list_metrics) >= 2:
                    # Test levene para comprobar homocedasticidad de los datos -> varianzas
                    estadistico, p_valor = levene(*list_metrics)
                    if p_valor < 0.05:
                        resultado_levene = "No pasa"
                    else:
                        resultado_levene = "Pasa"
                    
                    if p_valor > 0.05:
                        estadistico, p_valor = f_oneway(*list_metrics)

                        if p_valor < 0.05:
                            resultado_ANOVA = "No pasa"
                        else:
                            resultado_ANOVA = "Pasa"
                        resultado_kruskal = None
                    else:
                        resultado_ANOVA = None

                    estadistico, p_valor = kruskal(*list_metrics)

                    if p_valor < 0.05:
                        resultado_kruskal = "No pasa"
                    else:
                        resultado_kruskal = "Pasa"

                    # Almacenar los resultados en el diccionario
                    resultados['Metrica'].append(metrica)
                    resultados['Dispositivo'].append(dispositivo)
                    resultados['Shapiro'].append(resultado_shapiro)
                    resultados['Levene'].append(resultado_levene)
                    resultados['ANOVA'].append(resultado_ANOVA)
                    resultados['Kruskal'].append(resultado_kruskal)

                else:
                    nan[i] += 1

    posiciones = [i for i, x in enumerate(nan) if x == 3]

    """Creación de las gráficas de análisis de los datos"""
    # Crear subplots para cada combinación de métrica y dispositivo
    fig, axs = plt.subplots(nrows=len(metricas)-len(posiciones), ncols=len(dispositivos), figsize=(18, 11), layout='constrained')
    fig2, axs2 = plt.subplots(nrows=len(metricas)-len(posiciones), ncols=len(dispositivos), figsize=(18, 11), layout='constrained')
    fig3, axs3 = plt.subplots(nrows=len(metricas)-len(posiciones), figsize=(18, 11), layout='constrained', sharex=True)

    for j, dispositivo in enumerate(dispositivos):
        directorio = f"/home/usuario/Escritorio/{scenario}/metrics/{dispositivo}"

        # Preprocesamiento de los datos
        for metric_log in os.listdir(directorio):
            if not metric_log.endswith("summ.csv"):
                continue

            # Lee el archivo CSV
            df_summ = pd.read_csv(f"{directorio}/{metric_log}")
            
            # Analizamos cada una de las métricas que queremos
            for i, (metrica) in enumerate(metricas):
                if i in posiciones:
                    i -= 1
                    continue

                ax = axs[i][j]
                ax2 = axs2[i][j]

                for _, datos_grupo in df_summ.groupby('Ejecucion'):
                    if datos_grupo[metrica].nunique() == 1:
                        continue

                if j == 0 and metrica == 'Net_receive(B/s)':
                    cambios_ronda = sorted(df_summ.loc[df_summ['Ejecucion']==1][metrica].nlargest(60).index)

                means = df_summ.groupby('Ejecucion')[metrica].mean()

                sns.histplot(df_summ.loc[df_summ['Ejecucion']==1][metrica], ax=ax2, color=disp_color[j])
                sns.boxplot(x='Ejecucion', y=metrica, data=df_summ, ax=ax, color=disp_color[j])

                for l, mean in enumerate(means):
                    ax.scatter(l, mean, marker='x', s=125, facecolors='black', zorder=3)

                if j == len(dispositivos)-1:
                    ax3 = axs3[i]
                    sns.lineplot(df_summ.loc[df_summ['Ejecucion']==1], x=[i * 5 for i in range(1, len(df_summ.loc[df_summ['Ejecucion']==1]) + 1)], y=metrica, ax=ax3)
                    ax3.set_ylabel(metrica, fontsize=16)
                    ax3.set_xlabel('Tiempo (s)')
                    ax3.set_xlim(0, (len(df_summ.loc[df_summ['Ejecucion']==1]) + 1)*5)

                    # Añadir líneas verticales en las posiciones especificadas
                    for n, posicion in enumerate(cambios_ronda):
                        if n%2 == 0:
                            ax3.axvline(x=posicion*5, color='r', linestyle='--', linewidth=0.7)  # Puedes personalizar el color, estilo y grosor de la línea aquí

                if i == 0:
                    ax.set_title(f'{disp_name[j].capitalize()}', fontsize=16, pad=16)
                    ax2.set_title(f'{disp_name[j].capitalize()}', fontsize=16, pad=16)

                if i == len(axs)-1:
                    ax.set_xlabel('Valores')
                    ax2.set_xlabel('Valores')
                else:
                    ax.set_xlabel('')
                    ax2.set_xlabel('')

                if j == 0:
                    ax.set_ylabel(metrica.capitalize(), fontsize=16)
                    ax2.set_ylabel(metrica.capitalize(), fontsize=16)
                else:
                    ax.set_ylabel('')
                    ax2.set_ylabel('')
                    

    fig.savefig(f'{directorio_figuras}/boxandwhispers.png', dpi=300)
    fig2.savefig(f'{directorio_figuras}/histogram.png', dpi=300)
    fig3.savefig(f'{directorio_figuras}/metrics.png', dpi=300)

    """Tabla"""
    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.set_index('Dispositivo').loc[dispositivos].reset_index()
    df_resultados['Metrica'] = df_resultados['Dispositivo'] + '_' + df_resultados['Metrica']
    df_resultados.drop(columns=['Dispositivo'], inplace=True)


    # Crear la figura y el eje
    fig, ax = plt.subplots(figsize=(8, 4))

    # Ocultar los ejes
    ax.axis('off')

    colors = ["#65e830", "#e8be30", "#e84c30", "#cccccc"]
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
                else:
                    cell.set_facecolor(actual_colors[3])

    # Iterar sobre las cabeceras de fila y aplicar el color
    for i in range(len(df_resultados.index)):  # Iterar sobre las etiquetas de fila
        cell = tabla[(i + 1, 0)]  # Índices de la fila i y última columna
        if i%2 == 0:
            cell.set_facecolor(grey_color)  # Cambia '#your_color' con el color deseado


    plt.savefig(f'{directorio_figuras}/tabla.png', dpi=300)