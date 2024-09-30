#Universidad del Valle de Guatemala
#Metodos Matematicos para la Fisica II. Profesora: Yasmin Portillo Chang.
#Proyecto Funciones Beta y Gamma.
#---------------------------------------------------------------------#
#Integrantes:
#--Evelyn Fernanda López Peiro--21126
#--Pedro José Camposeco Ovalle--21360
#--Angel Josue Nij Culajay------21437
#---------------------------------------------------------------------#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from scipy.special import gamma

# Definición de la función Gamma#
def gamma_function(x, alpha, beta):
    return (beta ** alpha) * (x ** (alpha - 1)) * np.exp(-beta * x) / gamma(alpha)

# Definición de la función Beta#
def beta_function(x, alpha, beta):
    return (x ** (alpha - 1)) * ((1 - x) ** (beta - 1)) / (gamma(alpha) * gamma(beta) / gamma(alpha + beta))

# Importación de los archivos y lectura de los mismos:
gamma_file = 'C:\\Users\\Angel\\Desktop\\Histogramas de la distribucion Gamma.csv'
beta_file = 'C:\\Users\\Angel\\Desktop\\Histogramas de la distribucion beta.csv' 
gamma_data = pd.read_csv(gamma_file)
beta_data = pd.read_csv(beta_file)
gamma_data.columns = gamma_data.columns.str.strip()
beta_data.columns = beta_data.columns.str.strip()


results_list = []
#Funcion Gamma:---------------#
for i in range(1, 5):
    x_data = gamma_data['BinEdge']
    y_data = gamma_data[f'Histogram{i}']
    valid_indices = x_data > 0
    x_data = x_data[valid_indices]
    y_data = y_data[valid_indices]
    #Aplicacion de la funcion gamma:
    popt, _ = curve_fit(gamma_function, x_data, y_data, p0=(1, 1))
    # Parámetros optimizados
    alpha_opt, beta_opt = popt
    print(f'Histograma Gamma {i}: Parámetros optimizados: alpha = {alpha_opt}, beta = {beta_opt}')
    y_fit = gamma_function(x_data, alpha_opt, beta_opt)
    rmse = np.sqrt(mean_squared_error(y_data, y_fit))
    print(f'Histograma Gamma {i}: RMSE: {rmse}')
    results_list.append({'Histograma': f'Gamma {i}', 'Alpha': alpha_opt, 'Beta': beta_opt, 'RMSE': rmse})
    # Graficar de los datos:
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, marker='o', label='Datos de la función Gamma', color='blue', markersize=5)
    plt.plot(x_data, y_fit, label='Ajuste Gamma', color='red')
    plt.title(f'Ajuste de la Función Gamma - {results_list[-1]["Histograma"]}')
    plt.xlabel('Bin Edge')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid()
    plt.show()

# Función Beta--------#
for i in range(1, 5):
    x_data = beta_data['BinEdge']
    y_data = beta_data[f'Histogram{i}']
    valid_indices = (x_data > 0) & (x_data < 1)
    x_data = x_data[valid_indices]
    y_data = y_data[valid_indices]
    popt, _ = curve_fit(beta_function, x_data, y_data, p0=(1, 1))
    alpha_opt, beta_opt = popt
    print(f'Histograma Beta {i}: Parámetros optimizados: alpha = {alpha_opt}, beta = {beta_opt}')
    y_fit = beta_function(x_data, alpha_opt, beta_opt)
    rmse = np.sqrt(mean_squared_error(y_data, y_fit))
    print(f'Histograma Beta {i}: RMSE: {rmse}')
    results_list.append({'Histograma': f'Beta {i}', 'Alpha': alpha_opt, 'Beta': beta_opt, 'RMSE': rmse})
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, marker='o', label='Datos de la función Beta', color='green', markersize=5)
    plt.plot(x_data, y_fit, label='Ajuste Beta', color='orange')
    plt.title(f'Ajuste de la Función Beta - {results_list[-1]["Histograma"]}')
    plt.xlabel('Bin Edge')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid()
    plt.show()
    
#Resumiendo los datos obtenidos:
results = pd.DataFrame(results_list)
# Mostrar resultados en una tabla
print("\nResultados:")
print(results)
