# coding: utf-8

# Examen Claudio

import pandas as pd

# Se lee el conjunto de datos de Boston desde la URL
boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df = pd.read_csv(boston_url, header=0)

# Se muestra una vista previa de los primeros registros del conjunto de datos
print(boston_df.head())

# # 1. Valor mediano de las viviendas ocupadas por sus propietarios

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.boxplot(boston_df['MEDV'])
plt.xlabel('Valor Mediano de las Viviendas Ocupadas por Sus Propietarios')
plt.ylabel('Valor')
plt.title('Diagrama de Caja del Valor Mediano de las Viviendas Ocupadas por Sus Propietarios')
plt.show()

# # 2. Variable del río Charles

# Se cuentan las ocurrencias de cada valor en la variable del río Charles
charles_river_counts = boston_df['CHAS'].value_counts()

# Se crea un gráfico de barras
plt.figure(figsize=(6, 4))
charles_river_counts.plot(kind='bar')
plt.xlabel('Río Charles')
plt.ylabel('Recuento')
plt.title('Gráfico de Barras del Río Charles')
plt.xticks(rotation='horizontal')
plt.show()

# # 3. Diagrama de Caja de MEDV vs. AGE

# Se discretiza la variable AGE en tres grupos
boston_df['GRUPO_EDAD'] = pd.cut(boston_df['AGE'], bins=[0, 35, 70, float('inf')], labels=['35 años o menos', 'entre 35 y 70 años', '70 años o más'])

# Se crea el diagrama de caja
plt.figure(figsize=(8, 6))
plt.boxplot([boston_df[boston_df['GRUPO_EDAD'] == '35 años o menos']['MEDV'],
             boston_df[boston_df['GRUPO_EDAD'] == 'entre 35 y 70 años']['MEDV'],
             boston_df[boston_df['GRUPO_EDAD'] == '70 años o más']['MEDV']],
            labels=['35 años o menos', 'entre 35 y 70 años', '70 años o más'])
plt.xlabel('Grupo de Edad')
plt.ylabel('Valor Mediano de las Viviendas Ocupadas por Sus Propietarios')
plt.title('Diagrama de Caja de MEDV vs. AGE')
plt.show()

# # 4. Gráfico de Dispersión: NOX vs. INDUS

# Se crea el gráfico de dispersión
plt.figure(figsize=(8, 6))
plt.scatter(boston_df['NOX'], boston_df['INDUS'])
plt.xlabel('Concentración de Óxido Nítrico')
plt.ylabel('Proporción de Acres de Negocios No Minoristas')
plt.title('Gráfico de Dispersión: NOX vs. INDUS')
plt.show()

# # 5. Histograma de PTRATIO

# Se crea el histograma
plt.figure(figsize=(8, 6))
plt.hist(boston_df['PTRATIO'], bins=10, edgecolor='black')
plt.xlabel('Ratio de Alumnos por Profesor')
plt.ylabel('Frecuencia')
plt.title('Histograma de PTRATIO')
plt.show()

# # Tarea 5

# 1. ¿Existe una diferencia significativa en el valor mediano de las casas limitadas por el río Charles o no? (Prueba t para muestras independientes)

import scipy.stats as stats

# Se divide el conjunto de datos en dos grupos basados en el río Charles (0: no limitado, 1: limitado)
charles_limitado = boston_df[boston_df['CHAS'] == 1]['MEDV']
charles_no_limitado = boston_df[boston_df['CHAS'] == 0]['MEDV']

# Se realiza la prueba t para muestras independientes
estadistico_t, valor_p = stats.ttest_ind(charles_limitado, charles_no_limitado)

# Se define el nivel de significancia
alfa = 0.05

# Se verifica si el valor p es menor que el nivel de significancia
if valor_p < alfa:
    print("Existe una diferencia significativa en el valor mediano de las casas limitadas por el río Charles.")
else:
    print("No existe una diferencia significativa en el valor mediano de las casas limitadas por el río Charles.")

# 2. ¿Existe una diferencia en los valores medianos de las casas (MEDV) para cada proporción de unidades ocupadas por el propietario construidas antes de 1940 (AGE)? (ANOVA)

# Se crean tres grupos basados en la variable AGE
grupo_edad_1 = boston_df[boston_df['AGE'] <= 35]['MEDV']
grupo_edad_2 = boston_df[(boston_df['AGE'] > 35) & (boston_df['AGE'] <= 70)]['MEDV']
grupo_edad_3 = boston_df[boston_df['AGE'] > 70]['MEDV']

# Se realiza el ANOVA de una vía
estadistico_f, valor_p = stats.f_oneway(grupo_edad_1, grupo_edad_2, grupo_edad_3)

# Se define el nivel de significancia
alfa = 0.05

# Se verifica si el valor p es menor que el nivel de significancia
if valor_p < alfa:
    print("Existe una diferencia significativa en los valores medianos de las casas para cada proporción de unidades ocupadas por el propietario construidas antes de 1940.")
else:
    print("No existe una diferencia significativa en los valores medianos de las casas para cada proporción de unidades ocupadas por el propietario construidas antes de 1940.")

# 3. ¿Podemos concluir que no hay relación entre las concentraciones de óxido nítrico y la proporción de acres de negocios no minoristas por pueblo? (Correlación de Pearson)

# Se calcula el coeficiente de correlación de Pearson y el valor p
coef_corr, valor_p = stats.pearsonr(boston_df['NOX'], boston_df['INDUS'])

# Se define el nivel de significancia
alfa = 0.05

# Se verifica si el valor p es mayor que el nivel de significancia
if valor_p >= alfa:
    print("No podemos concluir que no hay una relación significativa entre las concentraciones de óxido nítrico y la proporción de acres de negocios no minoristas por pueblo.")
else:
    print("Existe una relación significativa entre las concentraciones de óxido nítrico y la proporción de acres de negocios no minoristas por ciudad.")
