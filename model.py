import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Cargar los datos del archivo data.csv en un marco de datos de Pandas
df = pd.read_csv("data.csv", sep=',')

# Imprimir las primeras 5 filas del marco de datos
print(df.head())


# Seleccionar las columnas que vamos a utilizar como variables predictoras y la variable objetivo
X = df[['contagios']]
y = df['trafico']

# Dividir los datos en un conjunto de entrenamiento y un conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo con el conjunto de entrenamiento
model.fit(X_train, y_train)

# Hacer predicciones con el modelo entrenado
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Error cuadrático medio: {mse}')
print(f'Coeficiente de determinación: {r2}')

# Utilizar el modelo para hacer predicciones
nuevos_contagios = [[1000]]
prediccion_trafico = model.predict(nuevos_contagios)
print(f'Predicción del tráfico aéreo para 1000 contagios: {prediccion_trafico}')
