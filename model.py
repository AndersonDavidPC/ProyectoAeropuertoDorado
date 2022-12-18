import pandas as pd
from sklearn.linear_model import LinearRegression

# Cargar los datos del archivo data.csv en un marco de datos de Pandas
df = pd.read_csv('data.csv', sep=';')

# Imprimir las primeras 5 filas del marco de datos
print(df.head())


X = df[['contagios', 'mes', 'anio']]
y = df['trafico']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse:.2f}')


# Hacer predicciones con el modelo entrenado
y_pred = model.predict(X_test)

# Crear un dataframe con los resultados actuales y las predicciones
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Mostrar los primeros 5 resultados
print(results.head())

print(results)


# Filtra el dataframe por aquellas filas en las que los contagios aumentan respecto al año anterior
results_increase = results[results['contagios'] > results['anio']]

# Imprime los resultados filtrados
print(results_increase)