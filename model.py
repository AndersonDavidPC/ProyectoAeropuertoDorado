import pandas as pd

# Cargar los datos del archivo data.csv en un marco de datos de Pandas
df = pd.read_csv('data.csv', sep=';')

# Imprimir las primeras 5 filas del marco de datos
print(df.head())

# Acceder a la columna contagios del marco de datos
contagios = df['contagios']

# Acceder a la columna trafico del marco de datos
trafico = df['trafico']

# Acceder a la columna mes del marco de datos
mes = df['mes']

# Acceder a la columna año del marco de datos
anio = df['anio']
