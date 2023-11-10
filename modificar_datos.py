import random

#* Script que use en un momento para modificar datos de entrenamiento y generar mas cantidad
#* si necesitaba. Ademas los desordena y los ordena primero a todos los que son para saltar,
#* luego todos los de agacharse y luego todos los de correr.
#* Devuelve en consola todos los datos para copiar y pegar en un csv.
#* Lo dejamos por si sirve en algun momento.

# Definir el conjunto de datos
data = """
278,14,340,220,97,68,0,1,0
264,14,340,220,97,68,0,1,0
...otros datos...
"""

# Dividir las filas y columnas
filas = data.strip().split('\n')
matriz = [fila.split(',') for fila in filas]

# Sumar 2 a la primera columna
for fila in matriz:
    fila[0] = str(int(fila[0]) + 2)

# Desordenar las filas
random.shuffle(matriz)

# Ordenar por columnas 7, 8 y 9
matriz_ordenada = sorted(matriz, key=lambda x: (int(x[6]), int(x[7]), int(x[8])))

# Unir la matriz ordenada en una cadena
datos_actualizados_ordenados = '\n'.join([','.join(fila) for fila in matriz_ordenada])

# Imprimir los datos actualizados y ordenados
print(datos_actualizados_ordenados)
