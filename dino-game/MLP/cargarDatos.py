import numpy as np
from desordenarDatos import *

def cargarDatos(nombreArchivo, cantSalidas):
    """
    Funcion para cargar los datos a partir de un archivo csv.
    Entrada: nombre del archivo de datos y la cantidad de salidas que tiene la red.
    Salida: entradas y salidas deseadas, por separado.
    Se agrega una entrada -1 al comienzo (del sesgo) y se separan las entradas de las 
    salidas deseadas.
    Se usa como (X, yd) = cargarDatos(datos, arquitectura[-1])
    """

    # Se usa la funcion genfromtxt de Numpy para cargar archivos csv
    datos = np.genfromtxt(nombreArchivo, delimiter=',')

    # Desordeno las filas de datos por si venian ordenados por categoria
    data = desordenarDatos(datos)  
    
    # --- Separo las entradas y las salidas deseadas ---
    
    X = data[:, :-cantSalidas]      # uso -cantSalidas: y -cantSalidas: porque ahora podemos tener mas de una salida
    Yd = data[:, -cantSalidas:]     # al final, entonces separamos la cantidad de salidas que tengamos.
                                    # [:, -cantSalidas:] todas las filas y las columnas desde -cantSalidas hasta el final,
                                    # es decir, contando -cantSalidas desde el final

    n, m = X.shape      # dimension de la matriz
    X0 = -1 * np.ones((n, 1))   # vector de -1 para agregar al comienzo (sesgo)
    Xnew = np.hstack((X0, X))   # concatena de forma horizontal los arreglos  

    # devuelve un vector de entradas X[x0, x1, ..., xn] con x0 = -1 y las salidas deseadas Yd por separado
    return (Xnew, Yd)