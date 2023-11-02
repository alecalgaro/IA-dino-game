import numpy as np
from cargarDatos import *
from sigmoidea import *
from winnerTakesAll import *
from graficas import * 

def prueba_MLP(nombreArchivo, W_mat, arquitectura, bSigmoidea, grafCategorias=False):
    """
    Algoritmo para probar el perceptron multicapa.
    Entradas: nombre del archivo de datos, pesos actualizados luego del entrenamiento,
    arquitectura de la red (capas), parametro de la sigmoidea y si se quiere graficar las 
    categorias como fueron clasificadas.
    Salida: porcentaje de aciertos en la etapa de pruebas con datos no vistos en el entrenamiento.
    """

    print("\n--- Comenzando etapa de prueba MLP ---")

    # Cargamos los datos de prueba
    (X, Yd) = cargarDatos(nombreArchivo, arquitectura[-1])
 
    cantCapas = len(arquitectura)

    # Inicializar vector de vectores de salidas
    Y_vec = []
    for i in arquitectura[:-1]:
        Y_vec.append(np.zeros(i+1))     # i+1 para que agregue una columna mas para el sesgo (primera columna)
    Y_vec.append(np.zeros(arquitectura[-1]))

    # ----- Inicializar variables a utilizar -----
    tasaErrorActual = 1      # tasa de error (100% para comenzar)
    cantPatrones = np.size(X, 0)    # cantidad de patrones (cantidad de filas)
    contErrores = 0
    errorProm = 0

    markerPorCategoria = []     # vector para agregar markadores (x / o) a los puntos que se grafican
    colorPorCategoria = [None] * len(X)     # vector para agregar colores a los puntos que se grafican
                                            # lo lleno con "None" para despues modificar cada posicion con un color

    # Recorremos cada patron
    # Recordar que es por cada patron, no por epoca. Es decir, hacemos propgacion hacia adelante, 
    # propagacion hacia atras y ajuste de pesos por cada patron.
    for i in range(cantPatrones):    
        # Capa inicial
        v1 = np.matmul(W_mat[0], X[i])
        Y_vec[0][0] = -1    # agrego -1 en la primera columna para el sesgo
        Y_vec[0][1:] = sigmoidea(v1, bSigmoidea)

        # Para el resto de capas
        for j in range(1, cantCapas-1):
            vj = np.matmul(W_mat[j], Y_vec[j-1])
            Y_vec[j][0] = -1 
            Y_vec[j][1:] = sigmoidea(vj, bSigmoidea)

        # Para la capa final
        vf = np.matmul(W_mat[-1], Y_vec[-2])
        Y_vec[-1] = sigmoidea(vf, bSigmoidea)

        if arquitectura[-1] > 1:    #* si hay mas de una salida al final (mas de una neurona en la capa final)
            newY = winnerTakesAll(Y_vec) 
            # .all() comprueba que todos sean True, entonces si no son todos True cuento un error
            contErrores += 1 if ((newY == Yd[i]).all()) == False else 0
            EC = np.sum(np.power(Yd[i]-Y_vec[-1], 2))
        else:   #* si hay una sola salida final
            clasificacion = 1 if Y_vec[-1] >= 0 else -1
            contErrores += Yd[i][0] != clasificacion
            EC = np.sum(np.power(Yd[i]-Y_vec[-1], 2))

        errorProm += EC    

        # Completo los vectores de colores y markers segun si fueron bien o mal clasificados los puntos
        if(grafCategorias):
            if (Yd[i][0] != clasificacion):  # si fue mal clasificado
                if(Yd[i] == 1):
                    colorPorCategoria[i] = 'green'   # +1 clasificado como -1
                    markerPorCategoria.append("x")
                else:
                    colorPorCategoria[i] = 'blue'   # -1 clasificado como +1
                    markerPorCategoria.append("o")
            else:     # si fue bien clasificado uso los colores originales
                if(Yd[i] == 1):
                    colorPorCategoria[i] = 'grey'   # +1 bien clasificado como +1
                    markerPorCategoria.append("x")
                else:
                    colorPorCategoria[i] = 'red'   # -1 bien clasificado como -1
                    markerPorCategoria.append("o")

    # Calculo de tasa de error en validacion
    tasaErrorActual = contErrores/cantPatrones
    # print(tasaErrorActual)

    errorCuadraticoPromedio = errorProm/cantPatrones
    # print(errorCuadraticoPromedio)

    if(grafCategorias):
        # llamo a la funcion para graficar:
        graficarDatosPorCategoria(X, colorPorCategoria, markerPorCategoria)

    return print("\nTasa de error en pruebas: ", round(tasaErrorActual*100, 2), "%\n")
