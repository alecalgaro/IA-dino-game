import numpy as np
from cargarDatos import *
from sigmoidea import *
from graficas import *
from winnerTakesAll import *

def entrenamiento_MLP(nombreArchivo, arquitectura, tasaErrorAceptable, numMaxEpocas,  
                      gamma, bSigmoidea=1, grafError=False):
    """
    Algoritmo de entrenamiento del perceptron multicapa.
    Entradas: datos, arquitectura de capas (por ejemplo [3 2]), tasa de error aceptable, 
    cant. max de epocas, tasa de aprendizaje, parametro "b" de la sigmoidea, un umbral para
    el error y si se quiere graficar el error.
    Salida: pesos W entrenados.
    """

    print("\n--- Comenzando entrenamiento MLP ---")

    # Cargar los datos
    (X, Yd) = cargarDatos(nombreArchivo, arquitectura[-1])

    #* ----- Inicializar variables a utilizar -----

    contEpocas = 0      # epocas realizadas
    tasaErrorActual = 1      # tasa de error (100% para comenzar)
    cantPatrones = np.size(X, 0)    # cantidad de patrones (filas)
    cantCapas = len(arquitectura)   # cantidad de capas que tiene la arquitectura definida (para usar en algunos for)
    errorCuadPlot = []      # vector para graficar error cuadratico por epocas
    tasaErrorPlot = []      # vector para graficar la tasa de error (porcentaje) por epocas
    
    #* ----- Inicializar estructuras de datos a utilizar -----

    Y_vec = []  # vector de vectores de salidas para cada capa (y^I, y^II, etc.)
    W_mat = []  # matriz con matrices de pesos para cada capa (W^I, W^II, etc.)
    deltas = [] # vector de vectores de deltas para cada capa (d^I, d^II, etc.)

    #* -- Salidas --

    # Vamos a tener un vector con vectores para las salidas de cada capa (y^I, y^II, etc.)
    # Se agrega una columna mas en todas menos en la salida final, para luego agregar el -1 del sesgo.
    # Por ejemplo, con arquitectura = [3, 2] voy a obtener:
    # [array([0.,  0.,  0.,  0.]), array([0., 0.])]
    for i in arquitectura[:-1]:     # para la salida de todas las capas menos la final ("i" seria la cant de neurona de la capa) 
        Y_vec.append(np.zeros(i+1))     # i+1 para que agregue una columna mas para el sesgo (primera columna)
    Y_vec.append(np.zeros(arquitectura[-1]))   # para la salida de la capa final (no lleva el -1 del sesgo)

    #* -- Matrices de pesos --

    # W_mat tendra tantas filas como neuronas tiene la capa i y tantas columnas
    # como entradas hay en dicha capa (recordar el sesgo que es una entrada mas).
    # Similar a lo anterior sobre las salidas, esto seria ir agregando las matrices 
    # de pesos para cada capa como fuimos viendo en teoria W^I, W^II, etc.
    # Inicializando las matrices de cada capa con pesos al azar entre -0.5 y 0.5

    # Para la primera capa usamos las entradas X:
    W_mat = [np.random.rand(arquitectura[0], np.size(X, 1)) - 0.5]

    # Para las capas ocultas y la final (desde 1 hasta cantCapas), las entradas seran las
    # salidas de la capa anterior, asi que como columnas tomamos el tamano de la salida anterior.
    # W_mat[1] para las columnas usa el tamano de Y_vec[0] que es y^I, W_mat[2] el de Y_vec[1] que es y^II, etc.
    # (la columna para el sesgo ya la inclui en las salidas Y_vec)
    for i in range(1, cantCapas):
        W_mat.append(np.random.rand(arquitectura[i], np.size(Y_vec[i-1])) - 0.5)

    #* -- Deltas --

    # Vector de vectores de deltas para todas las capas (d^I, d^II, etc.)
    # Un problema es que aca se definen los vectores como 1D y luego para hacer los productos 
    # matriz vector es un problema, por eso se usan distintas funciones de Numpy (matmul y multiply).
    # Para el ejemplo [3, 2] tendria deltas = [array([0., 0., 0.]), array([0., 0.])]
    for i in arquitectura:
        deltas.append(np.zeros(i))

    # Iniciar graficas para errores
    if(grafError):
        (ax, ax2) = iniciarGraficasErrores()

    #* ----- Bucle general ----- 
    while(contEpocas < numMaxEpocas and tasaErrorActual > tasaErrorAceptable):
        contEpocas += 1
        
        #* --- Etapa de entrenamiento ---
        
        # Recorremos cada patron
        # Recordar que es por cada patron, no por epoca. Es decir, hacemos propgacion hacia adelante, 
        # propagacion hacia atras y ajuste de pesos por cada patron.
        for i in range(cantPatrones):

            #* -- PROPAGACION HACIA ADELANTE --
            # matmul hace el producto matriz-vector (Numpy), asi es mas eficiente. Igual al @ de Python.
            # Otra opcion seria ir haciendo producto interno (como en el perceptron simple) entre
            # vector de pesos de una capa y vector de entradas de la capa (menos eficiente).

            # Primero multiplicamos la matriz de pesos de la capa I (W^I) por el vector de entrada a la red (X)
            # y lo pasamos por la funcion sigmoidea para tener la salida lineal y^I de la capa I. 
            v1 = np.matmul(W_mat[0], X[i])
            Y_vec[0][0] = -1    # agrego -1 en la primera columna para el sesgo
            Y_vec[0][1:] = sigmoidea(v1, bSigmoidea)    # se guarda en posicion 0 y todas las cols desde 1 hasta el       
                                                        # final, sin la primera col [0] que es del sesgo que agregamos antes
                                                        # recordar que es un vector tipo [array([0.,  0.,  0.,  0.]), array([0., 0.])]
                                                        # y lo vamos llenando con esta informacion.
            # Para el resto de capas, la entrada es la salida de la capa anterior
            for j in range(1, cantCapas-1):
                vj = np.matmul(W_mat[j], Y_vec[j-1])
                Y_vec[j][0] = -1    # agrego -1 en la primera columna para el sesgo
                Y_vec[j][1:] = sigmoidea(vj, bSigmoidea)

            # Para la capa final (se hace aparte porque no lleva el sesgo)
            vf = np.matmul(W_mat[-1], Y_vec[-2])
            Y_vec[-1] = sigmoidea(vf, bSigmoidea)

            #* -- PROPAGACION HACIA ATRAS --

            # Primero calculo el delta de la capa final
            # Seria error * derivada funcion sigmoidea en la capa final (0.5*(1+y)*(1-y))
            deltas[-1] = 0.5 * (Yd[i]-Y_vec[-1]) * (1+Y_vec[-1])*(1-Y_vec[-1])
            
            # For desde la capa final hasta la inicial
            # Comienza no del -1 (capa final) sino en -2 porque arriba ya calcule el delta de la capa final.
            # Y es hasta -1 para llegar hasta el 0, y el ultimo -1 es para que el for sea hacia atras.
            # La idea es ir pasando (retro propagando) ese gradiente de error que obtuvimos al final, 
            # hacia atras a traves de las matrices de pesos, para imputarle el error a las capas anteriores.
            for p in range(cantCapas-2, -1, -1):   
                # en la propagacion hacia atras, para obtener el delta de la capa actual, en cada capa 
                # multiplicamos el vector delta de la capa siguiente por la matriz de pesos de la capa 
                # siguiente, ignorando la primer columna que era el peso del sesgo y transponiendo dicha matriz.
                # Asi se hace un producto matriz por vector para que sea mas eficiente.
                # (explicado en pags. 5 y 7 de mi word de anotaciones de la practica)
                aux = np.multiply(deltas[p+1], np.transpose(W_mat[p+1][:, 1:]))
                # En W_mat[p+1][:, 1:]) los indices serian:
                # [p+1] -> matriz de pesos de la capa siguiente a la que estamos (p+1)
                # [:, 1:] -> todas las filas, todas las columnas excepto la primera (0) que es del sesgo,
                # asi luego obtener un delta para cada neurona, ordenados en un vector de deltas de cada capa.
                # "multiply" de Numpy equivale al "*" de Python, multiplica por elementos

                # Sumatoria de la formula (pag. 5 de mi word de practica).
                # axis=1 seria para que sume cada elemento de la columna, que como hicimos transpuesta se usa asi
                sumatoria = np.sum(aux, axis=1) 
                
                # En "probandoCosas.py" deje un ejemplo transponiendo y usando aux y sumatoria

                # la segunda parte de la formula para el delta actual es la derivada de la funcion sigmoidea
                # con la salida de la capa actual (ignorando el -1 del sesgo, por eso [1:] en el vector)
                derSigmoideaActual = 0.5*(1+Y_vec[p][1:]) * (1-Y_vec[p][1:])

                # Aplico la formula completa para obtener el delta de la capa actual
                deltas[p] = sumatoria*derSigmoideaActual
            
            #* -- CORREGIR PESOS --
            # Es Wnuevo = Wactual + velAprendizaje*deltaCapa*entradaCapa

            # Para la capa inicial multiplicamos las entradas por cada delta de la capa. 
            # Se transpone para adecuar el tamaño de la matriz para la multiplicacion.
            aux_dx = np.multiply(deltas[0], np.transpose([X[i]]))
            # Multiplicamos lo anterior por la tasa de aprendizaje y obtenemos el incremento de pesos dW
            dW = np.multiply(np.transpose(aux_dx), gamma)
            # Actualizamos los pesos sumando ese delta de pesos calculado
            W_mat[0] += dW

            # Lo mismo para el resto de capas pero las entradas seran la salida de la capa anterior
            for k in range(1, cantCapas):
                aux_dx = np.multiply(deltas[k], np.transpose([Y_vec[k-1]]))
                dW = np.multiply(np.transpose(aux_dx), gamma)

                W_mat[k] += dW      # actualizo los pesos

        #* --- Etapa de validacion ---
        contErrores = 0
        errorProm = 0
        for i in range(cantPatrones):    
            # Capa inicial
            v1 = np.matmul(W_mat[0], X[i])
            Y_vec[0][0] = -1    # agrego -1 en la primera columna para el sesgo
            Y_vec[0][1:] = sigmoidea(v1, bSigmoidea)

            # Para el resto de capas (capas ocultas) menos la capa final
            for j in range(1, cantCapas-1):
                vj = np.matmul(W_mat[j], Y_vec[j-1])
                Y_vec[j][0] = -1 
                Y_vec[j][1:] = sigmoidea(vj, bSigmoidea)

            # Para la capa final (no lleva la columna del sesgo al comienzo)
            vf = np.matmul(W_mat[-1], Y_vec[-2])
            Y_vec[-1] = sigmoidea(vf, bSigmoidea)

            if arquitectura[-1] > 1:    #* si hay mas de una salida al final (mas de una neurona en la capa final)
                newY = winnerTakesAll(Y_vec) 
                # .all() comprueba que todos sean True, entonces si no son todos True cuento un error
                contErrores += 1 if ((newY == Yd[i]).all()) == False else 0  
                EC = np.sum(np.power(Yd[i]-Y_vec[-1], 2))
                # en la formula del EC multplica por 1/2 pero no hace falta, solo escala. Cuando se usaba y luego
                # se saca la derivada ahi si tiene sentido usar el 1/2 para que se cancele con el ^2 al derivar
            else:   #* si hay una sola salida final
                clasificacion = 1 if Y_vec[-1] >= 0 else -1
                contErrores += Yd[i][0] != clasificacion
                EC = np.sum(np.power(Yd[i]-Y_vec[-1], 2))

            errorProm += EC

        # Calculo de tasa de error en validacion (error de clasificacion)
        tasaErrorActual = contErrores/cantPatrones
        tasaErrorPlot.append(tasaErrorActual)

        # Calculo el error cuadratico medio en validacion (error interno de la red)
        errorCuadraticoPromedio = errorProm/cantPatrones
        errorCuadPlot.append(errorCuadraticoPromedio)
        print(errorCuadraticoPromedio)

        # Actualizo las graficas de errores
        if(grafError):
            actualizarGraficasErrores(ax, ax2, errorCuadPlot, tasaErrorPlot)

    #* fin del while general

    if(contEpocas == numMaxEpocas): 
        print("\nCorto por cantidad de epocas (",contEpocas,")")
        print("Error logrado: ", round(tasaErrorActual*100, 2), "%")
    else:
        print("\nCorto por tasa de error aceptable")
        print("Epocas:", contEpocas)

    return W_mat    # retorno los pesos entrenados