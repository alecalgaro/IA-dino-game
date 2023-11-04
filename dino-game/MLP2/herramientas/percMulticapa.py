import numpy as np
import matplotlib.pyplot as plt
from .graficar import *
from .winnerTakesAll import *


def cargarDatos(nombreArchivo, nCapaFinal):
    datos = np.genfromtxt(nombreArchivo, delimiter=",", max_rows=None)

    # Definir la condicion del caso que la capa final no sea unico
    xData = datos[:, :-nCapaFinal]
    yData = datos[:, -nCapaFinal:]
    
    # Definir la columna de -1 de la entrada x0
    colX0 = -1 * np.ones(shape=(np.size(xData, 0), 1))
    xData = np.hstack((colX0, xData))

    return xData, yData

def sigmoidea(Wji, Xi, alpha):
    Vi = Wji@Xi
    Y = 2/(1 + np.exp(-alpha * Vi)) - 1

    return Y

def entrenar(nombreArchivo, capas, alpha, tasaAp, maxErr, maxEpoc, umbral = 1e-1):
    
    # np.random.seed(10000)
    # ===========[Inicializar los datos]===========
    
    # Pasar la ruta del archivo y la cantidad de neuronas de la capa de salida
    # Recibe 2 matrices 2D: Los patrones de entrada y las salidas esperadas 
    # como vector columna
    X, Yd = cargarDatos(nombreArchivo, capas[-1])

    cantCapas = len(capas)
    cantPatrones = X.shape[0]
    err = 1
    epoca = 0
    errPlot = np.array([])
    
    # =========[Inicializar los matrices de pesos]=========
    # La cantidad de columnas SIEMPRE debe coincidir con la cantidad de entrada Xi (o Yi)
    Wji = [np.random.rand(capas[0], X.shape[1]) - 0.5]

    for i in range(cantCapas - 1):
        # Numero de columna debe coincidir con la cantidad de entradas
        # Recordar que habia X0 como entrada adicional, por eso se suma 1
        # a la cantidad de columna
        Wji.append(np.random.rand(capas[i + 1], capas[i] + 1) - 0.5)
    
    # =========[Inicializar las salidas y las deltas de cada capa]=========
    yy = []
    deltas = []

    for i in capas:
        yy.append(np.empty(i))
        deltas.append(np.empty(i))
    

    # ===========[Ciclo de entrenamiento]===========
    while(err > maxErr and epoca < maxEpoc):

        # Recorrer todo los patrones de entrada 
        for i in range(cantPatrones):
            # ===========[Propagacion hacia adelante]===========

            entrada = X[i, :]

            for j in range(cantCapas):
                yy[j] = sigmoidea(Wji[j], entrada, alpha)
                
                # Recordar la entrada X0 = -1 existe en cada capa 
                entrada = np.hstack((np.array([-1]), yy[j]))
            
            # ===========[Propagacion hacia atras]===========

            # Asignar la delta de la capa de salida
            ySalida = yy[-1][:]
            deltas[-1] = 0.5 * (Yd[i, :] - ySalida) * (1 + ySalida) * (1 - ySalida)

            for j in range(cantCapas - 2, -1, -1):
                # Recordar que cada columna de Wji corresponde a una entrada Xi particular
                # Por ejemplo Wj1 corresponde a cada peso sinaptico que sale de 
                # X1 y se conectan a las j neuronas
                

                # delta de capa 1 = 1/2(dta2@Wji2)(1+y1)(1-y1)
                # En Octave seria como:

                # d2 = [0.1 0.2]
                # Wji = [1 2; 
                #        1 2]
                # d1 = 1/2(d2*Wji)(1+y1)(1-y1)
                # d1 = 1.2 * [0.3 0.6] * (1 + [y1(1) y1(2)]) ...
                #                      * (1 - [y1(1) y1(2)])

                # En python no te permite trabajar vector 1D junto con una matriz 2D
                # por esta razon habra que aumentar la dimension del vector 1D usando
                # np.newaxis

                # Mi version favorita
                sumatoria = (deltas[j + 1][np.newaxis]@Wji[j + 1][:, 1:])[0]

                ySalida = yy[j][:]
                deltas[j] = 0.5 * sumatoria * (1 + ySalida) * (1 - ySalida)


            # ===========[Ajuste de pesos]===========
            
            # Recorrer nuevamente cada capa ajustando los pesos
            # En Octave es como:
            # delta = [1;
            #          2]
            # entrada = [-1 2 3]

            # dWji = tasaAp * delta*entrada
            # dWji = tasaAp * [-1 2 3;
            #                  -2 4 6]

            # Hay que agregar dimension para que sean 2D
            # y aplicar producto matricial 

            # Delta Wji de capa 1
            dWji = tasaAp * deltas[0][:, np.newaxis] @ X[i, :][np.newaxis]

            for j in range(cantCapas - 1):
                Wji[j] += dWji 
                entradaCapa =  np.hstack((np.array([-1]), yy[j]))
                dWji = tasaAp * deltas[j + 1][:, np.newaxis] @ entradaCapa[np.newaxis]

            Wji[-1] += dWji
            
            # =======Fin del patron=======

        # ===========[Comprobar aciertos]===========
        cantErr = 0
        errPromedio = 0
        # Recorrer cada patron de entrada y despejar su salida y
        for i in range(cantPatrones):
            entrada = X[i, :]
            for j in range(cantCapas):
                yy[j] = sigmoidea(Wji[j], entrada, alpha)
               # Recordar la entrada X0 que existe en cada capa 
                entrada = np.hstack((np.array([-1]), yy[j]))
            

            E = 0.5 * np.sum(np.power(Yd[i, :] - yy[-1][:], 2))
            # Validar si hay multiple salida en la capa de salida 
            if(capas[-1] > 1):
                newY = winnerTakesAll(yy)
                cantErr += not((newY == Yd[i]).all())
            else:    
                # Comparar si el error del patron supera o no 
                # el umbral definido
                cantErr += E > umbral

            # Promedio de error para luego hacer la grafica
            errPromedio += E

        err = cantErr/cantPatrones

        errPromedio = errPromedio/cantPatrones

        errPlot = np.hstack((errPlot, errPromedio))
    
        epoca += 1
    # End while


    
    if(err < maxErr):
        print("Entrenamiento finalizado por tasa de acierto " +
                str(round((1 - err) * 100, 3)) + "%" + " en la epoca " +
                str(epoca-1))
    else:
        print("Entrenamiento finalizado por Maxima Epoca con ", 
              "un tasa de error ", err * 100, "%")

    return Wji

            

def probar(nombreArchivo, Wji, alpha, umbral, 
           graf = False, XOR = False):

    X, Yd = cargarDatos(nombreArchivo, Wji[-1].shape[0])
    

    # ===========[Comprobar acierto]===========
    cantErr = 0
    cantCapas = len(Wji)
    cantPatrones = X.shape[0]
    
    yy = []
    for i in range(cantCapas):
        yy.append(np.empty(Wji[i].shape[0]))
    
    # Recorrer cada patron de entrada y despejar su salida
    for i in range(cantPatrones):
        entrada = X[i, :]
        for j in range(cantCapas):
            yy[j] = sigmoidea(Wji[j], entrada, alpha)
            # Recordar la entrada X0 que existe en cada capa 
            entrada = np.hstack((np.array([-1]), yy[j]))
        
        E = 0.5 * np.sum(np.power(Yd[i, :] - yy[-1][:], 2))
        # Validar si hay multiple salida en la capa de salida 
        if(Wji[-1].shape[0] > 1):
            newY = winnerTakesAll(yy)
            cantErr += not((newY == Yd[i]).all())
        else:    
            cantErr += E > umbral


    err = cantErr/cantPatrones

        # print(err*100, "%", " de error")
    print("Prueba con tasa de acierto " + str((1 - err) * 100) + "%")

    plt.show()

