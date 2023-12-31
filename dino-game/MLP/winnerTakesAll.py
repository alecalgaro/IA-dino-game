import numpy as np

def winnerTakesAll(Y_vec):
    """
    Winner takes all: traducido seria "el ganador se lleva todo". Se utiliza cuando se tiene mas
    de una salida en la capa final.
    Entrada: vector de salidas en la capa final.
    Salida: vector con tantos elementos como cantidad de salidas al final de la red, y que contiene
    +1 en la posicion donde se encuentra la salida mas alta y -1 en las demas.
    """
    y_mayor = np.max(Y_vec[-1])   # busco la salida mayor
    indice_y_mayor = np.where(Y_vec[-1] == y_mayor)[0][0]   # busco el indice de esa salida
    # np.where busca las coincidencias dentro del array y luego accedo a la primera [0][0] que encuentra
    # newY = -1 * np.ones(len(Y_vec[-1]))     # creo un vector con 1s y multiplico por -1, del mismo tamano que Y_vec[-1]
    newY = np.zeros(len(Y_vec[-1]))
    newY[indice_y_mayor] = 1    # coloco un 1 solo en la posicion de la salida mayor

    return newY     