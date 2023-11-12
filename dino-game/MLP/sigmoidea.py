import numpy as np

def sigmoidea(v, b):
    """
    Funcion sigmoidea (formula sacada de la teoria).
    Entradas: "v" es la salida lineal (producto interno) y "b" es el parametro 
    para hacerla más o menos abrupta, si queremos que se acerque a la funcion signo
    """

    return 2/(1+np.exp(-b*v)) - 1