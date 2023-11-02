import numpy as np

def sigmoidea(v, b):
    """
    Funcion sigmoidea (formula sacada de la teoria).
    Entradas: "v" es la salida lineal (producto interno) y "b" es el parametro 
    para hacerla más o menos abrupta, si queremos que se acerque a la funcion signo
    """

    return 2/(1+np.exp(-b*v)) - 1

    # la formula de arriba es la que esta en el pdf de teoria, y en la teoria 1 vimos
    # la formula que dejo abajo comentada. 
    # Con las dos funciona, pero ¿es lo mismo usar una o la otra? ¿Debo cambiar la derivada?
    # Luego la derivada que se usa es de la funcion de arriba como en el pdf
    # return (1 - np.exp(-b*v)) / (1 + np.exp(-b*v))
