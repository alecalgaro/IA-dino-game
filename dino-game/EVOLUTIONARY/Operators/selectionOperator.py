import numpy as np

#* Operadores de seleccion: ventanas, competencia y ruleta.

def window(dataPopulation, numParent, replace=False):
    """
    Operador de seleccion mediante ventanas.
    Entradas: los datos de la poblacion, almacenado como un conjunto de tuplas [(puntaje, redNeuronal) * n],
    el numero de padres a seleccionar y si se admite o no reemplazo.
    Salida: padres seleccionados.
    """

    parent = []
    numIndiv = len(dataPopulation)

    # el num de padres viene como porcentaje, entonces lo convertimos a entero
    if(isinstance(numParent, float) and numParent <= 1):
        numParent = int(numIndiv * numParent)

    idxIndiv = np.arange(numIndiv)  # array de 0 a numIndiv-1 para indexar los elementos de la poblacion
    idxBool = np.full(shape=(numIndiv), fill_value=True, dtype=bool)    # array booleano para controlar los padres que se van seleccionando
    
    # Cantidad de indiv que se eliminan cada vez para achicar la ventana
    reduceAmount = numIndiv//numParent   
    start = 0

    # Seleccionar y agregar los padres a la lista 
    for _ in range(numParent):
        # Selecciona un indice aleatorio de los que aun no fueron seleccionados
        idxSelected = np.random.choice(idxIndiv[idxBool])
        start += reduceAmount

        # Si no admite reemplazo, se setea en Falso para que no vuelva a ser elegido
        if not(replace):
            idxBool[idxSelected] = False

            if(idxSelected >= start):
                start -= 1
                # while(not(idxBool[start])):
                #     start -= 1
        
        # Achica la ventana colocando en False todos hasta start
        idxBool[:start] = False
        # Se agrega a la lista de padres la red neuronal correspondiente al indice seleccionado 
        # dataPopulation[idxSelected][1] es la red neuronal
        parent.append(dataPopulation[idxSelected][1])
    
    return parent

def competition(dataPopulation, numParent, replace=False):
    """
    Operador de seleccion mediante competicion.
    Entradas: los datos de la poblacion, almacenado como un conjunto de tuplas [(puntaje, redNeuronal) * n],
    el numero de padres a seleccionar y si se admite o no reemplazo.
    Se elige k individuos a competir hasta obtener n padres.
    Salida: padres seleccionados.
    """

    parent = []
    numIndiv = len(dataPopulation)

    # el num de padres viene como porcentaje, entonces lo convertimos a entero
    if(isinstance(numParent, float) and numParent <= 1):
        numParent = int(numIndiv * numParent)

    idxIndiv = np.arange(numIndiv)
    idxBool = np.full(shape=(numIndiv), fill_value=True, dtype=bool)
    
    # Cantidad de individuos a competir
    competitionAmount = numIndiv//numParent

    # Seleccionar y agregar los padres a la lista 
    for _ in range(numParent):
        # Seleccionar n competidores 
        idxSelected = np.random.choice(idxIndiv[idxBool], size=(competitionAmount), replace=False)

        # Si no admite reemplazo, seteo en falso
        if(not(replace)):
            idxBool[idxSelected] = False

        # Se elige el indice maximo, ya que sabemos que los datos vienen ordenados porque se van 
        # guardando cada vez que pierden asi que el de mayor puntuacion estara ultimo.
        # Por ejemplo si los indices seleccionados son [1, 14, 5, 6, 30] sera seleccionado el 30
        idxWinner = np.max(idxSelected)
        # Se agrega a la lista de padres la red neuronal correspondiente al indice seleccionado 
        parent.append(dataPopulation[idxWinner][1])
    
    return parent

def roulette(dataPopulation, numParent, replace=False):
    """
    Operador de seleccion mediante ruleta.
    Entradas: los datos de la poblacion, almacenado como un conjunto de tuplas [(puntaje, redNeuronal) * n],
    el numero de padres a seleccionar y si se admite o no reemplazo.
    Salida: padres seleccionados.
    """
    
    parent = []
    numIndiv = len(dataPopulation)

    # el num de padres viene como porcentaje, entonces lo convertimos a entero
    if(isinstance(numParent, float) and numParent <= 1):
        numParent = int(numIndiv * numParent)

    # array de indices de 0 a numIndiv-1 para indexar los elementos de la poblacion
    idxIndiv = np.arange(numIndiv)  

    # Se recolecta los puntajes, se suman y despeja las probabilidades
    scores = []
    sum = 0
    for score, _ in dataPopulation:
        scores.append(score)
        sum += score

    probabilities = np.array(scores)/sum

    # Se seleccionan numParent indices de idxIndiv basados en las probabilidades calculadas
    idxParents = np.random.choice(idxIndiv, size=(numParent), p=probabilities, replace=replace)

    # Se recorren los indices seleccionados y se agregan las redes neuronales a la lista de padres
    for idx in idxParents:
        parent.append(dataPopulation[idx][1])
    
    return parent

#? Test operator

# datas = [(i, i) for i in range(20)]

# numPadres = 0.8
# replace = True
# parent = competition(datas, numParent=numPadres, replace=replace)

# for pp in parent:
#     print(pp)