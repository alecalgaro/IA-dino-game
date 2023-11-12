import numpy as np

#* Operadores de seleccion: ventanas, competencia y ruleta.

def window(dataPopulation, numParent, replace=False):
    """
    Operador de seleccion mediante ventanas.
    Entradas: los datos de la poblacion, almacenado como un conjunto tuplas [(puntaje, redesNeuronal) * n],
    el numero de padres a seleccionar y si se admite o no reemplazo.
    Salida: padres seleccionados.
    """

    parent = []
    numIndiv = len(dataPopulation)

    if(isinstance(numParent, float) and numParent <= 1):
        numParent = int(numIndiv * numParent)

    idxIndiv = np.arange(numIndiv)
    idxBool = np.full(shape=(numIndiv), fill_value=True, dtype=bool)
    
    # Cantidad a achicar la ventana
    reduceAmount = numIndiv//numParent
    start = 0

    # Coleccionar los indices 
    for _ in range(numParent):
        idxSelected = np.random.choice(idxIndiv[idxBool])
        start += reduceAmount

        # Si NO admite reemplazo, se setea en Falso
        if not(replace):
            idxBool[idxSelected] = False

            if(idxSelected >= start):
                start -= 1
                # while(not(idxBool[start])):
                #     start -= 1
        
        idxBool[:start] = False
        parent.append(dataPopulation[idxSelected][1])
    
    return parent

def competition(dataPopulation, numParent, replace=False):
    """
    Operador de seleccion mediante competicion.
    Entradas: datos de la poblacion almacenado como un conjunto tuplas [(puntaje, redesNeuronal) * n],
    numero de padres a seleccionar y si adminte o no reemplazo.
    Se elige k individuos a competir hasta obtener n padres.
    Salida: padres seleccionados.
    """

    parent = []
    numIndiv = len(dataPopulation)

    if(isinstance(numParent, float) and numParent <= 1):
        numParent = int(numIndiv * numParent)

    idxIndiv = np.arange(numIndiv)
    idxBool = np.full(shape=(numIndiv), fill_value=True, dtype=bool)
    
    # Cantidad de individuos a competir
    competitionAmount = numIndiv//numParent

    # Coleccionar los indices 
    for _ in range(numParent):
        # Seleccionar n competidores y desactivarlo 
        idxSelected = np.random.choice(idxIndiv[idxBool], size=(competitionAmount), replace=False)

        # Si NO admite reemplazo, seteo en falso
        if(not(replace)):
            idxBool[idxSelected] = False

        # Elegir directamente el maximo, ya que sabemos que los datos vienen ordenados
        # Por ejemplo si los indices seleccionado son [1, 14, 5, 6, 30] sera seleccionado el 30
        idxWinner = np.max(idxSelected)
        parent.append(dataPopulation[idxWinner][1])
    
    return parent

def roulette(dataPopulation, numParent, replace=False):
    """
    Operador de seleccion mediante ruleta.
    Entradas: datos de la poblacion almacenado como un conjunto tuplas [(puntaje, redesNeuronal) * n],
    numero de padres a seleccionar y si adminte o no reemplazo.
    Salida: padres seleccionados.
    """
    
    parent = []
    numIndiv = len(dataPopulation)

    if(isinstance(numParent, float) and numParent <= 1):
        numParent = int(numIndiv * numParent)

    idxIndiv = np.arange(numIndiv)

    # Recolectar los puntajes y despejar las probabilidades
    scores = []
    sum = 0
    for score, _ in dataPopulation:
        scores.append(score)
        sum += score

    probabilities = np.array(scores)/sum

    # Elegir n padres segun la probabilidad
    # Por defecto no admite reemplazo
    idxParents = np.random.choice(idxIndiv, size=(numParent), p=probabilities, replace=replace)

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