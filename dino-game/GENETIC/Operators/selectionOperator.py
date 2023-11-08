import numpy as np

def window(dataPopulation, numParent):
    """
    Recibe los datos de la poblacion, almacenado como un conjunto 
    tuplas [(puntaje, redesNeuronal) * n].
    Lo bueno es que ya viene ordenado
    """
    #! Tenemos que determinar si admite repeticion o no
    #! Por el momento SI admite 

    parent = []
    numIndiv = len(dataPopulation)

    if(isinstance(numParent, float) and numParent <= 1):
        numParent = int(numIndiv * numParent)

    idxIndiv = np.arange(numIndiv)
    
    # Cantidad a achatar la ventana
    reduceAmount = numIndiv//numParent
    start = 0

    # Coleccionar los indices 
    for _ in range(numParent):
        idxSelected = np.random.choice(idxIndiv[start:])
        start += reduceAmount
        parent.append(dataPopulation[idxSelected][1])
    
    return parent

def competition(dataPopulation, numParent):
    """
    Recibe los datos de la poblacion almacenado como un conjunto 
    tuplas [(puntaje, redesNeuronal) * n].
    Se elige k individuos a competir hasta obtener n padres
    """
    parent = []
    numIndiv = len(dataPopulation)

    if(isinstance(numParent, float) and numParent <= 1):
        numParent = int(numIndiv * numParent)

    idxIndiv = np.arange(numIndiv)
    idxBool = np.full(shape=(idxIndiv), fill_value=True, dtype=bool)
    
    # Cantidad de individuos a competir
    competitionAmount = numIndiv//numParent

    # Coleccionar los indices 
    for _ in range(numParent):
        # Seleccionar n competidores y desactivarlo 
        idxSelected = np.random.choice(idxIndiv[idxBool], size=(competitionAmount), replace=False)
        idxBool[idxSelected] = False

        # Elegir directamente el maximo, ya que sabemos que los datos vienen ordenado
        # Por ejemplo los indices seleccionado son [1, 14, 5, 6, 30], el 30 sera seleccionado
        idxWinner = np.max(idxSelected)
        parent.append(dataPopulation[idxWinner][1])
    
    return parent


def roulette(dataPopulation, numParent):
    """
    Recibe los datos de la poblacion almacenado como un conjunto 
    tuplas [(puntaje, redesNeuronal) * n].
    """
    #! Admite repeticion o no?
    
    parent = []
    numIndiv = len(dataPopulation)

    if(isinstance(numParent, float) and numParent <= 1):
        numParent = int(numIndiv * numParent)

    idxIndiv = np.arange(numIndiv)

    # Recolectar los puntajes y despejar las rpbabilidades
    scores = []
    sum = 0
    for score, _ in dataPopulation:
        scores.append(score)
        sum += score

    probabilities = np.array(scores)/sum

    # Elegir n padres segun la probabilidad
    # Por defecto no admite reemplazo
    idxParents = np.random.choice(idxIndiv, size=(numParent), p=probabilities, replace=False)

    for idx in idxParents:
        parent.append(dataPopulation[idx][1])
    
    return parent


