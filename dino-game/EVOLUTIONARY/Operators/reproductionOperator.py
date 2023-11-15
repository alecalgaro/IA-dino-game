import numpy as np

#* Operadores de reproduccion: cruza y mutacion.

def crossover(structure, parent, nDino, probability=0.85) -> list:
    """ 
    Funcion para realizar la cruza en algoritmo evolutivo.
    Entrada: estructura de la red, padres, cantidad de dinos usados y probabilidad de cruza.
    Salida: hijos obtenidos luego de haber aplicado la cruza.
    """
    child = []

    # Incluir la capa de entrada
    numStructure = len(structure)
    numParent = len(parent)
    numChild = nDino - numParent

    # array de indices de 0 a numParent-1
    idxs = np.arange(numParent)     

    # Bucle hasta generar todos los hijos
    while(numChild > 0):

        # Seleccionar al azar dos indices de idxs para elegir dos padres
        idxP1, idxP2 = np.random.choice(idxs, size=2, replace=False)

        # Se copian esos dos padres
        child1 = parent[idxP1].copy()
        child2 = parent[idxP2].copy()
    
        # Se tira un numero al azar y si es menor a la probabilidad de cruza se aplica la cruza
        if(np.random.rand() < probability):

            # Realizar cruza por cada capa de neuronas
            for i in range(numStructure - 1):
                # Cantidad de neuronas de la capa actual
                rowNum = child1[i].shape[0]     
                
                # Arreglo booleano con True o False de forma aleatoria, para elegir que neuronas se cruzan
                idxCross = np.random.choice([True, False], size=rowNum)

                # Cruza intercambiando los valores
                # child1[i][idxCross] seleccionaria las neuronas de child1[i] que se van a cruzar
                child1[i][idxCross], child2[i][idxCross] = child2[i][idxCross], child1[i][idxCross]
            
        numChild -= 1
        child.append(child1)

        if(numChild > 0):
            numChild -= 1
            child.append(child2)

    return child

def mutation(structure, childs, probability=0.1, score=10) -> list:
    """
    Funcion para realizar la mutacion en algoritmo evolutivo.
    Entradas: estructura de la red, hijos obtenidos luego de la cruza, probabilidad de mutacion y 
    la puntuacion obtenida.
    Salida: hijos luego de haber aplicado la mutacion.
    """

    # Definir el rango maximo y minimo del numero a sumar. 
    # Con eso hacemos que cuando la puntuacion de los dinos es baja se haga una mutacion sumando
    # un valor mas grande y cuando la puntuacion es alta se mute sumando valores mas chicos.
    base = 5
    maxRange = 1.2/np.emath.logn(base, score)

    # Numero de estructura, por defecto descontar la capa de entrada
    numStruct = len(structure) - 1

    # Recorrer cada hijo y aplicar la mutacion si el num elegido al azar es menor a la prob de mutacion
    for child in childs:
        if(np.random.rand() < probability):

            # Aplicar x mutacion a cada capa
            for i in range(numStruct):
                nPrev = structure[i]    # cantidad de neuronas de la capa previa
                nNext = structure[i + 1]    # cantidad de neuronas de la capa posterior

                # Definir cantidad de numeros a mutar, se puede usar max o min
                nMut = np.random.randint(low=1, high=min(nPrev, nNext))

                # Definir los indices i y j a mutar
                idxMut = np.random.randint(low=[0, 0], high=[nNext, nPrev + 1], size=(nMut, 2))

                # Sumar numero de mutacion a los indices ij definidos
                child[i][idxMut[:, 0], idxMut[:, 1]] += np.random.uniform(low=-maxRange, high=maxRange)

    return childs