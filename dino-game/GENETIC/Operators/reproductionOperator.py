import numpy as np

def crossover(structure, parent, nDino, probability=0.85) -> list:
    
    child = []

    # Incluir la capa de entrada
    numStructure = len(structure)
    numParent = len(parent)
    numChild = nDino - numParent

    idxs = np.arange(numParent)

    while(numChild > 0):

        # Seleccionar 2 padres
        idxP1, idxP2 = np.random.choice(idxs, size=2, replace=False)

        child1 = parent[idxP1].copy()
        child2 = parent[idxP2].copy()
    
        if(np.random.rand() < probability):

            # Realizar cruza por cada capa de neuronas
            for i in range(numStructure - 1):
                rowNum = child1[i].shape[0]
                
                idxCross = np.random.choice([True, False], size=rowNum)

                # Cruza
                child1[i][idxCross], child2[i][idxCross] = child2[i][idxCross], child1[i][idxCross]
            
        numChild -= 1
        child.append(child1)

        if(numChild > 0):
            numChild -= 1
            child.append(child2)

    return child


def mutation(structure, childs, probability=0.1, score=10) -> list:

    # Definir el rango maximo y minimo de numero a sumar 
    base = 5
    maxRange = 1.2/np.emath.logn(base, score)

    # Numero de estructura, por defecto descontar la capa de entrada
    numStruct = len(structure) - 1

    # Recorrer cada hijo y aplicar la mutacion
    for child in childs:
        if(np.random.rand() < probability):

            # Aplicar x mutacion a cada capa
            for i in range(numStruct):
                nPrev = structure[i]
                nNext = structure[i + 1]

                # Definir cantidad de numeros a mutar, se puede usar max o min
                nMut = np.random.randint(low=1, high=min(nPrev, nNext))

                # Definir los indices i y j a mutar
                idxMut = np.random.randint(low=[0, 0], high=[nNext, nPrev + 1], size=(nMut, 2))

                # Sumar numero de mutacion a los indices ij definidos
                child[i][idxMut[:, 0], idxMut[:, 1]] += np.random.uniform(low=-maxRange, high=maxRange)

    return childs







