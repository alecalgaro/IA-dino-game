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
                child1[idxCross], child2[idxCross] = child2[idxCross], child1[idxCross]
            
        numChild -= 1
        child.append(child1)

        if(numChild > 0):
            numChild -= 1
            child.append(child2)

    return child


def mutation(structure, childs, probability=0.1) -> list:
    # Recorrer cada hijo y aplicar la mutacion
    for child in childs:
        if(np.random.rand() < probability):
            