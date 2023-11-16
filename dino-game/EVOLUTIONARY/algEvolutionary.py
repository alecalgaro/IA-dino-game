from EVOLUTIONARY.Operators.selectionOperator import *
from EVOLUTIONARY.Operators.reproductionOperator import *
import numpy as np
import os

#* Funciones para realizar el proceso evolutivo

#* Funcion para actualizar la poblacion mediante seleccion, cruza, mutacion y reemplazo
def updatePopulation(SELECT_OPER, dataPopulation, NUM_PARENT, REPLACE,
                     NEURAL_STRUCTURE, N_DINO, PROB_CROSS, PROB_MUTA, 
                     points, elite, link, register):
    parent = []

    numIndiv = len(dataPopulation)

    # el num de padres viene como porcentaje, entonces lo convertimos a entero
    if(isinstance(NUM_PARENT, float) and NUM_PARENT <= 1):
        numParent = int(numIndiv * NUM_PARENT)

    match(SELECT_OPER):     # eleccion del operador de seleccion
        case 0:     # ventana
            parent = window(dataPopulation, numParent=numParent, numIndiv=numIndiv, replace=REPLACE)
        case 1:     # competicion
            parent = competition(dataPopulation, numParent=numParent, numIndiv=numIndiv, replace=REPLACE)
        case _:     # ruleta
            parent = roulette(dataPopulation, numParent=numParent, numIndiv=numIndiv, replace=REPLACE)
    
    parent[0] = elite

    # Cruza
    child = crossover(structure=NEURAL_STRUCTURE, parent=parent, 
                        nDino=N_DINO, probability=PROB_CROSS)
    # Mutacion
    child = mutation(structure=NEURAL_STRUCTURE, childs=child, 
                        probability=PROB_MUTA, score=points)

    # Generar poblacion uniendo padres e hijos
    population = parent + child

    # Almacenar poblacion en las carpetas que le correspondan
    savePopulation(population, link)
    saveRegister(register, link)

#* Almacenar Poblacion
def savePopulation(brains, link):
    for idx, Wji in enumerate(brains):
        path = link + 'brain_' + str(idx) + '.csv'

        with open(path, 'w') as file:
            for weight in Wji:
                np.savetxt(file, weight, delimiter=',')

#* Almacenar el registro de poblacion (tiene la cantidad de generaciones o puntuacion maxima)
def saveRegister(register, link) -> None:
    np.savetxt(link + 'register.csv', register, delimiter=',')

#* Obtener registro de la poblacion (cantidad de generaciones o puntuacion maxima)
def getPopulationRegister(EVOLUTIONARY, INIT_DINOBRAIN, link):
    """
    When it IS EVOLUTIONARY, it returns
    [0] -> generation 
    [1] -> maxScore
    """
    register = [int(0), int(0)]
    if(EVOLUTIONARY and not(INIT_DINOBRAIN)):
        path = link + "register.csv"    # se encuentra dentro de cada carpeta dinoBrain_estructura
        
        # Si existe el registro, extrae los datos
        if(os.path.exists(path)):
            register = np.genfromtxt(path, delimiter=',')

        # Sino crea el registro
        else:
            np.savetxt(path, register, delimiter=",")
    
    return register