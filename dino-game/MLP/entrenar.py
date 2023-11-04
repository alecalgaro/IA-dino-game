import numpy as np
from entrenamiento_MLP import *

nombreArchivo = 'dataSet.csv'
arquitectura = [10, 3]
tasaErrorAceptable = 0.1
numMaxEpocas = 550
gammab = 0.018
bSigm = 15


Wji = entrenamiento_MLP(nombreArchivo, 
                        arquitectura, 
                        tasaErrorAceptable, 
                        numMaxEpocas,
                        gamma=gammab,
                        bSigmoidea=bSigm)

print(Wji)

with open("neurWeightMLP.csv", 'w') as file:
    for weight in Wji:
        np.savetxt(file, weight, delimiter=',')
