import numpy as np
from entrenamiento_MLP_v1 import *
from entrenamiento_MLP_v2 import *

nombreArchivo = 'dataSet.csv'
arquitectura = [5, 3]
tasaErrorAceptable = 0.01
numMaxEpocas = 500
gammab = 0.01
bSigm = 5

Wji = entrenamiento_MLP_v2(nombreArchivo, 
                        arquitectura, 
                        tasaErrorAceptable, 
                        numMaxEpocas,
                        gamma=gammab,
                        bSigmoidea=bSigm)

print(Wji)

with open("neurWeightMLP.csv", 'w') as file:
    for weight in Wji:
        np.savetxt(file, weight, delimiter=',')
