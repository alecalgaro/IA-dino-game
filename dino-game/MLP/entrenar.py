import numpy as np
from entrenamiento_MLP_v1 import *
from entrenamiento_MLP_v2 import *

archivoTrain = 'dataSet.csv'
archivoValidation = 'dataSetValidation.csv'
arquitectura = [6, 3]
tasaErrorAceptable = 0.01
numMaxEpocas = 1000
gammab = 0.01
bSigm = 5

Wji = entrenamiento_MLP_v2(archivoTrain, 
                        archivoValidation,
                        arquitectura, 
                        tasaErrorAceptable, 
                        numMaxEpocas,
                        gamma=gammab,
                        bSigmoidea=bSigm)

print(Wji)

with open("neurWeightMLP.csv", 'w') as file:
    for weight in Wji:
        np.savetxt(file, weight, delimiter=',')
