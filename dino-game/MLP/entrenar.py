import numpy as np
from entrenamiento_MLP_v1 import *
from entrenamiento_MLP_v2 import *

archivoTrain = 'train.csv'
archivoValidation = 'test_.csv'
arquitectura = [6, 3]
tasaErrorAceptable = 0.01
numMaxEpocas = 500
gammab = 0.001
bSigm = 1
grafError = True

Wji = entrenamiento_MLP_v2(archivoTrain, 
                        archivoValidation,
                        arquitectura, 
                        tasaErrorAceptable, 
                        numMaxEpocas,
                        gamma=gammab,
                        bSigmoidea=bSigm,
                        grafError=grafError)

print(Wji)

with open("neurWeightMLP.csv", 'w') as file:
    for weight in Wji:
        np.savetxt(file, weight, delimiter=',')

# Para que permanezcan las graficas
plt.show()