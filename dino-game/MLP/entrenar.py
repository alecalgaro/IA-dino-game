import numpy as np
from entrenamiento_MLP import *

archivoTrain = 'datasets/train.csv'
archivoValidation = 'datasets/test.csv'
arquitectura = [6, 3]   # esta arquitectura no lleva la capa de entrada
tasaErrorAceptable = 0.01
numMaxEpocas = 300
gammab = 0.001
bSigm = 1
grafError = True

#* Ejecutar entrenamiento y obtener pesos de la red
Wji = entrenamiento_MLP(archivoTrain, 
                        archivoValidation,
                        arquitectura, 
                        tasaErrorAceptable, 
                        numMaxEpocas,
                        gamma=gammab,
                        bSigmoidea=bSigm,
                        grafError=grafError)

# print(Wji)

#* Guardar pesos obtenidos en archivo csv
# with open("neurWeightMLP.csv", 'w') as file:
#     for weight in Wji:
#         np.savetxt(file, weight, delimiter=',')

# Para que permanezcan las graficas
plt.show()