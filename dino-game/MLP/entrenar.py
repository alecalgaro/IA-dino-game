import numpy as np
from entrenamiento_MLP import *

archivoTrain = 'train.csv'
archivoValidation = 'test.csv'
arquitectura = [6, 3]
tasaErrorAceptable = 0.01
numMaxEpocas = 500
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

print(Wji)

#* Guardar pesos obtenidos en archivo csv
# with open("neurWeightMLP.csv", 'w') as file:
#     for weight in Wji:
#         np.savetxt(file, weight, delimiter=',')

# Para que permanezcan las graficas
plt.show()