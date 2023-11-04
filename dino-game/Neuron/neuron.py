import numpy as np
import pandas as pd

class Neuron:
    def __init__(self) -> None:
        self.Wji = []

    # Cargar pesos de redes neuronal 
    def loadNeuralWeight(self, structure):
        """
        Resibe una lista cuyo elementos indican la cantidad 
        de neuronas de cada capas.
        La lista NO incluye a la capa de entrada.
        """
        skipRow = 0
        for nRow in structure:
            layerWeight = pd.read_csv('neurWeightMLP.csv', delimiter=',', 
                                      header=None, skiprows=skipRow, nrows=nRow)
            skipRow += nRow
            self.Wji.append(layerWeight.to_numpy())
    
    def initNeuralWeight(self, structure):
        """
        Inicializacion aleatoria de los pesos en el rango [-0.5, 0.5]
        Asumimos que la capa de entrada cuenta con 5 dimensiones.
        """
        nPrev = 5 # Capa de entrada
        for nNext in structure:
            self.Wji.append(np.random.rand(nNext, nPrev + 1) - 0.5)
            nPrev = nNext

    def _sigmoidea(self, Wji, Xi, alpha):
        Vi = Wji@Xi
        Y = 2/(1 + np.exp(-alpha * Vi)) - 1
        return Y

    def forwardPropagation(self, Xi, alpha):

        # Concatenar el -1 de bias con el resto de las entradas
        input = np.concatenate([[-1], Xi])
        for Wji in self.Wji:
            Y = self._sigmoidea(Wji, input, alpha)
            input = np.concatenate([[-1], Y])

        # Debe terminar en un vector 1x3 y despejamos el winnerTakeAll
        idxMax = np.argmax(Y)
        Y = np.full(shape=(3), fill_value=False, dtype=bool)
        Y[idxMax] = True

        return Y

    def getNeuralWeight(self):
        return self.Wji


#? Test
import time
tt = time.time()
brain = Neuron()
brain.loadNeuralWeight([10, 5, 5, 3])
# print(brain.getNeuralWeight())
print(f"tiempo usado es: {time.time() - tt}")


Xi = [1,82,300,102,95]
alpha = 1
res = brain.forwardPropagation(Xi, alpha) # No tardo nada en terminar el proceso
print(res)
