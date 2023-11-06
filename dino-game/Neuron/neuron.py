import numpy as np
import pandas as pd

class Neuron:
    def __init__(self) -> None:
        self.Wji = []

    # Cargar pesos de redes neuronal 
    def loadNeuralWeight(self, link, structure) -> None:
        """
        Recibe una lista cuyos elementos indican la cantidad de neuronas de cada capa.
        La lista NO incluye a la capa de entrada.
        """
        skipRow = 0
        for nRow in structure:
            layerWeight = pd.read_csv(link, delimiter=',', 
                                      header=None, skiprows=skipRow, nrows=nRow)
            skipRow += nRow
            self.Wji.append(layerWeight.to_numpy())
    
    def initNeuralWeight(self, structure) -> None:
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

    def forwardPropagation(self, Xi, alpha) -> list:

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

    def setNeuralWeight(self, Wji) -> None:
        self.Wji = Wji

    def getNeuralWeight(self) -> list:
        return self.Wji


#? Test
# brain = Neuron()
# #* Cargar peso
# link = 'neurWeightMLP.csv'
# brain.loadNeuralWeight(link, [5, 3])

# #* Inicializar pesos aleatorios
# # brain.initNeuralWeight([10, 5, 5, 3])

# Wji = brain.getNeuralWeight()

# for ww in Wji:
#     print(ww.shape)


#? Forward propagation test
# Xi = [1,82,300,102,95]
# Xi = [2.682203389830511497e+01,3.100000000000000000e+02,3.000000000000000000e+02,4.800000000000000000e+01,9.500000000000000000e+01]
# alpha = 5
# res = brain.forwardPropagation(Xi, alpha)
# print(res)
