import numpy as np
import pandas as pd

class Neuron:
    def __init__(self) -> None:
        self.Wji = []

    def loadNeuralWeight(self, link, structure) -> None:
        """
        Funcion para cargar los pesos de la red neuronal.
        Recibe una lista cuyos elementos indican la cantidad de neuronas de cada capa:
        [capaEntrada, capasOcultas, capaSalida]
        """
        skipRow = 0
        for nRow in structure[1:]:
            layerWeight = pd.read_csv(link, delimiter=',', 
                                      header=None, skiprows=skipRow, nrows=nRow)
            skipRow += nRow
            self.Wji.append(layerWeight.to_numpy())
    
    def initNeuralWeight(self, structure) -> None:
        """
        Funcion para inicializacion aleatoria de los pesos en el rango [-0.5, 0.5].
        """
        nPrev = structure[0]    # Capa de entrada
        for nNext in structure[1:]:
            self.Wji.append(np.random.rand(nNext, nPrev + 1) - 0.5)
            nPrev = nNext

    def _sigmoidea(self, Wji, Xi, alpha):
        """
        Funcion de activacion no lineal: sigmoidea.
        """
        # print(f"Wji.shape = {Wji.shape}")
        # print(f"Xi.shape = {Xi.shape}")

        Vi = Wji@Xi

        # print(f"Vi = {Vi}") # Puede aparecer exponentes como 580, e^580 da error

        Y = 2/(1 + np.exp(-alpha * Vi)) - 1

        return Y

    def forwardPropagation(self, Xi, alpha) -> list:
        """
        Funcion para aplicar la propagacion hacia adelante.
        Entradas: vector de entradas y parametro de la funcion de activacion no lineal.
        Salida: salida de la red.
        """
        
        input = np.concatenate([[-1], Xi])  # Concatenar el -1 de bias con el resto de las entradas
        for Wji in self.Wji:
            Y = self._sigmoidea(Wji, input, alpha)
            input = np.concatenate([[-1], Y])

        # Debe terminar en un vector 1x3 y despejamos con winnerTakeAll o one hot
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
# link = 'neurWeightMLP.csv'
# brain.loadNeuralWeight(link, [5, 3])      # Cargar peso
# brain.initNeuralWeight([10, 5, 5, 3])     # Inicializar pesos aleatorios

# Wji = brain.getNeuralWeight()

# for ww in Wji:
#     print(ww.shape)

#? Forward propagation test
# Xi = [165,14,278,325,40,71,1,0,0]
# alpha = 1
# res = brain.forwardPropagation(Xi, alpha)
# print(res)