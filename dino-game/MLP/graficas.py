import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches  # Importar para crear objetos proxy

def iniciarGraficasErrores():
    fig, ax = plt.subplots()
    ax.set_title("Error cuadratico promedio por epoca")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Error cuadrático promedio")
    # fig.show()    # la muestro cuando llamo al algoritmo de entrenamiento, para que no se cierre al terminar

    fig2, ax2 = plt.subplots()
    ax2.set_title("Porcentaje de error por epoca")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Porcentaje de error")
    # fig2.show()   

    return (ax, ax2)

def actualizarGraficasErrores(ax, ax2, errorCuadPlot, tasaErrorPlot):
    plt.sca(ax)
    plt.cla()
    ax.set_title("Error cuadratico promedio por epoca")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Error cuadrático promedio")
    ax.plot(errorCuadPlot)

    plt.sca(ax2)
    plt.cla()
    ax2.set_title("Porcentaje de error por epoca")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Porcentaje de error")
    ax2.plot(tasaErrorPlot)

    plt.pause(0.01)

# "scatter" se usa especialmente para graficar puntos, por ejemplo, para graficos de dispersion,
# mientras que "plot" se usa generalmente para lineas o curvas
def graficarDatosPorCategoria(X, colorPorCategoria, markerPorCategoria):
    fig3, ax3 = plt.subplots()
    ax3.set_title("Clasificación de datos por categoria")
    ax3.set_xlabel("X1")
    ax3.set_ylabel("X2")

    # Funciona pero ver si hay una mejor opcion porque demora un poco mas de esta manera.
    # El problema es que "marker" no recibe un arreglo directamente como markerPorCategoria,
    # sino que hay que recorrerlo para pasar cada valor dentro del for (y eso demora mas). En 
    # cambio "c" si permite el arreglo de colores directamente.
    # El parametro "s" es size, para asignar el tamano de los puntos
    
    for i in range(len(X)):
        ax3.scatter(X[i, 1], X[i, 2], c=colorPorCategoria[i], marker=markerPorCategoria[i], s=15)

    # Crear objetos proxy para la leyenda
    legend_labels = ['Verdadero +', 'Verdadero -', 'Falso -', 'Falso +']
    legend_handles = [mpatches.Patch(color=color, label=label) for color, label in zip(['grey', 'red', 'green', 'blue'], legend_labels)]

    # Agregar la leyenda
    ax3.legend(handles=legend_handles)

    # scatter = ax3.plot(X[:, 1], X[:, 2], c=colorPorCategoria)
    # fig3.show()