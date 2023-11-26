import numpy as np
from matplotlib import ticker
import matplotlib.pyplot as plt


def plot_cross_validation_analysis():
    font = {'fontname': 'Arial'}

    def setup_figure(title):
        fig, axs = plt.subplots(3, 1)

        fig.suptitle(title, fontsize=14, **font)

        for ax in axs:
            ax.set_ylim([0, 0])
            ax.yaxis.set_visible(False)
            ax.spines['left'].set_color('None')
            ax.spines['right'].set_color('None')
            ax.spines['top'].set_color('None')
            ax.spines['bottom'].set_position(('data', 0))
            ax.tick_params(labelbottom=True)

        return fig, axs


    # Gradient-boosted Trees
    # Iteración 1
    fig1, axs1 = setup_figure("Gradient-boosted Trees")

    # maxDepth
    maxDepth1 = np.array([3, 10, 17])
    maxDepth1Limits = [0, 20]
    axs1[0].set_title("maxDepth", fontsize=10, loc="left", **font)
    axs1[0].set_xlim(maxDepth1Limits)
    axs1[0].xaxis.set_major_locator(ticker.FixedLocator(maxDepth1Limits))
    for xy in zip(maxDepth1, np.zeros(3)):
        axs1[0].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .15, -.018), textcoords='data', c='red')

    axs1[0].plot(maxDepth1, np.zeros(3), "r|", linewidth=15)

    # maxBins
    maxBins1 = np.array([38, 100, 200])
    maxBins1Limits =[0, 250]
    axs1[1].set_title("maxBins", fontsize=10, loc="left", **font)
    axs1[1].set_xlim(maxBins1Limits)
    axs1[1].xaxis.set_major_locator(ticker.FixedLocator(maxBins1Limits))
    for xy in zip(maxBins1, np.zeros(3)):
        axs1[1].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='red')

    axs1[1].plot(maxBins1, np.zeros(3), "r|", linewidth=15)

    # maxIter
    maxIter1 = np.array([5, 15, 25])
    maxIter1Limits = [0, 30]
    axs1[2].set_title("maxIter", fontsize=10, loc="left", **font)
    axs1[2].set_xlim(maxIter1Limits)
    axs1[2].xaxis.set_major_locator(ticker.FixedLocator(maxIter1Limits))
    for xy in zip(maxIter1, np.zeros(3)):
        axs1[2].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .25, -.018), textcoords='data', c='red')

    axs1[2].plot(maxIter1, np.zeros(3), "r|", linewidth=15)

    # Primera figura
    # plt.show()
    # Para la segunda, y posteriores figuras con los círculos verdes, se dibujaron los círculos a mano,
    # basado en las figuras idénticas que no llevan círculos.
    # Esto debido a que aquí los ejes Y fueron reducidos para graficar una recta,
    # y por ello no se pueden dibujar estructuras bidimensionales

    ##############################################################################################################

    # Random Forest
    # Iteración 1
    fig2, axs2 = setup_figure("Random Forest")

    # maxDepth
    maxDepth1 = np.array([3, 10, 17])
    maxDepth1Limits = [0, 20]
    axs2[0].set_title("maxDepth", fontsize=10, loc="left", **font)
    axs2[0].set_xlim(maxDepth1Limits)
    axs2[0].xaxis.set_major_locator(ticker.FixedLocator(maxDepth1Limits))
    for xy in zip(maxDepth1, np.zeros(3)):
        axs2[0].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .15, -.018), textcoords='data', c='red')

    axs2[0].plot(maxDepth1, np.zeros(3), "r|", linewidth=15)

    # maxBins
    maxBins1 = np.array([38, 100, 200])
    maxBins1Limits =[0, 250]
    axs2[1].set_title("maxBins", fontsize=10, loc="left", **font)
    axs2[1].set_xlim(maxBins1Limits)
    axs2[1].xaxis.set_major_locator(ticker.FixedLocator(maxBins1Limits))
    for xy in zip(maxBins1, np.zeros(3)):
        axs2[1].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='red')

    axs2[1].plot(maxBins1, np.zeros(3), "r|", linewidth=15)

    # numTrees
    numTrees1 = np.array([10, 100, 200])
    numTrees1Limits = [0, 250]
    axs2[2].set_title("numTrees", fontsize=10, loc="left", **font)
    axs2[2].set_xlim(numTrees1Limits)
    axs2[2].xaxis.set_major_locator(ticker.FixedLocator(numTrees1Limits))
    for xy in zip(numTrees1, np.zeros(3)):
        axs2[2].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='red')

    axs2[2].plot(numTrees1, np.zeros(3), "r|", linewidth=15)

    # Tercera figura
    # plt.show()

    ##############################################################################################################

    # Gradient-boosted Trees
    # Iteración 2
    fig3, axs3 = setup_figure("Gradient-boosted Trees")

    # maxDepth
    maxDepth1 = np.array([7, 10, 13])
    maxDepth1Limits = [0, 20]
    axs3[0].set_title("maxDepth", fontsize=10, loc="left", **font)
    axs3[0].set_xlim(maxDepth1Limits)
    axs3[0].xaxis.set_major_locator(ticker.FixedLocator(maxDepth1Limits))
    for xy in zip(maxDepth1, np.zeros(3)):
        axs3[0].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .15, -.018), textcoords='data', c='red')

    axs3[0].plot(maxDepth1, np.zeros(3), "r|", linewidth=15)

    # maxBins
    maxBins1 = np.array([60, 100, 150])
    maxBins1Limits =[0, 250]
    axs3[1].set_title("maxBins", fontsize=10, loc="left", **font)
    axs3[1].set_xlim(maxBins1Limits)
    axs3[1].xaxis.set_major_locator(ticker.FixedLocator(maxBins1Limits))
    for xy in zip(maxBins1, np.zeros(3)):
        axs3[1].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='red')

    axs3[1].plot(maxBins1, np.zeros(3), "r|", linewidth=15)

    # maxIter
    maxIter1 = np.array([10, 15, 20])
    maxIter1Limits = [0, 30]
    axs3[2].set_title("maxIter", fontsize=10, loc="left", **font)
    axs3[2].set_xlim(maxIter1Limits)
    axs3[2].xaxis.set_major_locator(ticker.FixedLocator(maxIter1Limits))
    for xy in zip(maxIter1, np.zeros(3)):
        axs3[2].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .25, -.018), textcoords='data', c='red')

    axs3[2].plot(maxIter1, np.zeros(3), "r|", linewidth=15)

    # Quinta figura
    # plt.show()

    ##############################################################################################################

    # Gradient-boosted Trees
    # Iteración 2 vs Iteración 1
    fig4, axs4 = setup_figure("Gradient-boosted Trees")

    # maxDepth
    maxDepth1 = np.array([7, 10, 13])
    maxDepth1Limits = [0, 20]
    axs4[0].set_title("maxDepth", fontsize=10, loc="left", **font)
    axs4[0].set_xlim(maxDepth1Limits)
    axs4[0].xaxis.set_major_locator(ticker.FixedLocator(maxDepth1Limits))
    for xy in zip(maxDepth1, np.zeros(3)):
        axs4[0].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .15, -.018), textcoords='data', c='red')

    axs4[0].plot(maxDepth1, np.zeros(3), "r|", linewidth=15)

    # Mejor parámetro anterior iteración
    for xy in zip(np.array([10]), np.zeros(1)):
        axs4[0].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .15, -.018), textcoords='data', c='blue')
    axs4[0].plot(np.array([10]), np.zeros(1), "b|", linewidth=15)

    # maxBins
    maxBins1 = np.array([60, 100, 150])
    maxBins1Limits =[0, 250]
    axs4[1].set_title("maxBins", fontsize=10, loc="left", **font)
    axs4[1].set_xlim(maxBins1Limits)
    axs4[1].xaxis.set_major_locator(ticker.FixedLocator(maxBins1Limits))
    for xy in zip(maxBins1, np.zeros(3)):
        axs4[1].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='red')

    axs4[1].plot(maxBins1, np.zeros(3), "r|", linewidth=15)

    # Mejor parámetro anterior iteración
    for xy in zip(np.array([100]), np.zeros(1)):
        axs4[1].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='blue')
    axs4[1].plot(np.array([100]), np.zeros(1), "b|", linewidth=15)

    # maxIter
    maxIter1 = np.array([10, 15, 20])
    maxIter1Limits = [0, 30]
    axs4[2].set_title("maxIter", fontsize=10, loc="left", **font)
    axs4[2].set_xlim(maxIter1Limits)
    axs4[2].xaxis.set_major_locator(ticker.FixedLocator(maxIter1Limits))
    for xy in zip(maxIter1, np.zeros(3)):
        axs4[2].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .25, -.018), textcoords='data', c='red')

    axs4[2].plot(maxIter1, np.zeros(3), "r|", linewidth=15)

    # Mejor parámetro anterior iteración
    for xy in zip(np.array([15]), np.zeros(1)):
        axs4[2].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .25, -.018), textcoords='data', c='blue')
    axs4[2].plot(np.array([15]), np.zeros(1), "b|", linewidth=15)

    # Sexta figura (sin dibujar círculos verdes, ésto se hace a mano)
    # plt.show()

    ##############################################################################################################

    # Random Forest
    # Iteración 2
    fig5, axs5 = setup_figure("Random Forest")

    # maxDepth
    maxDepth1 = np.array([7, 10, 13])
    maxDepth1Limits = [0, 20]
    axs5[0].set_title("maxDepth", fontsize=10, loc="left", **font)
    axs5[0].set_xlim(maxDepth1Limits)
    axs5[0].xaxis.set_major_locator(ticker.FixedLocator(maxDepth1Limits))
    for xy in zip(maxDepth1, np.zeros(3)):
        axs5[0].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .15, -.018), textcoords='data', c='red')

    axs5[0].plot(maxDepth1, np.zeros(3), "r|", linewidth=15)

    # maxBins
    maxBins1 = np.array([60, 100, 150])
    maxBins1Limits =[0, 250]
    axs5[1].set_title("maxBins", fontsize=10, loc="left", **font)
    axs5[1].set_xlim(maxBins1Limits)
    axs5[1].xaxis.set_major_locator(ticker.FixedLocator(maxBins1Limits))
    for xy in zip(maxBins1, np.zeros(3)):
        axs5[1].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='red')

    axs5[1].plot(maxBins1, np.zeros(3), "r|", linewidth=15)

    # numTrees
    numTrees1 = np.array([60, 100, 150])
    numTrees1Limits = [0, 250]
    axs5[2].set_title("numTrees", fontsize=10, loc="left", **font)
    axs5[2].set_xlim(numTrees1Limits)
    axs5[2].xaxis.set_major_locator(ticker.FixedLocator(numTrees1Limits))
    for xy in zip(numTrees1, np.zeros(3)):
        axs5[2].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='red')

    axs5[2].plot(numTrees1, np.zeros(3), "r|", linewidth=15)

    # Séptima figura
    # plt.show()

    ##############################################################################################################

    # Random Forest
    # Iteración 2 vs Iteración 1
    fig6, axs6 = setup_figure("Random Forest")

    # maxDepth
    maxDepth1 = np.array([7, 10, 13])
    maxDepth1Limits = [0, 20]
    axs6[0].set_title("maxDepth", fontsize=10, loc="left", **font)
    axs6[0].set_xlim(maxDepth1Limits)
    axs6[0].xaxis.set_major_locator(ticker.FixedLocator(maxDepth1Limits))
    for xy in zip(maxDepth1, np.zeros(3)):
        axs6[0].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .15, -.018), textcoords='data', c='red')

    axs6[0].plot(maxDepth1, np.zeros(3), "r|", linewidth=15)

    # Mejor parámetro anterior iteración
    for xy in zip(np.array([10]), np.zeros(1)):
        axs6[0].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .15, -.018), textcoords='data', c='blue')

    axs6[0].plot(np.array([10]), np.zeros(1), "b|", linewidth=15)

    # maxBins
    maxBins1 = np.array([60, 100, 150])
    maxBins1Limits =[0, 250]
    axs6[1].set_title("maxBins", fontsize=10, loc="left", **font)
    axs6[1].set_xlim(maxBins1Limits)
    axs6[1].xaxis.set_major_locator(ticker.FixedLocator(maxBins1Limits))
    for xy in zip(maxBins1, np.zeros(3)):
        axs6[1].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='red')

    axs6[1].plot(maxBins1, np.zeros(3), "r|", linewidth=15)

    # Mejor parámetro anterior iteración
    for xy in zip(np.array([100]), np.zeros(1)):
        axs6[1].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='blue')

    axs6[1].plot(np.array([100]), np.zeros(1), "b|", linewidth=15)

    # numTrees
    numTrees1 = np.array([60, 100, 150])
    numTrees1Limits = [0, 250]
    axs6[2].set_title("numTrees", fontsize=10, loc="left", **font)
    axs6[2].set_xlim(numTrees1Limits)
    axs6[2].xaxis.set_major_locator(ticker.FixedLocator(numTrees1Limits))
    for xy in zip(numTrees1, np.zeros(3)):
        axs6[2].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='red')

    axs6[2].plot(numTrees1, np.zeros(3), "r|", linewidth=15)

    # Mejor parámetro anterior iteración
    for xy in zip(np.array([100]), np.zeros(1)):
        axs6[2].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='blue')

    axs6[2].plot(np.array([100]), np.zeros(1), "b|", linewidth=15)

    # Octava figura (sin dibujar círculos verdes, ésto se hace a mano)
    # plt.show()

    ##############################################################################################################

    # Gradient-boosted Trees
    # Iteración 3
    fig7, axs7 = setup_figure("Gradient-boosted Trees")

    # maxDepth
    maxDepth1 = np.array([3, 5, 7])
    maxDepth1Limits = [0, 20]
    axs7[0].set_title("maxDepth", fontsize=10, loc="left", **font)
    axs7[0].set_xlim(maxDepth1Limits)
    axs7[0].xaxis.set_major_locator(ticker.FixedLocator(maxDepth1Limits))
    for xy in zip(maxDepth1, np.zeros(3)):
        axs7[0].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .15, -.018), textcoords='data', c='red')

    axs7[0].plot(maxDepth1, np.zeros(3), "r|", linewidth=15)

    # maxBins
    maxBins1 = np.array([38, 50, 60])
    maxBins1Limits =[0, 250]
    axs7[1].set_title("maxBins", fontsize=10, loc="left", **font)
    axs7[1].set_xlim(maxBins1Limits)
    axs7[1].xaxis.set_major_locator(ticker.FixedLocator(maxBins1Limits))
    for xy in zip(maxBins1, np.zeros(3)):
        axs7[1].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='red')

    axs7[1].plot(maxBins1, np.zeros(3), "r|", linewidth=15)

    # maxIter
    maxIter1 = np.array([20, 22, 25])
    maxIter1Limits = [0, 30]
    axs7[2].set_title("maxIter", fontsize=10, loc="left", **font)
    axs7[2].set_xlim(maxIter1Limits)
    axs7[2].xaxis.set_major_locator(ticker.FixedLocator(maxIter1Limits))
    for xy in zip(maxIter1, np.zeros(3)):
        axs7[2].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .25, -.018), textcoords='data', c='red')

    axs7[2].plot(maxIter1, np.zeros(3), "r|", linewidth=15)

    # Novena figura
    # plt.show()

    ##############################################################################################################

    # Gradient-boosted Trees
    # Iteración 3 vs Iteraciones 1 y 2
    fig8, axs8 = setup_figure("Gradient-boosted Trees")

    # maxDepth
    maxDepth1 = np.array([3, 5, 7])
    maxDepth1Limits = [0, 20]
    axs8[0].set_title("maxDepth", fontsize=10, loc="left", **font)
    axs8[0].set_xlim(maxDepth1Limits)
    axs8[0].xaxis.set_major_locator(ticker.FixedLocator(maxDepth1Limits))
    for xy in zip(maxDepth1, np.zeros(3)):
        axs8[0].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .15, -.018), textcoords='data', c='red')

    axs8[0].plot(maxDepth1, np.zeros(3), "r|", linewidth=15)

    # Mejor parámetro iteración 1
    for xy in zip(np.array([10]), np.zeros(1)):
        axs8[0].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .15, -.018), textcoords='data', c='blue')
    axs8[0].plot(np.array([10]), np.zeros(1), "b|", linewidth=15)

    # Mejor parámetro iteración 2
    for xy in zip(np.array([7]), np.zeros(1)):
        axs8[0].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .15, -.018), textcoords='data', c='purple')
    axs8[0].plot(np.array([7]), np.zeros(1), "m|", linewidth=15)

    # maxBins
    maxBins1 = np.array([38, 50, 60])
    maxBins1Limits =[0, 250]
    axs8[1].set_title("maxBins", fontsize=10, loc="left", **font)
    axs8[1].set_xlim(maxBins1Limits)
    axs8[1].xaxis.set_major_locator(ticker.FixedLocator(maxBins1Limits))
    for xy in zip(maxBins1, np.zeros(3)):
        axs8[1].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='red')

    axs8[1].plot(maxBins1, np.zeros(3), "r|", linewidth=15)

    # Mejor parámetro iteración 1
    for xy in zip(np.array([100]), np.zeros(1)):
        axs8[1].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='blue')
    axs8[1].plot(np.array([100]), np.zeros(1), "b|", linewidth=15)

    # Mejor parámetro iteración 2
    for xy in zip(np.array([60]), np.zeros(1)):
        axs8[1].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='purple')
    axs8[1].plot(np.array([60]), np.zeros(1), "m|", linewidth=15)

    # maxIter
    maxIter1 = np.array([20, 22, 25])
    maxIter1Limits = [0, 30]
    axs8[2].set_title("maxIter", fontsize=10, loc="left", **font)
    axs8[2].set_xlim(maxIter1Limits)
    axs8[2].xaxis.set_major_locator(ticker.FixedLocator(maxIter1Limits))
    for xy in zip(maxIter1, np.zeros(3)):
        axs8[2].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .25, -.018), textcoords='data', c='red')

    axs8[2].plot(maxIter1, np.zeros(3), "r|", linewidth=15)

    # Mejor parámetro anterior iteración
    for xy in zip(np.array([15]), np.zeros(1)):
        axs8[2].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .25, -.018), textcoords='data', c='blue')
    axs8[2].plot(np.array([15]), np.zeros(1), "b|", linewidth=15)

    # Mejor parámetro anterior iteración
    for xy in zip(np.array([20]), np.zeros(1)):
        axs8[2].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .25, -.018), textcoords='data', c='purple')
    axs8[2].plot(np.array([20]), np.zeros(1), "m|", linewidth=15)

    # Décima figura (sin dibujar círculos verdes, ésto se hace a mano)
    # plt.show()

    ##############################################################################################################

    # Random Forest
    # Iteración 3
    fig9, axs9 = setup_figure("Random Forest")

    # maxDepth
    maxDepth1 = np.array([9, 10, 11])
    maxDepth1Limits = [0, 20]
    axs9[0].set_title("maxDepth", fontsize=10, loc="left", **font)
    axs9[0].set_xlim(maxDepth1Limits)
    axs9[0].xaxis.set_major_locator(ticker.FixedLocator(maxDepth1Limits))
    for xy in zip(maxDepth1, np.zeros(3)):
        axs9[0].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .15, -.018), textcoords='data', c='red')

    axs9[0].plot(maxDepth1, np.zeros(3), "r|", linewidth=15)

    # maxBins
    maxBins1 = np.array([150, 170, 200])
    maxBins1Limits =[0, 250]
    axs9[1].set_title("maxBins", fontsize=10, loc="left", **font)
    axs9[1].set_xlim(maxBins1Limits)
    axs9[1].xaxis.set_major_locator(ticker.FixedLocator(maxBins1Limits))
    for xy in zip(maxBins1, np.zeros(3)):
        axs9[1].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='red')

    axs9[1].plot(maxBins1, np.zeros(3), "r|", linewidth=15)

    # numTrees
    numTrees1 = np.array([150, 170, 200])
    numTrees1Limits = [0, 250]
    axs9[2].set_title("numTrees", fontsize=10, loc="left", **font)
    axs9[2].set_xlim(numTrees1Limits)
    axs9[2].xaxis.set_major_locator(ticker.FixedLocator(numTrees1Limits))
    for xy in zip(numTrees1, np.zeros(3)):
        axs9[2].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='red')

    axs9[2].plot(numTrees1, np.zeros(3), "r|", linewidth=15)

    # Undécima figura
    # plt.show()

    ##############################################################################################################

    # Random Forest
    # Iteración 3 vs Iteraciones 1 y 2
    fig10, axs10 = setup_figure("Random Forest")

    # maxDepth
    maxDepth1 = np.array([9, 10, 11])
    maxDepth1Limits = [0, 20]
    axs10[0].set_title("maxDepth", fontsize=10, loc="left", **font)
    axs10[0].set_xlim(maxDepth1Limits)
    axs10[0].xaxis.set_major_locator(ticker.FixedLocator(maxDepth1Limits))
    for xy in zip(maxDepth1, np.zeros(3)):
        axs10[0].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .15, -.018), textcoords='data', c='red')

    axs10[0].plot(maxDepth1, np.zeros(3), "r|", linewidth=15)

    # Mejor parámetro iteración 1
    for xy in zip(np.array([10]), np.zeros(1)):
        axs10[0].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .15, -.018), textcoords='data', c='blue')

    # Este 9.85 debería ser 10, pero para visualización se varió un poco
    axs10[0].plot(np.array([9.85]), np.zeros(1), "b|", linewidth=15)

    # Mejor parámetro iteración 2
    for xy in zip(np.array([10]), np.zeros(1)):
        axs10[0].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * .15, -.018), textcoords='data', c='purple')

    axs10[0].plot(np.array([10]), np.zeros(1), "m|", linewidth=15)

    # maxBins
    maxBins1 = np.array([150, 170, 200])
    maxBins1Limits =[0, 250]
    axs10[1].set_title("maxBins", fontsize=10, loc="left", **font)
    axs10[1].set_xlim(maxBins1Limits)
    axs10[1].xaxis.set_major_locator(ticker.FixedLocator(maxBins1Limits))
    for xy in zip(maxBins1, np.zeros(3)):
        axs10[1].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='red')

    axs10[1].plot(maxBins1, np.zeros(3), "r|", linewidth=15)

    # Mejor parámetro iteración 1
    for xy in zip(np.array([100]), np.zeros(1)):
        axs10[1].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='blue')

    axs10[1].plot(np.array([100]), np.zeros(1), "b|", linewidth=15)

    # Mejor parámetro iteración 2
    for xy in zip(np.array([150]), np.zeros(1)):
        axs10[1].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='purple')

    axs10[1].plot(np.array([150]), np.zeros(1), "m|", linewidth=15)

    # numTrees
    numTrees1 = np.array([150, 170, 200])
    numTrees1Limits = [0, 250]
    axs10[2].set_title("numTrees", fontsize=10, loc="left", **font)
    axs10[2].set_xlim(numTrees1Limits)
    axs10[2].xaxis.set_major_locator(ticker.FixedLocator(numTrees1Limits))
    for xy in zip(numTrees1, np.zeros(3)):
        axs10[2].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='red')

    axs10[2].plot(numTrees1, np.zeros(3), "r|", linewidth=15)

    # Mejor parámetro iteración 1
    for xy in zip(np.array([100]), np.zeros(1)):
        axs10[2].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='blue')

    axs10[2].plot(np.array([100]), np.zeros(1), "b|", linewidth=15)

    # Mejor parámetro iteración 2
    for xy in zip(np.array([150]), np.zeros(1)):
        axs10[2].annotate(xy[0], xy=xy, xytext=(xy[0] - (len(str(xy[0]))) * 2, -.018), textcoords='data', c='purple')

    axs10[2].plot(np.array([150]), np.zeros(1), "m|", linewidth=15)

    # Duodécima figura (sin dibujar círculos verdes, ésto se hace a mano)
    plt.show()

def plot_roc_curves():
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.style'] = 'normal'

    # De acuerdo a la memoria:

    # (0.0,0.0)
    # (0.02295505981247979,0.6179682824655894)
    # (0.03281603621079858,0.6783576985551851)
    # (0.03863562883931458,0.7222792168932205)
    # (0.053831231813773035,0.7576942805847653)
    # (0.0720982864532816,0.7966465760451398)
    # (0.09618493372130618,0.8257459177566897)
    # (0.13223407694794698,0.8595580063264084)
    # (0.177335919818946,0.8885825425322732)
    # (0.22017458777885549,0.9113875352654527)
    # (0.2824118978338183,0.9364580661708131)
    # (0.361946330423537,0.9555227836197315)
    # (0.4571613320400905,0.9702701547405318)
    # (0.5801810539928872,0.9847396768402155)
    # (0.7419980601357905,0.9947422416004104)
    # (0.9843194309731652,0.9999893134991878)
    # (1.0,1.0)

    fig1 = plt.figure()




    x1 = np.array([0.0, 0.02295505981247979, 0.03281603621079858, 0.03863562883931458, 0.053831231813773035, 0.0720982864532816, 0.09618493372130618, 0.13223407694794698, 0.177335919818946, 0.22017458777885549, 0.2824118978338183, 0.361946330423537, 0.4571613320400905, 0.5801810539928872, 0.7419980601357905, 0.9843194309731652, 1.0])
    y1 = np.array([0.0, 0.6179682824655894, 0.6783576985551851, 0.7222792168932205, 0.7576942805847653, 0.7966465760451398, 0.8257459177566897, 0.8595580063264084, 0.8885825425322732, 0.9113875352654527, 0.9364580661708131, 0.9555227836197315, 0.9702701547405318, 0.9847396768402155, 0.9947422416004104, 0.9999893134991878, 1.0])

    xyb = np.linspace(start=0, stop=1, num=17)

    plt.plot(x1, y1, 'r-')
    plt.plot(xyb, xyb, 'k--')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.tick_params(axis='both', which='major', labelsize=10)

    fig2 = plt.figure()

    x2 = np.array([0.0, 0.00048496605237633366, 0.0019398642095053346, 0.004849660523763337, 0.00921435499515034, 0.0187520206918849, 0.026996443582282575, 0.03572583252505658, 0.041060459101196246, 0.05754930488199159, 0.08567733591981895, 0.12415130940834142, 0.18800517297122535, 0.28192693178144196, 0.4277400581959263, 0.7258325250565794, 1.0])
    y2 = np.array([0.0, 0.3249337436949645, 0.40872659656322136, 0.471082328802257, 0.5266521330255621, 0.5856843635120116, 0.634083525690348, 0.6794690946396512, 0.7002009062152689, 0.7452979396426435, 0.7896362315123536, 0.8341455073950585, 0.8808562024450713, 0.9268722749422929, 0.9655680943831751, 0.9931072069761477, 1.0])

    plt.plot(x2, y2, 'g-')
    plt.plot(xyb, xyb, 'k--')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.tick_params(axis='both', which='major', labelsize=10)

    fig3 = plt.figure()

    plt.plot(x1, y1, 'r-', label="Gradient-boosted Trees")
    plt.plot(x2, y2, 'g-', label="Random Forest")
    plt.plot(xyb, xyb, 'k--')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.legend(loc="lower right")

    plt.show()

# Inicio script
# Descomentar el método que se quiere usar

# plot_cross_validation_analysis()

plot_roc_curves()