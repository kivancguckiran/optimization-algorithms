from NiaPy.algorithms.basic import MothFlameOptimizer, DifferentialEvolution
from NiaPy.benchmarks import Ackley, Alpine1, Levy, Rastrigin, \
    Rosenbrock, StyblinskiTang

import matplotlib.pyplot as plt
import numpy as np


def plot(bench):
    def function(x1, x2):
        bench = Ackley().function()
        return bench(2, [x1, x2])

    plotN = 100
    x1 = np.linspace(bench.Lower, bench.Upper, plotN)
    x2 = np.linspace(bench.Lower, bench.Upper, plotN)

    x1, x2 = np.meshgrid(x1, x2)

    vecFunction = np.vectorize(function)
    z = vecFunction(x1, x2)

    fig = plt.figure()

    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x1, x2, z, rstride=1, cstride=1,
                           cmap=plt.cm.rainbow, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=7, cmap=plt.cm.coolwarm)

    ax.zaxis.set_major_locator(plt.LinearLocator(10))
    ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.01f'))

    plt.show()


def optimize(bench):
    mfo = MothFlameOptimizer(D=2, NP=20, nGEN=1000, benchmark=bench)
    de = DifferentialEvolution(D=2, NP=20, nGEN=1000, benchmark=bench)

    best_de = de.run()
    best_mfo = mfo.run()

    print('DE Best: ', best_de)
    print('MFO Best: ', best_mfo)


bench = Ackley(Lower=-5, Upper=5)
# bench = Alpine1(Lower=-10, Upper=10)
# bench = Levy(Lower=-7, Upper=7)
# bench = Rastrigin(Lower=-5, Upper=5)
# bench = Rosenbrock(Lower=-1, Upper=1)
# bench = StyblinskiTang(Lower=-5, Upper=5)

plot(bench)
optimize(bench)
