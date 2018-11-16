from NiaPy.algorithms.basic import MothFlameOptimizer, \
    DifferentialEvolution, ArtificialBeeColonyAlgorithm, \
    ParticleSwarmAlgorithm, BatAlgorithm, FireflyAlgorithm, \
    GeneticAlgorithm
from NiaPy.benchmarks import Ackley, Alpine1, Rastrigin, \
    Rosenbrock
from mpl_toolkits.mplot3d import axes3d, Axes3D

import matplotlib.pyplot as plt
import numpy as np


maxIter = 200
pop = 50
dim = 10
epoch = 10


def plot(bench, filename):
    def function(x1, x2):
        vec = bench.function()
        return vec(2, [x1, x2])

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

    plt.savefig(filename)


def optimize(bench, algo):
    average_mfo = 0
    average_de = 0
    average_abc = 0
    average_pso = 0
    average_ba = 0
    average_fa = 0
    average_ga = 0

    for i in np.arange(epoch):
        mfo = MothFlameOptimizer(D=dim, NP=pop, nGEN=maxIter, benchmark=bench)
        de = DifferentialEvolution(D=dim, NP=pop, nGEN=maxIter, benchmark=bench)
        abc = ArtificialBeeColonyAlgorithm(D=dim, NP=pop, nFES=maxIter, benchmark=bench)
        pso = ParticleSwarmAlgorithm(D=dim, NP=pop, nGEN=maxIter, benchmark=bench)
        ba = BatAlgorithm(D=dim, NP=pop, nFES=maxIter, benchmark=bench)
        fa = FireflyAlgorithm(D=dim, NP=pop, nFES=maxIter, benchmark=bench)
        ga = GeneticAlgorithm(D=dim, NP=pop, nFES=maxIter, benchmark=bench)

        gen, best_de = de.run()
        gen, best_mfo = mfo.run()
        gen, best_abc = abc.run()
        gen, best_pso = pso.run()
        gen, best_ba = ba.run()
        gen, best_fa = fa.run()
        gen, best_ga = ga.run()

        average_mfo += best_de / epoch
        average_de += best_mfo / epoch
        average_abc += best_abc / epoch
        average_pso += best_pso / epoch
        average_ba += best_ba / epoch
        average_fa += best_fa / epoch
        average_ga += best_ga / epoch

    print(algo, ': DE Average of Bests over', epoch, 'run: ', average_de)
    print(algo, ': MFO Average of Bests over', epoch, 'run: ', average_mfo)
    print(algo, ': ABC Average of Bests over', epoch, 'run: ', average_abc)
    print(algo, ': PSO Average of Bests over', epoch, 'run: ', average_pso)
    print(algo, ': BA Average of Bests over', epoch, 'run: ', average_ba)
    print(algo, ': FA Average of Bests over', epoch, 'run: ', average_fa)
    print(algo, ': GA Average of Bests over', epoch, 'run: ', average_ga)

    return [average_de, average_mfo, average_abc, average_pso, average_ba, average_fa, average_ga]


results = {}

bench = Ackley(Lower=-5, Upper=5)
# plot(bench, 'ackley.png')
de, mfo, abc, pso, ba, fa, ga = optimize(bench, 'Ackley')
results["ackley"] = {"de": de, "mfo": mfo, "abc": abc, "pso": pso, "ba": ba, "fa": fa, "ga": ga}
bench = Alpine1(Lower=-10, Upper=10)
# plot(bench, 'alpine.png')
de, mfo, abc, pso, ba, fa, ga = optimize(bench, 'Alpine')
results["alpine"] = {"de": de, "mfo": mfo, "abc": abc, "pso": pso, "ba": ba, "fa": fa, "ga": ga}
bench = Rastrigin(Lower=-5, Upper=5)
# plot(bench, 'rastrigin.png')
de, mfo, abc, pso, ba, fa, ga = optimize(bench, 'Rastrigin')
results["rastrigin"] = {"de": de, "mfo": mfo, "abc": abc, "pso": pso, "ba": ba, "fa": fa, "ga": ga}
bench = Rosenbrock(Lower=-1, Upper=1)
# plot(bench, 'rosenbrock.png')
de, mfo, abc, pso, ba, fa, ga = optimize(bench, 'Rosenbrock')
results["rosenbrock"] = {"de": de, "mfo": mfo, "abc": abc, "pso": pso, "ba": ba, "fa": fa, "ga": ga}

print(results)
