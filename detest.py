#
#
#

import numpy
import time

import desolver



class MySolve(desolver.DESolver):
    #some test data
    xData = numpy.array([5.357, 9.861, 5.457, 5.936, 6.161, 6.731])
    yData = numpy.array([0.376, 7.104, 0.489, 1.049, 1.327, 2.077])

    def error_func(self, indiv, *args):
        # inverse exponential with offset, y = a * exp(b/x) + c
        predicted = indiv[0] * numpy.exp(indiv[1] / self.xData) + indiv[2]

        # sum of squared error
        error = predicted - self.yData
        return numpy.sum(error*error)


if __name__ == '__main__':

    # parameterRanges, populationSize, maxGenerations, deStrategy,
    # diffScale, crossoverProb, cutoffEnergy,
    # useClassRandomNumberMethods, polishTheBestTrials
    tStart = time.time()
    #solver = desolver.DESolver([(-100,100)]*3, 30, 600,
    solver = MySolve([(-100,100)]*3, 30, 600,
                               "Rand2Exp",
                 args=None, scale=0.7, crossover_prob=0.6,
                 goal_val=.01, polish=True, use_pp = True)
    tElapsed = time.time() - tStart
    print "Best solution:", solver.best_ind
    print "Best energy:", solver.best_val, \
          ": Elapsed time", tElapsed, \
          'seconds for', solver.gen, 'generation(s)'
    print tElapsed / solver.gen, 'seconds per generation with polisher off'
    print
