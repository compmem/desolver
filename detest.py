#
#
#

import numpy as np
import time

import desolve

#some test data
xData = np.array([5.357, 9.861, 5.457, 5.936, 6.161, 6.731])
yData = np.array([0.376, 7.104, 0.489, 1.049, 1.327, 2.077])


def err_func(trial, *args):
    # inverse exponential with offset, y = a * exp(b/x) + c
    predicted = trial[0] * np.exp(trial[1] / xData) + trial[2]
    
    # sum of squared error
    error = predicted - yData
    return np.sum(error*error)




if __name__ == '__main__':

    # parameterRanges, populationSize, maxGenerations, deStrategy,
    # diffScale, crossoverProb, cutoffEnergy,
    # useClassRandomNumberMethods, polishTheBestTrials
    tStart = time.time()
    solver = desolve.DESolver([(-100,100)]*3, 30, 600,
                        "Rand2Exp",
                 err_func, args=None, scale=0.7, crossover_prob=0.6,
                 goal_val=.01, polish=True)
    tElapsed = time.time() - tStart
    print "Best solution:", solver.best_ind
    print "Best energy:", solver.best_val, \
          ": Elapsed time", tElapsed, \
          'seconds for', solver.gen, 'generation(s)'
    print tElapsed / solver.gen, 'seconds per generation with polisher off'
    print
