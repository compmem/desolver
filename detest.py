#
#
#

import numpy
import time

import desolver



class MySolve(desolver.DESolver):
    #some test data
    #xData = numpy.array([5.357, 9.861, 5.457, 5.936, 6.161, 6.731])
    #yData = numpy.array([0.376, 7.104, 0.489, 1.049, 1.327, 2.077])

    def error_func(self, indiv, *args):
        # inverse exponential with offset, y = a * exp(b/x) + c
        #predicted = indiv[0] * numpy.exp(indiv[1] / self.xData) + indiv[2]
        predicted = indiv[0] * numpy.exp(indiv[1] / args[1]) + indiv[2]

        # sum of squared error
        #error = predicted - self.yData
        error = predicted - args[2]
        return numpy.sum(error*error)


if __name__ == '__main__':

    # parameterRanges, populationSize, maxGenerations, deStrategy,
    # diffScale, crossoverProb, cutoffEnergy,
    # useClassRandomNumberMethods, polishTheBestTrials
    tStart = time.time()

    xData = numpy.array([5.357, 9.861, 5.457, 5.936, 6.161, 6.731])
    yData = numpy.array([0.376, 7.104, 0.489, 1.049, 1.327, 2.077])


    #solver = desolver.DESolver([(-100,100)]*3, 30, 600,
    solver = MySolve([(-100,100)]*3, 30, 600,
                     method = desolver.DE_RAND_1,
                     args=[xData,yData], scale=0.7, crossover_prob=0.6,
                     goal_error=.01, polish=True, verbose=False,
                     use_pp = True, pp_modules=['numpy'])
    tElapsed = time.time() - tStart
    print "Best individual:", solver.best_individual
    print "Best error:", solver.best_error, \
          ": Elapsed time", tElapsed, \
          'seconds for', solver.generation+1, 'generation(s)'
    print tElapsed / (solver.generation+1), 'seconds per generation.'
