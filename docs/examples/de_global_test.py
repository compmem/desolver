import desolver
import time
import numpy

import global_fun

def dep_error_func(indiv, *args):
    if global_fun.fun is None:
        print "Called init"
        global_fun.fun = global_fun.FunToOpt()
    return global_fun.fun(indiv, *args)

if __name__ == '__main__':

    # set the data
    xData = numpy.array([5.357, 9.861, 5.457, 5.936, 6.161, 6.731])
    yData = numpy.array([0.376, 7.104, 0.489, 1.049, 1.327, 2.077])

    # start timer
    tStart = time.time()

    # set up the solver
    solver = desolver.DESolver(dep_error_func,
        [(-100,100)]*3, 30, 600,
                     #method = desolver.DE_BEST_1,
                     #method = desolver.DE_BEST_1_JITTER,
                     #method = desolver.DE_LOCAL_TO_BEST_1,
                     method = desolver.DE_RAND_1,
                     args=[xData,yData], 
                     scale=[0.5,1.0], 
                     crossover_prob=0.9,
                     goal_error=.01, polish=False, verbose=True,
                     use_pp = True, pp_modules=['global_fun'],
                               pp_depfuncs=[])
    tElapsed = time.time() - tStart
    print
    print "Best generation:", solver.best_generation
    print "Best individual:", solver.best_individual
    print "Best error:", solver.best_error, \
          ": Elapsed time", tElapsed, \
          'seconds for', solver.generation+1, 'generation(s)'
    print tElapsed / (solver.generation+1), 'seconds per generation.'



