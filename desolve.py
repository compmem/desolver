#
#
#

import numpy as np

# http;//www.parallelpython.com -
# can be single CPU, multi-core SMP, or cluster parallelization
try:
    import pp
    USE_PP = True
except ImportError:
    USE_PP = False
    
# Import Psyco if available
try:
    import psyco
    psyco.full()
except ImportError:
    print "psyco not loaded"

class DESolver:
    """
    Genetic minimization based on Differential Evolution.
    """

    def __init__(self, param_ranges, pop_size, max_gen, method,
                 func, args=None, diff_scale=0.8, crossover_prob=0.9,
                 cutoff_val=1e-5, polish=True):
        """
        """
        # set the internal vars
        self.param_ranges = param_ranges
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.method = method
        self.func = func
        # prepend self to the args
        if args is None:
            args = [self]
        else:
            args.insert(0,self)
        self.args = args
        self.diff_scale = diff_scale
        self.crossover_prob = crossover_prob
        self.cutoff_val = cutoff_energy
        self.polish = polish

        # set status vars
        self.gen = -1

        # set up the population
        # eventually we can allow for unbounded min/max values with None
        self.pop = np.hstack([np.random.uniform(p[0],p[1],
                                                size=[self.pop_size,1])
                              for p in param_ranges])
        self.pop_vals = self._eval_pop(self.pop)

        # set the best
        self.best_val = self.pop_vals.min()
        self.best_ind = np.copy(self.pop[self.pop.argmin(),:])


    def _eval_pop(self,pop):
        """
        Evals the provided population, returning the vals from the
        function.
        """
        # allocate for the popvals
        pop_vals = np.empty(len(pop))

        # eval the function for the initial population
        for i in xrange(len(pop)):
            pop_vals[i] = func(pop[i,:],*(self.args))


    def _evolve_pop(self,pop):
        """
        Evolove to new generation of population.
        """
        pass
    
    def _solve(self):
        """
        Optimize the parameters of the function.
        """
