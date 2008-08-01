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
    pass #print "psyco not loaded"

class DESolver:
    """
    Genetic minimization based on Differential Evolution.
    """

    def __init__(self, param_ranges, pop_size, max_gen, method,
                 func, args=None, scale=0.8, crossover_prob=0.9,
                 goal_val=1e-3, polish=True):
        """
        """
        # set the internal vars
        self.param_ranges = param_ranges
        self.num_params = len(self.param_ranges)
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
        self.scale = scale
        self.crossover_prob = crossover_prob
        self.goal_val = goal_val
        self.polish = polish

        # set helper vars
        self.rot_ind = np.arange(self.pop_size)

        # set status vars
        self.gen = -1

        # set up the population
        # eventually we can allow for unbounded min/max values with None
        self.pop = np.hstack([np.random.uniform(p[0],p[1],
                                                size=[self.pop_size,1])
                              for p in param_ranges])
        self.pop_vals = np.empty(len(self.pop))
        self._eval_pop()

        # set the best
        self.best_val = self.pop_vals.min()
        self.best_ind = np.copy(self.pop[self.pop_vals.argmin(),:])

        # now solve
        self._solve()

    def _eval_pop(self):
        """
        Evals the provided population, returning the vals from the
        function.
        """
        # eval the function for the initial population
        for i in xrange(len(self.pop)):
            self.pop_vals[i] = self.func(self.pop[i,:],*(self.args))

    def _evolve_pop(self):
        """
        Evolove to new generation of population.
        """
        # save the old pop
        self.pop_old = self.pop.copy()
        self.pop_old_vals = self.pop_vals.copy()

        # index pointers
        rind = np.random.permutation(4)+1

        # shuffle the locations of the individuals
        ind1 = np.random.permutation(self.pop_size)
        pop1 = self.pop_old[ind1,:]
        
        # rotate for remaining indices
        rot = np.remainder(self.rot_ind + rind[0], self.pop_size)
        ind2 = ind1[rot,:]
        pop2 = self.pop_old[ind2,:]

        rot = np.remainder(self.rot_ind + rind[1], self.pop_size)
        ind3 = ind2[rot,:]
        pop3 = self.pop_old[ind3,:]

        rot = np.remainder(self.rot_ind + rind[2], self.pop_size)
        ind4 = ind3[rot,:]
        pop4 = self.pop_old[ind4,:]

        rot = np.remainder(self.rot_ind + rind[3], self.pop_size)
        ind5 = ind4[rot,:]
        pop5 = self.pop_old[ind5,:]
        
        # pop filled with best individual
        pop_best = self.best_ind[np.newaxis,:].repeat(self.pop_size,axis=0)

        # figure out the crossover ind
        xold_ind = np.random.rand(self.pop_size,self.num_params) >= \
            self.crossover_prob

        # get new pop based on desired strategy
        # DE/rand/1
        pop = pop3 + self.scale*(pop1 - pop2)
        pop_orig = pop3

        # crossover
        pop[xold_ind] = self.pop_old[xold_ind]

        # apply the boundary constraints
        for p in xrange(self.num_params):
            # get min and max
            min_val = self.param_ranges[p][0]
            max_val = self.param_ranges[p][1]

            # find where exceeded max
            ind = pop[:,p] > max_val
            if ind.sum() > 0:
                # bounce back
                pop[ind,p] = max_val + np.random.rand(ind.sum())*(pop_orig[ind,p]-max_val)

            # find where below min
            ind = pop[:,p] < min_val
            if ind.sum() > 0:
                # bounce back
                pop[ind,p] = min_val + np.random.rand(ind.sum())*(pop_orig[ind,p]-min_val)

        # set the class members
        self.pop = pop
        self.pop_orig = pop

    
    def _solve(self):
        """
        Optimize the parameters of the function.
        """

        # loop over generations
        for g in xrange(self.max_gen):
            # set the generation
            self.gen = g

            # update the population
            self._evolve_pop()
            
            # evaluate the population
            self._eval_pop()

            # decide what stays
            ind = self.pop_vals > self.pop_old_vals
            self.pop[ind,:] = self.pop_old[ind,:]
            self.pop_vals[ind] = self.pop_old_vals[ind]

            # update what is best
            self.best_val = self.pop_vals.min()
            self.best_ind = np.copy(self.pop[self.pop_vals.argmin(),:])

            # see if done
            if self.best_val < self.goal_val:
                break
