#
#
#

import numpy
import scipy.optimize
import sys

# http;//www.parallelpython.com -
# can be single CPU, multi-core SMP, or cluster parallelization
try:
    import pp
    HAS_PP = True
except ImportError:
    HAS_PP = False
    
# # Import Psyco if available
# try:
#     import psyco
#     psyco.full()
# except ImportError:
#     pass #print "psyco not loaded"

# set up the enumerated DE method types
DE_RAND_1 = 0
DE_BEST_1 = 1
DE_BEST_2 = 2
DE_BEST_1_JITTER = 3
DE_LOCAL_TO_BEST_1 = 4

class DESolver(object):
    """
    Genetic minimization based on Differential Evolution.

    See http://www.icsi.berkeley.edu/~storn/code.html

    """

    def __init__(self, error_func, param_ranges, population_size, max_generations,
                 method = DE_RAND_1, args=None, seed=None,
                 param_names = None, scale=[0.5,1.0], crossover_prob=0.9,
                 goal_error=1e-3, polish=True, verbose=True,
                 use_pp=True, pp_depfuncs=None, pp_modules=None,
                 pp_proto=2, pp_ncpus='autodetect'):
        """
        Initialize and solve the minimization problem.

        
        """
        # set the internal vars
        #self.error_func = error_func # don't keep b/c can't pickle it
        self.param_ranges = param_ranges
        self.num_params = len(self.param_ranges)
        if param_names is None:
            # generate dummy names
            param_names = ['p_%d'%(x) for x in range(self.num_params)]
        self.param_names = param_names
        self.population_size = population_size
        self.max_generations = max_generations
        self.method = method
        if args is None:
            args = ()
        self.args = args
        self.scale = scale
        self.crossover_prob = crossover_prob
        self.goal_error = goal_error
        self.polish = polish
        self.verbose = verbose

        # set helper vars
        self.rot_ind = numpy.arange(self.population_size)

        # set status vars
        self.generation = 0

        # set the seed (must be 2D)
        self.seed = seed

        # set up the population
        if self.seed is None:
            # we'll be creating an entirely new population
            num_to_gen = self.population_size
        else:
            self.seed = numpy.atleast_2d(self.seed)
            num_to_gen = self.population_size - len(self.seed)

        # generate random population
        # eventually we can allow for unbounded min/max values with None
        self.population = numpy.hstack( \
            [numpy.random.uniform(p[0],p[1], size=[num_to_gen,1]) \
                 for p in param_ranges])

        # add in the seed if necessary
        if not self.seed is None:
            self.population = numpy.vstack([self.population,seed])

        self.population_errors = numpy.empty(self.population_size)

        # save the best error from each generation
        self.best_gen_errors = numpy.zeros(max_generations)*numpy.nan
        self.best_gen_indv = numpy.zeros((max_generations,self.num_params))*numpy.nan

        # check for pp
        if use_pp and not HAS_PP:
            print "WARNING: ParallelPython was not found on your system, "\
                  "so no parallelization will be performed."
            use_pp = False

        if use_pp:
            # auto-detects number of SMP CPU cores (will detect 1 core on
            # single-CPU systems)
            job_server = pp.Server(proto=pp_proto, ncpus=pp_ncpus)
            #job_server = pp.Server()

            # if self.verbose:
            #     print "Setting up %d pp_cpus" % (job_server.get_ncpus())

            # set up lists of depfuncs and modules
            depfuncs = []
            if not pp_depfuncs is None:
                depfuncs.extend(pp_depfuncs)
            self.depfuncs = tuple(depfuncs)
            #modules = ['desolver']
            modules = []
            if not pp_modules is None:
                modules.extend(pp_modules)
            self.modules = tuple(modules)

        else:
            job_server = None

        # the rest is now in a try block
        
        # try/finally block is to ensure remote worker processes are
        # killed if they were started
        try:
            # eval the initial population to fill errors
            self._eval_population(error_func, job_server=job_server)

            # set the index of the best individual
            best_ind = self.population_errors.argmin()
            self.best_error = self.population_errors[best_ind]
            self.best_individual = numpy.copy(self.population[best_ind,:])
            self.best_generation = self.generation

            # save the best for that gen
            self.best_gen_errors[0] = self.population_errors[best_ind]
            self.best_gen_indv[0,:] = self.population[best_ind,:]

            if self.verbose:
                self._report_best()

            # now solve
            self._solve(error_func, job_server=job_server)
        finally:
            # destroy the server if it was started
            if use_pp:
                job_server.destroy()

    def _indv_to_dictstr(self,indv):
        return '{' + \
            ', '.join(["'%s': %f" % (name,val) \
                           for name,val \
                           in zip(self.param_names,indv)]) + '}'

    def _report_best(self):
        print "Current generation: %g" % (self.generation)
        print "Current Best Error: %g" % (self.best_gen_errors[self.generation])
        print "Current Best Indiv: " + \
            self._indv_to_dictstr(self.best_gen_indv[self.generation,:])
        print "Overall Best generation: %g" % (self.best_generation)
        print "Overall Best Error: %g" % (self.best_error)
        #print "Best Indiv: " + str(self.best_individual)
        print "Overall Best Indiv: " + self._indv_to_dictstr(self.best_individual)
        print

    def get_scale(self):
        # generate random scale in range if desired
        if isinstance(self.scale,list):
            # return range
            return numpy.random.uniform(self.scale[0],self.scale[1])
        else:
            return self.scale


    def _eval_population(self, error_func, job_server=None):
        """
        Evals the provided population, returning the errors from the
        function.
        """
        # see if use job_server
        if self.verbose:
            print "Generation: %d (%d)" % (self.generation,self.max_generations)
            sys.stdout.write('Evaluating population (%d): ' % (self.population_size))
        if not job_server:
            # eval the function for the initial population
            for i in xrange(self.population_size):
                if self.verbose:
                    sys.stdout.write('%d ' % (i))
                    sys.stdout.flush()
                self.population_errors[i] = error_func(self.population[i,:],*(self.args))
        else:
            # submit the functions to the job server
            jobs = []
            for i in xrange(self.population_size):
                jobs.append(job_server.submit(error_func, 
                                              args=tuple([self.population[i,:]]+self.args),
                                              depfuncs=self.depfuncs,
                                              modules=self.modules))

            for i,job in enumerate(jobs):
                if self.verbose:
                    sys.stdout.write('%d ' % (i))
                    sys.stdout.flush()
                error = job()
                self.population_errors[i] = error

        if self.verbose:
            sys.stdout.write('\n')
            sys.stdout.flush()

    def _evolve_population(self):
        """
        Evolove to new generation of population.
        """
        # save the old population
        self.old_population = self.population.copy()
        self.old_population_errors = self.population_errors.copy()

        # index pointers
        rind = numpy.random.permutation(4)+1

        # shuffle the locations of the individuals
        ind1 = numpy.random.permutation(self.population_size)
        pop1 = self.old_population[ind1,:]
        
        # rotate for remaining indices
        rot = numpy.remainder(self.rot_ind + rind[0], self.population_size)
        ind2 = ind1[rot,:]
        pop2 = self.old_population[ind2,:]

        rot = numpy.remainder(self.rot_ind + rind[1], self.population_size)
        ind3 = ind2[rot,:]
        pop3 = self.old_population[ind3,:]

        rot = numpy.remainder(self.rot_ind + rind[2], self.population_size)
        ind4 = ind3[rot,:]
        pop4 = self.old_population[ind4,:]

        rot = numpy.remainder(self.rot_ind + rind[3], self.population_size)
        ind5 = ind4[rot,:]
        pop5 = self.old_population[ind5,:]
        
        # population filled with best individual
        best_population = self.best_individual[numpy.newaxis,:].repeat(self.population_size,axis=0)

        # figure out the crossover ind
        xold_ind = numpy.random.rand(self.population_size,self.num_params) >= \
            self.crossover_prob

        # get new population based on desired strategy
        # DE/rand/1
        if self.method == DE_RAND_1:
            population = pop3 + self.get_scale()*(pop1 - pop2)
            population_orig = pop3
        # DE/BEST/1
        if self.method == DE_BEST_1:
            population = best_population + self.get_scale()*(pop1 - pop2)
            population_orig = best_population
        # DE/best/2
        elif self.method == DE_BEST_2:
            population = best_population + self.get_scale() * \
                         (pop1 + pop2 - pop3 - pop4)
            population_orig = best_population
        # DE/BEST/1/JITTER
        elif self.method == DE_BEST_1_JITTER:
            population = best_population + (pop1 - pop2) * \
                         ((1.0-0.9999) * \
                          numpy.random.rand(self.population_size,self.num_params) + \
                          self.get_scale())
            population_orig = best_population
        # DE/LOCAL_TO_BEST/1
        elif self.method == DE_LOCAL_TO_BEST_1:
            population = self.old_population + \
                         self.get_scale()*(best_population - self.old_population) + \
                         self.get_scale()*(pop1 - pop2)
            population_orig = self.old_population
            
        # crossover
        population[xold_ind] = self.old_population[xold_ind]

        # apply the boundary constraints
        for p in xrange(self.num_params):
            # get min and max
            min_val = self.param_ranges[p][0]
            max_val = self.param_ranges[p][1]

            # find where exceeded max
            ind = population[:,p] > max_val
            if ind.sum() > 0:
                # bounce back
                population[ind,p] = max_val + \
                                    numpy.random.rand(ind.sum())*\
                                    (population_orig[ind,p]-max_val)

            # find where below min
            ind = population[:,p] < min_val
            if ind.sum() > 0:
                # bounce back
                population[ind,p] = min_val + \
                                    numpy.random.rand(ind.sum())*\
                                    (population_orig[ind,p]-min_val)

        # set the class members
        self.population = population
        self.population_orig = population

    
    def _solve(self, error_func, job_server=None):
        """
        Optimize the parameters of the function.
        """

        # loop over generations
        for g in xrange(1,self.max_generations):
            # set the generation
            self.generation = g

            # update the population
            self._evolve_population()
            
            # evaluate the population
            self._eval_population(error_func, job_server=job_server)

            # set the index of the best individual
            best_ind = self.population_errors.argmin()

            # update what is best
            if self.population_errors[best_ind] < self.best_error:
                self.best_error = self.population_errors[best_ind]
                self.best_individual = numpy.copy(self.population[best_ind,:])
                self.best_generation = self.generation

            # save the best indv for that generation
            self.best_gen_errors[g] = self.population_errors[best_ind]
            self.best_gen_indv[g,:] = self.population[best_ind,:]

            if self.verbose:
                self._report_best()

            # see if done
            if self.best_error < self.goal_error:
                break

            # decide what stays 
            # (don't advance individuals that did not improve)
            ind = self.population_errors > self.old_population_errors
            self.population[ind,:] = self.old_population[ind,:]
            self.population_errors[ind] = self.old_population_errors[ind]

        # see if polish with fmin search after the last generation
        if self.polish:
            if self.verbose:
                print "Polishing best result: %g" % (self.best_error)
                iprint = 1
            else:
                iprint = -1
            # polish with bounded min search
            polished_individual, polished_error, details = \
                                 scipy.optimize.fmin_l_bfgs_b(self.error_func,
                                                              #self.population[best_ind,:],
                                                              self.best_individual,
                                                              args=self.args,
                                                              bounds=self.param_ranges,
                                                              approx_grad=True,
                                                              iprint=iprint)
            if self.verbose:
                print "Polished Result: %g" % (polished_error)
                print "Polished Indiv: " + str(polished_individual)
            if polished_error < self.population_errors[best_ind]:
                # it's better, so keep it
                #self.population[best_ind,:] = polished_individual
                #self.population_errors[best_ind] = polished_error

                # update what is best
                #self.best_error = self.population_errors[best_ind]
                #self.best_individual = numpy.copy(self.population[best_ind,:])
                self.best_error = polished_error
                self.best_individual = polished_individual
                self.best_generation = -1

        if job_server:
            self.pp_stats = job_server.get_stats()
