import numpy

fun = None

# class-dependent error
class FunToOpt():
    def __init__(self):
        pass
    def __call__(self, indiv, *args):
        # inverse exponential with offset, y = a * exp(b/x) + c
        predicted = indiv[0] * numpy.exp(indiv[1] / args[0]) + indiv[2]        

        # sum of squared error
        error = predicted - args[1]
        return numpy.sum(error*error)

