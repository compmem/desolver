========
DESolver
========
------------------------------------------------------------
Differential Evolution genetic minimization solver in Python
------------------------------------------------------------

Overview
========

This Python module provides genetic function minimization via
`differential evolution
<http://www.icsi.berkeley.edu/~storn/code.html>`_.  The module makes
use of NumPy and optionally SciPy and Parallel Python to provide
additional optimization and parallelization support, repectively.


Installation
============

For now the best thing to do is to make sure you have NumPy installed
(and optionally SciPy and Parallel Python) and then just check out the
latest version of the code from github.  The DistUtils-based setup.py
should work as expected::

  python setup.py install


Usage
=====

Until we have full sphinx-based documentation, please look at the
detest.py in the docs/examples directory for a simple use-case.  There
are additional examples, as well as unit tests that should be
informative.
