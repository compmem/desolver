

from distutils.core import setup, Extension
from distutils.sysconfig import get_config_var
import os
import sys

setup(name='desolver', 
      version=vstr, 
      package_dir={"desolver":"desolver"},
      packages=['desolver','desolver.tests'],
      author=['Per B. Sederberg'],
      maintainer=['Per B. Sederberg'],
      maintainer_email=['psederberg@gmail.com'],
      url=['http://code.google.com/p/desolver/'])

