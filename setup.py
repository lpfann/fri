from __future__ import print_function
import sys
import versioneer
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]


try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)

setup(name='fri',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Feature relevance interval method',
      author='Lukas Pfannschmidt',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      dependency_links=[
                        "git+https://github.com/cvxgrp/cvxpy.git@1.0#egg=cvxpy-1.0"
                       ],
      author_email='lpfannschmidt@techfak.uni-bielefeld.de',
      )
