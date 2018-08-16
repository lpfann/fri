from __future__ import print_function

import sys

from setuptools import setup, find_packages

import versioneer

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

from os import path

here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='fri',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Feature relevance interval method',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/lpfann/fri',
      author='Lukas Pfannschmidt',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      author_email='lpfannschmidt@techfak.uni-bielefeld.de',
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',

          # Pick your license as you wish
          'License :: OSI Approved :: MIT License',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
      ],
      keywords="feature selection relevance bounds machine learning bioinformatics"
      )
