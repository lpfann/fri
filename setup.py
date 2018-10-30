from setuptools import setup, find_packages

import versioneer

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

from os import path
here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'nbsphinx'
    ]
}

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
      extras_require=EXTRAS_REQUIRE,
      author_email='lpfannschmidt@techfak.uni-bielefeld.de',
      license='MIT',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Topic :: Scientific/Engineering :: Medical Science Apps.',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ],
      keywords="feature selection relevance bounds machine learning bioinformatics"
      )
