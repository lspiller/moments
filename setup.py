from setuptools import setup
from Cython.Build import cythonize
#setup(ext_modules=cythonize('moments.pyx', annotate=True))
setup(name='moments', version='1.0.0',  ext_modules=cythonize('moments.pyx', annotate=True, line_directives=True))
