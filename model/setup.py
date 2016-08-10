

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import sys

# setup variable updates 
ext_modules = [Extension("abundance", sources=["abundance.pyx"]),
               Extension("prior", sources=["prior.pyx"]),
               Extension("footprint", sources=["footprint.pyx", "pi.c", "tau.c"],
                libraries=["gsl","gslcblas"],
                extra_compile_args=["-O3"]),
               Extension("utils", ["utils.pyx"])]
ext_modules = cythonize(ext_modules)

setup(
    name = 'variables',
    author = 'Anil Raj',
    version = '1.1',
    author_email = 'rajanil@stanford.edu',
    cmdclass = {'build_ext': build_ext},
    include_dirs=[numpy.get_include(), '.', '/usr/local/include'],
    ext_modules = ext_modules
)
