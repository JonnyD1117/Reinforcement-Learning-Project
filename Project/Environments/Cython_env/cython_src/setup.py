from setuptools import setup, Extension
from Cython.Build import cythonize

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
    ext_modules=cythonize("Cythonized_SPMe_STEP.pyx", annotate=True)
)

# setup(
#     ext_modules=cythonize("Opt_Cython_SPMe_STEP.pyx", annotate=True)
# )