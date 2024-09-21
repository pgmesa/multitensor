from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extensions = [
    Extension(
        "cytensor",
        sources=["cytensor.pyx", "tensor.c"],  # Include C and Cython files
        include_dirs=['.'],                    # Include directory with header file tensor.h
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
