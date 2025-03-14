from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    name="project2",
    version="0.1",
    packages=["project2"],
    ext_modules=cythonize("project2/project2.py"),  # Compile with Cython
    zip_safe=False,
)
