"""
Setup for genetic_algorithm
"""

from setuptools import setup, find_packages

setup(
    name='Genetic algorithm',
    version='0.0.0',
    packages=['genetic_alg'],
    install_requires=[
       'numpy',
       'tensorflow',
       'matplotlib',
    ]
)
