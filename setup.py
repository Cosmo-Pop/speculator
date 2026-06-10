#!/usr/bin/env python

from setuptools import setup, find_packages
import sys
import platform

install_requires = ["tqdm>=4.41.1", "numpy", "scikit-learn", "torch", "wandb"]

setup(name='speculator',
      version='v0.3',
      description='SPS emulation',
      author='Justin Alsing',
      url='https://github.com/Cosmo-Pop/speculator',
      packages=find_packages(),
      install_requires=install_requires)
