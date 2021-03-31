# File: setup.py
# Created Date: 2020-04-08
# Author: Steven Atkinson (steven@atkinson.mn)

from setuptools import setup, find_packages

requirements = ["jax", "jaxlib"]

setup(
    name="madgrad",
    version="0.0.0",
    description="JAX implementation of MADGRAD",
    author="Steven Atkinson",
    author_email="steven@atkinson.mn",
    url="https://github.com/sdatkinson/madgrad",
    install_requires=requirements,
    packages=find_packages(),
)
