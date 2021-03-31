# File: setup.py
# Created Date: 2020-04-08
# Author: Steven Atkinson (steven@atkinson.mn)

from setuptools import setup, find_packages

requirements = [
    "jax",
    "jaxlib",
    # "git+ssh://git@github.com/sdatkinson/snets.git@v0.1.0"
]

setup(
    name="mypackage",
    version="0.0.0",
    description="My package",
    author="Steven Atkinson",
    author_email="steven@atkinson.mn",
    url="https://github.com/sdatkinson/",
    install_requires=requirements,
    packages=find_packages(),
)
