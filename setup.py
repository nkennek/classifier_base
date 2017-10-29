#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='classifier-base',
    version='0.1',
    description='Scripts for training, collecting data, and web app for online prediction',
    author='Kenichi Nakahara',
    author_email='chris.nk.0802@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'beautifulsoup4',
        'chainer',
        'Flask',
        'h5py',
        'numpy',
        'opencv-python',
        'pandas',
        'requests',
        'scikit-image',
        'scikit-learn',
        'scipy'
    ],
    extras_require={
        'dev': [
            'cupy',
            'matplotlib',
            'ipdb',
            'flake8',
            'pylint',
            'pep8',
            'mypy',
            'pytest',
            'pytest-asyncio'
        ],
        'test': [
            'pytest',
            'pytest-asyncio'
        ],
    },
)
