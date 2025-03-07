from setuptools import setup


setup(name='mlpro',
version='1.9.5',
description='MLPro - The integrative middleware framework for standardized machine learning',
author='MLPro Team',
author_mail='mlpro@listen.fh-swf.de',
license='Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)',
packages=['mlpro'],

# Package dependencies for full installation
extras_require={
    "full": [
        "dill>=0.3.9",
        "multiprocess>=0.70.17",
        "numpy>=1.24.2",
        "torch>=2.0.0",
        "PySide6>=6.8.1",
        "matplotlib>=3.10.0",
        "scipy>=1.8.1",
        "pandas>=2.1.3"
    ],
},

zip_safe=False)