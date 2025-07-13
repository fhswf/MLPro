from setuptools import setup


setup(name='mlpro',
version='2.0.3',
description='MLPro - The integrative middleware framework for standardized machine learning',
author='MLPro Team',
author_mail='mlpro@listen.fh-swf.de',
license='Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)',
packages=['mlpro'],

# Package dependencies for full installation
extras_require={
    "full": [
        "dill>=0.4.0",
        "multiprocess>=0.70.18",
        "numpy>=2.2.5",
        "torch>=2.7.0",
        "PySide6>=6.9.0",
        "matplotlib>=3.10.1",
        "scipy>=1.10.1",
        "pandas>=2.1.3"
    ],
},

zip_safe=False)