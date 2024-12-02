from setuptools import setup


setup(name='mlpro',
version='1.9.4',
description='MLPro - The Integrative Middleware Framework for Standardized Machine Learning',
author='MLPro Team',
author_mail='mlpro@listen.fh-swf.de',
license='Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)',
packages=['mlpro'],

# Package dependencies for full installation
extras_require={
    "full": [
        "dill>=0.3.6",
        "numpy>=1.23.5",
        "torch>=1.13.1",
        "matplotlib>=3.6.3",
        "scipy>=1.8.1",
        "multiprocess>=0.70.14",
        "pandas>=2.1.3"
    ],
},

zip_safe=False)