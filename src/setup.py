from setuptools import setup


setup(name='mlpro',
version='1.0.2',
description='MLPro - The Integrative Middleware Framework for Standardized Machine Learning',
author='MLPro Team',
author_mail='mlpro@listen.fh-swf.de',
license='Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)',
packages=['mlpro'],

# Package dependencies for full installation
extras_require={
    "full": [
        "dill==0.3.6",
        "numpy==1.23.5",
        "torch==1.13.1",
        "matplotlib==3.6.3",
        "transformations==2022.9.26",
        "stable-baselines3==1.7.0",
        "gym<=0.25.0",
        "scipy==1.8.1",
        "pettingzoo==1.22.3",
        "pygame==2.1.2",
        "pymunk==6.4.0",
        "multiprocess==0.70.14",
        "river==0.14.0",
        "scikit-learn==1.2.0",
        "optuna==3.0.5",
        "hyperopt==0.2.7",
        "pyglet==1.5.27"
    ],
},

zip_safe=False)