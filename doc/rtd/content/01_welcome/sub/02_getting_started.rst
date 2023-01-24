.. _target_mlpro_getstarted:
Getting Started
===============

**1. Installation using PIP**

  * Without any dependencies

    MLPro is listed in the `Python Package Index (PyPI) <https://pypi.org/project/mlpro/>`_ and also in `Anaconda <https://anaconda.org/mlpro/mlpro/>`_ . If the user wants to install
    without any dependencies, MLPro can be installed with the following command:

    .. code-block:: bash

        pip install mlpro
  
  * Full installation with all dependencies

    There is also an optional option to include all the dependencies. This option will ensure that all the functionality, including wrapper and examples, works appropriately out of the box. The user does not need to take care of the dependencies; we will install them automatically. MLPro with its dependencies can be installed with the following command:

    .. code-block:: bash

        pip install mlpro[full]


**2. Installation using conda**

  If you are using conda environment, MLPro can be installed with the following command:

  .. code-block:: bash

      conda install -c mlpro mlpro


**3. Installation using johnnydep**

  The dependencies of MLPro can be checked by using the johnnydep package.

  .. code-block:: bash

      johnnydep mlpro[full]

      
**4. List of dependencies**

  Below are the dependencies list that will be installed:

  .. list-table:: Dependencies List
    :widths: 25 25
    :header-rows: 1

    * - Package Name
      - Version
    * - dill
      - 0.3.6
    * - numpy
      - 1.23.5
    * - torch
      - 1.13.1
    * - transformations
      - 2022.9.26
    * - stable-baselines3
      - 1.7.0
    * - gym
      - 0.21.0
    * - scipy
      - 1.8.1
    * - pettingzoo
      - 1.22.3
    * - pygame
      - 2.1.2
    * - pymunk
      - 6.4.0
    * - multiprocess
      - 0.70.14
    * - river
      - 0.14.0
    * - scikit-learn
      - 1.2.0
    * - optuna
      - 3.0.5
    * - hyperopt
      - 0.2.7
    * - pyglet
      - 1.5.27

      
**5. Get to know MLPro Functionality**

  The easiest way to become familiar with the concepts and functions of MLPro is to browse through the numerous :ref:`example programs <target_appendix1>`.
  Each sub-package has also an individual getting started page, such as
  :ref:`MLPro-RL <target_getstarted_RL>`, :ref:`MLPro-GT <target_getstarted_gt>`, and :ref:`MLPro-GT <target_getstarted_OA>`. 