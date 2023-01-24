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

      
**3. List of dependencies**

   The list of the dependencies can be seen on the table below.

  .. tabularcolumns:: |p{1cm}|p{7cm}|
  
  .. csv-table::
    :file: deps.txt
    :class: longtable
    :widths: 1 1
    :header: "Package", "Version"

      
**4. Get to know MLPro Functionality**

  The easiest way to become familiar with the concepts and functions of MLPro is to browse through the numerous :ref:`example programs <target_appendix1>`.
  Each sub-package has also an individual getting started page, such as
  :ref:`MLPro-RL <target_getstarted_RL>`, :ref:`MLPro-GT <target_getstarted_gt>`, and :ref:`MLPro-GT <target_getstarted_OA>`. 