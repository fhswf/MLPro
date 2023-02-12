.. _target_mlpro_getstarted:
Getting Started
===============

Installation from PyPI
----------------------

MLPro is listed in the `Python Package Index (PyPI) <https://pypi.org/project/mlpro/>`_ and can be installed using the package
installer for Python (pip) in two variants:

  * **Without any dependencies**

    The following command installs the latest version of MLPro. 

    .. code-block:: bash

        pip install mlpro

    Additional packages may need to be installed manually (depending on the functionalities you intend to use).
  
  * **Full installation with all dependencies**

    There is also an option to automatically install MLPro and all depending packages in validated versions 
    (see Subsection *Dependencies* below). This option will ensure that all the functionalities of MLPro, including 
    wrappers and examples, work appropriately out of the box. 

    .. code-block:: bash

        pip install mlpro[full]


Installation from Anaconda
--------------------------

MLPro is also available on `Anaconda <https://anaconda.org/mlpro/mlpro/>`_ and can be installed 
with the following command:

  .. code-block:: bash

      conda install -c mlpro mlpro

      
.. _target_dependencies:      
Dependencies
------------

The table below shows all packages that MLPro has dependencies on. Additionally, the versions 
with which MLPro is compatible are listed. Since we cannot influence incompatible changes on 
dependent packages, we unfortunately cannot rule out the possibility of problems occurring 
with different versions. We review and update the list with each new release.

Which packages are actually required depends on the functionalities of MLPro that are used.

.. tabularcolumns:: |p{1cm}|p{7cm}|
  
.. csv-table::
  :file: deps.txt
  :class: longtable
  :widths: 1 1
  :header: "Package", "Version"

      
First Steps
-----------

The easiest way to become familiar with the concepts and functions of MLPro is to browse 
through the numerous :ref:`example programs <target_appendix1>`. 
We can also recommend taking a closer look at the :ref:`key features <target_key_features>` 
of MLPro and following the links.