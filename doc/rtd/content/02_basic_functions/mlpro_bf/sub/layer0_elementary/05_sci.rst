Scientific Reference
-----------------------

MLPro integrates scientific referencing in any classes using a class **ScientificObject**.
This class provides elementary functionality for storing a scientific reference.
For example, when the users create a custom reinforcement learning policy or a custom environment, then the users can simply inherit ScientificObject class and add scientific reference to the related elements.
This class can be accessed as follows:

.. code-block:: python

    from mlpro.bf.various import ScientificObject

MLPro provides various forms of scientific references, which are:
    * ``C_SCIREF_TYPE_NONE`` : None
    * ``C_SCIREF_TYPE_ARTICLE`` : Journal Article
    * ``C_SCIREF_TYPE_BOOK`` : Book
    * ``C_SCIREF_TYPE_ONLINE`` : Online
    * ``C_SCIREF_TYPE_PROCEEDINGS`` : Proceedings
    * ``C_SCIREF_TYPE_TECHREPORT`` : Technical Report
    * ``C_SCIREF_TYPE_UNPUBLISHED`` : Unpublished

After selecting the type of the reference, the users can add more details, such as authors, title, volume, DOI, and many more.

**Cross Reference**
    + :ref:`API Reference <Various>`