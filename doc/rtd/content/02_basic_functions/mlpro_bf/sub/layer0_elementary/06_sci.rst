Scientific Reference
-----------------------

MLPro integrates scientific referencing in any class using a class **ScientificObject**.
This class provides elementary functionality for storing scientific references.
For example, when the users create a custom reinforcement learning policy or a custom environment, then the users can simply inherit ScientificObject class and add a scientific reference to the related elements.
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

After selecting the type of reference, the users can add more details, such as authors, titles, volume, DOI, and many more.

The type and detail of the related scientific reference in a class can be initialized, as follows:

.. code-block:: python

    from mlpro.bf.various import ScientificObject

    class MyClass(ScientificObject):

        def __init__(self):
            self.C_SCIREF_TYPE = self.C_SCIREF_TYPE_ARTICLE
            self.C_SCIREF_AUTHOR  = "Max Mustermann"
            self.C_SCIREF_TITLE   = "Analysis of MLPro"
            self.C_SCIREF_JOURNAL = "My Journal"
            self.C_SCIREF_YEAR    = "2023"
            self.C_SCIREF_MONTH   = "01"
            self.C_SCIREF_DAY     = "01"
            self.C_SCIREF_VOLUME  = "01"
            self.C_SCIREF_DOI     = "10.XXXX"

Shortly, the MLPro team is planning to add a citing functionality. Therefore, the users can obtain the citation of the specific class in the form of BibTeX. 

**Cross Reference**
    + :ref:`API Reference <target_api_bf_various>`