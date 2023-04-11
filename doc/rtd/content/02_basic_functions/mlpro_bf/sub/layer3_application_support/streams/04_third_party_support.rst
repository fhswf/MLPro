.. _target_bf_streams_3rd_party_support:
3rd Party Support
=================


MLPro's stream sub-package is a comprehensive package that supports third-party platforms in addition
to its native stream object implementation. Third-party platforms provide access to various datasets, enabling to
design, test, and validate stream processing workflows and online machine learning algorithms. These thrird party
platforms are called StreamProvider in context of Stream sub-package of MLPro. Currently the streams sub-package
supports following 3rd party packages:

OpenML
------
OpenML is a collaborative platform that provides access to a vast range of datasets and algorithms for machine
learning. OpenML's datasets are typically larger in size and contain more complex data, enabling more accurate
results. MLPro supports OpenML's datasets as stream objects.

These datasets are provided as streams containing metadata information about the datasets, including their
names, descriptions, and the attributes they contain.

Learn More: `OpenML <https://www.openml.org/>`_

.. note::
    Unfortunately, there have been some issues with OpenML's API integration in MLPro, causing it to not work as intended in some cases. Users may encounter errors or limitations when using OpenML data sources through MLPro's stream provider functionality. It is recommended to keep an eye out for updates.

.. note::
    MLPro's stream provider functionality not only allows users to access third-party platforms but also provides additional options for loading stream objects from these platforms. For instance, when accessing datasets from OpenML, MLPro users can choose between default or custom input and target features.

ScikitLearn
-----------
ScikitLearn is a popular and well-established machine learning package that provides various algorithms for
classification, regression, clustering, and dimensionality reduction. ScikitLearn also provides a wide range of
datasets which includes popular examples such as the Iris dataset for classification, and the California Housing
dataset for regression that are frequently used for training and testing machine learning models. MLPro currently
supports ScikitLearn's datasets as stream objects, which can be used in stream processing and online machine learning.


Learn More: `ScikitLearn <https://scikit-learn.org/>`_

River
-----
River is a machine learning library designed for streaming data. It is used for real-time analytics and online
machine learning where data is constantly evolving. River provides various online learning algorithms that are
optimized for processing streaming data. MLPro supports River's datasets as data sources, which
can be used to build machine learning models that are optimized for streaming data.

River datasets include specialized datasets for online machine learning which is commonly used for evaluating online
learning algorithms. MLPro standardizes the access to the River streams, making it easier to incorporate them into
stream processing workflows. While MLPro currently supports River as a data resource, future updates will include
standardized support for algorithms in River.

Learn More: `River ML <https://riverml.xyz/latest/>`_


Accessing the Datasets
----------------------
ScikitLearn, River, and OpenML datasets are all available through MLPro-BF-Stream's stream provider functionality.
Each dataset is provided as a stream package containing metadata information about the Streams, including their
names, descriptions, and the attributes they contain.

The stream providers can be included in your script as following:

.. code-block:: python


    # Importing ScikitLearn Stream Provider
    from mlpro.wrappers.sklearn import WrStreamProviderSKLearn

    # Importing River Stream Provider
    from mlpro.wrappers.river import WrStreamProviderRiver

    # Importing OpenML Stream Provider
    from mlpro.wrappers.openml import WrStreamProviderOpenML

Learn more in the Howto section of MLPro, in appendix 01 section.


**Cross References**

- :ref:`Howto BF-STREAMS-051: Accessing Data from OpenML <Howto BF STREAMS 051>`

- :ref:`Howto BF-STREAMS-052: Accessing Data from Scikit-Learn <Howto BF STREAMS 052>`

- :ref:`Howto BF-STREAMS-053: Accessing Data from River <Howto BF STREAMS 053>`

- :ref:`OpenML <Wrapper OpenML>`

- :ref:`ScikitLearn <Wrapper River>`

- :ref:`ScikitLearn <Wrapper sklearn>`