Normalizer
----------------
MLPro provides two normalization classes for MinMax normalization and Z transform. These classes can imported
respectively by executing

.. code-block:: python

    from mlpro.bf.math.normalizers import NormalizerMinMax
    from mlpro.bf.math.normalizers import NormalizerZTransform

Both the normalizations provide following operations:
 * Normalization : Normalization of the given data element as MLPro Element or a Numpy array
 * Denormalization : Denormalization of the given data element as MLPro Element or a Numpy array
 * Update Parameters : The normalizers in MLPro are based on normalization parameters, which can be changed in events like change MinMax boundaries, change of data distribution, etc.
 * Renormalizing : Following the update of the normalization parameters, the previously normalized data can be renormalized based on the current parameters


.. note::
Please refer to :ref:`Howto BF 23 <Howto BF 023>` to know more about normalizers in MLPro