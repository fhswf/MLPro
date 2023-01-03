Normalizer
----------
MLPro's normalizer classes can be used to normalize data based on MinMax Normalization and Z-transformation. These
normalizer classes can be imported by incorporating following lines in your script.

.. code-block:: python

    from mlpro.bf.math.normalizers import NormalizerMinMax
    from mlpro.bf.math.normalizers import NormalizerZTransform


Both normalizers store the parameters required for normalization based on the data provided for normalization, with
the possibility to set/update the parameters when required, based on data instances or direct parameters for e.g
boundaries for MinMax normalizers.

Both the normalizers provide following operations:
 * Normalize : Normalize a given data element based on the set parameters.
 * Denormalize : Denormalize a given data element based on the set parameters.
 * Update Parameters : Upadte the normalization parameters based on data characteristics such as boundaries or statistical properties.
 * Renormalize : MLPro's normalizers also provide the possibility to renormalize the previously normalized data elements on new normalization parameters.


* Please refer to :ref:`Howto BF 23 <Howto BF 023>` to know more about normalizers in MLPro
* Please refer to the class diagram at :ref:`Normalizers <Normalizers>`