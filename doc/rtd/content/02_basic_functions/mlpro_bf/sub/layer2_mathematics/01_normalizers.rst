.. _target_bf_math_normalizer:
Normalization
=============

Normalization is a process of scaling different parameters to a common scale. The way the parameters are scaled depends
on the type of normalization being performed. For e.g. parameters are scaled within a range of -1 to 1 in case of a
minmax normalization.
MLPro's normalizer classes can be used to normalize data based on MinMax Normalization and Z-transformation. These
normalizer classes can be imported by incorporating following lines in your script.

.. code-block:: python

    from mlpro.bf.math.normalizers import NormalizerMinMax
    from mlpro.bf.math.normalizers import NormalizerZTransform


Both normalizers store the parameters required for normalization based on the data provided for normalization. MLPro
also provides the possibility to set/update the parameters when required, based on data instances or direct parameters for e.g
boundaries for MinMax normalizers.

Both the normalizers provide following operations:
 * Normalize : Normalize a given data element based on the set parameters.
 * Denormalize : Denormalize a given data element based on the set parameters.
 * Update Parameters : Upadte the normalization parameters based on data characteristics such as boundaries or statistical properties.
 * Renormalize : MLPro's normalizers also provide the possibility to renormalize the previously normalized data elements on new normalization parameters.


**Cross Reference**
    + :ref:`Howto BF-MATH-010: Normalizers <Howto BF MATH 010>`
    + :ref:`API Reference <target_ap_bf_math_norm>`