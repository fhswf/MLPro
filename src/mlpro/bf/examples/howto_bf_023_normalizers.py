 ## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_023_normalizers.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-09-16  0.0.0     LSB      Creation
## -- 2022-mm-dd  1.0.0     LSB      Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-09-dd)
Example file for demonstrating the use of MLPro's normalizer objects normalizing and denormalizing data.
"""


from mlpro.bf.math.normalizers import *
from mlpro.bf.various import Log





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyNormalizerZTrans(NormalizerZTrans, Log):
    def __init__(self):
        super(MyNormalizerZTrans, self).__init__()
        Log.__init__(self)


## -------------------------------------------------------------------------------------------------
    def normalize(self, p_data:Union[Element, np.ndarray], p_param=None):
        normalized_value = super().normalize(p_data=p_data)
        self.log("Normalized Value:", normalized_value)
        return normalized_value


## -------------------------------------------------------------------------------------------------
    def denormalize(self, p_data:Union[Element, np.ndarray], p_param=None):
        denormalized_value = super().denormalize(p_data=p_data)
        self.log("Denormalized Value:", denormalized_value)
        return denormalized_value




my_dataset = np.array([[45,-7,65,-87],[21.3,47.1,-41.02,89],[0.12,98.11,11,-56.01]])
    


my_nomalizer = MyNormalizerZTrans()
# 1. Normalizing a numpy array/ a dataset (as an array)
normalized_data = my_nomalizer.normalize(p_data=my_dataset)
# 2. Denormalizing a numpy array/ a dataset (as an array)
denormalized_data = my_nomalizer.denormalize(p_data=normalized_data)