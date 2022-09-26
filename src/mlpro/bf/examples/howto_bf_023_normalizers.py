 ## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_023_normalizers.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-09-16  0.0.0     LSB      Creation
## -- 2022-09-25  1.0.0     LSB      Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-09-25)
Example file for demonstrating the use of MLPro's normalizer for normalizing and de-normalizing data.
"""


from mlpro.bf.math.normalizers import *
from mlpro.bf.various import Log

# checking for internal unit tests
if __name__ == '__main__':
    p_logging = Log.C_LOG_ALL
else:
    p_logging = Log.C_LOG_NOTHING


# Creating Numpy dummy Dataset
my_dataset = np.array(([45,-7,65,-87],[21.3,47.1,-41.02,89],[0.12,98.11,11,-56.01]))


# Creating a dummy set with dummy dimensions
my_data = Set()
my_data.add_dim(Dimension(p_name_short='1', p_boundaries=[10,19]))
my_data.add_dim(Dimension(p_name_short='2', p_boundaries=[-9,10]))


# Creating a dummy element to normalize
my_state = Element(my_data)
my_state.set_values([19,8])




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyNormalizerMinMax(NormalizerMinMax, Log):
    def __init__(self):
        super(MyNormalizerMinMax, self).__init__()
        Log.__init__(self, p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def normalize(self, p_data:Union[Element, np.ndarray], p_param=None):
        normalized_value = super().normalize(p_element=p_data)
        self.log(self.C_LOG_TYPE_I,"Normalized Value:\n", normalized_value.get_values())
        return normalized_value


## -------------------------------------------------------------------------------------------------
    def denormalize(self, p_data:Union[Element, np.ndarray], p_param=None):
        denormalized_value = super().denormalize(p_element=p_data)
        self.log(self.C_LOG_TYPE_I,"Denormalized Value:\n", denormalized_value.get_values())
        return denormalized_value





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyNormalizerZTransform(NormalizerZTrans, Log):
    def __init__(self):
        super(MyNormalizerZTransform, self).__init__()
        Log.__init__(self, p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def normalize(self, p_data:Union[Element, np.ndarray], p_param=None):
        normalized_value = super().normalize(p_element=p_data)
        self.log(self.C_LOG_TYPE_I,"Normalized Value:\n", normalized_value)
        return normalized_value


## -------------------------------------------------------------------------------------------------
    def denormalize(self, p_data:Union[Element, np.ndarray], p_param=None):
        denormalized_value = super().denormalize(p_element=p_data)
        self.log(self.C_LOG_TYPE_I,"Denormalized Value:\n", denormalized_value)
        return denormalized_value





# 1. Creating Normalizer Object
my_normalizerZTrans = MyNormalizerZTransform()


# 2. Setting parameters for NormalizationZTrans
my_normalizerZTrans.set_parameters(my_dataset)


# 3. Normalizing a numpy array/ a dataset (as an array) in Z transformation
normalized_data = my_normalizerZTrans.normalize(p_data=my_dataset)


# 4. De-normalizing a numpy array/ a dataset (as an array) in Z transformation
denormalized_data = my_normalizerZTrans.denormalize(p_data=normalized_data)


# 5. Creating a MinMax normalizer object
my_normalizerMinMax = MyNormalizerMinMax()


# 6. Setting parameters for Normalization
my_normalizerMinMax.set_parameters(my_data)


# 7. Normalizing using MinMax
normalized = my_normalizerMinMax.normalize(my_state)


# 8. De-normalizing using MinMAx
denormalized = my_normalizerMinMax.denormalize(normalized)


