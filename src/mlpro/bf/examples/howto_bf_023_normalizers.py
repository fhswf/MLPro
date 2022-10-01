## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_023_normalizers.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-09-16  0.0.0     LSB      Creation
## -- 2022-09-25  1.0.0     LSB      Release of first version
## -- 2022-10-01  1.0.1     LSB      Renormalization
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-09-25)
Example file for demonstrating the use of MLPro's normalizer for normalizing and de-normalizing data.


You will learn:

1. How to update parameters for Normalization

2. How to normalize a data element (ndarray/mlpro eleement) by MinMax or ZTransofrm

3. How to denormalize a data element (ndarray/mlpro eleement) by MinMax or ZTransofrm

4. How to renormalize the data element (ndarray/mlpro eleement) with respect to the changed paramaters
"""


from mlpro.bf.math.normalizers import *
from mlpro.bf.various import Log

# checking for internal unit tests
if __name__ == '__main__':
    p_logging = Log.C_LOG_ALL
else:
    p_logging = Log.C_LOG_NOTHING


# Creating Numpy dummy Dataset
my_dataset = np.array(([45,-7,65,-87],[21.3,47.1,-41.02,89],[0.12,98.11,11,-56.01],[7.12,55.01,4.78,5.3],
                       [-44.371,-0.521,14.12,8.5],[77.13,-23.14,-7.54,12.32],[8.1,27.61,-31.01,17.8],
                       [4.22,-84.21,47.12,82.11],[-53.22,1.024,5.044,71.23],[9.4,-4.39,12.51,83.01]))


# Creating a dummy set with dummy dimensions
my_set = Set()
my_set.add_dim(Dimension(p_name_short='1', p_boundaries=[10,19]))
my_set.add_dim(Dimension(p_name_short='2', p_boundaries=[-9,10]))


# Creating a dummy element to normalize
my_state = Element(my_set)
my_state.set_values([19,8])





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NormalizerDemo(Log):

    C_NAME = 'Normalizer'
    C_TYPE = 'Demo Class'


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_normalizer_minmax:NormalizerMinMax, p_normalizer_Ztransform:NormalizerZTrans, p_logging=Log.C_LOG_ALL):


        self.normalizer_minmax = p_normalizer_minmax
        self.normalizer_ztransform = p_normalizer_Ztransform
        super().__init__(p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def normalize_minmax(self, p_data: Union[Element, np.ndarray]):
        normalized_value = self.normalizer_minmax.normalize(p_data=p_data)
        self.log(self.C_LOG_TYPE_I, "Normalized Value:\n", normalized_value.get_values())
        return normalized_value


## -------------------------------------------------------------------------------------------------
    def denormalize_minmax(self, p_data:Union[Element, np.ndarray]):
        denormalized_value = self.normalizer_minmax.denormalize(p_data=p_data)
        self.log(self.C_LOG_TYPE_I,"Denormalized Value:\n", denormalized_value.get_values())
        return denormalized_value


## -------------------------------------------------------------------------------------------------
    def renormalize_minmax(self, p_data:Union[Element, np.ndarray]):
        re_normalized = self.normalizer_minmax.renormalize(p_data)
        self.log(self.C_LOG_TYPE_I, 'Renoramalized Value:\n', re_normalized.get_values())


## -------------------------------------------------------------------------------------------------
    def normalize_ztrans(self, p_data: Union[Element, np.ndarray]):
        normalized_value = self.normalizer_ztransform.normalize(p_data=p_data)
        self.log(self.C_LOG_TYPE_I, "Normalized Value:\n", normalized_value)
        return normalized_value


## -------------------------------------------------------------------------------------------------
    def denormalize_ztrans(self, p_data:Union[Element, np.ndarray]):
        denormalized_value = self.normalizer_ztransform.denormalize(p_data=p_data)
        self.log(self.C_LOG_TYPE_I,"Denormalized Value:\n", denormalized_value)
        return denormalized_value







# 1. Creating Normalizer Object
my_normalizer = NormalizerDemo(p_normalizer_minmax=NormalizerMinMax(),p_normalizer_Ztransform=NormalizerZTrans())


# 2. Setting parameters for NormalizationZTrans
my_normalizer.normalizer_ztransform.update_parameters(my_dataset)


# 3. Normalizing a numpy array/ a dataset (as an array) in Z transformation
normalized_data = my_normalizer.normalize_ztrans(p_data=my_dataset)


# 4. De-normalizing a numpy array/ a dataset (as an array) in Z transformation
denormalized_data = my_normalizer.denormalize_ztrans(p_data=normalized_data)


# 6. Setting parameters for Normalization
my_normalizer.normalizer_minmax.update_parameters(my_set)


# 7. Normalizing using MinMax
normalized_state = my_normalizer.normalize_minmax(my_state)


# 8. De-normalizing using MinMAx
denormalized_state = my_normalizer.denormalize_minmax(normalized_state)


# 9. Updating the boundaries of the dimension
my_set.get_dim(p_id=my_set.get_dim_ids()[0]).set_boundaries([-10,51])
my_set.get_dim(p_id=my_set.get_dim_ids()[1]).set_boundaries([-5,10])


# 10. updating tbe normalization parameters for the new set
my_normalizer.normalizer_minmax.update_parameters(my_set)


# 11. renormalizing the previously normalized data with the new parameters
re_normalized_state = my_normalizer.renormalize_minmax(normalized_state)


# 12. Validating the renormalization
normalized_state = my_normalizer.normalize_minmax(my_state)