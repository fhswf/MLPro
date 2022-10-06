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
## -- 2022-10-06  1.0.1     LSB      Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2022-10-06)
Example file for demonstrating the use of MLPro's normalizer for normalizing and de-normalizing data.


You will learn:

1. How to update parameters for Normalization

2. How to normalize a data element (ndarray/mlpro eleement) by MinMax or ZTransofrm

3. How to denormalize a data element (ndarray/mlpro eleement) by MinMax or ZTransofrm

4. How to renormalize the data element (ndarray/mlpro eleement) with respect to the changed paramaters
"""


from mlpro.bf.math.normalizers import *



# checking for internal unit tests
if __name__ == '__main__':
    p_printing = True




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


# Creating Normalizer object
my_normalizer_minmax = NormalizerMinMax()
my_normalizer_ztrans = NormalizerZTrans()



# 1. Setting parameters for NormalizationZTrans
my_normalizer_ztrans.update_parameters(my_dataset)
if p_printing:
    print('Parameters updated for the Z transformer\n\n')


# 2. Normalizing a numpy array/ a dataset (as an array) in Z transformation
normalized_data = my_normalizer_ztrans.normalize(p_data=my_dataset)
if p_printing:
    print('Normalized value:\n', normalized_data,'\n\n')


# 3. De-normalizing a numpy array/ a dataset (as an array) in Z transformation
denormalized_data = my_normalizer_ztrans.denormalize(p_data=normalized_data)
if p_printing:
    print('Deormalized value:\n', denormalized_data,'\n\n')


# 4. Setting parameters for Normalization
my_normalizer_minmax.update_parameters(my_set)
if p_printing:
    print('Parameters updated for the MinMax Normalizer\n\n')


# 5. Normalizing using MinMax
normalized_state = my_normalizer_minmax.normalize(my_state)
if p_printing:
    print('Normalized value:\n', normalized_state.get_values(),'\n\n')


# 6. De-normalizing using MinMAx
denormalized_state = my_normalizer_minmax.denormalize(normalized_state)
if p_printing:
    print('Denormalized value:\n', denormalized_state.get_values(),'\n\n')


# 7. Updating the boundaries of the dimension
my_set.get_dim(p_id=my_set.get_dim_ids()[0]).set_boundaries([-10,51])
my_set.get_dim(p_id=my_set.get_dim_ids()[1]).set_boundaries([-5,10])
if p_printing:
    print('Boundaries updated\n\n')


# 8. updating tbe normalization parameters for the new set
my_normalizer_minmax.update_parameters(my_set)
if p_printing:
    print('Parameters updated for minmax normalizer\n\n')


# 9. renormalizing the previously normalized data with the new parameters
re_normalized_state = my_normalizer_minmax.renormalize(normalized_state)
if p_printing:
    print('Renormalized value:\n', re_normalized_state.get_values(),'\n\n')


# 10. Validating the renormalization
normalized_state = my_normalizer_minmax.normalize(my_state)
if p_printing:
    print('Normalized value:\n', normalized_state.get_values(),'\n\n')