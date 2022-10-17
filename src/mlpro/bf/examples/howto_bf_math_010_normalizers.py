## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_math_010_normalizers.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-09-16  0.0.0     LSB      Creation
## -- 2022-09-25  1.0.0     LSB      Release of first version
## -- 2022-10-01  1.0.1     LSB      Renormalization
## -- 2022-10-06  1.0.2     LSB      Refactoring
## -- 2022-10-16  1.0.3     LSB      Updating z-transform parameters based on a new data/element(np.ndarray)
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.3 (2022-10-16)
Example file for demonstrating the use of MLPro's normalizer for normalizing and de-normalizing data.


You will learn:

1. How to update parameters for Normalization

2. How to update parameters based on single element/data

3. How to normalize a data element (ndarray/mlpro element) by MinMax or ZTransofrm

4. How to denormalize a data element (ndarray/mlpro element) by MinMax or ZTransofrm

5. How to renormalize the data element (ndarray/mlpro element) with respect to the changed parameters
"""


from mlpro.bf.math.normalizers import *



# checking for internal unit tests
if __name__ == '__main__':
    p_printing = True
else:
    p_printing = False




# Creating Numpy dummy Dataset
my_dataset = np.asarray(([45,-7,65,-87],[21.3,47.1,-41.02,89],[0.12,98.11,11,-56.01],[7.12,55.01,4.78,5.3],
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
my_normalizer_ztrans.update_parameters(p_dataset = my_dataset)
if p_printing:
    print('01. Parameters updated for the Z transformer\n\n')


# 2. Normalizing a numpy array/ a dataset (as an array) in Z transformation
normalized_data = my_normalizer_ztrans.normalize(p_data=my_dataset)
if p_printing:
    print('02. Normalized value(Z transoformer):\n', normalized_data,'\n\n')


# 3. De-normalizing a numpy array/ a dataset (as an array) in Z transformation
denormalized_data = my_normalizer_ztrans.denormalize(p_data=normalized_data)
if p_printing:
    print('03. Deormalized value(Z transoformer):\n', denormalized_data,'\n\n')


# 4. Updating the parameters using a new element to the dataset
new_data = np.asarray([12,-71,74,-12]).reshape(1,4)
my_normalizer_ztrans.update_parameters(p_data = new_data)
if p_printing:
    print('04. Parameters updated for the Z transformer\n\n')


# 5. Normalizing the new element with new parameters
normalize_new = my_normalizer_ztrans.normalize(new_data)
if p_printing:
    print('\n05. Normalized Data (Z transoformer):', normalize_new,'\n\n')


# 6. Validating the changed parameters
#    6.1 Adding the new element in the dataset
my_dataset = np.append(my_dataset, new_data, axis=0)
#    6.2 Setting up the parameters based on the new dataset
my_normalizer_ztrans.update_parameters(p_dataset=my_dataset)
#    6.3 Normalizing the element for validation
normalized_val = my_normalizer_ztrans.normalize(new_data)
if p_printing:
    print('\n06. Normalized Data (validation Z transoformer): ', normalized_val, '\n\n')


# 7. Setting parameters for Normalization
my_normalizer_minmax.update_parameters(p_set = my_set)
if p_printing:
    print('07. Parameters updated for the MinMax Normalizer\n\n')


# 8. Normalizing using MinMax
normalized_state = my_normalizer_minmax.normalize(my_state)
if p_printing:
    print('08. Normalized value(Minmax Normalizer):\n', normalized_state.get_values(),'\n\n')


# 9. De-normalizing using MinMAx
denormalized_state = my_normalizer_minmax.denormalize(normalized_state)
if p_printing:
    print('09. Denormalized value(Minmax Normalizer):\n', denormalized_state.get_values(),'\n\n')


# 10. Updating the boundaries of the dimension
my_set.get_dim(p_id=my_set.get_dim_ids()[0]).set_boundaries([-10,51])
my_set.get_dim(p_id=my_set.get_dim_ids()[1]).set_boundaries([-5,10])
if p_printing:
    print('10. Boundaries updated(Minmax Normalizer)\n\n')


# 11. Updating tbe normalization parameters for the new set
my_normalizer_minmax.update_parameters(my_set)
if p_printing:
    print('11. Parameters updated for minmax normalizer(Minmax Normalizer)\n\n')


# 12. Renormalizing the previously normalized data with the new parameters
re_normalized_state = my_normalizer_minmax.renormalize(normalized_state)
if p_printing:
    print('12. Renormalized value(Minmax Normalizer):\n', re_normalized_state.get_values(),'\n\n')


# 13. Validating the renormalization
normalized_state = my_normalizer_minmax.normalize(my_state)
if p_printing:
    print('13. Normalized value(Validation renormalization):\n', normalized_state.get_values(),'\n\n')