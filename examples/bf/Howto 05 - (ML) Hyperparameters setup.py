## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : Howto 05 - (ML) Hyperparameters setup
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-08-31  0.0.0     SY       Creation
## -- 2021-09-01  1.0.0     SY       Release of first version
## -- 2021-09-11  1.0.0     MRD      Change Header information to match our new library name
## -- 2021-12-10  1.0.1     DA       Refactoring, little beautifying
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2021-12-10)

This module demonstrates how to set-up hyperparameters using available HyperParamTuple, 
HyperParamSpace, and HyperParam classes.
"""



from mlpro.bf.ml import *


# 1 Setup a class that requires a tuple of hyperparameters 
class MyHyperparameter:
    
    def __init__(self):
        # 1.1 Construct a hyperparameter space using HyperParamSpace() and an empty tuple 
        self._hyperparam_space  = HyperParamSpace()
        self._hyperparam_tuple  = None
        self._init_hyperparam()
        

    def _init_hyperparam(self):
        # 1.2 Declare hyperparameters with unique id, names, and data type   
        self._hyperparam_space.add_dim(HyperParam(0,'num_states','Z'))
        self._hyperparam_space.add_dim(HyperParam(1,'smoothing','R'))
        self._hyperparam_space.add_dim(HyperParam(2,'lr_rate','R'))
        self._hyperparam_space.add_dim(HyperParam(3,'buffer_size','Z'))
        self._hyperparam_space.add_dim(HyperParam(4,'update_rate','Z'))
        self._hyperparam_space.add_dim(HyperParam(5,'sampling_size','Z'))
        self._hyperparam_tuple = HyperParamTuple(self._hyperparam_space)
        
        # 1.3 Set the hyperparameter with a default value 
        self._hyperparam_tuple.set_value(0, 100)
        self._hyperparam_tuple.set_value(1, 0.035)
        self._hyperparam_tuple.set_value(2, 0.0001)
        self._hyperparam_tuple.set_value(3, 100000)
        self._hyperparam_tuple.set_value(4, 100)
        self._hyperparam_tuple.set_value(4, 256)


    def get_hyperparam(self) -> HyperParamTuple:
        return self._hyperparam_tuple
    


# 2 Get value from the hyperparameter tuple 
myParameter         = MyHyperparameter()
for idx in myParameter.get_hyperparam().get_dim_ids():
    print('Variable with ID %s = %.2f'%(idx, myParameter.get_hyperparam().get_value(idx)))
    


# 3 Overwrite current value with new desired value
myParameter.get_hyperparam().set_value(0, 50)
print('\nA new value for variable ID 0')
print('Variable with ID 0 = %.2f'%(myParameter.get_hyperparam().get_value(0)))