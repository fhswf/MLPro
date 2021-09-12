## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : ml
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-08-20  0.0.0     DA       Creation 
## -- 2021-08-25  1.0.0     DA       Release of first version
## -- 2021-09-11  1.0.0     MRD      Change Header information to match our new library name
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-08-25)

This module provides common machine learning functionalities and properties.
"""

from mlpro.bf.various import *
from mlpro.bf.math import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class HyperParam(Dimension):
    """
    ...
    """

## -------------------------------------------------------------------------------------------------
    def register_callback(self, p_cb):
        self._cb = p_cb


## -------------------------------------------------------------------------------------------------
    def callback_on_change(self, p_value):
        try:
            self._cb(p_value)
        except:
            pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class HyperParamSpace(ESpace):
    """
    ...
    """

    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class HyperParamTupel(Element):
    """
    ...
    """

## -------------------------------------------------------------------------------------------------
    def set_value(self, p_dim_id, p_value):
        super().set_value(p_dim_id, p_value)
        self._set.get_dim(p_dim_id).callback_on_change(p_value)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class HyperParamTuning(Log):
    """
    Template class for hyperparameter tuning.
    """

    C_TYPE          = 'Hyperparameter Tuning'
    C_NAME          = '????'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_path:str, p_logging=True):
        super().__init__(p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def optimize(self, *p_hp):
        """
        Mathematical function to be optimized.

        Parameters:
            *p_hp       Hyperparameters

        Returns:
            Real value to be optimized
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Adaptive(Log, LoadSave):
    """
    Property class for adapativity. And if something can be adapted it should be loadable and saveable
    so that this class provides load/save properties as well.
    """

    C_TYPE          = 'Adaptive'
    C_NAME          = '????'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_ada=True, p_logging=True):
        """
        Parameters:
            p_ada               Boolean switch for adaptivity
            p_logging           Boolean switch for logging functionality
        """

        Log.__init__(self, p_logging=p_logging)
        self.switch_adaptivity(p_ada)
        self._hyperparam_space  = HyperParamSpace()
        self._hyperparam_tupel  = None
        self._init_hyperparam()

        self._attrib_hp1 = 0


## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self):
        """
        Implementation specific hyperparameters can be added here. Please follow these steps:
        a) Add each hyperparameter as an object of type HyperParam to the internal hyperparameter
           space object self._hyperparam_space
        b) Create hyperparameter tuple and bind to self._hyperparam_tupel
        c) Set default value for each hyperparameter
        """


## -------------------------------------------------------------------------------------------------
    def get_hyperparam(self) -> HyperParamTupel:
        return self._hyperparam_tupel


## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada:bool):
        """
        Switches adaption functionality on/off.
        
        Parameters:
            p_ada               Boolean switch for adaptivity
        """

        self._adaptivity = p_ada
        if self._adaptivity:
            self.log(self.C_LOG_TYPE_I, 'Adaptivity switched on')
        else:
            self.log(self.C_LOG_TYPE_I, 'Adaptivity switched off')


## -------------------------------------------------------------------------------------------------
    def adapt(self, *p_args) -> bool:
        """
        Adapts something inside in the sense of machine learning. The number and types of parameters 
        depend on the specific implementation. Please redefine and describe the exact number and types
        of parameters in your own implementation. It is recommended to call this implementation stub
        by calling super().adapt() and to continue adapting something only if it returns True. So it 
        is ensured that an adaption takes place only it adaptivity is switched on. Furthermore adaption
        will be logged.

        Parameters:
            p_args          All parameters that are needed for the adaption. 

        Returns:
            True, if something has been adapted. False otherwise.
        """

        if not self._adaptivity: return False
        self.log(self.C_LOG_TYPE_I, 'Adaption started')
        return True
