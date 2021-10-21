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
## -- 2021-09-18  1.0.1     MRD      Buffer Class Implementation. Add new parameter buffer
## --                                to the Adaptive Class
## -- 2021-09-19  1.0.1     MRD      Improvement on Buffer Class. Implement new base class
## --                                Buffer Element and BufferRnd
## -- 2021-09-25  1.0.2     MRD      Add __len__ functionality for SARBuffer
## -- 2021-10-06  1.0.3     DA       Extended class Adaptive by new methods _adapt(), get_adapted(),
## --                                _set_adapted(); moved Buffer classes to mlpro.bf.data.py
## -- 2021-10-21  1.1.0     DA       New class AdaptiveFunction
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2021-10-21)

This module provides fundamental machine learning functionalities and properties.
"""

from mlpro.bf.various import *
from mlpro.bf.math import *
from mlpro.bf.data import Buffer





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

    C_BUFFER_CLS    = Buffer            

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_buffer_size=0, p_ada=True, p_logging=True):
        """
        Parameters:
            p_buffer_size       Initial size of internal data buffer (0=no buffering)
            p_ada               Boolean switch for adaptivity
            p_logging           Boolean switch for logging functionality
        """

        Log.__init__(self, p_logging=p_logging)
        self._adapted           = False
        self.switch_adaptivity(p_ada)
        self._hyperparam_space  = HyperParamSpace()
        self._hyperparam_tupel  = None
        self._init_hyperparam()

        if p_buffer_size > 0:
            self._buffer = self.C_BUFFER_CLS(p_size=p_buffer_size)
        else:
            self._buffer = None

        self._attrib_hp1        = 0


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
    def get_adapted(self) -> bool:
        return self._adapted


## -------------------------------------------------------------------------------------------------
    def _set_adapted(self, p_adapted:bool):
        self._adapted = p_adapted


## -------------------------------------------------------------------------------------------------
    def adapt(self, *p_args) -> bool:
        """
        Adapts something inside in the sense of machine learning. Please redefine and describe the 
        protected method _adapt() that is called here. 

        Parameters:
            p_args          All parameters that are needed for the adaption. 

        Returns:
            True, if something has been adapted. False otherwise.
        """

        if not self._adaptivity: return False
        self.log(self.C_LOG_TYPE_I, 'Adaptation started')
        self._set_adapted(self._adapt(*p_args))
        return self.get_adapted()


## -------------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool:
        """
        Please redefine and implement your specific adaptation algorithm. Furthermore please describe 
        the type and purpose of all parameters needed by your implementation. This method will be 
        called by public method adapt() if adaptivity is switched on. 

        Parameters:
            p_args[0]           ...
            p_args[1]           ...
            ...

        Returns:
            True, if something has been adapted. False otherwise.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
        if self._buffer is not None: self._buffer.clear()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AdaptiveFunction (Adaptive, Function):
    """
    Model class for an adaptive bi-multivariate mathematical function.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_input_space:MSpace, p_output_space:MSpace, p_threshold=0, p_buffer_size=0, p_ada=True, p_logging=True):
        Adaptive.__init__(self, p_buffer_size=p_buffer_size, p_ada=p_ada, p_logging=p_logging)
        Function.__init__(self, p_input_space=p_input_space, p_output_space=p_output_space)
        self._threshold         = p_threshold
        self._mappings_total    = 0             # Number of mappings since last adaptation
        self._mappings_good     = 0             # Number of 'good' mappings since last adaptation


## -------------------------------------------------------------------------------------------------
    def adapt(self, p_input:Element, p_output:Element) -> bool:
        """
        Parameters:
            p_input         Abscissa/input element object (type Element)
            p_output        Setpoint ordinate/output element (type Element)
        """

        if not self._adaptivity: return False
        self.log(self.C_LOG_TYPE_I, 'Adaptation started')

        # Quality check
        if self._output_space.distance(p_output, self.map(p_input)) <= self._threshold:
            # Quality of function ok. No need to adapt.
            self._mappings_total    += 1
            self._mappings_good     += 1

        else:
            # Quality of function not ok. Adapation is to be triggered.
            self._set_adapted(self._adapt(p_input, p_output))
            if self.get_adapted():
                self._mappings_total    = 1
                self._mappings_good     = 1
            else:
                self._mappings_total    += 1

        return self.get_adapted()


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_input:Element, p_output:Element) -> bool:
        """
        Protected custom adaptation algorithm that is called by public adaptation method. Please redefine.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_maturity(self):
        """
        Returns the maturity of the adaptive function. The maturity is defined as the relation 
        between the number of successful mapped inputs and the total number of mappings since the 
        last adaptation.
        """

        if self._mappings_total == 0: return 0
        return self._mappings_good / self._mappings_total