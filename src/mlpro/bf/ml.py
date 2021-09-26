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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2021-09-25)

This module provides common machine learning functionalities and properties.
"""

from mlpro.bf.various import *
from mlpro.bf.math import *
import random




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
class BufferElement:
    """
    Base class implementation for buffer element
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_element:dict) -> None:
        """
        Parameters:
            p_element (dict): Buffer element in dictionary
        """

        self._element = {}

        self.add_value_element(p_element)

## -------------------------------------------------------------------------------------------------
    def add_value_element(self, p_val:dict):
        """
        Adding new value to the element container

        Parameters:
            p_val (dict): Elements in dictionary
        """
        self._element = {**self._element, **p_val}


## -------------------------------------------------------------------------------------------------
    def get_data(self):
        """
        Get the buffer element.

        Returns:
            Returns the buffer element.
        """

        return self._element

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Buffer:
    """
    Base class implementation for buffer management.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_size=1):
        """
        Parameters:
            p_size (int, optional): Buffer size. Defaults to 1.
        """
        self._size = p_size
        self._data_buffer = {}

## -------------------------------------------------------------------------------------------------
    def add_element(self, p_elem:BufferElement):
        """
        Add element to the buffer.

        Parameters:
            p_elem (BufferElement): Element of Buffer
        """
        self._data_buffer = {**p_elem.get_data(), **self._data_buffer}
        for key, value in self._data_buffer.items():
            if key in p_elem.get_data() and key in self._data_buffer:
                if not isinstance(self._data_buffer[key], list):
                    self._data_buffer[key] = [p_elem.get_data()[key]]
                else:
                    self._data_buffer[key].append(p_elem.get_data()[key])

                if len(self._data_buffer[key]) > self._size:
                    self._data_buffer[key].pop(-len(self._data_buffer[key]))

## -------------------------------------------------------------------------------------------------
    def clear(self):
        """
        Resets buffer.
        """

        self._data_buffer.clear()

## -------------------------------------------------------------------------------------------------
    def get_latest(self):
        """
        Returns latest buffered element. 
        """

        try:
            return {key: self._data_buffer[key][-1] for key in self._data_buffer}
        except:
            return None

## -------------------------------------------------------------------------------------------------
    def get_all(self):
        """
        Return all buffered elements.

        """
        return self._data_buffer

## -------------------------------------------------------------------------------------------------
    def get_sample(self, p_num:int):
        """
        Sample some element from the buffer.

        Parameters:
            p_num (int): Number of sample

        Returns:
            Samples in dictionary
        """
        return self._extract_rows(self._gen_sample_ind(p_num))

## -------------------------------------------------------------------------------------------------
    def _gen_sample_ind(self, p_num:int) -> list:
        """
        Generate random indices from the buffer.

        Parameters:
            p_num (int): Number of sample

        Returns:
            List of incides
        """
        raise NotImplementedError

## -------------------------------------------------------------------------------------------------
    def _extract_rows(self, p_list_idx:list):
        """
        Extract the element in the buffer based on a
        list of indices.

        Parameters:
            p_list_idx (list): List of indices

        Returns:
            Samples in dictionary
        """
        rows = {}
        for key in self._data_buffer:
            rows[key] = [self._data_buffer[key][i] for i in p_list_idx]
        return rows

## -------------------------------------------------------------------------------------------------
    def is_full(self) -> bool:
        """
        Check if the buffer is full.

        Returns:
            True, if the buffer is full
        """
        keys = list(self._data_buffer.keys())
        return len(self._data_buffer[keys[0]]) >= self._size

## -------------------------------------------------------------------------------------------------
    def __len__(self):
        keys = list(self._data_buffer.keys())
        return len(self._data_buffer[keys[0]])

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BufferRnd(Buffer):
    """
    Buffer implmentation with random sampling
    """

## -------------------------------------------------------------------------------------------------
    def _gen_sample_ind(self, p_num: int) -> list:
        """
        Generate random indicies

        Parameters:
            p_num (int): Number of sample

        Returns:
            List of indicies
        """
        keys = list(self._data_buffer.keys())
        return random.sample(list(range(0,len(self._data_buffer[keys[0]]))),p_num)

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
    def __init__(self, p_buffer:Buffer=None, p_ada=True, p_logging=True):
        """
        Parameters:
            p_buffer            Buffer
            p_ada               Boolean switch for adaptivity
            p_logging           Boolean switch for logging functionality
        """

        Log.__init__(self, p_logging=p_logging)
        self.switch_adaptivity(p_ada)
        self._hyperparam_space  = HyperParamSpace()
        self._hyperparam_tupel  = None
        self._init_hyperparam()

        self._buffer = p_buffer

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

