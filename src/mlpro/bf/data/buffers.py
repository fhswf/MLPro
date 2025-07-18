## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.data
## -- Module  : data
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-04-16  0.0.0     DA       Creation
## -- 2021-05-29  1.0.0     DA       Release of first version
## -- 2021-06-16  1.1.0     SY       Adding the first version of data storing,
## --                                data plotting, and data saving classes
## -- 2021-06-21  1.2.0     SY       Add extensions in classes Loadable,
## --                                Saveable, DataPlotting & DataStoring.
## -- 2021-08-28  1.2.1     DA       Added constant C_VAR0 to class DataStoring
## -- 2021-09-18  1.2.1     MRD      Buffer Class Implementation. Add new parameter buffer
## --                                to the Adaptive Class
## -- 2021-09-19  1.2.2     MRD      Improvement on Buffer Class. Implement new base class
## --                                Buffer Element and BufferRnd
## -- 2021-09-22  1.3.0     MRD      New classes BufferElement, Buffer, BufferRnd
## -- 2021-09-25  1.3.1     MRD      Add __len__ functionality for SARBuffer
## -- 2023-02-09  1.3.2     MRD      Beautify
## -- 2023-03-02  1.3.3     SY       Update load_data in DataStoring
## -- 2024-04-28  1.4.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.4.0 (2024-04-28)

This module provides various elementary buffer management classes.

"""


import random



# Export list for public API
__all__ = [ 'BufferElement',
            'Buffer',
            'BufferRnd' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BufferElement:
    """
    Base class implementation for buffer element
    """


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_element: dict) -> None:
        """
        Parameters:
            p_element (dict): Buffer element in dictionary
        """

        self._element = {}
        self.add_value_element(p_element)


## -------------------------------------------------------------------------------------------------
    def add_value_element(self, p_val: dict):
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
    def add_element(self, p_elem: BufferElement):
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
    def get_sample(self, p_num: int):
        """
        Sample some element from the buffer.

        Parameters:
            p_num (int): Number of sample

        Returns:
            Samples in dictionary
        """
        
        return self._extract_rows(self._gen_sample_ind(p_num))


## -------------------------------------------------------------------------------------------------
    def _gen_sample_ind(self, p_num: int) -> list:
        """
        Generate random indices from the buffer.

        Parameters:
            p_num (int): Number of sample

        Returns:
            List of incides
        """
        
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _extract_rows(self, p_list_idx: list):
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
        return random.sample(list(range(0, len(self._data_buffer[keys[0]]))), p_num)
