## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro.pool.sarbuffers
## -- Module  : randombuffer
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-19  0.0.0     MRD      Creation
## -- 2021-09-19  1.0.0     MRD      Release first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-09-19)

This module provides the implementation of random sampling on SARBuffer.
"""

from mlpro.rl.models import *
import random



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SARRandomBuffer(SARBuffer):
    """
    State-Action-Reward Buffer with random sampling.

    """
    def __init__(self, p_size: int):
        """
        Parameters:
            p_size (int): Buffer size
        """
        super().__init__(p_size)

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
    def _extract_rows(self, p_list_idx: list) -> dict:
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
    