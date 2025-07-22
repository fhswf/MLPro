## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.streams.tasks.windows
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-10-16  0.0.0     LSB      Creation
## -- 2022-11-04  0.1.0     LSB      Removing class WindowR
## -- 2022-11-24  0.2.0     LSB      Implementations and release of nd plotting
## -- 2022-11-26  0.3.0     LSB      Implementations and release of 3-d plotting
## -- 2022-12-08  0.4.0     DA       Refactoring after changes on bf.streams
## -- 2022-12-08  1.0.0     LSB      Release
## -- 2022-12-08  1.0.1     LSB      Compatilbility for both Instance and Element object
## -- 2022-12-16  1.0.2     LSB      Delay in delivering the buffered data
## -- 2022-12-16  1.0.3     DA       Refactoring after changes on bf.streams
## -- 2022-12-18  1.0.4     LSB      Bug Fixes
## -- 2022-12-18  1.1.0     LSB      New plot updates -
##                                   - single rectangle
##                                   - transparent patch on obsolete data
## -- 2022-12-19  1.1.1     DA       New parameter p_duplicate_data
## -- 2022-12-28  1.1.2     DA       Refactoring of plot settings
## -- 2022-12-29  1.1.3     DA       Removed method Window.init_plot()
## -- 2022-12-31  1.1.4     LSB      Refactoring
## -- 2023-02-02  1.1.5     DA       Methods Window._init_plot_*: removed figure creation
## -- 2024-05-22  1.2.0     DA       Refactoring and splitting
## -- 2025-04-11  1.2.1     DA       Code review/cleanup
## -- 2025-06-01  2.0.0     DA       Refactoring of class Window:
## --                                - events removed
## --                                - method get_boundaries(): new parameters
## -- 2025-06-05  2.0.1     DA       Bugfix in Window.get_variance()
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.0.1 (2025-06-05)

This module provides pool of window objects further used in the context of online adaptivity.
"""

from typing import Tuple

import numpy as np

from mlpro.bf.various import Log
from mlpro.bf.events import *
from mlpro.bf.math.statistics import *
from mlpro.bf.streams import *



# Export list for public API
__all__ = [ 'Window' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Window (StreamTask, BoundaryProvider):
    """
    This is the abstract root class for window implementations. 

    Parameters
    ----------
    p_buffer_size:int
        the size/length of the buffer/window.
    p_delay:bool, optional
        Set to true if full buffer is desired before passing the window data to next step. Default is false.
    p_name:str, optional
        Name of the Window. Default is None.
    p_range_max     -Optional
        Maximum range of task parallelism for window task. Default is set to multithread.
    p_duplicate_data : bool
        If True, instances will be duplicated before processing. Default = False.
    p_ada:bool, optional
        Adaptivity property of object. Default is True.
    p_logging      -Optional
        Log level for the object. Default is log everything.
    """

    C_NAME                  = 'Window'

    C_PLOT_STANDALONE       = False

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_buffer_size:int,
                  p_delay:bool = False,
                  p_enable_statistics:bool = False,
                  p_name:str   = None,
                  p_range_max  = StreamTask.C_RANGE_THREAD,
                  p_duplicate_data : bool = False,
                  p_visualize:bool = False,
                  p_logging    = Log.C_LOG_ALL,
                  **p_kwargs ):

        super().__init__( p_name = p_name,
                          p_range_max = p_range_max,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging = p_logging,
                          p_kwargs = p_kwargs )

        self.buffer_size         = p_buffer_size
        self._delay              = p_delay
        self._buffer             = {}
        self._buffer_pos         = 0
        self._statistics_enabled = p_enable_statistics


## -------------------------------------------------------------------------------------------------
    def get_buffered_data(self) -> Tuple[dict,int]:
        """
        Method to fetch the date from the window buffer

        Returns
        -------
        buffer:dict
            the buffered data in the form of dictionary
        buffer_pos:int
            the latest buffer position
        """

        return (self._buffer, self._buffer_pos)


## -------------------------------------------------------------------------------------------------
    def get_mean(self):
        """
        Method to get the mean of the data in the Window.

        Returns
        -------
        mean:np.ndarray
            Returns the mean of the current data in the window in the form of a Numpy array.
        """

        return np.mean(self._buffer.values(), axis=0, dtype=np.float64)


## -------------------------------------------------------------------------------------------------
    def get_variance(self):
        """
        Method to get the variance of the data in the Window.

        Returns
        -------
        variance:np.ndarray
            Returns the variance of the current data in the window as a numpy array.
        """

        return np.var(self._buffer.values(), axis=0, dtype=np.float64)


## -------------------------------------------------------------------------------------------------
    def get_std_deviation(self):
        """
        Method to get the standard deviation of the data in the window.

        Returns
        -------
        std:np.ndarray
            Returns the standard deviation of the data in the window as a numpy array.
        """

        return np.std(self._buffer.values(), axis=0, dtype=np.float64)
