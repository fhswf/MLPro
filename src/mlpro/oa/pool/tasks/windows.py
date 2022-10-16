## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.pool.tasks.windows
## -- Module  : windows.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-10-16  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2022-10-16)
This module provides pool of window objects further used in the context of online adaptivity.
"""

import numpy as np
from mlpro.oa import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Window(OATask):
    """
    This is the base class for window implementations

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
        p_ada:bool, optional
            Adaptivity property of object. Default is True.
        p_logging      -Optional
            Log level for the object. Default is log everything.
    """
    C_NAME = 'Window'
    C_EVENT_BUFFER_FULL = 'BUFFER_FULL'
    C_EVENT_DATA_REMOVED = 'DATA_REMOVED'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_buffer_size:int,
                 p_delay:bool = False,
                 p_name:str = None,
                 p_range_max = OATask.C_RANGE_THREAD,
                 p_ada:bool = True,
                 p_logging = Log.C_LOG_ALL,
                 **p_kwargs):

        self._kwargs = p_kwargs.copy()

        super().__init__(p_name = p_name,
                         p_range_max=p_range_max,
                         p_ada=p_ada,
                         p_logging=p_logging
                         **p_kwargs)

        self._buffer = {}
        self._buffer_pos = 0


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new:list, p_inst_del:list):
        """
        Method to run the window including adding and deleting of elements

        Parameters
        ----------
            p_inst_new:list
                Instance/s to be added to the window
            p_inst_del:list
                Instance/s to be deleted from the window
        """
        pass


## -------------------------------------------------------------------------------------------------
    def get_buffered_data(self):
        """
        Method to fetch the date from the window buffer

        Returns
        -------
            buffer:dict
                the buffered data in the form of dictionary
            buffer_pos:int
                the latest buffer position
        """
        pass


## -------------------------------------------------------------------------------------------------
    def get_boundaries(self):
        """
        Method to get the current boundaries of the Window

        Returns
        -------
            boundaries:np.ndarray
                Returns the current window boundaries in the form of a Numpy array.
        """
        pass


## -------------------------------------------------------------------------------------------------
    def get_mean(self):
        """
        Method to get the mean of the data in the Window.

        Returns
        -------
            mean:np.ndarray
                Returns the mean of the current data in the window in the form of a Numpy array.
        """
        pass


## -------------------------------------------------------------------------------------------------
    def get_variance(self):
        """
        Method to get the variance of the data in the Window.

        Returns
        -------
            variance:np.ndarray
                Returns the variance of the current data in the window as a numpy array.
        """
        pass


## -------------------------------------------------------------------------------------------------
    def get_std_deviation(self, p_mean):
        """
        Method to get the standard deviation of the data in the window.

        Returns
        -------
            std:np.ndarray
                Returns the standard deviation of the data in the window as a numpy array.
        """
        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WindowR(Window):
    """
    Custom implementation of window class for high performance based on numpy.

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
        p_ada:bool, optional
            Adaptivity property of object. Default is True.
        p_logging      -Optional
            Log level for the object. Default is log everything.
    """
    C_NAME = 'Window (Real)'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_buffer_size:int,
                 p_delay:bool = False,
                 p_name:str = None,
                 p_range_max = OATask.C_RANGE_THREAD,
                 p_ada:bool = True,
                 p_logging = Log.C_LOG_ALL,
                 **p_kwargs):

        self._kwargs = p_kwargs.copy()

        super().__init__(p_buffer_size=p_buffer_size,
                         p_delay = p_delay,
                         p_name = p_name,
                         p_range_max=p_range_max,
                         p_ada=p_ada,
                         p_logging=p_logging
                         **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new:list, p_inst_old:list):
        """
        Method to run the window including adding and deleting of elements

        Parameters
        ----------
            p_inst_new:list
                Instance/s to be added to the window
            p_inst_del:list
                Instance/s to be deleted from the window
        """
        pass


## -------------------------------------------------------------------------------------------------
    def get_boundaries(self):
        """
        Method to get the current boundaries of the Window

        Returns
        -------
            boundaries:np.ndarray
                Returns the current window boundaries in the form of a Numpy array.
        """
        pass


## -------------------------------------------------------------------------------------------------
    def get_mean(self):
        """
        Method to get the mean of the data in the Window.

        Returns
        -------
            mean:np.ndarray
                Returns the mean of the current data in the window in the form of a Numpy array.
        """
        pass


## -------------------------------------------------------------------------------------------------
    def get_variance(self):
        """
        Method to get the variance of the data in the Window.

        Returns
        -------
            variance:np.ndarray
                Returns the variance of the current data in the window as a numpy array.
        """
        pass


## -------------------------------------------------------------------------------------------------
    def get_std_deviation(self):
        """
        Method to get the standard deviation of the data in the window.

            Returns
            -------
                std:np.ndarray
                    Returns the standard deviation of the data in the window as a numpy array.
        """
        pass
