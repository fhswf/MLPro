## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.math
## -- Module  : geometry.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-04-18  0.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-04-18)

This module provides class for geometric objects like points, etc.

""" 


from mlpro.bf.plot import *
from mlpro.bf.math import *
from datetime import datetime
from typing import Union, Tuple




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Point (Plottable):
    """
    Implementation of a point in a hyper space. Properties like the current position, velocity and
    acceleration are managed.

    Parameters
    ----------
    p_pos : Union[ list, np.ndarray ] = None
        Optional initial position of the point.
    p_time_stamp : datetime = None
        Optional initial time stamp of the initial position.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.

    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_pos : Union[list, np.ndarray] = None, p_time_stamp : datetime = None, p_visualize: bool = False):
        self._point_pos              = None
        self._point_vel : np.ndarray = None
        self._point_acc : np.ndarray = None
        self._last_update : datetime = p_time_stamp
        super().__init__(p_visualize = p_visualize)

        if p_pos != None:
            self.set_pos( p_pos = p_pos, p_timestamp = p_time_stamp)


## -------------------------------------------------------------------------------------------------
    def get_details(self) -> Tuple[ Union[list, np.ndarray], np.ndarray, np.ndarray ]:
        """
        Returns details of the points.

        Returns
        -------
        point_pos : Union[ list, np.ndarray ]
            Current position of the point.
        point_vel : np.ndarray
            Current velocity of the point in unit/sec for each dimension.
        point_acc : np.ndarray
            Current accelation of the point in unit/sec² for each dimension.
        """

        return self._point_pos, self._point_vel, self._point_acc
    

## -------------------------------------------------------------------------------------------------
    def set_pos(self, p_pos : Union[ list, np.ndarray ], p_time_stamp : datetime = None):

        if self._point_pos != None:

            if ( p_time_stamp != None ) and ( self._last_update != None ):
                delta_t = (p_time_stamp - self._last_update).total_seconds()
            else:
                delta_t = 2

            # Update velocity
            vel_old         = self._point_vel
            self._point_vel = ( np.array(p_pos) - np.array(self._point_pos) ) / delta_t

            # Update acceleration
            if vel_old != None:
                self._point_acc = ( self._point_vel - vel_old ) / delta_t


        # Update point position and time stamp
        self._point_pos   = p_pos
        self._last_update = p_time_stamp


## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
        return super()._init_plot_2d(p_figure, p_settings)
    

## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
        return super()._init_plot_3d(p_figure, p_settings)
    

## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure: Figure, p_settings: PlotSettings):
        return super()._init_plot_nd(p_figure, p_settings)
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings: PlotSettings, **p_kwargs):
        return super()._update_plot_2d(p_settings, **p_kwargs)
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings: PlotSettings, **p_kwargs):
        return super()._update_plot_3d(p_settings, **p_kwargs)
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings: PlotSettings, **p_kwargs):
        return super()._update_plot_nd(p_settings, **p_kwargs)
    
