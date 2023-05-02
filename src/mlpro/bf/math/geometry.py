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

    C_PLOT_ACTIVE   = True

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_pos : Union[list, np.ndarray] = None, p_time_stamp : datetime = None, p_visualize: bool = False):
        self._point_pos : np.ndarray = None
        self._point_vel : np.ndarray = None
        self._point_acc : np.ndarray = None
        self._plot_pos               = None
        self._plot_vel               = None
        self._last_update : datetime = p_time_stamp
        super().__init__(p_visualize = p_visualize)

        if p_pos != None:
            self.set_pos( p_pos = p_pos, p_timestamp = p_time_stamp)


## -------------------------------------------------------------------------------------------------
    def get_details(self) -> Tuple[ np.ndarray, np.ndarray, np.ndarray ]:
        """
        Returns details of the point.

        Returns
        -------
        point_pos : np.ndarray
            Current position of the point.
        point_vel : np.ndarray
            Current velocity of the point in unit/sec for each dimension.
        point_acc : np.ndarray
            Current accelation of the point in unit/sec² for each dimension.
        """

        return self._point_pos, self._point_vel, self._point_acc
    

## -------------------------------------------------------------------------------------------------
    def set_pos(self, p_pos : Union[ list, np.ndarray ], p_time_stamp : datetime = None):
        """
        Set/updates the point position and computes the resulting velocity and acceleration.

        Parameters
        ----------
        p_pos : Union[ list, np.ndarray ]
            New position of the point.
        p_time_stamp : datetime = None
            Optional time stamp.
        """

        # 1 Update point data
        if self._point_pos is None:
            self._point_pos   = np.array(p_pos)

            if ( self.get_visualization() ) and ( self._plot_settings is None ):
                if self._point_pos.size == 2:
                    view = PlotSettings.C_VIEW_2D
                elif self._point_pos.size == 3:
                    view = PlotSettings.C_VIEW_3D
                else:
                    view = PlotSettings.C_VIEW_ND

                self.set_plot_settings( p_plot_settings=PlotSettings( p_view = view ) )

        else:
            if ( p_time_stamp != None ) and ( self._last_update != None ):
                delta_t = (p_time_stamp - self._last_update).total_seconds()
            else:
                delta_t = 1

            # 1.1 Update velocity
            pos_old         = self._point_pos
            self._point_pos = np.array(p_pos)
            vel_old         = self._point_vel

            self._point_vel = ( self._point_pos - pos_old ) / delta_t

            # 1.2 Update acceleration
            if vel_old is not None:
                self._point_acc = ( self._point_vel - vel_old ) / delta_t


        # 2 Update time stamp
        self._last_update = p_time_stamp


        # 3 Update plot
        self.update_plot()


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings: PlotSettings, **p_kwargs):

        if self._plot_pos is None:
            self._plot_pos,  = p_settings.axes.plot( self._point_pos[0], 
                                                     self._point_pos[1], 
                                                     marker='+', 
                                                     color='blue', 
                                                     linestyle='',
                                                     markersize=3 )
            
        else:
            self._plot_pos.set_xdata( self._point_pos[0] )    
            self._plot_pos.set_ydata( self._point_pos[1] )    

            if self._plot_vel is not None:
                self._plot_vel.remove()

            self._plot_vel  = p_settings.axes.quiver( np.array([self._point_pos[0]]), 
                                                      np.array([self._point_pos[1]]),
                                                      np.array([self._point_vel[0]]),
                                                      np.array([self._point_vel[1]]),
                                                      scale = 1,
                                                      color='red' )


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings: PlotSettings, **p_kwargs):

        if self._plot_pos is not None:
            self._plot_pos.remove()

        self._plot_pos,  = p_settings.axes.plot( self._point_pos[0], 
                                                 self._point_pos[1], 
                                                 self._point_pos[2],
                                                 marker='+', 
                                                 color='blue', 
                                                 linestyle='',
                                                 markersize=3 )
            
        if self._plot_vel is not None:
            self._plot_vel.remove()
        elif self._point_vel is None:
            return

        self._plot_vel  = p_settings.axes.quiver( np.array([self._point_pos[0]]), 
                                                  np.array([self._point_pos[1]]),
                                                  np.array([self._point_pos[2]]),
                                                  np.array([self._point_vel[0]]),
                                                  np.array([self._point_vel[1]]),
                                                  np.array([self._point_vel[2]]),
                                                  length = 0.1,
                                                  normalize = True,
                                                  color='red' )
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings: PlotSettings, **p_kwargs):
        pass
    
