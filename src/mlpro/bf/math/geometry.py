## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.math
## -- Module  : geometry.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-04-18  0.0.0     DA       Creation
## -- 2023-12-28  1.0.0     DA       Finalized class Point
## -- 2024-02-23  1.1.0     DA       Class Point: implementation of methods _renmove_plot*
## -- 2024-04-29  1.2.0     DA       Class Point: replaced parent Element by Properties
## -- 2024-04-30  1.3.0     DA       Class Point: re-normalization added
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.0 (2024-04-30)

This module provides class for geometric objects like points, etc.

""" 


from mlpro.bf.data import Properties
from mlpro.bf.plot import *
from mlpro.bf.math import *
from mlpro.bf.math.normalizers import Normalizer, Renormalizable
from datetime import datetime
from typing import Union




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Point (Properties, Plottable, Renormalizable):
    """
    Implementation of a point in a hyper space. Properties like the current position, velocity and
    acceleration are managed.

    Parameters
    ----------
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    """

    C_PROPERTY_POS      = 'Position'
    C_PLOT_ACTIVE       = True

## -------------------------------------------------------------------------------------------------
    def __init__( self, p_visualize : bool = False ):

        Properties.__init__( self )
        self._property_pos = self.define_property( p_property = self.C_PROPERTY_POS, p_derivative_order_max = 2  )

        self._plot_pos = None
        self._plot_vel = None
        Plottable.__init__( self, p_visualize=p_visualize )


## -------------------------------------------------------------------------------------------------
    def get_velocity(self):
        """
        Returns current velocity of the point.

        Returns
        -------
        point_vel 
            Current velocity of the point in unit/sec for each dimension.
        """

        try:
            return self._property_pos.derivatives[1]
        except:
            return None
    

## -------------------------------------------------------------------------------------------------
    def get_acceleration(self):
        """
        Returns current acceleration of the point.

        Returns
        -------
        point_acc 
            Current accelation of the point in unit/sec² for each dimension.
        """

        try:
            return self._property_pos.derivatives[2]
        except:
            return None


## -------------------------------------------------------------------------------------------------
    def get_position(self):
        """
        Returns current position of the point.

        Returns
        -------
        point_pos
            Current position of the point.
        """

        try:
            return self._property_pos.value
        except:
            return None
    

## -------------------------------------------------------------------------------------------------
    def set_position(self, p_pos : Union[list, np.ndarray], p_time_stamp : datetime = None):
        """
        Set/updates the point position and computes the resulting velocity and acceleration. It also
        updates the visualization.

        Parameters
        ----------
        p_pos : Union[ list, np.ndarray ]
            New position of the point.
        p_time_stamp : datetime = None
            Optional time stamp.
        """
        
        # 1 Update point data (position, velocity, acceleration)
        if self._property_pos.value is None:
            # 1.1 First call prepares the visualization
            self.set_property( p_property=self.C_PROPERTY_POS, p_value=p_pos, p_time_stamp=p_time_stamp )

            if ( self.get_visualization() ) and ( self._plot_settings is None ):
                if self._property_pos.dim == 2:
                    view = PlotSettings.C_VIEW_2D
                elif self._property_pos.dim == 3:
                    view = PlotSettings.C_VIEW_3D
                else:
                    view = PlotSettings.C_VIEW_ND

                self.set_plot_settings( p_plot_settings=PlotSettings( p_view = view ) )

        else:
            # 1.2 Subsequent calls just update the point data
            self.set_property( p_property=self.C_PROPERTY_POS, p_value=p_pos, p_time_stamp=p_time_stamp )

        # 2 Update plot
        self.update_plot()


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings: PlotSettings, **p_kwargs):

        point_pos = self._property_pos.value

        if self._plot_pos is None:
            self._plot_pos,  = p_settings.axes.plot( point_pos[0], 
                                                     point_pos[1], 
                                                     marker='+', 
                                                     color='red', 
                                                     linestyle='',
                                                     markersize=3 )
            
        else:
            self._plot_pos.set_xdata( point_pos[0] )    
            self._plot_pos.set_ydata( point_pos[1] )    

            if self._plot_vel is not None:
                self._plot_vel.remove()

            try:
                point_vel = self._property_pos.derivatives[1]
            except:
                return           

            self._plot_vel  = p_settings.axes.quiver( np.array([point_pos[0]]), 
                                                      np.array([point_pos[1]]),
                                                      np.array([point_vel[0]]),
                                                      np.array([point_vel[1]]),
                                                      scale = 1,
                                                      color='red' )


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings: PlotSettings, **p_kwargs):

        point_pos = self._property_pos.value

        if self._plot_pos is not None:
            self._plot_pos.remove()

        self._plot_pos,  = p_settings.axes.plot( point_pos[0], 
                                                 point_pos[1], 
                                                 point_pos[2],
                                                 marker='+', 
                                                 color='red',  
                                                 linestyle='',
                                                 markersize=3 )
            
        if self._plot_vel is not None:
            self._plot_vel.remove()

        try:
            point_vel = self._property_pos.derivatives[1]
        except:
            return
            
               
        len = ( point_vel[0]**2 + point_vel[1]**2 + point_vel[2]**2 ) **0.5

        self._plot_vel  = p_settings.axes.quiver( np.array([point_pos[0]]), 
                                                  np.array([point_pos[1]]),
                                                  np.array([point_pos[2]]),
                                                  np.array([point_vel[0]]),
                                                  np.array([point_vel[1]]),
                                                  np.array([point_vel[2]]),
                                                  length = len,
                                                  normalize = True,
                                                  color='red' )
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings: PlotSettings, **p_kwargs):
        pass


## -------------------------------------------------------------------------------------------------
    def _remove_plot_2d(self):
        if self._plot_pos is not None: 
            self._plot_pos.remove()
            self._plot_pos = None

            if self._plot_vel is not None:
                self._plot_vel.remove()
                self._plot_vel = None


## -------------------------------------------------------------------------------------------------
    def _remove_plot_3d(self):
        self._remove_plot_2d()


## -------------------------------------------------------------------------------------------------
    def _remove_plot_nd(self):
        pass        


## -------------------------------------------------------------------------------------------------
    def renormalize(self, p_normalizer: Normalizer):
        self._property_pos.value = p_normalizer.renormalize( p_data=self._property_pos.value )

        # 2024-04-30/DA Renormalization of derivates currently not implemented...

