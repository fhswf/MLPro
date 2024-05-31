## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.math.geometry
## -- Module  : point.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-04-18  0.0.0     DA       Creation
## -- 2023-12-28  1.0.0     DA       Finalized class Point
## -- 2024-02-23  1.1.0     DA       Class Point: implementation of methods _renmove_plot*
## -- 2024-04-29  1.2.0     DA       Class Point: replaced parent Element by Properties
## -- 2024-04-30  1.3.0     DA       Class Point: re-normalization added
## -- 2024-05-06  1.4.0     DA       Class Point: refactoring
## -- 2024-05-07  1.4.1     DA       Bugfix in method Point.renormalize()
## -- 2024-05-24  1.4.2     DA       Bugfix in method _update_plot_2d()
## -- 2024-05-29  1.5.0     DA       Cleaned the code and completed the documentation
## -- 2024-05-30  1.6.0     DA       Global aliases: new boolean param ValuePrev
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.6.0 (2024-05-30)

This module provides class for geometric objects like points, etc.

""" 


from matplotlib.figure import Figure
from mlpro.bf.plot import *
from mlpro.bf.math.properties import *
from mlpro.bf.math.normalizers import Normalizer
from mlpro.bf.plot import PlotSettings




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Point (Property):
    """
    Implementation of a point in a hyper space. Current position, velocity and acceleration are managed.

    Attributes
    ----------
    values
        Current point coordinates
    """

    C_PLOT_ACTIVE       = True
    C_PLOT_STANDALONE   = False
    C_PLOT_VALID_VIEWS  = [PlotSettings.C_VIEW_2D, PlotSettings.C_VIEW_3D, PlotSettings.C_VIEW_ND]
    C_PLOT_DEFAULT_VIEW = PlotSettings.C_VIEW_ND

## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure: Figure = None, p_plot_settings: PlotSettings = None, **p_kwargs):
        self._plot_pos = None
        self._plot_vel = None
        super().init_plot(p_figure, p_plot_settings, **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings: PlotSettings, **p_kwargs):

        point_pos = self.value

        if self._plot_pos is not None:
            self._plot_pos.remove()

        self._plot_pos,  = p_settings.axes.plot( point_pos[0], 
                                                 point_pos[1], 
                                                 marker='+', 
                                                 color='red', 
                                                 linestyle='',
                                                 markersize=3 )
            
        if self._plot_vel is not None:
            self._plot_vel.remove()

        try:
            point_vel = self.derivatives[1]
        except:
            return

        if point_vel is not None:
            self._plot_vel  = p_settings.axes.arrow( point_pos[0], 
                                                     point_pos[1], 
                                                     point_vel[0], 
                                                     point_vel[1],
                                                     color='red' )
                                                          
                                                         
## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings: PlotSettings, **p_kwargs):

        point_pos = self.value

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
            point_vel = self.derivatives[1]
        except:
            return
        
        if point_vel is not None:
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
        self._value = p_normalizer.renormalize( p_data=np.array(self.value) )
        self._derivatives[0] = self._value

        # 2024-04-30/DA Renormalization of derivates currently not implemented...





cprop_point  : PropertyDefinition = ( 'point', 0, False, Point )
cprop_point1 : PropertyDefinition = ( 'point', 1, False, Point )
cprop_point2 : PropertyDefinition = ( 'point', 2, False, Point )