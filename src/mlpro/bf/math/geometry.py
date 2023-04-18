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
from typing import Tuple




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Point (Plottable):

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_pos : Element, p_visualize: bool = False):
        self._point_pos : Element = None
        self._point_vel : Element = None
        self._point_acc : Element = None
        super().__init__(p_visualize = p_visualize)


## -------------------------------------------------------------------------------------------------
    def get_point(self) -> Tuple[ Element, Element, Element ]:
        return self._point_pos, self._point_vel, self._point_acc
    

## -------------------------------------------------------------------------------------------------
    def set_point(self, p_pos : Element):
        pass


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
    
