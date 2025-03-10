## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.clusteranalyzers.clusters
## -- Module  : body.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-10  1.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-03-10)

This module provides a template class for clusters with a centroid and a body.

"""

from datetime import datetime

import numpy as np

from mlpro.bf.streams import Instance
from mlpro.bf.math.geometry import cprop_center_geo, cprop_size_geo

from sparccstream.clusters.properties.basics import CellComponent




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Body (CellComponent):
    """
    Template class for cell bodies of SPARCCStream cells.

    Attributes
    ----------
    boundaries : np.ndarray[dim][2]
        Left/right boundaries of the body per dimension. 
    num_inst : np.ndarray[dim][2]
        Number of instances within the body per dimension and side of the nucleolus.
    """

    C_PROPERTIES        = [ cprop_center_geo, cprop_size_geo ]

    C_PLOT_DETAIL_LEVEL = 2

## -------------------------------------------------------------------------------------------------
    def init(self, p_num_dim: int):
        self._set_num_inst( p_num_inst = np.zeros( shape=(p_num_dim, 2) ) )
        self._set_boundaries( p_boundaries = np.zeros( shape=(p_num_dim, 2) ) )


## -------------------------------------------------------------------------------------------------
    def reset(self):
        self.remove_plot()
        try:
            self.value.fill(0)
            self._set_boundaries( p_boundaries = None )
        except:
            pass


## -------------------------------------------------------------------------------------------------
    def equalize(self, p_dim: int, p_side: int, p_defector_pos: float):

        if self.num_inst[p_dim][p_side] > 1:
            self.num_inst[p_dim][p_side] -= 1
        else:
            self.num_inst[p_dim][p_side]   = 0
            self.boundaries[p_dim][p_side] = self._cell.centroid.value[p_dim]

        self.num_inst[p_dim][1-p_side] += 1

       
## -------------------------------------------------------------------------------------------------
    def get_membership(self, p_inst : Instance) -> float:
        """
        Custom method to determine an absolute membership value for the specified instance based
        on the individual geometry.

        Parameters
        ----------
        p_inst : Instance
            Instance for which the membership value is determined.

        Returns
        -------
        float
            Absolute membership value.
        """

        raise NotImplementedError
    

## -------------------------------------------------------------------------------------------------
    def check_collision(self, p_body : CellComponent ) -> bool:
        """
        Custom method to detect a collision with a specified cell body.

        Parameters
        ----------
        p_body : CellComponent
            Other cell body for with a collision is detected.

        Returns
        -------
        bool
            True, if both bodies are colliding. False otherwise.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _get_boundaries(self):
        return self._boundaries


## -------------------------------------------------------------------------------------------------
    def _set_boundaries(self, p_boundaries : np.ndarray):
        self._boundaries = p_boundaries


## -------------------------------------------------------------------------------------------------
    def _get_num_inst(self):
        return super()._get()
    

## -------------------------------------------------------------------------------------------------
    def _set_num_inst( self, 
                       p_num_inst: np.ndarray, 
                       p_time_stamp: datetime | int | float = None, 
                       p_upd_time_stamp: bool = True, 
                       p_upd_derivatives: bool = True ):
        
        return super().set( p_value = p_num_inst, 
                            p_time_stamp = p_time_stamp, 
                            p_upd_time_stamp = p_upd_time_stamp, 
                            p_upd_derivatives = p_upd_derivatives )
    

## -------------------------------------------------------------------------------------------------
    def divide(self, p_dim_id) -> object:
        
        new_body = type(self)( p_name = self.name,
                               p_derivative_order_max = self._derivative_order_max, 
                               p_value_prev = self._sw_value_prev,
                               p_visualize = self.get_visualization() )
        
        new_body.init( p_num_dim=self.num_inst.shape[0] )
        self.reset()
        return new_body


## -------------------------------------------------------------------------------------------------
    boundaries = property( fget = _get_boundaries, fset = _set_boundaries )
    num_inst   = property( fget = _get_num_inst, fset = _set_num_inst )
