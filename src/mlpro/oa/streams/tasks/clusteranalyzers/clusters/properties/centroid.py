## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties
## -- Module  : centroid.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-05-04  0.1.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2024-05-04)

This module provides ...

"""

from matplotlib.figure import Figure
from matplotlib.text import Text
from mpl_toolkits.mplot3d.art3d import Line3D, Text3D
from mlpro.bf.mt import Figure, PlotSettings
from mlpro.bf.various import *
from mlpro.bf.plot import *
from mlpro.bf.streams import *
from mlpro.bf.various import Id
from mlpro.bf.math.geometry import Point
from mlpro.bf.math.properties import *
from mlpro.oa.streams.tasks.clusteranalyzers.clusters import Cluster




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Centroid (Point, Id): 
    """
    This managed property provides full centroid functionality including 
    - managing its position
    - optionally its velocity and acceleration as auto-derivatives
    - plot functionality
    - renormalization

    Hint: please assign the id of the cluster to the centroid as well to get a proper visualization.

    Parameters
    ----------
    p_derivative_order_max : DerivativeOrderMax
        Maximum order of auto-generated derivatives (numeric properties only).
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    """

    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = False
    C_PLOT_VALID_VIEWS      = [ PlotSettings.C_VIEW_2D, 
                                PlotSettings.C_VIEW_3D, 
                                PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW     = PlotSettings.C_VIEW_ND

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_derivative_order_max : DerivativeOrderMax = 0, 
                  p_visualize : bool = False ):

        Point.__init__( self, p_derivative_order_max=p_derivative_order_max, p_visualize=p_visualize )
        Id.__init__( self, p_id = 0 )


## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
        super()._init_plot_2d(p_figure=p_figure, p_settings=p_settings)
        self._plot_line1 = None
        self._plot_line2 = None
        self._plot_line1_t1 : Text = None
        self._plot_line1_t2 : Text = None
        self._plot_line1_t3 : Text = None
        self._plot_line2_t1 : Text = None
        self._plot_line2_t2 : Text = None
    

## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
        super()._init_plot_3d(p_figure=p_figure, p_settings=p_settings)
        self._plot_line1 : Line3D = None
        self._plot_line1_t1 : Text3D = None
        self._plot_line1_t2 : Text3D = None

        self._plot_line2 : Line3D = None
        self._plot_line2_t1 : Text3D = None

        self._plot_line3 : Line3D = None
        self._plot_line3_t1 : Text3D = None
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings: PlotSettings, **p_kwargs):
        super()._update_plot_2d(p_settings, **p_kwargs)

        # 1 Get coordinates
        centroid = self.value
        ax_xlim  = p_settings.axes.get_xlim()
        ax_ylim  = p_settings.axes.get_ylim()
        xlim     = [ min( ax_xlim[0], centroid[0] ), max(ax_xlim[1], centroid[0] ) ]
        ylim     = [ min( ax_ylim[0], centroid[1] ), max(ax_ylim[1], centroid[1] ) ]

        # 2 Plot a crosshair
        if self._plot_line1 is None:
            # 2.1 Add initial crosshair lines
            cluster_id = self.get_id()
            col_id = cluster_id % len(Cluster.C_CLUSTER_COLORS)
            color = Cluster.C_CLUSTER_COLORS[col_id]
            label = ' C' + str(cluster_id) + ' '
            self._plot_line1 = p_settings.axes.plot( xlim, [centroid[1],centroid[1]], color=color, linestyle='dashed', lw=1, label=label)[0]
            self._plot_line2 = p_settings.axes.plot( [centroid[0],centroid[0]], ylim, color=color, linestyle='dashed', lw=1)[0]
            self._plot_line1_t1 = p_settings.axes.text(centroid[0], centroid[1], label, color=color )
            self._plot_line1_t2 = p_settings.axes.text(xlim[0], centroid[1], label, ha='right', va='center', color=color )
            self._plot_line1_t3 = p_settings.axes.text(xlim[1], centroid[1], label, ha='left',va='center', color=color )
            self._plot_line2_t1 = p_settings.axes.text(centroid[0], ylim[0], label, ha='center', va='top', color=color )
            self._plot_line2_t2 = p_settings.axes.text(centroid[0], ylim[1], label, ha='center', va='bottom',color=color )
            p_settings.axes.legend(title='Clusters', alignment='left', loc='upper right', shadow=True, draggable=True)
        else:
            # 2.2 Update data of crosshair lines
            self._plot_line1.set_data( xlim, [centroid[1],centroid[1]] )
            self._plot_line2.set_data( [centroid[0],centroid[0]], ylim )
            self._plot_line1_t1.set(position=(centroid[0], centroid[1]) )
            self._plot_line1_t2.set(position=(xlim[0], centroid[1]))
            self._plot_line1_t3.set(position=(xlim[1], centroid[1]))
            self._plot_line2_t1.set(position=(centroid[0], ylim[0]))
            self._plot_line2_t2.set(position=(centroid[0], ylim[1]))


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings: PlotSettings, **p_kwargs):
        super()._update_plot_3d(p_settings, **p_kwargs) 

        # 1 Get coordinates
        centroid = self.value
        ax_xlim  = p_settings.axes.get_xlim()
        ax_ylim  = p_settings.axes.get_ylim()
        ax_zlim  = p_settings.axes.get_zlim()
        xlim     = [ min( ax_xlim[0], centroid[0] ), max(ax_xlim[1], centroid[0] ) ]
        ylim     = [ min( ax_ylim[0], centroid[1] ), max(ax_ylim[1], centroid[1] ) ]
        zlim     = [ min( ax_zlim[0], centroid[2] ), max(ax_zlim[1], centroid[2] ) ]


        # 2 Determine label text alignments
        ap = p_settings.axes.get_axis_position()

        if ap[0]: 
            l1_t2_ha='left' 
        else: 
            l1_t2_ha='right'

        if ap[1]: 
            l2_t1_ha='right' 
        else: 
            l2_t1_ha='left'

        l3_t1_va='top' 


        # 3 Plot a crosshair with label texts
        if self._plot_line1 is None:
            # 3.1 Add initial crosshair lines
            cluster_id = self.get_id()
            col_id = cluster_id % len(Cluster.C_CLUSTER_COLORS)
            color = Cluster.C_CLUSTER_COLORS[col_id]
            label = ' C' + str(cluster_id) + ' '
            self._plot_line1 = p_settings.axes.plot( xlim, [centroid[1],centroid[1]], [centroid[2],centroid[2]], color=color, linestyle='dashed', lw=1, label=label)[0]
            self._plot_line2 = p_settings.axes.plot( [centroid[0],centroid[0]], ylim, [centroid[2],centroid[2]], color=color, linestyle='dashed', lw=1)[0]
            self._plot_line3 = p_settings.axes.plot( [centroid[0],centroid[0]], [centroid[1],centroid[1]], zlim, color=color, linestyle='dashed', lw=1)[0]

            self._plot_line1_t1 = p_settings.axes.text(centroid[0], centroid[1], centroid[2], label, color=color )
            self._plot_line1_t2 = p_settings.axes.text(xlim[0], centroid[1], centroid[2], label, ha=l1_t2_ha, va='center', color=color )
            self._plot_line2_t1 = p_settings.axes.text(centroid[0], ylim[0], centroid[2], label, ha=l2_t1_ha, va='center', color=color )
            self._plot_line3_t1 = p_settings.axes.text(centroid[0], centroid[1], zlim[0], label, ha='center', va=l3_t1_va, color=color )

            p_settings.axes.legend(title='Clusters', alignment='left', loc='right', shadow=True, draggable=True)
        else:
            # 3.2 Update data of crosshair lines
            self._plot_line1.set_data_3d( xlim, [centroid[1],centroid[1]], [centroid[2],centroid[2]] )
            self._plot_line2.set_data_3d( [centroid[0],centroid[0]], ylim, [centroid[2],centroid[2]] )
            self._plot_line3.set_data_3d( [centroid[0],centroid[0]], [centroid[1],centroid[1]], zlim )

            self._plot_line1_t1.set(position_3d=(centroid[0], centroid[1], centroid[2]))
            self._plot_line1_t2.set(position_3d=(xlim[0], centroid[1], centroid[2]), ha=l1_t2_ha)
            self._plot_line2_t1.set(position_3d=(centroid[0], ylim[0], centroid[2]), ha=l2_t1_ha)
            self._plot_line3_t1.set(position_3d=(centroid[0], centroid[1], zlim[0]), va=l3_t1_va)


## -------------------------------------------------------------------------------------------------
    def _remove_plot_2d(self):
        super()._remove_plot_2d()

        if self._plot_line1 is None: return

        self._plot_line1.remove()
        self._plot_line1 = None

        self._plot_line1_t1.remove()
        self._plot_line_t1 = None
        
        self._plot_line1_t2.remove()
        self._plot_line1_t2 = None

        self._plot_line1_t3.remove()
        self._plot_line1_t3 = None

        self._plot_line2.remove()
        self._plot_line2 = None

        self._plot_line2_t1.remove()
        self._plot_line2_t1 = None

        self._plot_line2_t2.remove()
        self._plot_line2_t2 = None


## -------------------------------------------------------------------------------------------------
    def _remove_plot_3d(self):
        super()._remove_plot_3d()

        if self._plot_line1 is None: return
        
        self._plot_line1.remove()
        self._plot_line1 = None

        self._plot_line1_t1.remove()
        self._plot_line1_t1 = None

        self._plot_line1_t2.remove()
        self._plot_line1_t2 = None

        self._plot_line2.remove()
        self._plot_line2 = None

        self._plot_line2_t1.remove()
        self._plot_line2_t1 = None

        self._plot_line3.remove()
        self._plot_line3 = None

        self._plot_line3_t1.remove()
        self._plot_line3_t1 = None
  

## -------------------------------------------------------------------------------------------------
    def _remove_plot_nd(self):
        super()._remove_plot_nd()


