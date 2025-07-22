## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties
## -- Module  : centroid.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-05-04  0.1.0     DA       Creation
## -- 2024-05-29  0.2.0     DA       Refactoring
## -- 2024-05-30  0.3.0     DA       Global aliases: new boolean param ValuePrev
## -- 2024-05-31  0.4.0     DA       Improved the stability of the plot methods
## -- 2024-06-13  0.5.0     DA       New property definitions cprop_centroid_prev*
## -- 2024-06-26  0.6.0     DA       Refactoring
## -- 2024-07-13  0.7.0     DA       Refactoring
## -- 2024-10-31  0.8.0     DA       New parent class Crosshair
## -- 2024-12-11  0.8.1     DA       Pseudo classes if matplotlib is not installed
## -- 2025-03-19  0.8.2     DA       Removed property definitions cprop_center_geo*
## -- 2025-06-08  0.9.0     DA       Refactoring of Centroid._update_plot*: new return parameter
## -- 2025-06-25  1.0.0     DA       Class Crosshair: implementation of nD plot methods
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-06-25)

This module provides the cluster property class 'Centroid'.

"""

try:
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    from mpl_toolkits.mplot3d.art3d import Line3D, Text3D
except:
    class Figure : pass
    class Text : pass
    class Line3D : pass
    class Text3D : pass

from mlpro.bf.mt import Figure, PlotSettings
from mlpro.bf.various import *
from mlpro.bf.plot import *
from mlpro.bf.streams import *
from mlpro.bf.various import Id
from mlpro.bf.math.geometry import Crosshair
from mlpro.bf.math.properties import *
from mlpro.oa.streams.tasks.clusteranalyzers.clusters import Cluster



# Export list for public API
__all__ = [ 'Centroid',
            'cprop_centroid',
            'cprop_centroid1',
            'cprop_centroid2',
            'cprop_centroid_prev',
            'cprop_centroid_prev1',
            'cprop_centroid_prev2' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Centroid (Crosshair, Id): 
    """
    This managed property provides full centroid functionality including 
    - managing its position
    - optionally its velocity and acceleration as auto-derivatives
    - plot functionality
    - renormalization

    Hint: please assign the id of the cluster to the centroid as well to get a proper visualization.

    Parameters
    ----------
    p_name : str
        Name of the property
    p_derivative_order_max : DerivativeOrderMax
        Maximum order of auto-generated derivatives (numeric properties only).
    p_value_prev : bool
        If True, the previous value is stored in value_prev whenever value is updated.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_kwargs : dict
        Keyword parameters.
    """

    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = False
    C_PLOT_VALID_VIEWS      = [ PlotSettings.C_VIEW_2D, 
                                PlotSettings.C_VIEW_3D, 
                                PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW     = PlotSettings.C_VIEW_ND

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name : str,
                  p_derivative_order_max : DerivativeOrderMax = 0, 
                  p_value_prev : ValuePrev = False,
                  p_visualize : bool = False,
                  **p_kwargs ):

        Crosshair.__init__( self, 
                            p_name = p_name, 
                            p_derivative_order_max = p_derivative_order_max,
                            p_value_prev = p_value_prev, 
                            p_visualize = p_visualize,
                            **p_kwargs )
        
        Id.__init__( self, p_id = 0 )

        self.color = None


## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
        Crosshair._init_plot_2d(self, p_figure=p_figure, p_settings=p_settings)
        self._plot_line1_t1 : Text = None
        self._plot_line1_t2 : Text = None
        self._plot_line1_t3 : Text = None
        self._plot_line2_t1 : Text = None
        self._plot_line2_t2 : Text = None
    
        self._plot_label = None
            

## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
        Crosshair._init_plot_3d(self, p_figure=p_figure, p_settings=p_settings)
        self._plot_line1_t1 : Text3D = None
        self._plot_line1_t2 : Text3D = None
        self._plot_line2_t1 : Text3D = None
        self._plot_line3_t1 : Text3D = None

        self._plot_label = None


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure, p_settings):
        Crosshair._init_plot_nd(self, p_figure=p_figure, p_settings=p_settings)

        self._plot_line_texts = []


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings: PlotSettings, **p_kwargs) -> bool:

        # 0 Intro
        if self.value is None: return False


        # 1 Determine the color of the crosshair
        cluster_id = self.id
        if self.color is None:
            col_id     = cluster_id % len(Cluster.C_CLUSTER_COLORS)
            self.color = Cluster.C_CLUSTER_COLORS[col_id]

        
        # 2 Plot the crosshair
        Crosshair._update_plot_2d(self, p_settings, **p_kwargs)
        

        # 3 Get line coordinates
        centroid = self.value
        ax_xlim  = p_settings.axes.get_xlim()
        ax_ylim  = p_settings.axes.get_ylim()
        xlim     = [ min( ax_xlim[0], centroid[0] ), max(ax_xlim[1], centroid[0] ) ]
        ylim     = [ min( ax_ylim[0], centroid[1] ), max(ax_ylim[1], centroid[1] ) ]
              

        # 4 Add label elements
        if self._plot_label is None:

            # 4.1 Add all label elements
            self._plot_label = ' C' + str(cluster_id) + ' '
            self._plot_line1.set_label( self._plot_label )
            self._plot_line1_t1 = p_settings.axes.text(centroid[0], centroid[1], self._plot_label, color=self.color )
            self._plot_line1_t2 = p_settings.axes.text(xlim[0], centroid[1], self._plot_label, ha='right', va='center', color=self.color )
            self._plot_line1_t3 = p_settings.axes.text(xlim[1], centroid[1], self._plot_label, ha='left',va='center', color=self.color )
            self._plot_line2_t1 = p_settings.axes.text(centroid[0], ylim[0], self._plot_label, ha='center', va='top', color=self.color )
            self._plot_line2_t2 = p_settings.axes.text(centroid[0], ylim[1], self._plot_label, ha='center', va='bottom',color=self.color )

            p_settings.axes.legend(title='Clusters', alignment='left', loc='upper right', draggable=True)

        else:
            # 4.2 Update color and labels of the crosshair lines
            self._plot_line1_t1.set(position=(centroid[0], centroid[1]), color=self.color)
            self._plot_line1_t2.set(position=(xlim[0], centroid[1]), color=self.color)
            self._plot_line1_t3.set(position=(xlim[1], centroid[1]), color=self.color)
            self._plot_line2_t1.set(position=(centroid[0], ylim[0]), color=self.color)
            self._plot_line2_t2.set(position=(centroid[0], ylim[1]), color=self.color)

        return True


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings: PlotSettings, **p_kwargs) -> bool:

        # 0 Intro
        if self.value is None: return False

        
        # 1 Determine the color of the crosshair
        cluster_id = self.id
        if self.color is None:
            col_id     = cluster_id % len(Cluster.C_CLUSTER_COLORS)
            self.color = Cluster.C_CLUSTER_COLORS[col_id]

        
        # 2 Plot the crosshair
        Crosshair._update_plot_3d(self, p_settings, **p_kwargs)

        
        # 3 Get coordinates
        centroid = self.value
        ax_xlim  = p_settings.axes.get_xlim()
        ax_ylim  = p_settings.axes.get_ylim()
        ax_zlim  = p_settings.axes.get_zlim()
        xlim     = [ min( ax_xlim[0], centroid[0] ), max(ax_xlim[1], centroid[0] ) ]
        ylim     = [ min( ax_ylim[0], centroid[1] ), max(ax_ylim[1], centroid[1] ) ]
        zlim     = [ min( ax_zlim[0], centroid[2] ), max(ax_zlim[1], centroid[2] ) ]


        # 4 Determine label text alignments
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
            

        # 5 Add label elements
        if self._plot_label is None:

            # 5.1 Add all label elements
            cluster_id = self.get_id()
            self._plot_label = ' C' + str(cluster_id) + ' '
            self._plot_line1.set_label(self._plot_label)
            self._plot_line1_t1 = p_settings.axes.text(centroid[0], centroid[1], centroid[2], self._plot_label, color=self.color )
            self._plot_line1_t2 = p_settings.axes.text(xlim[0], centroid[1], centroid[2], self._plot_label, ha=l1_t2_ha, va='center', color=self.color )
            self._plot_line2_t1 = p_settings.axes.text(centroid[0], ylim[0], centroid[2], self._plot_label, ha=l2_t1_ha, va='center', color=self.color )
            self._plot_line3_t1 = p_settings.axes.text(centroid[0], centroid[1], zlim[0], self._plot_label, ha='center', va=l3_t1_va, color=self.color )

            p_settings.axes.legend(title='Clusters', alignment='left', loc='right', draggable=True)

        else:
            # 5.2 Update color and labels of the crosshair lines
            self._plot_line1_t1.set(position_3d=(centroid[0], centroid[1], centroid[2]), color=self.color)
            self._plot_line1_t2.set(position_3d=(xlim[0], centroid[1], centroid[2]), ha=l1_t2_ha, color=self.color)
            self._plot_line2_t1.set(position_3d=(centroid[0], ylim[0], centroid[2]), ha=l2_t1_ha, color=self.color)
            self._plot_line3_t1.set(position_3d=(centroid[0], centroid[1], zlim[0]), va=l3_t1_va, color=self.color)

        return True
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings, **p_kwargs) -> bool:

        # 0 Intro
        if self.value is None: return False


        # 1 Determine the color of the crosshair
        cluster_id = self.id
        if self.color is None:
            col_id     = cluster_id % len(Cluster.C_CLUSTER_COLORS)
            self.color = Cluster.C_CLUSTER_COLORS[col_id]


        
        # 2 Plot the crosshair
        Crosshair._update_plot_nd(self, p_settings, **p_kwargs)
        

        # 3 Get line coordinates
        centroid = self.value
        xpos     = p_settings.axes.get_xlim()[1]
              

        # 4 Add label elements
        if not self._plot_line_texts:

            # 4.1 Add all label elements
            plot_label_feature = len(centroid) > 1

            plot_label_stub    = ' C' + str(cluster_id) + ' '

            plot_label         = plot_label_stub
            self._plot_lines[0].set_label( plot_label_stub )

            for i, centroid_pos in enumerate(centroid):
                if plot_label_feature:
                    plot_label = plot_label_stub + 'F' + str(i) + ' '

                self._plot_line_texts.append( p_settings.axes.text( xpos, centroid_pos, plot_label, color=self.color ) )
            
            p_settings.axes.legend(title='Feat./Clust.', alignment='left', loc='upper right', draggable=True)

        else:
            # 4.2 Update color and labels of the crosshair lines
            for i, plot_line_text in enumerate(self._plot_line_texts):
                plot_line_text.set( position=(xpos, centroid[i]), color=self.color)
    

## -------------------------------------------------------------------------------------------------
    def _remove_plot_2d(self):

        if self._plot_line1 is None: return

        Crosshair._remove_plot_2d(self)
        
        self._plot_line1_t1.remove()
        self._plot_line_t1 = None
        
        self._plot_line1_t2.remove()
        self._plot_line1_t2 = None

        self._plot_line1_t3.remove()
        self._plot_line1_t3 = None

        self._plot_line2_t1.remove()
        self._plot_line2_t1 = None

        self._plot_line2_t2.remove()
        self._plot_line2_t2 = None


## -------------------------------------------------------------------------------------------------
    def _remove_plot_3d(self):

        if self._plot_line1 is None: return
        
        Crosshair._remove_plot_3d(self)

        self._plot_line1_t1.remove()
        self._plot_line1_t1 = None

        self._plot_line1_t2.remove()
        self._plot_line1_t2 = None

        self._plot_line2_t1.remove()
        self._plot_line2_t1 = None

        self._plot_line3_t1.remove()
        self._plot_line3_t1 = None


## -------------------------------------------------------------------------------------------------
    def _remove_plot_nd(self):
        
        if not self._plot_line_texts: return

        for plot_line_text in self._plot_line_texts: 
            plot_line_text.remove()

        self._plot_line_texts.clear()

        Crosshair._remove_plot_nd(self)
  




# Centroid with 0,1,2 order derivatives and plot functionality with/without storing previous values
cprop_centroid       : PropertyDefinition = ( 'centroid', 0, False, Centroid )
cprop_centroid1      : PropertyDefinition = ( 'centroid', 1, False, Centroid )
cprop_centroid2      : PropertyDefinition = ( 'centroid', 2, False, Centroid )

cprop_centroid_prev  : PropertyDefinition = ( 'centroid', 0, True, Centroid )
cprop_centroid_prev1 : PropertyDefinition = ( 'centroid', 1, True, Centroid )
cprop_centroid_prev2 : PropertyDefinition = ( 'centroid', 2, True, Centroid )
