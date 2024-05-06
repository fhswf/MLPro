## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.clusteranalyzers
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-01-24  0.0.0     DA       Creation
## -- 2023-04-18  0.1.0     DA       First implementation of classes ClusterMembership, ClusterAnalyzer
## -- 2023-05-06  0.2.0     DA       New class ClusterCentroid
## -- 2023-05-14  0.3.0     DA       Class ClusterAnalyzer: simplification
## -- 2023-05-30  0.3.1     DA       Further comments, docstrings
## -- 2023-06-03  0.4.0     DA       Method ClusterAnalyzer.get_cluster_memberships():
## --                                - renaming
## --                                - new parameter p_scope
## --                                - refactoring
## --                                New Method ClusterAnalyzer.new_cluster_allowed()
## -- 2023-11-18  0.5.0     DA       Class ClusterCentroid: added plot functionality
## -- 2023-12-08  0.6.0     DA       Class ClusterAnalyzer: 
## --                                - changed internal cluster storage from list to dictionary
## --                                - added method _remove_cluster()
## -- 2023-12-10  0.6.1     DA       Bugfix in method ClusterAnalyzer.get_cluster_membership()
## -- 2023-12-20  0.7.0     DA       Renormalization
## -- 2024-02-23  0.8.0     DA       Class ClusterCentroid: implementation of methods _remove_plot*
## -- 2024-02-24  0.8.1     DA       Method ClusterAnalyzer._remove_cluster() explicitely removes
## --                                the plot of a cluster before removal of the cluster itself
## -- 2024-02-24  0.8.2     DA       Class ClusterCentroid: redefined method remove_plot()
## -- 2024-04-10  0.8.3     DA       Refactoring
## -- 2024-04-29  0.9.0     DA       Refactoring after changes on class Point
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.9.0 (2024-04-29)

This module provides templates for cluster analysis to be used in the context of online adaptivity.
"""

# from matplotlib.figure import Figure
# from matplotlib.text import Text
# from mpl_toolkits.mplot3d.art3d import Line3D, Text3D
# from mlpro.bf.mt import Figure, PlotSettings
# from mlpro.bf.math.properties import PropertyDefinitions
from mlpro.bf.various import *
from mlpro.bf.plot import *
from mlpro.bf.streams import *
from mlpro.bf.math.normalizers import Normalizer
from typing import List, Tuple

from mlpro.oa.streams.tasks.clusteranalyzers.clusters import Cluster
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterCentroid (Cluster):
    """
    Extended cluster class with a centroid.

    Parameters
    ----------
    p_id
        Optional external id
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_cls_centroid = Point
        Name of a point class. Default = Point
    **p_kwargs
        Further optional keyword arguments.

    Attributes
    ----------
    centroid : Centroid
        Centroid object.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_id=None, p_properties: List[Tuple[str, int, type]] = ..., p_visualize: bool = False):
        super().__init__( p_id=p_id, p_properties=p_properties, p_visualize=p_visualize )

        self.add_properties( p_property_definitions = [ cprop_centroid ], p_visualize = p_visualize )
        self.centroid.set_id( p_id = self.get_id() )
        self._centroid_elem : Element = None


# ## -------------------------------------------------------------------------------------------------
    def set_id(self, p_id=None):
        super().set_id( p_id = p_id )
        try:
            self.centroid.set_id( p_id = p_id )
        except:
            pass


## -------------------------------------------------------------------------------------------------
    def get_membership(self, p_inst: Instance) -> float:
        feature_data = p_inst.get_feature_data()

        if self._centroid_elem is None:
            self._centroid_elem = Element( p_set=feature_data.get_related_set() )

        self._centroid_elem.set_values( p_value=self.centroid.value )

        return feature_data.get_related_set().distance( p_e1 = feature_data, p_e2 = self._centroid_elem )


# ## -------------------------------------------------------------------------------------------------
#     def init_plot(self, p_figure: Figure = None, p_plot_settings: PlotSettings = None, **p_kwargs):
#         super().init_plot(p_figure, p_plot_settings, **p_kwargs)
#         self._centroid.init_plot( p_figure=p_figure, p_plot_settings=p_plot_settings)


# ## -------------------------------------------------------------------------------------------------
#     def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
#         self._plot_line1 = None
#         self._plot_line2 = None
#         self._plot_line1_t1 : Text = None
#         self._plot_line1_t2 : Text = None
#         self._plot_line1_t3 : Text = None
#         self._plot_line2_t1 : Text = None
#         self._plot_line2_t2 : Text = None
    

# ## -------------------------------------------------------------------------------------------------
#     def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
#         self._plot_line1 : Line3D = None
#         self._plot_line1_t1 : Text3D = None
#         self._plot_line1_t2 : Text3D = None

#         self._plot_line2 : Line3D = None
#         self._plot_line2_t1 : Text3D = None

#         self._plot_line3 : Line3D = None
#         self._plot_line3_t1 : Text3D = None
    

# ## -------------------------------------------------------------------------------------------------
#     def update_plot(self, **p_kwargs):
#         super().update_plot(**p_kwargs)
#         self._centroid.update_plot( p_kwargs=p_kwargs)   
    

# ## -------------------------------------------------------------------------------------------------
#     def _update_plot_2d(self, p_settings: PlotSettings, **p_kwargs):
#         super()._update_plot_2d(p_settings, **p_kwargs)

#         # 1 Get coordinates
#         centroid = self._centroid.get_position()
#         ax_xlim  = p_settings.axes.get_xlim()
#         ax_ylim  = p_settings.axes.get_ylim()
#         xlim     = [ min( ax_xlim[0], centroid[0] ), max(ax_xlim[1], centroid[0] ) ]
#         ylim     = [ min( ax_ylim[0], centroid[1] ), max(ax_ylim[1], centroid[1] ) ]

#         # 2 Plot a crosshair
#         if self._plot_line1 is None:
#             # 2.1 Add initial crosshair lines
#             cluster_id = self.get_id()
#             col_id = cluster_id % len(self.C_CLUSTER_COLORS)
#             color = self.C_CLUSTER_COLORS[col_id]
#             label = ' C' + str(cluster_id) + ' '
#             self._plot_line1 = p_settings.axes.plot( xlim, [centroid[1],centroid[1]], color=color, linestyle='dashed', lw=1, label=label)[0]
#             self._plot_line2 = p_settings.axes.plot( [centroid[0],centroid[0]], ylim, color=color, linestyle='dashed', lw=1)[0]
#             self._plot_line1_t1 = p_settings.axes.text(centroid[0], centroid[1], label, color=color )
#             self._plot_line1_t2 = p_settings.axes.text(xlim[0], centroid[1], label, ha='right', va='center', color=color )
#             self._plot_line1_t3 = p_settings.axes.text(xlim[1], centroid[1], label, ha='left',va='center', color=color )
#             self._plot_line2_t1 = p_settings.axes.text(centroid[0], ylim[0], label, ha='center', va='top', color=color )
#             self._plot_line2_t2 = p_settings.axes.text(centroid[0], ylim[1], label, ha='center', va='bottom',color=color )
#             p_settings.axes.legend(title='Clusters', alignment='left', loc='upper right', shadow=True, draggable=True)
#         else:
#             # 2.2 Update data of crosshair lines
#             self._plot_line1.set_data( xlim, [centroid[1],centroid[1]] )
#             self._plot_line2.set_data( [centroid[0],centroid[0]], ylim )
#             self._plot_line1_t1.set(position=(centroid[0], centroid[1]) )
#             self._plot_line1_t2.set(position=(xlim[0], centroid[1]))
#             self._plot_line1_t3.set(position=(xlim[1], centroid[1]))
#             self._plot_line2_t1.set(position=(centroid[0], ylim[0]))
#             self._plot_line2_t2.set(position=(centroid[0], ylim[1]))


# ## -------------------------------------------------------------------------------------------------
#     def _update_plot_3d(self, p_settings: PlotSettings, **p_kwargs):
#         super()._update_plot_3d(p_settings, **p_kwargs) 

#         # 1 Get coordinates
#         centroid = self._centroid.get_position()
#         ax_xlim  = p_settings.axes.get_xlim()
#         ax_ylim  = p_settings.axes.get_ylim()
#         ax_zlim  = p_settings.axes.get_zlim()
#         xlim     = [ min( ax_xlim[0], centroid[0] ), max(ax_xlim[1], centroid[0] ) ]
#         ylim     = [ min( ax_ylim[0], centroid[1] ), max(ax_ylim[1], centroid[1] ) ]
#         zlim     = [ min( ax_zlim[0], centroid[2] ), max(ax_zlim[1], centroid[2] ) ]


#         # 2 Determine label text alignments
#         ap = p_settings.axes.get_axis_position()

#         if ap[0]: 
#             t2_ha='left' 
#             t3_ha='right'
#         else: 
#             t2_ha='right'
#             t3_ha='left'

#         if ap[1]: 
#             t4_ha='right' 
#             t5_ha='left'
#         else: 
#             t4_ha='left'
#             t5_ha='right'

#         t6_va='top' 
#         t7_va='bottom'


#         # 3 Plot a crosshair with label texts
#         if self._plot_line1 is None:
#             # 3.1 Add initial crosshair lines
#             cluster_id = self.get_id()
#             col_id = cluster_id % len(self.C_CLUSTER_COLORS)
#             color = self.C_CLUSTER_COLORS[col_id]
#             label = ' C' + str(cluster_id) + ' '
#             self._plot_line1 = p_settings.axes.plot( xlim, [centroid[1],centroid[1]], [centroid[2],centroid[2]], color=color, linestyle='dashed', lw=1, label=label)[0]
#             self._plot_line2 = p_settings.axes.plot( [centroid[0],centroid[0]], ylim, [centroid[2],centroid[2]], color=color, linestyle='dashed', lw=1)[0]
#             self._plot_line3 = p_settings.axes.plot( [centroid[0],centroid[0]], [centroid[1],centroid[1]], zlim, color=color, linestyle='dashed', lw=1)[0]

#             self._plot_line1_t1 = p_settings.axes.text(centroid[0], centroid[1], centroid[2], label, color=color )
#             self._plot_line1_t2 = p_settings.axes.text(xlim[0], centroid[1], centroid[2], label, ha=t2_ha, va='center', color=color )
#             self._plot_line2_t1 = p_settings.axes.text(centroid[0], ylim[0], centroid[2], label, ha=t4_ha, va='center', color=color )
#             self._plot_line3_t1 = p_settings.axes.text(centroid[0], centroid[1], zlim[0], label, ha='center', va=t6_va, color=color )

#             p_settings.axes.legend(title='Clusters', alignment='left', loc='right', shadow=True, draggable=True)
#         else:
#             # 3.2 Update data of crosshair lines
#             self._plot_line1.set_data_3d( xlim, [centroid[1],centroid[1]], [centroid[2],centroid[2]] )
#             self._plot_line2.set_data_3d( [centroid[0],centroid[0]], ylim, [centroid[2],centroid[2]] )
#             self._plot_line3.set_data_3d( [centroid[0],centroid[0]], [centroid[1],centroid[1]], zlim )

#             self._plot_line1_t1.set(position_3d=(centroid[0], centroid[1], centroid[2]))
#             self._plot_line1_t2.set(position_3d=(xlim[0], centroid[1], centroid[2]), ha=t2_ha)
#             self._plot_line2_t1.set(position_3d=(centroid[0], ylim[0], centroid[2]), ha=t4_ha)
#             self._plot_line3_t1.set(position_3d=(centroid[0], centroid[1], zlim[0]), va=t6_va)


# ## -------------------------------------------------------------------------------------------------
#     def remove_plot(self, p_refresh: bool = True):
#         self._centroid.remove_plot(p_refresh=p_refresh)
#         return super().remove_plot(p_refresh=p_refresh)


# ## -------------------------------------------------------------------------------------------------
#     def _remove_plot_2d(self):
#         if self._plot_line1 is None: return

#         self._plot_line1.remove()
#         self._plot_line1 = None

#         self._plot_line1_t1.remove()
#         self._plot_line_t1 = None
        
#         self._plot_line1_t2.remove()
#         self._plot_line1_t2 = None

#         self._plot_line1_t3.remove()
#         self._plot_line1_t3 = None

#         self._plot_line2.remove()
#         self._plot_line2 = None

#         self._plot_line2_t1.remove()
#         self._plot_line2_t1 = None

#         self._plot_line2_t2.remove()
#         self._plot_line2_t2 = None


# ## -------------------------------------------------------------------------------------------------
#     def _remove_plot_3d(self):
#         if self._plot_line1 is None: return
        
#         self._plot_line1.remove()
#         self._plot_line1 = None

#         self._plot_line1_t1.remove()
#         self._plot_line1_t1 = None

#         self._plot_line1_t2.remove()
#         self._plot_line1_t2 = None

#         self._plot_line2.remove()
#         self._plot_line2 = None

#         self._plot_line2_t1.remove()
#         self._plot_line2_t1 = None

#         self._plot_line3.remove()
#         self._plot_line3 = None

#         self._plot_line3_t1.remove()
#         self._plot_line3_t1 = None
  

# ## -------------------------------------------------------------------------------------------------
#     def _remove_plot_nd(self):
#         pass


## -------------------------------------------------------------------------------------------------
    def renormalize(self, p_normalizer: Normalizer):
        self.centroid.renormalize( p_normalizer=p_normalizer)

