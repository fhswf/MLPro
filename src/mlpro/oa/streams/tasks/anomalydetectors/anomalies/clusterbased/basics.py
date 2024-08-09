## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies.clusterbased
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2023-11-21  1.0.1     SK       Time Stamp update
## -- 2024-02-25  1.1.0     SK       Visualisation update
## -- 2024-04-10  1.2.0     DA/SK    Refactoring
## -- 2024-05-28  1.3.0     SK       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.0 (2024-05-28)

This module provides a template class for cluster-based anomalies to be used in anomaly detection algorithms.
"""

from mlpro.oa.streams.basics import Instance
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.basics import Anomaly
from mlpro.bf.mt import Figure, PlotSettings
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.basics import Cluster
from matplotlib.figure import Figure





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CBAnomaly (Anomaly):
    """
    Event class to be raised when cluster-based anomalies are detected.
    
    Parameters
    ----------
    p_id : int
        Anomaly ID. Default value = 0.
    p_instances : Instance
        List of instances. Default value = None.
    p_clusters : dict[Cluster]
        Clusters associated with the anomaly. Default = None.
    p_properties : dict
        Poperties of clusters associated with the anomaly. Default = None.
    p_ano_scores : list
        List of anomaly scores of instances. Default = None.
    p_det_time : str
        Time of occurance of anomaly. Default = None.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_raising_object : object
        Reference of the object raised. Default = None.
    **p_kwargs
        Further optional keyword arguments.
    """
    
    C_NAME      = 'Cluster based Anomaly'
    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = False
    C_PLOT_VALID_VIEWS      = [ PlotSettings.C_VIEW_2D, 
                                PlotSettings.C_VIEW_3D, 
                                PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW     = PlotSettings.C_VIEW_ND

    C_ANOMALY_COLORS        = [ 'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan' ]

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_id : int = 0,
                 p_instances : list[Instance] = None,
                 p_clusters : dict[Cluster] = None,
                 p_properties : dict = None,
                 p_ano_scores : list = None,
                 p_det_time : str = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 **p_kwargs):
        
        super().__init__(p_id=p_id,
                         p_instances=p_instances,
                         p_ano_scores=p_ano_scores,
                         p_visualize=p_visualize, 
                         p_raising_object=p_raising_object,
                         p_det_time=p_det_time,
                         **p_kwargs)
        
        self._colour_id = 0
        self._clusters : dict[Cluster] = p_clusters
        self._properties : dict = p_properties

## -------------------------------------------------------------------------------------------------
    def get_clusters(self) -> dict[Cluster]:
        """
        Method that returns the clusters associated with the anomaly.
        
        Returns
        -------
        dict[Cluster]
            Dictionary of clusters.
        """
        return self._clusters
    

## -------------------------------------------------------------------------------------------------
    def get_properties(self) -> dict:
        """
        Method that returns the properties of clusters associated with the anomaly.
        
        Returns
        -------
        dict
            Dictionary of properties.
        """
        return self._properties
    

## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
        super()._init_plot_2d(p_figure=p_figure, p_settings=p_settings)

        cluster : Cluster = None

        for cluster in self._clusters.values(): 

            cluster.color = "red"


## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
        super()._init_plot_3d(p_figure=p_figure, p_settings=p_settings)
    
        cluster : Cluster = None

        for cluster in self._clusters.values(): 

            cluster.color = "red"



