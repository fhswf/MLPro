## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.changedetectors.clusterbased
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-06-03  0.1.0     DA/DS    Creation
## -- 2025-06-09  1.0.0     DA       Design updates
## -- 2025-06-10  1.1.0     DA       Review/rework of ChangeDetectorCB._run()
## -- 2025-06-11  1.1.1     DA       Workaround in ChangeDetectorCB.__init__(): parent/super()
## -- 2025-06-13  1.2.0     DA       Class Change: param p_id is now initialized to -1
## -- 2025-07-15  1.2.1     DA       Class ChangeCB: bugfix in 
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2025-06-13)

This module provides templates for cluster-based change detection to be used in the context of 
online-adaptive data stream processing.
"""

try:
    from matplotlib.figure import Figure
except:
    class Figure : pass

from mlpro.bf.various import Log, TStampType
from mlpro.bf.plot import PlotSettings
from mlpro.bf.math.properties import PropertyDefinitions
from mlpro.bf.streams import InstDict, InstTypeNew, Instance

from mlpro.oa.streams.tasks.clusteranalyzers.basics import Cluster, ClusterAnalyzer
from mlpro.oa.streams.tasks.changedetectors.basics import Change, ChangeDetector



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ChangeCB (Change):
    """
    Base class for cluster-based change events raised by cluster-based change detectors.

    Parameters
    ----------
    p_id : int = -1
        Change ID. Default value = -1, indicating that the ID is not set. In that case, the id is
        automatically generated when raising the change.
    p_status : bool = True
        Status of the change.
    p_tstamp : TStampType = None
        Time of occurance of change. Default = None.
    p_visualize : bool = False
        Boolean switch for visualisation. Default = False.
    p_raising_object : object = None
        Reference of the object raised. Default = None.
    p_clusters : dict[Cluster] = {}
        Clusters associated with the anomaly.
    p_properties : PropertyDefinitions = []
        List of properties of clusters associated with the anomaly.
    **p_kwargs
        Further optional keyword arguments.
    """

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_id : int = -1,
                  p_status : bool = True,
                  p_tstamp : TStampType = None,
                  p_visualize : bool = False,
                  p_raising_object : object = None,
                  p_clusters : dict[Cluster] = {},
                  p_properties : PropertyDefinitions = [],
                  **p_kwargs ):
        
        super().__init__( p_id = p_id,
                          p_status = p_status,
                          p_tstamp = p_tstamp,
                          p_visualize = p_visualize, 
                          p_raising_object = p_raising_object,
                          **p_kwargs )

        self.clusters : dict[Cluster] = p_clusters
        self.properties : dict       = p_properties


## -------------------------------------------------------------------------------------------------
    def add_clusters( self, p_clusters : dict ):
        """
        Adds clusters. Existing entries are replaced by new ones.

        Parameters
        ----------
        p_clusters : dict
            Dictionary with clusters.
        """
        
        self.clusters.update(p_clusters)

## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings, **p_kwargs):
        super()._update_plot_2d(p_settings, **p_kwargs)

        cluster : Cluster = None

        for cluster in self.clusters.values(): 
            if self.status:
                try:
                    if cluster.color_bak is None:
                        cluster.color_bak = cluster.color
                except:
                    cluster.color_bak = cluster.color

                cluster.color = "red"
            else:
                try:
                    cluster.color     = cluster.color_bak
                    cluster.color_bak = None
                except:
                    pass

        return True


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings, **p_kwargs):
        super()._update_plot_3d(p_settings, **p_kwargs)

        cluster : Cluster = None

        for cluster in self.clusters.values(): 
            color = 'red' if self.status else None
            if cluster.color != color: cluster.color = color
            
        return True
    




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ChangeDetectorCB (ChangeDetector):
    """
    Base class for cluster-based change detectors.

    Parameters
    ----------
    p_clusterer : ClusterAnalyzer
        Associated cluster analyzer.
    p_name : str = None
        Optional name of the task. Default is None.
    p_range_max : int
        Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_PROCESS.
    p_ada : bool
        Boolean switch for adaptivitiy. Default = True.
    p_duplicate_data : bool
        If True, instances will be duplicated before processing. Default = False.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_change_buffer_size : int = 100
        Size of the internal change buffer self.changes. Default = 100.
    p_thrs_inst : int = 0
        The algorithm is only executed after this number of instances.
    p_thrs_clusters : int = 1
        The algorithm is only executed with this minimum number of clusters.
    p_kwargs : dict
        Further optional named parameters.
    """

    C_TYPE = 'Cluster-based Change Detector'

    C_REQ_CLUSTER_PROPERTIES : PropertyDefinitions = []

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_clusterer : ClusterAnalyzer,
                  p_name:str = None,
                  p_range_max = ChangeDetector.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL,
                  p_change_buffer_size : int = 100,
                  p_thrs_inst : int = 0,
                  p_thrs_clusters : int = 1,
                  **p_kwargs ):

        ChangeDetector.__init__( self, 
                                 p_name = p_name,
                                 p_range_max = p_range_max,
                                 p_ada = p_ada,
                                 p_duplicate_data = p_duplicate_data,
                                 p_visualize = p_visualize,
                                 p_logging = p_logging,
                                 p_change_buffer_size = p_change_buffer_size,
                                 p_thrs_inst = p_thrs_inst,
                                 **p_kwargs )
        
        self._clusterer           = p_clusterer
        self._thrs_clusters : int = p_thrs_clusters
        self.cb_changes           ={}

        unknown_prop = self._clusterer.align_cluster_properties(p_properties=self.C_REQ_CLUSTER_PROPERTIES)

        if len(unknown_prop) > 0:
            raise RuntimeError("The following cluster properties need to be provided by the clusterer: ", unknown_prop)
        

## -------------------------------------------------------------------------------------------------
    def _detect(self, p_clusters : dict, p_instance: Instance, **p_kwargs):
        """
        Custom method for the main detection algorithm. Use the _raise_change_event() method to raise
        a change detected by your algorithm.

        Parameters
        ----------
        p_instance : Instance
            Instance that triggered the detection.
        **p_kwargs
            Optional keyword arguments (originally provided to the constructor).
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _run(self, p_instances: InstDict):
        """
        This method is called by the stream task to process the incoming instance.

        Parameters
        ----------
        p_instances : InstDict
            The incoming instance to be processed.
        """

        # 0 Check whether the minimum number of instances has been reached
        if self._chk_num_inst:
            self._num_inst += len( p_instances )
            if self._num_inst < self._thrs_inst: return
            self._chk_num_inst = False


        # 1 Check for the minimum number of clusters
        current_clusters = self._clusterer.clusters
        if len(current_clusters) < self._thrs_clusters: return


        # 2 Execution of the main detection algorithm once for the latest new instance
        inst_new = None
        for inst_type, inst in reversed(list(p_instances.values())):
            if inst_type == InstTypeNew: 
                inst_new = inst
                break

        if inst_new is not None:
            self._detect( p_clusters = current_clusters, 
                          p_instance = inst_new, 
                          **self.kwargs )


        # 3 Clean-up loop ('triage')
        triage_list = []

        # 3.1 Collect changes to be deleted
        for change in self.changes.values():

            # 3.1.1 Extended triage check
            if change.clusters.keys() <= current_clusters.keys():
                # Case 1: all clusters of the change still exist. Here the custom triage method makes the decision
                triage = self._triage( p_change = change, **self.kwargs )
            else:
                # Case 2: at least one cluster of the change disappeared -> triage
                triage = True

            if triage: triage_list.append( change )

        # 3.2 Raise and remove all obsolete changes
        for change in triage_list:
            if change.status:
                change.status = False
                change.tstamp = None
                self._raise_change_event( p_change = change, 
                                          p_instance = inst_new, 
                                          p_buffer = False )

            self._remove_change( p_change = change )


## -------------------------------------------------------------------------------------------------
    def _buffer_change(self, p_change:Change):
        """
        Method to be used to add a new anomaly. Please use as part of your algorithm.

        Parameters
        ----------
        p_anomaly : AnomalyCB
            Anomaly object to be added.
        """

        super()._buffer_change(p_change= p_change)

        for cluster in p_change.clusters.values():
            self.cb_changes[cluster.id] = p_change


## -------------------------------------------------------------------------------------------------
    def _remove_change(self, p_change:Change):
        """
        Method to remove an existing anomaly. Please use as part of your algorithm.

        Parameters
        ----------
        p_anomaly : AnomalyCB
            Anomaly object to be removed.
        """

        super()._remove_change(p_change = p_change)

        for cluster in p_change.clusters.values():
            del self.cb_changes[cluster.id]
