## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.cb_detectors
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
## -- 2025-04-22  1.3.1     DA/DS    New methods - _run_algorithm & _run added
## -- 2025-04-24  1.3.2     DS       New classes - AnomalyDetectorCBSingle & AnomalyDetectorCBMulti added
## -- 2025-04-27  1.4.0     DA/DS    Design extensions/refactoring
## -- 2025-05-06  1.5.0     DA/DS    Design reduction
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.5.0 (2025-05-06)

This module provides template for cluster-based anomaly detection algorithms to be used in the context of online adaptivity.
"""

from mlpro.oa.streams.basics import *
from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.basics import AnomalyDetector
from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.anomalies.clusterbased import *
from mlpro.oa.streams.tasks.clusteranalyzers.basics import ClusterAnalyzer
from mlpro.bf.math.properties import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorCB (AnomalyDetector):
    """
    This is the base class for cluster-based online anomaly detectors. It raises an event when an
    anomaly is detected in a cluster dataset.

    """

    C_TYPE = 'Cluster based Anomaly Detector'

    # List of cluster properties necessary for the algorithm
    C_REQ_CLUSTER_PROPERTIES : PropertyDefinitions = []

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_clusterer : ClusterAnalyzer,
                  p_name : str = None,
                  p_range_max = StreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  **p_kwargs):

        super().__init__(p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging = p_logging,
                         **p_kwargs)
        
        self._clusterer = p_clusterer

        unknown_prop = self._clusterer.align_cluster_properties(p_properties=self.C_REQ_CLUSTER_PROPERTIES)

        if len(unknown_prop) >0:
            raise RuntimeError("The following cluster properties need to be provided by the clusterer: ", unknown_prop)


## -------------------------------------------------------------------------------------------------
    def _detect_cb_anomalies(self, p_inst: Instance) -> None:
        """
        Custom method for the main detection algorithm.

        Parameters
        ----------
        p_inst : Instance
            Instance that triggered the detection.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _triage_anomaly( self, 
                         p_anomaly: AnomalyCB,
                         **p_kwargs ) -> bool:
        """
        Custom method for extended anomaly triage.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst: InstDict) -> None:
        """
        This method is called by the stream task to process the incoming instance.

        Parameters
        ----------
        p_inst : InstDict
            The incoming instance to be processed.

        Returns
        -------
        None

        """

        # 1 Execution of the main detection algorithm        
        try:
            inst_type, inst = list(p_inst.values())[-1]
            if inst_type != InstTypeNew:
                inst = None
        except:
            inst = None

        self._detect_cb_anomalies( p_inst = inst )


        # 2 Clean-up loop ('triage')
        clusters    = self._clusterer.clusters
        triage_list = []

        # 2.1 Collect anomalies to be deleted
        for anomaly in self.cb_anomalies.values():

            # 2.1.1 Check whether the related clusters still exist
            remove_anomaly = True
            for related_cluster in anomaly.clusters.values():
                try:
                    clusters[related_cluster.id]
                    remove_anomaly = False
                    break
                except:
                    pass

            if not remove_anomaly:
                remove_anomaly = self._triage_anomaly( p_anomaly = anomaly )

            if remove_anomaly:
                # 2.1.2 Anomaly is prepared to be removed
                triage_list.append( anomaly )

        # 2.2 Remove all obsolete anomalies from the triage list
        for anomaly in triage_list:
            self._remove_anomaly( p_anomaly = anomaly )
