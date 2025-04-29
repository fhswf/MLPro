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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.4.0 (2025-04-27)

This module provides template for cluster-based anomaly detection algorithms to be used in the context of online adaptivity.
"""

from mlpro.oa.streams.basics import *
from mlpro.oa.streams.tasks.anomalydetectors.basics import AnomalyDetector
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased import *
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
    def _run_algorithm(self, p_inst: InstDict) -> None:
        """
        Custom method for the main detection algorithm.

        Parameters
        ----------
        p_inst : InstDict
            The incoming instance to be processed.

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
        self._run_algorithm(p_inst=p_inst)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorCBSingle(AnomalyDetectorCB):
    """
    This is the class for detect anomalies related to a single cluster.

    """

    C_TYPE = 'Cluster based Anomaly Detector - Single Cluster'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_clusterer : ClusterAnalyzer,
                 p_name : str = None,
                 p_range_max = StreamTask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):
        
        self.cb_anomalies ={}

        super().__init__(p_clusterer = p_clusterer,
                         p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging= p_logging,
                         **p_kwargs)


## -------------------------------------------------------------------------------------------------

    def _buffer_anomaly(self, p_anomaly:AnomalyCB):
        """
        Method to be used to add a new anomaly. Please use as part of your algorithm.

        Parameters
        ----------
        p_anomaly : AnomalyCB
            Anomaly object to be added.
        """

        super()._buffer_anomaly(p_anomaly)
        self.cb_anomalies[p_anomaly.clusters.values()[0].id] = p_anomaly


## -------------------------------------------------------------------------------------------------
    def _remove_anomaly(self, p_anomaly:AnomalyCB):
        """
        Method to remove an existing anomaly. Please use as part of your algorithm.

        Parameters
        ----------
        p_anomaly : AnomalyCB
            Anomaly object to be removed.
        """

        super()._remove_anomaly(p_anomaly)
        del self.cb_anomalies[p_anomaly.clusters.values()[0].id]


## -------------------------------------------------------------------------------------------------
    def _triage_anomaly( self, 
                         p_anomaly: AnomalyCB,
                         **p_kwargs ) -> bool:
        """
        Custom method for extended anomaly triage.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _run( p_inst: InstDict ):
       
        # 1 Main detection algorithm
        super()._run( p_inst = p_inst )


        # 2 Clean-up loop ('triage')
        clusters    = self._clusterer.clusters
        triage_list = []

        # 2.1 Collect anomalies to be deleted
        for anomaly in self.cb_anomalies.values():

            # 2.1.1 Check whether the related cluster still exists
            related_cluster = next(iter(anomaly.clusters.keys()))
            try:
                clusters[related_cluster.id]
                remove_anomaly = self._triage_anomaly( p_anomaly = anomaly )
            except:
                remove_anomaly = True

            if remove_anomaly:
                # 2.1.2 Anomaly is prepared to be removed
                triage_list.append( anomaly )

        # 2.2 Remove all obsolete anomalies from the triage list
        for anomaly in triage_list:
            self._remove_anomaly( p_anomaly = p_anomaly )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorCBMulti(AnomalyDetectorCB):
    """
    This is the class for detect anomalies related to multiple clusters.

    """
    pass