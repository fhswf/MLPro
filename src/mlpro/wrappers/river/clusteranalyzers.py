## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.wrappers.river
## -- Module  : clusteranalyzers.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-05-12  0.0.0     DA       Creation
## -- 2023-05-23  1.0.0     SY       First version release
## -- 2023-05-25  1.0.1     SY       Refactoring related to ClusterCentroid
## -- 2023-06-03  1.0.2     DA       Renaming of method ClusterAnalyzer.get_cluster_memberships
## -- 2023-06-05  1.0.3     SY       Updating get_cluster_memberships, p_cls_cluster, and _adapt
## -- 2023-11-20  1.0.4     SY       Update due to intorduction of visualization for ClusterCentroid
## -- 2023-12-08  1.0.5     SY       Update due to new data type in ClusterCentroid
## -- 2023-12-15  1.0.6     SY/DA    Adjustments on classes WrClusterAnalyzerRiver2MLPro and 
## --                                WrRiverClustStream2MLPro
## -- 2023-12-17  1.0.7     SY       Updates on classes WrRiverKMeans2MLPro, WrRiverDBStream2MLPro
## --                                WrRiverStreamKMeans2MLPro, WrRiverDenStream2MLPro
## -- 2023-12-21  1.0.8     SY       Add renormalization method on all wrapped algorithms
## -- 2023-12-22  1.0.9     DA/SY    Bugfix 
## -- 2023-12-25  1.0.10    DA       Bugfix in WrClusterAnalyzerRiver2MLPro._adapt()
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.10 (2023-12-25)

This module provides wrapper classes from River to MLPro, specifically for cluster analyzers. This
module includes three clustering algorithms from River that are embedded to MLPro, such as:

1) DBSTREAM (https://riverml.xyz/latest/api/cluster/DBSTREAM/)

2) CluStream (https://riverml.xyz/latest/api/cluster/CluStream/)

3) DenStream (https://riverml.xyz/latest/api/cluster/DenStream/)

4) KMeans (https://riverml.xyz/latest/api/cluster/KMeans/)

5) STREAMKMeans (https://riverml.xyz/latest/api/cluster/STREAMKMeans/)

Learn more:
https://www.riverml.xyz/

"""


from mlpro.bf.streams import Instance, List
from mlpro.wrappers.river.basics import WrapperRiver
from mlpro.bf.math.normalizers import Normalizer
from mlpro.oa.streams.tasks.clusteranalyzers import ClusterAnalyzer, Cluster, ClusterCentroid
from mlpro.bf.mt import Task as MLTask
from mlpro.bf.various import Log
from mlpro.bf.streams import *
from river import base, cluster
from typing import List, Tuple
import numpy as np





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrClusterAnalyzerRiver2MLPro (WrapperRiver, ClusterAnalyzer):
    """
    This is the base wrapper class for each River-based cluster analyzer to MLPro.

    Parameters
    ----------
    p_cls_cluster 
        Cluster class (Class Cluster or a child class).
    p_river_algo : river.base.Clusterer
        Instantiated river-based clusterer.
    p_name : str
        Name of the clusterer. Default: None.
    p_range_max :
        MLPro machine learning task, either process or thread. Default: MLTask.C_RANGE_THREAD.
    p_ada : bool
        Turn on adaptivity. Default: True.
    p_visualize : bool
        Turn on visualization. Default: False.
    p_logging :
        Set up type of logging. Default: Log.C_LOG_ALL.
    p_kwargs : dict
        Further optional named parameters. 
        
    """

    C_TYPE              = 'River Cluster Analyzer'
    C_NAME              = '????'
    
    C_WRAPPED_PACKAGE   = 'river'
    C_MINIMUM_VERSION   = '0.15.0'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_cls_cluster,
                 p_river_algo:base.Clusterer,
                 p_name:str = None,
                 p_range_max = MLTask.C_RANGE_THREAD,
                 p_ada:bool = True,
                 p_visualize:bool = True,
                 p_logging = Log.C_LOG_ALL,
                 **p_kwargs):
        
        self._river_algo = p_river_algo

        WrapperRiver.__init__(self, p_logging=p_logging)

        ClusterAnalyzer.__init__(self,
                                 p_cls_cluster=p_cls_cluster,
                                 p_name=p_name,
                                 p_range_max=p_range_max,
                                 p_ada=p_ada,
                                 p_visualize=p_visualize,
                                 **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new:List[Instance]) -> bool:
        """
        This method is to adapt the current clusters according to the incoming instances.

        Parameters
        ----------
        p_inst_new : List[Instance]
            incoming instances.

        Returns
        -------
        adapted : bool
            True, if something has been adapted. False otherwise.
            
        """
        
        # extract features data from instances
        first_instance = True
        for inst in p_inst_new:
            if first_instance:
                feature_data    = np.array(inst.get_feature_data().get_values())
                first_instance  = False
            else:
                feature_data    = np.append(feature_data, inst.get_feature_data().get_values())

        # transform np array to dict with enumeration
        input_data = dict(enumerate(feature_data.flatten(), 1))

        # update the model with a set of features
        self.log(self.C_LOG_TYPE_I, 'Cluster is adapted...')
        self._river_algo.learn_one(input_data)
        
        # update MLPro clusters from river
        self.get_clusters()
        self._update_clusters()

        return True
    

## -------------------------------------------------------------------------------------------------
    def _update_clusters(self):
        """
        This method is to update the centroids of each introduced cluster.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_clusters(self) -> dict[Cluster]:
        """
        This method returns the current list of clusters. To be defined according to each clusterer
        mechanism.

        Returns
        -------
        list_of_clusters : List[Cluster]
            Current list of clusters.
            
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_algorithm(self) -> base.Clusterer:
        """
        This method returns the river algorithm of the clusterer.

        Returns
        -------
        base.Clusterer
            The river algorithm of the clusterer.
            
        """

        return self._river_algo


## -------------------------------------------------------------------------------------------------
    def get_cluster_memberships( self, 
                                 p_inst: Instance, 
                                 p_scope: int = ClusterAnalyzer.C_MS_SCOPE_MAX ) -> List[Tuple[str, float, Cluster]]:
        """
        Public custom method to determine the membership of the given instance to each cluster as
        a value in percent.

        Parameters
        ----------
        p_inst : Instance
            Instance to be evaluated.
        p_scope : int
            Scope of the result list. See class attributes C_MS_SCOPE_* for possible values. Default
            value is C_MS_SCOPE_MAX.

        Returns
        -------
        membership : List[Tuple[str, float, Cluster]]
            List of membership tuples for each cluster. A tuple consists of a cluster id, a
            relative membership value in percent and a reference to the cluster.
            
        """
        
        # extract features data from instances
        first_instance = True
        for inst in p_inst:
            if first_instance:
                feature_data    = inst.get_feature_data().get_values()
                first_instance  = False
            else:
                feature_data    = np.append(feature_data, inst.get_feature_data().get_values())

        # transform np array to dict with enumeration
        input_data = dict(enumerate(feature_data.flatten(), 1))

        # predict the cluster number according to a set of features
        cluster_idx = self._river_algo.predict_one(input_data)

        # get the corresponding cluster
        list_clusters = self.get_clusters().values()

        # return the cluster membership
        memberships_rel = []
        if list_clusters is not None:
            for x in range(len(list_clusters)):
                cluster = list_clusters[cluster_idx]
                if x == cluster_idx:
                    memberships_rel.append((cluster.get_id(), 1, cluster))
                    self.log(self.C_LOG_TYPE_I,
                             'Actual instances belongs to cluster %s'%(cluster.get_id()))
                else:
                    if p_scope == ClusterAnalyzer.C_MS_SCOPE_ALL:
                        memberships_rel.append((cluster.get_id(), 0, cluster))

        return memberships_rel





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrRiverDBStream2MLPro (WrClusterAnalyzerRiver2MLPro):
    """
    This is the wrapper class for DBSTREAM clusterer.
    
    According to https://riverml.xyz/latest/api/cluster/DBSTREAM/ :
    DBSTREAM is a clustering algorithm for evolving data streams.
    It is the first micro-cluster-based online clustering component that explicitely captures the
    density between micro-clusters via a shared density graph. The density information in the graph
    is then exploited for reclustering based on actual density between adjacent micro clusters.
    
    The algorithm is divided into two parts:
        1) Online micro-cluster maintenance (learning)
        2) Offline generation of macro clusters (clustering)

    Parameters
    ----------
    p_name : str
        Name of the clusterer. Default: None.
    p_range_max :
        MLPro machine learning task, either process or thread. Default: MLTask.C_RANGE_THREAD.
    p_ada : bool
        Turn on adaptivity. Default: True.
    p_visualize : bool
        Turn on visualization. Default: False.
    p_logging :
        Set up type of logging. Default: Log.C_LOG_ALL.
    p_clustering_threshold : float
        DBStream represents each micro cluster by a leader (a data point defining the micro cluster's
        center) and the density in an area of a user-specified radius (clustering_threshold) around
        the center. Default: 1.0.
    p_fading_factor : float
        Parameter that controls the importance of historical data to current cluster.
        Note that fading_factor has to be different from 0. Default: 0.01.
    p_cleanup_interval : float
        The time interval between two consecutive time points when the cleanup process is conducted.
        Default: 2.
    p_intersection_factor : float
        The intersection factor related to the area of the overlap of the micro clusters relative
        to the area cover by micro clusters. This parameter is used to determine whether a micro
        cluster or a shared density is weak. Default: 0.3.
    p_minimum_weight : float
        The minimum weight for a cluster to be not "noisy". Default: 1.0.
    p_kwargs : dict
        Further optional named parameters. 
        
    """

    C_NAME          = 'DBSTREAM'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str = None,
                 p_range_max = MLTask.C_RANGE_THREAD,
                 p_ada:bool = True,
                 p_visualize:bool = False,
                 p_logging = Log.C_LOG_ALL,
                 p_clustering_threshold:float = 1.0,
                 p_fading_factor:float = 0.01,
                 p_cleanup_interval:float = 2,
                 p_intersection_factor:float = 0.3,
                 p_minimum_weight:float = 1.0,
                 **p_kwargs):
        
        alg = cluster.DBSTREAM(clustering_threshold=p_clustering_threshold,
                               fading_factor=p_fading_factor,
                               cleanup_interval=p_cleanup_interval,
                               intersection_factor=p_intersection_factor,
                               minimum_weight=p_minimum_weight)

        super().__init__(p_cls_cluster=ClusterCentroid(),
                         p_river_algo=alg,
                         p_name=p_name,
                         p_range_max=p_range_max,
                         p_ada=p_ada,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _update_clusters(self):
        """
        This method is to update the centroids of each introduced cluster.
        """
        
        for _, (key, val) in enumerate(self._river_algo.micro_clusters.items()):
            related_cluster = self._clusters[id(val)]
            related_cluster.get_centroid().set_values(list(self._river_algo.centers[key].values()))


## -------------------------------------------------------------------------------------------------
    def get_clusters(self) -> dict[Cluster]:
        """
        This method returns the current list of clusters.

        Returns
        -------
        dict_of_clusters : dict[Cluster]
            Current list of clusters.
            
        """

        list_keys_river = []
        for _, (key, val) in enumerate(self._river_algo.micro_clusters.items()):
            list_keys_river.append(id(val))
            try:
                related_cluster = self._clusters[id(val)]
            except:
                related_cluster = ClusterCentroid(
                    p_id = id(val),
                    p_visualize=self.get_visualization(),
                    p_cluster=self._river_algo.clusters[key],
                    p_micro_cluster=val
                    )
                
                if self.get_visualization():
                    related_cluster.init_plot(p_figure = self._figure, p_plot_settings=self._plot_settings)             

                related_cluster.get_centroid().set_values(list(self._river_algo.centers[key].values()))
                self._add_cluster( p_cluster = related_cluster )
        
        list_keys_mlpro = list(self._clusters.keys())
        for x in list_keys_mlpro:
            if x not in list_keys_river:
                self._remove_cluster(self._clusters[x])

        return self._clusters


## -------------------------------------------------------------------------------------------------
    def _renormalize(self, p_normalizer:Normalizer):
        """
        Internal renormalization of all clusters. See method OATask.renormalize_on_event() for further
        information.

        Parameters
        ----------
        p_normalizer : Normalizer
            Normalizer object to be applied on task-specific 
        """
        
        super()._renormalize(p_normalizer)

        for cluster in self._clusters.values():
            for val in self._river_algo.micro_clusters.values():
                if id(val) == cluster.get_id():
                    for mlpro_idx, river_idx in enumerate(val.center):
                        val.center[river_idx] = cluster._centroid.get_values()[mlpro_idx]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrRiverCluStream2MLPro (WrClusterAnalyzerRiver2MLPro):
    """
    This is the wrapper class for CluStream clusterer.
    
    According to https://riverml.xyz/latest/api/cluster/CluStream/ :
    The CluStream algorithm maintains statistical information about the data using micro-clusters.
    These micro-clusters are temporal extensions of cluster feature vectors. The micro-clusters are
    stored at snapshots in time following a pyramidal pattern. This pattern allows to recall summary
    statistics from different time horizons.

    Parameters
    ----------
    p_name : str
        Name of the clusterer. Default: None.
    p_range_max :
        MLPro machine learning task, either process or thread. Default: MLTask.C_RANGE_THREAD.
    p_ada : bool
        Turn on adaptivity. Default: True.
    p_visualize : bool
        Turn on visualization. Default: False.
    p_logging :
        Set up type of logging. Default: Log.C_LOG_ALL.
    p_n_macro_clusters : int
        The number of clusters (k) for the k-means algorithm. Default: 5.
    p_max_micro_clusters : int
        The maximum number of micro-clusters to use. Default: 100.
    p_micro_cluster_r_factor : int
        Multiplier for the micro-cluster radius. When deciding to add a new data point to a
        micro-cluster, the maximum boundary is defined as a factor of the micro_cluster_r_factor
        of the RMS deviation of the data points in the micro-cluster from the centroid. Default: 2.
    p_time_window : int
        If the current time is T and the time window is h, we only consider the data that arrived
        within the period (T-h,T). Default: 1000.
    p_time_gap : int
        An incremental k-means is applied on the current set of micro-clusters after each time_gap
        to form the final macro-cluster solution. Default: 100.
    p_seed : int
        Random seed used for generating initial centroid positions. Default: None.
    p_halflife : float
        Amount by which to move the cluster centers, a reasonable value if between 0 and 1.
        Default: 0.5.
    p_mu : float
        Mean of the normal distribution used to instantiate cluster positions. Default: 1.
    p_sigma : float
        Standard deviation of the normal distribution used to instantiate cluster positions.
        Default: 1.
    p_p : int
        Power parameter for the Minkowski metric. When p=1, this corresponds to the Manhattan
        distance, while p=2 corresponds to the Euclidean distance. Default: 2.
    p_kwargs : dict
        Further optional named parameters. 
        
    """

    C_NAME          = 'CluStream'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str = None,
                 p_range_max = MLTask.C_RANGE_THREAD,
                 p_ada:bool = True,
                 p_visualize:bool = False,
                 p_logging = Log.C_LOG_ALL,
                 p_n_macro_clusters:int = 5,
                 p_max_micro_clusters:int = 100,
                 p_micro_cluster_r_factor:int = 2,
                 p_time_window:int = 1000,
                 p_time_gap:int = 100,
                 p_seed:int = None,
                 p_halflife:float = 0.5,
                 p_mu:float = 1,
                 p_sigma:float = 1,
                 p_p:int = 2,
                 **p_kwargs):
        
        alg = cluster.CluStream(n_macro_clusters=p_n_macro_clusters,
                                max_micro_clusters=p_max_micro_clusters,
                                micro_cluster_r_factor=p_micro_cluster_r_factor,
                                time_window=p_time_window,
                                time_gap=p_time_gap,
                                seed=p_seed,
                                halflife=p_halflife,
                                mu=p_mu,
                                sigma=p_sigma,
                                p=p_p)

        super().__init__(p_cls_cluster=ClusterCentroid(),
                         p_river_algo=alg,
                         p_name=p_name,
                         p_range_max=p_range_max,
                         p_ada=p_ada,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _update_clusters(self):
        """
        This method is to update the centroids of each introduced cluster.
        """
        
        for x in self._river_algo.centers.keys():
            related_cluster = self._clusters[x]
            related_cluster.get_centroid().set_values(list(self._river_algo.centers[x].values()))


## -------------------------------------------------------------------------------------------------
    def get_clusters(self) -> dict[Cluster]:
        """
        This method returns the current list of clusters.

        Returns
        -------
        dict_of_clusters : dict[Cluster]
            Current list of clusters.
            
        """
        
        for x in self._river_algo.centers.keys():
            try:
                related_cluster = self._clusters[x]
            except:
                related_cluster = ClusterCentroid(p_id=x, p_visualize=self.get_visualization())   

                if self.get_visualization():
                    related_cluster.init_plot(p_figure = self._figure, p_plot_settings=self._plot_settings)             

                related_cluster.get_centroid().set_values(list(self._river_algo.centers[x].values()))
                self._add_cluster( p_cluster = related_cluster )

        return self._clusters


## -------------------------------------------------------------------------------------------------
    def _renormalize(self, p_normalizer:Normalizer):
        """
        Internal renormalization of all clusters. See method OATask.renormalize_on_event() for further
        information.

        Parameters
        ----------
        p_normalizer : Normalizer
            Normalizer object to be applied on task-specific 
        """
        
        super()._renormalize(p_normalizer)

        for cluster in self._clusters.values():
            related_cluster = self._river_algo.centers[cluster.get_id()]
            for mlpro_idx, river_idx in enumerate(related_cluster):
                related_cluster[river_idx] = cluster._centroid.get_values()[mlpro_idx]





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrRiverDenStream2MLPro (WrClusterAnalyzerRiver2MLPro):
    """
    This is the wrapper class for DenStream clusterer.
    
    According to https://riverml.xyz/latest/api/cluster/DenStream/ :
    DenStream is a clustering algorithm for evolving data streams. DenStream can discover clusters
    with arbitrary shape and is robust against noise (outliers).

    "Dense" micro-clusters (named core-micro-clusters) summarise the clusters of arbitrary shape.
    A pruning strategy based on the concepts of potential and outlier micro-clusters guarantees the
    precision of the weights of the micro-clusters with limited memory.
    
    The algorithm is divided into two parts:
        1) Online micro-cluster maintenance (learning)
        2) Offline generation of macro clusters (clustering)

    Parameters
    ----------
    p_name : str
        Name of the clusterer. Default: None.
    p_range_max :
        MLPro machine learning task, either process or thread. Default: MLTask.C_RANGE_THREAD.
    p_ada : bool
        Turn on adaptivity. Default: True.
    p_visualize : bool
        Turn on visualization. Default: False.
    p_logging :
        Set up type of logging. Default: Log.C_LOG_ALL.
    p_decaying_factor : float
        Parameter that controls the importance of historical data to current cluster. Note that
        decaying_factor has to be different from 0. Default: 0.25.
    p_beta : float
        Parameter to determine the threshold of outlier relative to core micro-clusters. The value
        of beta must be within the range (0,1). Default: 0.75.
    p_mu : float
        Parameter to determine the threshold of outliers relative to core micro-cluster.
        As beta * mu must be greater than 1, mu must be within the range (1/beta, inf). Default: 2.
    p_epsilon : float
        Defines the epsilon neighborhood. Default: 0.02.
    p_n_samples_init : int
        Number of points to to initiqalize the online process. Default: 1000.
    p_stream_speed : int
        Number of points arrived in unit time. Default: 100.
    p_kwargs : dict
        Further optional named parameters. 
        
    """

    C_NAME          = 'DenStream'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str = None,
                 p_range_max = MLTask.C_RANGE_THREAD,
                 p_ada:bool = True,
                 p_visualize:bool = False,
                 p_logging = Log.C_LOG_ALL,
                 p_decaying_factor:float = 0.25,
                 p_beta:float = 0.75,
                 p_mu:float = 2,
                 p_epsilon:float = 0.02,
                 p_n_samples_init:int = 1000,
                 p_stream_speed:int = 100,
                 **p_kwargs):
        
        alg = cluster.DenStream(decaying_factor=p_decaying_factor,
                                beta=p_beta,
                                mu=p_mu,
                                epsilon=p_epsilon,
                                n_samples_init=p_n_samples_init,
                                stream_speed=p_stream_speed)
        
        self.n_dummy_prediction = 0

        super().__init__(p_cls_cluster=ClusterCentroid(),
                         p_river_algo=alg,
                         p_name=p_name,
                         p_range_max=p_range_max,
                         p_ada=p_ada,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _update_clusters(self):
        """
        This method is to update the centroids of each introduced cluster.
        """
        
        pass


## -------------------------------------------------------------------------------------------------
    def get_clusters(self) -> dict[Cluster]:
        """
        This method returns the current list of clusters.

        Returns
        -------
        dict_of_clusters : dict[ClusterCentroid]
            Current list of clusters.
            
        """

        # dummy prediction
        if self.n_dummy_prediction == self._river_algo.n_samples_init+1:
            self._river_algo.predict_one({1: 0, 2: 0})
        else:
            self.n_dummy_prediction += 1

        list_keys_river = []
        for _, (key, val) in enumerate(self._river_algo.p_micro_clusters.items()):
            list_keys_river.append(id(val))
            try:
                related_cluster = self._clusters[id(val)]
            except:
                try:
                    o_micro_cluster = self._river_algo.o_micro_clusters[key]
                except:
                    o_micro_cluster = None
                related_cluster = ClusterCentroid(
                    p_id = id(val),
                    p_visualize=self.get_visualization(),
                    p_cluster=self._river_algo.clusters[key],
                    p_micro_cluster=val,
                    p_o_micro_cluster=o_micro_cluster
                    )
                
                if self.get_visualization():
                    related_cluster.init_plot(p_figure = self._figure, p_plot_settings=self._plot_settings)

                self._add_cluster( p_cluster = related_cluster )
        
        list_keys_mlpro = list(self._clusters.keys())
        for x in list_keys_mlpro:
            if x not in list_keys_river:
                self._remove_cluster(self._clusters[x])

        return self._clusters


## -------------------------------------------------------------------------------------------------
    def _renormalize(self, p_normalizer:Normalizer):
        """
        Internal renormalization of all clusters. See method OATask.renormalize_on_event() for further
        information.

        Parameters
        ----------
        p_normalizer : Normalizer
            Normalizer object to be applied on task-specific 
        """
        
        super()._renormalize(p_normalizer)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrRiverKMeans2MLPro (WrClusterAnalyzerRiver2MLPro):
    """
    This is the wrapper class for KMeans clusterer.
    
    According to https://riverml.xyz/latest/api/cluster/KMeans/ :
    Incremental k-means. The most common way to implement batch k-means is to use Lloyd's algorithm,
    which consists in assigning all the data points to a set of cluster centers and then moving the
    centers accordingly.

    Parameters
    ----------
    p_name : str
        Name of the clusterer. Default: None.
    p_range_max :
        MLPro machine learning task, either process or thread. Default: MLTask.C_RANGE_THREAD.
    p_ada : bool
        Turn on adaptivity. Default: True.
    p_visualize : bool
        Turn on visualization. Default: False.
    p_logging :
        Set up type of logging. Default: Log.C_LOG_ALL.
    p_n_clusters : int
        Maximum number of clusters to assign. Default: 5.
    p_seed : int
        Random seed used for generating initial centroid positions. Default: None.
    p_halflife : float
        Amount by which to move the cluster centers, a reasonable value if between 0 and 1.
        Default: 0.5.
    p_mu : float
        Mean of the normal distribution used to instantiate cluster positions. Default: 1.
    p_sigma : float
        Standard deviation of the normal distribution used to instantiate cluster positions.
        Default: 1.
    p_p : int
        Power parameter for the Minkowski metric. When p=1, this corresponds to the Manhattan
        distance, while p=2 corresponds to the Euclidean distance. Default: 2.
    p_kwargs : dict
        Further optional named parameters. 
        
    """

    C_NAME          = 'KMeans'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str = None,
                 p_range_max = MLTask.C_RANGE_THREAD,
                 p_ada:bool = True,
                 p_visualize:bool = False,
                 p_logging = Log.C_LOG_ALL,
                 p_n_clusters:int = 5,
                 p_halflife:float = 0.5,
                 p_mu:float = 0,
                 p_sigma:float = 1,
                 p_p:int = 2,
                 p_seed:int = None,
                 **p_kwargs):
        
        alg = cluster.KMeans(n_clusters=p_n_clusters,
                             halflife=p_halflife,
                             mu=p_mu,
                             sigma=p_sigma,
                             p=p_p,
                             seed=p_seed)

        super().__init__(p_cls_cluster=ClusterCentroid(),
                         p_river_algo=alg,
                         p_name=p_name,
                         p_range_max=p_range_max,
                         p_ada=p_ada,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _update_clusters(self):
        """
        This method is to update the centroids of each introduced cluster.
        """
        for x in self._river_algo.centers.keys():
            related_cluster = self._clusters[x]
            list_center = []
            for y in range(len(self._river_algo.centers[x])):
                list_center.append(self._river_algo.centers[x][y+1])
            related_cluster.get_centroid().set_values(list_center)


## -------------------------------------------------------------------------------------------------
    def get_clusters(self) -> dict[Cluster]:
        """
        This method returns the current list of clusters.

        Returns
        -------
        dict_of_clusters : dict[Cluster]
            Current list of clusters.
            
        """

        for x in self._river_algo.centers.keys():
            try:
                related_cluster = self._clusters[x]
            except:
                related_cluster = ClusterCentroid(p_id=x, p_visualize=self.get_visualization())  

                if self.get_visualization(): 
                    related_cluster.init_plot(p_figure = self._figure, p_plot_settings=self._plot_settings)  

                list_center = []
                for y in range(len(self._river_algo.centers[x])):
                    list_center.append(self._river_algo.centers[x][y+1])           
                related_cluster.get_centroid().set_values(list_center)
                self._add_cluster( p_cluster = related_cluster )

        return self._clusters


## -------------------------------------------------------------------------------------------------
    def _renormalize(self, p_normalizer:Normalizer):
        """
        Internal renormalization of all clusters. See method OATask.renormalize_on_event() for further
        information.

        Parameters
        ----------
        p_normalizer : Normalizer
            Normalizer object to be applied on task-specific 
        """
        
        super()._renormalize(p_normalizer)

        for cluster in self._clusters.values():
            related_cluster = self._river_algo.centers[cluster.get_id()]
            for mlpro_idx, river_idx in enumerate(related_cluster):
                related_cluster[river_idx] = cluster._centroid.get_values()[mlpro_idx]





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrRiverStreamKMeans2MLPro (WrClusterAnalyzerRiver2MLPro):
    """
    This is the wrapper class for STREAMKMeans clusterer.
    
    According to https://riverml.xyz/latest/api/cluster/STREAMKMeans/ :
    STREAMKMeans is an alternative version of the original algorithm STREAMLSEARCH proposed by
    O'Callaghan et al., by replacing the k-medians using LSEARCH by the k-means algorithm.

    However, instead of using the traditional k-means, which requires a total reclustering each time
    the temporary chunk of data points is full, the implementation of this algorithm uses an
    increamental k-means.

    Parameters
    ----------
    p_name : str
        Name of the clusterer. Default: None.
    p_range_max :
        MLPro machine learning task, either process or thread. Default: MLTask.C_RANGE_THREAD.
    p_ada : bool
        Turn on adaptivity. Default: True.
    p_visualize : bool
        Turn on visualization. Default: False.
    p_logging :
        Set up type of logging. Default: Log.C_LOG_ALL.
    p_chunk_size : int
        Maximum size allowed for the temporary data chunk. Default: 10.
    p_n_clusters : int
        Number of clusters generated by the algorithm. Default: 5.
    p_seed : int
        Random seed used for generating initial centroid positions. Default: None.
    p_halflife : float
        Amount by which to move the cluster centers, a reasonable value if between 0 and 1.
        Default: 0.5.
    p_mu : float
        Mean of the normal distribution used to instantiate cluster positions. Default: 1.
    p_sigma : float
        Standard deviation of the normal distribution used to instantiate cluster positions.
        Default: 1.
    p_p : int
        Power parameter for the Minkowski metric. When p=1, this corresponds to the Manhattan
        distance, while p=2 corresponds to the Euclidean distance. Default: 2.
    p_kwargs : dict
        Further optional named parameters. 
        
    """

    C_NAME          = 'STREAMKMeans'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str = None,
                 p_range_max = MLTask.C_RANGE_THREAD,
                 p_ada:bool = True,
                 p_visualize:bool = False,
                 p_logging = Log.C_LOG_ALL,
                 p_chunk_size:int = 10,
                 p_n_clusters:int = 5,
                 p_halflife:float = 0.5,
                 p_mu:float = 0,
                 p_sigma:float = 1,
                 p_p:int = 2,
                 p_seed:int = None,
                 **p_kwargs):
        
        alg = cluster.STREAMKMeans(chunk_size=p_chunk_size,
                                   n_clusters=p_n_clusters,
                                   halflife=p_halflife,
                                   mu=p_mu,
                                   sigma=p_sigma,
                                   p=p_p,
                                   seed=p_seed)

        super().__init__(p_cls_cluster=ClusterCentroid(),
                         p_river_algo=alg,
                         p_name=p_name,
                         p_range_max=p_range_max,
                         p_ada=p_ada,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _update_clusters(self):
        """
        This method is to update the centroids of each introduced cluster.
        """
        for x in self._river_algo.centers.keys():
            related_cluster = self._clusters[x]
            list_center = []
            for y in range(len(self._river_algo.centers[x])):
                list_center.append(self._river_algo.centers[x][y+1])
            try:
                related_cluster.get_centroid().set_values(list_center)
            except:
                pass



## -------------------------------------------------------------------------------------------------
    def get_clusters(self) -> dict[Cluster]:
        """
        This method returns the current list of clusters.

        Returns
        -------
        dict_of_clusters : dict[Cluster]
            Current list of clusters.
            
        """

        for x in self._river_algo.centers.keys():
            try:
                related_cluster = self._clusters[x]
            except:
                related_cluster = ClusterCentroid(p_id=x, p_visualize=self.get_visualization()) 

                if self.get_visualization():  
                    related_cluster.init_plot(p_figure = self._figure, p_plot_settings=self._plot_settings)  

                list_center = []
                for y in range(len(self._river_algo.centers[x])):
                    list_center.append(self._river_algo.centers[x][y+1])
                try:          
                    related_cluster.get_centroid().set_values(list_center)
                except:
                    pass
                self._add_cluster( p_cluster = related_cluster )

        return self._clusters


## -------------------------------------------------------------------------------------------------
    def _renormalize(self, p_normalizer:Normalizer):
        """
        Internal renormalization of all clusters. See method OATask.renormalize_on_event() for further
        information.

        Parameters
        ----------
        p_normalizer : Normalizer
            Normalizer object to be applied on task-specific 
        """
        
        super()._renormalize(p_normalizer)

        for cluster in self._clusters.values():
            related_cluster = self._river_algo.centers[cluster.get_id()]
            for mlpro_idx, river_idx in enumerate(related_cluster):
                related_cluster[river_idx] = cluster._centroid.get_values()[mlpro_idx]


    