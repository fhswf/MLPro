## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.streams.streams
## -- Module  : clusters.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-09-17  0.1.0     DA       Creation of class StreamMLProClusterGenerator2
## --                                - New parameter p_cluster_specs enabling full specification of clusters
## --                                - Initial radii per cluster and feature
## --                                - Generation of real hyper-ellipsoidal clusters even in higher
## --                                  dimensional spaces
## --                                - Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2025-09-17)

This module provides the native stream class StreamMLProClusterGenerator2.
It generates instances with n-dimensional random feature data, placed around
a variable number of (random) centers. The resulting clusters may or may not 

- move
- change size
- change density and/or change distribution_bias 

over time.

"""

import random
from dataclasses import dataclass, field
from typing import Dict 
from copy import deepcopy

import numpy as np

from mlpro.bf import Log
from mlpro.bf.events import Event, EventManager
from mlpro.bf.exceptions import ParamError
from mlpro.bf.math import Element, MSpace, ESpace
from mlpro.bf.streams import Feature, Instance
from mlpro.bf.streams.streams.provider_mlpro import StreamMLProBase



# Export list for public API
__all__ = [ 'ClusterSpec', 'Cluster', 'ClusterStatistics', 'StreamMLProClusterGenerator2' ]



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
@dataclass
class Cluster:
    """
    This class contains runtime information about a single cluster.
    """

    center : np.array = None
    size : int = 0
    radii : np.array = None
    velocities : np.array = None
    roc_of_radius : float = 0.0
    distribution_bias : int = 1





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
@dataclass
class ClusterSpec:
    """
    This class enables the specification of a single cluster. All parameters refer to the normalized 
    feature space [-1000,1000] in each dimension.

    Parameters
    ----------
    p_center : list[float] = None
        Optional list of initial center coordinates. If not provided, a random center will be generated.
    p_radii : list[float] = [100.0]
        Optional list of initial radii per dimension. If not provided, a default radius of 100.0 will be used for all dimensions.
    p_velocity : list[float] = [0.0]
        Optional list of initial velocities per dimension. If not provided, a default velocity of 0.0 will be used for all dimensions.
    p_distribution_bias : int = 1
        Optional integer distribution_bias. For example, a value of 2 causes the cluster to be
        flooded with two times more instances than a cluster with distribution_bias 1. Default = 1.
    """

    p_center : list[float] = None
    p_radii : list[float] = field(default_factory=lambda: [100.0])
    p_velocities : list[float] = field(default_factory=lambda: [0.0])
    p_distribution_bias : int = 1

## -------------------------------------------------------------------------------------------------
    def create_cluster( self, 
                        p_num_dim : int,
                        p_boundaries : list) -> Cluster:

        # 1 Initial cluster center
        if self.p_center is None:
            center = np.array( [random.uniform(p_boundaries[0], p_boundaries[1]) for d in range(p_num_dim)], dtype=float )
        elif len(self.p_center) == p_num_dim:
            center = np.array(self.p_center, dtype=float)
        else:
            raise ParamError(f"Number of dimensions of provided cluster center ({len(self.p_center)}) does not match number of dimensions ({p_num_dim}).")


        # 2 Initial cluster radii
        if self.p_radii is None:
            radii = np.array( [100.0]*p_num_dim, dtype=float )
        elif len(self.p_radii) == 1:
            radii = np.array( self.p_radii * p_num_dim, dtype=float )
        elif len(self.p_radii) == p_num_dim:
            radii = np.array(self.p_radii, dtype=float)
        else:
            raise ParamError(f"Number of dimensions of provided cluster radii ({len(self.p_radii)}) does not match number of dimensions ({p_num_dim}).")


        # 3 Velocities
        if self.p_velocities is None:
            velocities = np.array([0.0]*p_num_dim, dtype=float)
        elif len(self.p_velocities) == 1:
            velocities = np.array(self.p_velocities * p_num_dim, dtype=float)
        elif len(self.p_velocities) == p_num_dim:
            velocities = np.array(self.p_velocities, dtype=float)
        else:
            raise ParamError(f"Number of dimensions of provided cluster velocities ({len(self.p_velocities)}) does not match number of dimensions ({p_num_dim}).")


        # 4 Create and return cluster
        return Cluster( center = center,
                        radii = radii,
                        velocities = velocities,
                        distribution_bias = self.p_distribution_bias )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
@dataclass
class ClusterStatistics:
    """
    This class provides statistics about the generated clusters.
    """

    feature_boundaries : list = None
    feature_rescale_params : np.array = None
    clusters: Dict[int, Cluster] = field(default_factory=dict)

## -------------------------------------------------------------------------------------------------
    @property
    def num_clusters(self):
        return len(self.clusters)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProClusterGenerator2 (StreamMLProBase, EventManager):
    """
    This benchmark stream class generates freely configurable random point clusters of any number,
    dimensionality,  size, velocity, acceleration and distribution_bias.

    This class is a version 2 of the StreamMLProClusterGenerator class and includes additional features
    and improvements. The behavior if single clusters can be fully specified via the new parameter
    p_cluster_specs.

    Parameters
    ----------
    p_num_dim : int
        The number of dimensions or features of the data.
    p_num_instances : int
        Total number of instances.
    p_num_clusters : int
        Number of clusters.
    p_cluster_specs : list[ClusterSpec] = None
        Optional list of cluster specifications. If not provided, default specifications will be used.
    p_boundaries_rescale : list = None
        Optional list of alternative boundaries per dimension. The generated clusters will be rescaled
        to fit within these boundaries. If not provided, the boundaries [-1000,1000] will be used for all
        dimensions.
    p_outlier_rate : float = 0.0
        The rate of the interval [0,1] at which outliers occur. A value of 0.0 disables outlier generation. Default = 0.0
    p_change_radii : bool
        If there are changes in radii of clusters. Default= False.
    p_rate_of_change_of_radius : float
        Rate of change of radius per cycle. Default = 0.001.
    p_points_of_change_radii : list
        Instances at which change of radii occur. Default = None.
    p_num_clusters_for_change_radii : list
        The number of clusters for which the radius changes. Default = None.
    p_change_velocities : bool
        If there are changes in velocities of clusters. Default = False.
    p_points_of_change_velocities : list
        Instances at which change of velocities occur. Default = None.
    p_num_clusters_for_change_velocities : list
        The number of clusters for which the velocity changes. Default = None.
    p_changed_velocities : list
        The changed value of velocities. Default = None.
    p_change_distribution_bias : bool
        If there are changes in distribution_bias of clusters. Default = False.
    p_points_of_change_distribution_bias : list
        Instances at which change of distribution_biass occur. Default = None.
    p_num_clusters_for_change_distribution_bias : list
        The number of clusters for which the radius changes. Default = None.
    p_appearance_of_clusters : bool
        If new clusters appear. Default = False.
    p_points_of_appearance_of_clusters : list
        Instances at which new clusters appears. Deafult = None.
    p_num_new_clusters_to_appear : int
        The number of clusters to appear. Default = None.
    p_disappearance_of_clusters : bool
        If clusters disappears. Default = False.
    p_points_of_disappearance_of_clusters : list
        Instances at which clusters disappear. Default = None.
    p_num_clusters_to_disappear : int
        The number of clusters which disappear in time. Default = None.
    p_split_and_merge_of_clusters : bool
        If clusters split and merge. Default = False.
    p_num_of_clusters_for_split_and_merge : int
        Number of clusters that split split at first and then later on merge. Default = 2.
    p_max_clusters_affected : float
        Fraction of maximum number of clusters affected by changes in properties. Default = 0.75.
    p_seed 
        Seeding value for the random generator. Default = None (no seeding).
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.


    Attributes
    ----------
    cluster_statistics : ClusterStatistics
        Public attribute providing statistics about the generated clusters.
    num_outliers : int
        Public attribute providing the number of generated outliers.
    """

    C_ID                    = 'ClustersNDim'
    C_NAME                  = 'Clusters N-Dim'
    C_TYPE                  = 'Benchmark'
    C_VERSION               = '2.0.0'
    C_SCIREF_ABSTRACT       = 'Demo stream provides self.C_NUM_INSTANCES self._no_dim-dimensional instances randomly positioned around centers which form clusters whose numbers, size, velocity, acceleration and density may or may not change over time.'
    C_BOUNDARIES            = [-1000,1000]
    C_EVENT_ID_OUTLIER      = 'Outlier'

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_num_dim : int,
                  p_num_instances : int,
                  p_num_clusters : int,
                  p_cluster_specs : list[ClusterSpec] = None,
                  p_boundaries_rescale : list = None,
                  p_outlier_rate : float = 0.0,
                  p_change_radii : bool = False,
                  p_rate_of_change_of_radius : float = 0.001,
                  p_points_of_change_radii : list = None,
                  p_num_clusters_for_change_radii : int = None,
                  p_change_velocities : bool = False,
                  p_points_of_change_velocities : list = None,
                  p_num_clusters_for_change_velocities : int = None,
                  p_changed_velocities : list = None,
                  p_change_distribution_bias : bool = False,
                  p_points_of_change_distribution_bias : list = None,
                  p_num_clusters_for_change_distribution_bias : int = None,
                  p_appearance_of_clusters : bool = False,
                  p_points_of_appearance_of_clusters : list = None,
                  p_num_new_clusters_to_appear : int = None,
                  p_disappearance_of_clusters : bool = False,
                  p_points_of_disappearance_of_clusters : list = None,
                  p_num_clusters_to_disappear : int = None,
                  p_clusters_split_and_merge : bool = False,
                  p_clusters_split : bool = False,
                  p_points_of_split : list = None,
                  p_velocities_after_split : list = None,
                  p_num_clusters_to_split_into: int = 2,
                  p_max_clusters_affected : float = 0.75,
                  p_seed = None,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        # 1 Initialize private attributes
        self._num_dim           = p_num_dim
        self.C_NUM_INSTANCES    = p_num_instances
        self._num_clusters      = p_num_clusters

        self._cluster_ids           = [x for x in range(self._num_clusters)]
        self._current_cluster_id    = None
        self._cycle                 = None


        # 1.1 New style cluster specifications
        self._cluster_specs     = self._create_cluster_specs( p_cluster_specs, **p_kwargs )


        # 1.2 Optional explicit boundaries per dimension and resulting rescaling parameters
        if p_boundaries_rescale is not None:
            if len(p_boundaries_rescale) != self._num_dim:
                raise ParamError(f"Expected {self._num_dim} dimensions for feature boundaries.")
            self._boundaries_rescale    = p_boundaries_rescale
        else:
            self._boundaries_rescale    = [self.C_BOUNDARIES]*self._num_dim

        self._rescaling_params      = self._get_rescaling_params(self._boundaries_rescale)


        # 1.3 Outlier generation
        self._outlier_appearance = p_outlier_rate > 0.0
        self._outlier_rate       = p_outlier_rate


        # 1.4 Changes in radii
        self._change_in_radius      = self._change_in_property(p_change=p_change_radii,
                                                               p_point_of_change=p_points_of_change_radii,
                                                               p_num_clusters_affected=p_num_clusters_for_change_radii)
        self._roco_radius           = p_rate_of_change_of_radius


        # 1.5 Changes in velocities
        self._change_in_velocity    = self._change_in_property(p_change=p_change_velocities,
                                                               p_point_of_change=p_points_of_change_velocities,
                                                               p_num_clusters_affected=p_num_clusters_for_change_velocities,
                                                               p_changed_values=p_changed_velocities)
        

        # 1.6 Changes in distribution_bias
        self._change_in_distribution_bias = self._change_in_property(p_change=p_change_distribution_bias,
                                                               p_point_of_change=p_points_of_change_distribution_bias,
                                                               p_num_clusters_affected=p_num_clusters_for_change_distribution_bias)
        

        # 1.7 Appearance of new clusters
        self._new_cluster           = self._change_in_property(p_change=p_appearance_of_clusters,
                                                               p_point_of_change=p_points_of_appearance_of_clusters,
                                                               p_num_clusters_affected=p_num_new_clusters_to_appear)
        
        
        # 1.8 Disappearance of clusters
        self._remove_cluster        = self._change_in_property(p_change=p_disappearance_of_clusters,
                                                               p_point_of_change=p_points_of_disappearance_of_clusters,
                                                               p_num_clusters_affected=p_num_clusters_to_disappear)
        

        # 1.9 Split and merge of clusters
        if p_clusters_split:
            self._split_and_merge_of_clusters= self._change_in_property(p_change=p_clusters_split,
                                                                        p_point_of_change=p_points_of_split,
                                                                        p_num_clusters_affected=p_num_clusters_to_split_into)
        else:
            self._split_and_merge_of_clusters= self._change_in_property(p_change=p_clusters_split_and_merge,
                                                                        p_point_of_change=p_points_of_split,
                                                                        p_num_clusters_affected=p_num_clusters_to_split_into)
        self._split_of_clusters = p_clusters_split

        if self._split_and_merge_of_clusters:
            self._split_and_merge_of_clusters["start"] = self._split_and_merge_of_clusters["start"][:1]
            self._split_and_merge_of_clusters["end"] = [self.C_NUM_INSTANCES*0.8]
            self._velocities_split = []
            self._center_split = []
            if p_velocities_after_split:
                if len(p_velocities_after_split) == 1:
                    self._velocities_split = p_num_clusters_to_split_into*p_num_clusters_to_split_into
                elif len(p_velocities_after_split) < p_num_clusters_to_split_into:
                    self._velocities_split = p_velocities_after_split.extend([p_velocities_after_split[0]] *
                                                    (p_num_clusters_to_split_into-len(p_velocities_after_split)))
                else:
                    self._velocities_split = p_velocities_after_split[:p_num_clusters_to_split_into]

            else:
                for x in range(p_num_clusters_to_split_into):
                    if max(self._velocities)==0 and min(self._velocities)==0:
                        self._velocities_split.append(random.random())
                    else:
                        a = random.randint(0,1)
                        if a == 0:
                            self._velocities_split.append(max(self._velocities)*random.random())
                        else:
                            self._velocities_split.append(max(self._velocities)*(1+random.random()))
        
        self._max_clusters_affected = p_max_clusters_affected


        # 3 Initialize public attributes
        self.num_outliers = 0

        if p_boundaries_rescale is not None:
            boundaries = p_boundaries_rescale
        else:
            boundaries = [self.C_BOUNDARIES]*p_num_dim

        self.cluster_statistics = ClusterStatistics( feature_boundaries = boundaries,
                                                     feature_rescale_params = self._rescaling_params )
        
        self._clusters = self.cluster_statistics.clusters


        # 4 Initialize parent classes
        StreamMLProBase.__init__ ( self, p_logging=p_logging, **p_kwargs)
        
        EventManager.__init__(self, p_logging=p_logging)    


        # 5 Initialize seeding
        if p_seed is not None:
            self.set_random_seed(p_seed=p_seed)
        else:
            random.seed()
            np.random.seed()


## -------------------------------------------------------------------------------------------------
    def _create_cluster_specs(self, p_cluster_specs : list[ClusterSpec], **p_kwargs):
        """
        Create cluster specifications from provided cluster specs or legacy parameters.
        If p_cluster_specs is provided, it takes precedence over legacy parameters.

        Parameters
        ----------
        p_cluster_specs : list[ClusterSpec]
            List of cluster specifications to use.
        **p_kwargs : dict
            Additional keyword arguments for legacy parameters.

        Returns
        -------
        list[ClusterSpec]
            List of created cluster specifications.
        """

        # 1 Take over provided cluster specifications if provided
        if p_cluster_specs:

            len_specs = len(p_cluster_specs)
            if len_specs == 1:
                return [deepcopy(p_cluster_specs[0]) for _ in range(self._num_clusters)]
            
            if len(p_cluster_specs) != self._num_clusters:
                raise ParamError(f"Number of provided cluster specifications ({len(p_cluster_specs)}) does not match number of clusters ({self._num_clusters}).")
            
            for spec in p_cluster_specs:
                if spec.p_center and len(spec.p_center) != self._num_dim:
                    raise ParamError(f"Number of dimensions of provided cluster center ({len(spec.p_center)}) does not match number of dimensions ({self._num_dim}).")
                
                if spec.p_radii and len(spec.p_radii) != self._num_dim:
                    raise ParamError(f"Number of dimensions of provided cluster radii ({len(spec.p_radii)}) does not match number of dimensions ({self._num_dim}).")
                
            return p_cluster_specs
                

        # 2 Create cluster specifications from provided legacy parameters
        cluster_specs = []

        for c in range(self._num_clusters):

            # 2.1 Center
            try:
                centers = p_kwargs['p_centers']
                try:
                    center  = centers[c]
                except IndexError:
                    raise ParamError(f"Number of provided cluster centers ({len(centers)}) does not match number of clusters ({self._num_clusters}).")  
            except KeyError:
                center = None

            # 2.2 Radii
            try:
                radii = p_kwargs['p_radii']

                if len(radii) == 1:
                    radii_cluster = [radii[0]] * self._num_dim
                elif len(radii) == self._num_clusters:
                    radii_cluster = [radii[c]] * self._num_dim
                else:
                    raise ParamError(f"Number of provided cluster radii ({len(radii)}) does not match number of clusters ({self._num_clusters}).")

            except KeyError:
                radii = None

            cluster_spec = ClusterSpec( p_center = center, 
                                        p_radii = radii_cluster )
            
            cluster_specs.append(cluster_spec)


        return cluster_specs
        

## -------------------------------------------------------------------------------------------------
    def _get_rescaling_params(self, p_boundaries_rescale : list):
        if p_boundaries_rescale is None: return None

        if len(p_boundaries_rescale) != self._num_dim:
            raise ParamError(f"Expected {self._num_dim} dimensions for rescale boundaries.")

        params = np.zeros((self._num_dim, 2))

        for dim in range(self._num_dim):
            params[dim,0] = ( p_boundaries_rescale[dim][1] - p_boundaries_rescale[dim][0] ) / ( self.C_BOUNDARIES[1] - self.C_BOUNDARIES[0] )
            params[dim,1] = p_boundaries_rescale[dim][0] - self.C_BOUNDARIES[0] * params[dim,0]

        return params


## -------------------------------------------------------------------------------------------------
    def _extend_property_list(self, property : list, size : int):
        """
        Function to extend the property list for all clusters.
        """
        if len(property) == 1:
            property *= size
        elif len(property) < size:
            property.extend([property[0]] * (size - len(property)))
        elif len(property) > size:
            property = property[:size]
        return property


## -------------------------------------------------------------------------------------------------
    def _change_in_property( self, 
                             p_change = False, 
                             p_point_of_change = None, 
                             p_num_clusters_affected = None,
                             p_changed_values = None ):
        """
        Function to assign clusters and the point of change, when there is a change in property.

        Parameters
        ----------
        p_change : bool
            If there is a change in property.
        p_point_of_change : list
            List of points of change.
        p_num_clusters_affected : int
            Number of clusters affected by the change.
        p_changed_values : list
            List of changed values. If only one value is provided, it will be applied to all affected clusters.

        Returns
        -------
        dict or bool
            Dictionary with keys "clusters", "start", "end", and "changed_values" if there is a change, otherwise False.
        """

        if not p_change:
            return False
        
        if p_num_clusters_affected:
            num_clusters = p_num_clusters_affected
        else:
            num_clusters = random.randint(1, int(self._max_clusters_affected*self._num_clusters))

        clusters = list(random.sample(self._cluster_ids, num_clusters))

        if p_point_of_change is None:           
            start = [random.randint(int(0.1*self.C_NUM_INSTANCES), int(0.5*self.C_NUM_INSTANCES))
                     for _ in clusters]
        else:
            start = p_point_of_change
            if len(start) == 1:
                start *= p_num_clusters_affected
            elif len(start) < p_num_clusters_affected:
                start.extend([start[0]] * (p_num_clusters_affected - len(start)))
            elif len(start) >= p_num_clusters_affected:
                start = start[:p_num_clusters_affected]


        y = 200 if self.C_NUM_INSTANCES > 1000 else int(0.2 * self.C_NUM_INSTANCES)
        z = [x+y for x in start]
        end = [random.randint(x, self.C_NUM_INSTANCES) for x in z]

        if p_changed_values:
            if len(p_changed_values) == 1:
                p_changed_values = p_changed_values*num_clusters

        return {"clusters": clusters, "start": start, "end": end, "changed_values": p_changed_values}


## -------------------------------------------------------------------------------------------------
    def _setup_feature_space(self) -> MSpace:
        feature_space : MSpace = ESpace()

        for i in range(self._num_dim):
            feature_space.add_dim( Feature( p_name_short = 'f_' + str(i),
                                            p_base_set = Feature.C_BASE_SET_R,
                                            p_name_long = 'Feature #' + str(i),
                                            p_name_latex = '',
                                            p_description = '',
                                            p_symmetrical = False,
                                            p_logging=Log.C_LOG_NOTHING ) )
        return feature_space


## -------------------------------------------------------------------------------------------------
    def _init_dataset(self):
        for cluster_id in self._cluster_ids:
            # Add cluster to dictionary
            self._clusters[cluster_id] = self._define_cluster(cluster_id)


## -------------------------------------------------------------------------------------------------
    def _define_cluster(self, p_cluster_id):
        """
        Function to define the cluster.
        """

        return self._cluster_specs[p_cluster_id].create_cluster( p_num_dim = self._num_dim ,
                                                                 p_boundaries = self.C_BOUNDARIES )


## -------------------------------------------------------------------------------------------------
    def _find_velocity(self, velocity, init_point : list = [], final_point : list = [], steps : int = 1):
        """
        Function to calculate velocity of a cluster.
        """

        if len(final_point) != 0:
            return [(float(final_point[d]) - init_point[d]) / steps for d in range(self._num_dim)]

        velocity_vector = np.array( [random.uniform(self.C_BOUNDARIES[0], self.C_BOUNDARIES[1]) for _ in range(self._num_dim)] )
            
        dist = np.linalg.norm(velocity_vector)
        return velocity_vector * (velocity / dist)
    

## -------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:
        """
        Generate the next instance.
        
        Returns
        -------
        Instance
            The next generated instance.
        """

        # 0 Iteration control
        if self.C_NUM_INSTANCES== 0: pass
        elif self._index == self.C_NUM_INSTANCES: raise StopIteration


        # 1 Update of cluster properties
        self._prepare_clusters_for_changes()
        self._update_cluster_properties()


        # 2 Determine next cluster to provide an instance
        cluster_id = self._get_next_cluster()
        self._clusters[cluster_id].size += 1


        # 3 Generate and return new instance for the selected cluster
        feature_data = Element(self._feature_space)

        raised_outlier = False

        if self._outlier_appearance and (random.uniform(0, 1) <= self._outlier_rate):
            point_values = np.array( [random.uniform(self.C_BOUNDARIES[0], self.C_BOUNDARIES[1]) for _ in range(self._num_dim)] )
       
            raised_outlier = True
            self.num_outliers += 1

        else:
            point_values = self._generate_random_point_around_cluster(cluster_id)
        
        # 3.1 Optional rescaling of generated point values
        if self._rescaling_params is not None:
            point_values = point_values * self._rescaling_params[:,0] + self._rescaling_params[:,1]

        feature_data.set_values(point_values)

        self._index += 1
        inst = Instance(p_feature_data=feature_data,
                        p_tstamp= self.get_tstamp())
    
        if raised_outlier:
            self._raise_event(p_event_id = self.C_EVENT_ID_OUTLIER,
                              p_event_object = Event(p_raising_object= self,
                                                     p_tstamp = inst.tstamp,
                                                     p_instance = inst ))
            
        return inst
    

## -------------------------------------------------------------------------------------------------
    def _prepare_clusters_for_changes(self):
        """
        Prepare clusters for changes in properties.
        """

        if self._change_in_radius:
            self._handle_change_in_property("radius", self._change_in_radius, self._roco_radius)

        if not self._split_and_merge_of_clusters:
            if self._change_in_velocity:
                self._handle_change_in_property("velocity", self._change_in_velocity, 0)
        
        if self._change_in_distribution_bias:
            self._handle_change_in_property("distribution_bias", self._change_in_distribution_bias, 0)

        if self._new_cluster:
            self._handle_new_clusters()

        if self._remove_cluster:
            self._handle_remove_clusters()

        if self._split_and_merge_of_clusters:
            self._handle_split_and_merge()
    

## -------------------------------------------------------------------------------------------------
    def _handle_change_in_property( self, 
                                    p_property_name, 
                                    p_property_change, 
                                    p_change_value ):
        """
        Handle changes in cluster properties.
        
        Parameters
        ----------
        p_property_name : str
            Name of the property to change ("radius", "velocity", or "distribution_bias").
        p_property_change : dict
            Dictionary containing change information with keys "clusters", "start", "end", and "changed_values".
        p_change_value : float
            Value to apply for the change (e.g., rate of change of radius).
        """

        if self._index in p_property_change["start"]:
            ids = [i for i, x in enumerate(p_property_change["start"]) if x == self._index]

            for idx in ids:
                cluster_id = p_property_change["clusters"][idx]

                if p_property_name == "radius":
                    self._clusters[cluster_id].roc_of_radius = self._clusters[cluster_id].radii * p_change_value

                elif p_property_name == "velocity":
                    if p_property_change["changed_values"]:
                        self._clusters[cluster_id].velocities = self._find_velocity(velocity=p_property_change["changed_values"][idx])
                    else:
                        if max(self._velocities)==0 and min(self._velocities)==0:
                            velocity = random.random()*0.25
                        else:
                            a = random.randint(0,1)
                            if a == 0:
                                velocity = max(self._velocities)*random.random()
                            else:
                                velocity = max(self._velocities)*(1+random.random())
                        self._clusters[cluster_id].velocities = self._find_velocity(velocity=velocity)

                elif p_property_name == "distribution_bias":
                    if max(self._distribution_bias)==1 and min(self._distribution_bias)==1:
                        distribution_bias = random.randint(1,10)
                    else:
                        distribution_bias = random.randint(1,max(self._distribution_bias)+2)
                    self._clusters[cluster_id].distribution_bias = distribution_bias

        if self._index in p_property_change["end"]:
            ids = [i for i, x in enumerate(p_property_change["end"]) if x == self._index]
            for idx in ids:
                cluster_id = p_property_change["clusters"][idx]
                if p_property_name == "radius":
                    self._clusters[cluster_id].roc_of_radius = 0


## -------------------------------------------------------------------------------------------------
    def _handle_new_clusters(self):
        
        if self._index in self._new_cluster["start"]:
            n = self._new_cluster["start"].count(self._index)

            for _ in range(n):
                self._cluster_ids.append(self._num_clusters)
                self._radii.append((min(self._radii) + max(self._radii))/2)
                a = random.randint(0, 1)
                self._velocities.append(min(self._velocities) if a == 0 else (min(self._velocities) + max(self._velocities)) / 2)
                self._distribution_bias.append(random.randint(1, max(self._distribution_bias)))
                self._clusters[self._cluster_ids[-1]] = self._define_cluster(p_cluster_id=self._cluster_ids[-1])
                self._num_clusters += 1 


## -------------------------------------------------------------------------------------------------
    def _handle_remove_clusters(self):

        if self._index in self._remove_cluster["start"]:
            ids = [i for i, x in enumerate(self._remove_cluster["start"]) if x == self._index]
            j=1

            for i in ids:
                self._radii.pop(i-j)
                self._velocities.pop(i-j)
                self._distribution_bias.pop(i-j)
                j+=1
            self._num_clusters -= len(ids)
            self._cluster_ids = self._cluster_ids[:self._num_clusters-len(ids)]
            self._current_cluster_id = 0

            for id in ids:
                del self._clusters[self._remove_cluster["clusters"][id]]
                for i in range(len(self._remove_cluster["clusters"])):
                    if self._remove_cluster["clusters"][i] > self._remove_cluster["clusters"][id]:
                        self._remove_cluster["clusters"][i] -= 1
            a = 1
            clusters = {}

            for id in self._clusters.keys():
                clusters[a] = self._clusters[id]
                a += 1
            self._clusters = clusters


## -------------------------------------------------------------------------------------------------
    def _handle_split_and_merge(self):
        if self._index < 2:
            self._center_split = self._clusters[self._split_and_merge_of_clusters["clusters"][0]].center
            for id in self._split_and_merge_of_clusters["clusters"]:
                self._clusters[id].velocities = np.zeros(self._num_dim)
                self._clusters[id].center = self._center_split

        if self._index in self._split_and_merge_of_clusters["start"]:
            x = 0
            for id in self._split_and_merge_of_clusters["clusters"]:
                self._clusters[id].velocities = self._find_velocity(velocity=self._velocities_split[x])
                x += 1

        if not self._split_of_clusters:

            if self._index == int(self._split_and_merge_of_clusters["start"][0]+
                                  (self._split_and_merge_of_clusters["end"][0] -
                                  self._split_and_merge_of_clusters["start"][0])/2):
                for id in self._split_and_merge_of_clusters["clusters"]:
                    initial_point = list(self._clusters[id].center)
                    final_point = self._center_split
                    self._clusters[id].velocities = self._find_velocity(velocity=0,
                                                                     init_point=initial_point,
                                                                     final_point=final_point,
                                                                     steps=int((self._split_and_merge_of_clusters["end"][0] -
                                                                                self._split_and_merge_of_clusters["start"][0])/2)-1)

            if self._index in self._split_and_merge_of_clusters["end"]:
                for id in self._split_and_merge_of_clusters["clusters"]:
                    self._clusters[id].velocities = np.zeros(self._num_dim)
                    self._clusters[id].center = self._center_split


## -------------------------------------------------------------------------------------------------
    def _update_cluster_properties(self):
        """
        Function to update the properties of clusters.
        """

        for cluster in self._clusters.values():

            # 1 Radius based on rate of change of radius (roc_of_radius)
            if cluster.roc_of_radius != 0:
                np.add(cluster.roc_of_radius, cluster.radii, out=cluster.radii)

            # 2 Center based on velocity
            np.add(cluster.velocities, cluster.center, out=cluster.center)


## -------------------------------------------------------------------------------------------------
    def _get_next_cluster(self) -> Cluster:
        """
        Function to determine the cluster for the next data point to be added.

        Returns
        -------
        cluster_id : int
            ID of the selected cluster.
        """

        if self._cycle is None:
            self._cycle = 1
            self._current_cluster_id = 0
            return self._current_cluster_id

        if ( self._cycle % self._clusters[self._current_cluster_id].distribution_bias) == 0:
            self._current_cluster_id = ( self._current_cluster_id + 1 ) % self._num_clusters
            self._cycle  = 1
        else:
            self._cycle += 1

        return self._current_cluster_id


## -------------------------------------------------------------------------------------------------
    def _generate_random_point_around_cluster(self, cluster_id):
        """
        Function to generate a random point around the cluster center.

        Parameters
        ----------
        cluster_id : int
            ID of the cluster around which the point is to be generated.

        Returns
        -------
        point : np.array
            Generated point.
        """

        # 0 Preparation
        cluster = self._clusters[cluster_id]
        center  = cluster.center
        radii   = cluster.radii

        # 1 Generate and return random point
        v       = np.random.normal(size=self._num_dim)
        v      /= np.linalg.norm(v)
        r       = np.random.rand() ** (1.0 / self._num_dim)

        return center + v * r * radii   
