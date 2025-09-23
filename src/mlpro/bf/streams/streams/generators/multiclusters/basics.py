## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.streams.streams.clusters
## -- Module  : cluster.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-09-21  1.0.0     DA       Creation 
## -- 2025-09-23  1.1.0     DA       Class StreamGenCluster: renamed parameter p_durations to 
## --                                p_transition_steps
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-09-23)

This module provides an elementary stream generator shaping a single cluster of random points in the
d-dimensional feature space [-1000,1000]^d. The cluster can be static or dynamic (moving and/or changing its size)
over the total number of instances. Its behavior is defined by a list of states, each defining the center and the radii
of the cluster at the beginning of the state. If multiple states are provided, the cluster will move and/or change its size
linearly between the states over the specified number of instances. Overall, the following parameters can be defined:

- number of instances
- list of states with center and radii
- list of durations for each state transition
- random seed
- data type (float32, float64)

"""

from typing import Union, Literal, List
from dataclasses import dataclass, field
from typing import Dict 

import numpy as np

from mlpro.bf import Log
from mlpro.bf.exceptions import ParamError
from mlpro.bf.math import Element
from mlpro.bf.streams import Instance, Sampler, Stream
from mlpro.bf.streams.streams.generators.basics import *



# Export list for public API
__all__ = [ 'ClusterState', 'ClusterStatistics', 'StreamGenCluster', 'MultiStreamGenCluster' ]



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
@dataclass
class ClusterState:
    """
    Class defining a state of the cluster with center and radii.
    """

    p_center : Union[ Literal['rnd'], list, np.ndarray ] = 'rnd'
    p_radii: Union[ Literal['rnd'], list, np.ndarray ] = 'rnd'





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
@dataclass
class ClusterStatistics:
    """
    This class provides statistics about the generated clusters.
    """

    feature_boundaries : list = None
    feature_rescale_params : np.array = None
    clusters: Dict[int, object] = field(default_factory=dict)

## -------------------------------------------------------------------------------------------------
    @property
    def num_clusters(self):
        return len(self.clusters)
    




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamGenCluster(StreamGenerator):
    """
    Elementary stream generator shaping a single cluster of random points in the d-dimensional 
    feature space [-1000,1000]^d. The cluster can be static or dynamic (moving and/or changing its size) 
    over the total number of instances.

    Parameters
    ----------
    p_num_dim : int
        Number of dimensions of the feature space (d).
    p_id : int = 0
        Optional unique ID of the stream within an application. Default is 0.
    p_seed : int = 0    
        Optional random seed for reproducibility. Use different seeds for different streams to avoid identical 
        sequences. Default is 0.
    p_num_instances : int = 0
        Total number of instances to be generated. If 0, the stream will generate instances indefinitely by
        repeating the defined states.
    p_states : List[ClusterState], default: [ ClusterState() ]
        List of states defining the behavior of the cluster over time. Each state defines the center and
        the radii of the cluster at the beginning of the state. If multiple states are provided, the
        cluster will move and/or change its size linearly between the states over the number of instances
        defined by p_transition_steps. If only one state is provided, the cluster will remain static
        at the defined center and radii.
    p_transition_steps : List[int], default: None
        List of numbers of instances for each state transition step. The length of p_transition_steps must
        be equal to len(p_states) - 1. If None, the cluster will stay in the first state for all instances.
    p_dtype : type, default: np.float32
        Data type of the feature values (np.float32 or np.float64).
    p_logging : int, default: Log.C_LOG_NOTHING
        Logging level. See :class:`mlpro.bf.Log` for details.

    Attributes
    ----------
    center : np.array
        Current center of the cluster.
    radii : np.array
        Current radii of the cluster.
    velocities : np.array
        Current velocities of the cluster.
    roc_of_radii : np.array
        Current rate of change of the radii of the cluster.
    size : int
        Number of instances generated so far.
    cluster_stats : ClusterStatistics
        Statistics about the generated cluster.
    """

    C_NAME              = 'Cluster'
    C_RADII_RND_FACTOR  = 0.2           # Factor for random radii generation within boundary range
  
## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_num_dim : int,
                  p_id : int = 0,
                  p_seed : int = 0,
                  p_num_instances : int = 0,
                  p_states : List[ClusterState] = [ ClusterState() ],
                  p_transition_steps : List[int] = None,
                  p_boundaries_rescale : list = None,
                  p_outlier_rate : float = 0.0,
                  p_sampler : Sampler = None,
                  p_dtype : type = np.float32,
                  p_logging : int = Log.C_LOG_NOTHING ):
        
        # 1 Parameter checks

        # 1.1 States
        if p_states is None or len(p_states) < 1:
            raise ParamError('At least one state must be provided in p_states!')
        
        for state in p_states:
            if not isinstance(state, ClusterState):
                raise ParamError('All elements in p_states must be of type ClusterState!')
            
            if state.p_center != 'rnd' and ( not isinstance(state.p_center, (list, np.ndarray) or len(state.p_center) != p_num_dim) ):
                raise ParamError('state.center must be "rnd" or a list/array of length p_num_dim!')
            
            if state.p_radii != 'rnd' and ( not isinstance(state.p_radii, (list, np.ndarray) or len(state.p_radii) != p_num_dim) ):
                raise ParamError('state.radii must be "rnd" or a list/array of length p_num_dim!')
            
        # 1.2 Durations
        if isinstance(p_states, list) and len(p_states) > 1:
            if p_transition_steps is None or ( not isinstance(p_transition_steps, list) or len(p_transition_steps) != len(p_states) - 1 ):
                raise ParamError('If multiple states are provided in p_states, p_transition_durations must be a list of length len(p_states) - 1!')
        
        
        # 2 Init all attributes  
        self._states                          = p_states
        self._transition_steps                = p_transition_steps
        self._array_centers : np.ndarray      = None
        self._array_velocities : np.ndarray   = None
        self._array_radii : np.ndarray        = None
        self._array_roc_of_radii : np.ndarray = None

        self.center : np.ndarray              = None
        self.velocities : np.ndarray          = None
        self.radii  : np.ndarray              = None
        self.roc_of_radii : np.ndarray        = None
        self.size   : int                     = 0


        # 3 Call parent initializations
        super().__init__( p_num_dim = p_num_dim,
                          p_id = p_id,
                          p_seed = p_seed,
                          p_num_instances = p_num_instances,
                          p_boundaries_rescale = p_boundaries_rescale,
                          p_outlier_rate = p_outlier_rate,
                          p_sampler = p_sampler,
                          p_dtype = p_dtype,
                          p_logging = p_logging )


        self.cluster_stats = ClusterStatistics( feature_boundaries = self._boundaries_rescale,
                                                feature_rescale_params = self._rescaling_params )
        
        self.cluster_stats.clusters[self.id] = self



## -------------------------------------------------------------------------------------------------
    def _reset(self):

        # 0 Call parent reset
        super()._reset()


        # 1 Cluster size and process variables
        self.size                     = 0
        self._current_phase : int  = 0
        self._transition_counter      = 0


        # 2 Arrays of centers and radii
        if self._array_centers is None:
            self.center         = np.zeros( shape = ( self._num_dim ), dtype=self._dtype )
            self.velocities     = np.zeros( shape = ( self._num_dim ), dtype=self._dtype )
            self.radii          = np.zeros( shape = ( self._num_dim ), dtype=self._dtype )
            self.roc_of_radii   = np.zeros( shape = ( self._num_dim ), dtype=self._dtype )

            self._array_centers = np.empty( shape = ( len(self._states), self._num_dim ), dtype=self._dtype )
            self._array_radii   = np.empty( shape = ( len(self._states), self._num_dim ), dtype=self._dtype )

        for i, state in enumerate(self._states):

            # 2.1 Center of state i
            if state.p_center == 'rnd':
                self._array_centers[i,:] = self._gen_rnd_array( p_low = self.C_BOUNDARIES[0], 
                                                                p_high = self.C_BOUNDARIES[1], 
                                                                p_size = self._num_dim )
            else:
                self._array_centers[i,:] = np.array( state.p_center, dtype=self._dtype )

            # 2.2 Radii of state i
            if state.p_radii == 'rnd':
                self._array_radii[i,:] = self._gen_rnd_array( p_low = 0,
                                                              p_high = ( self.C_BOUNDARIES[1] - self.C_BOUNDARIES[0] ) * self.C_RADII_RND_FACTOR, 
                                                              p_size = self._num_dim )
            else:
                self._array_radii[i,:] = np.array( state.p_radii, dtype=self._dtype )


        # 3 Array of velocities and rates of change of radii
        if self._transition_steps is not None:
            if self._array_velocities is None:
                self._array_velocities    = np.empty( shape = ( len(self._transition_steps), self._num_dim ), dtype=self._dtype )
                self._array_roc_of_radii  = np.empty( shape = ( len(self._transition_steps), self._num_dim ), dtype=self._dtype )

            for i, duration in enumerate(self._transition_steps):
                self._array_velocities[i,:]   = ( self._array_centers[i+1,:] - self._array_centers[i,:] ) / ( duration - 1 )
                self._array_roc_of_radii[i,:] = ( self._array_radii[i+1,:] - self._array_radii[i,:] ) / ( duration - 1 )

        else:
            self._array_velocities    = np.zeros( shape = ( 1, self._num_dim ), dtype=self._dtype )
            self._array_roc_of_radii  = np.zeros( shape = ( 1, self._num_dim ), dtype=self._dtype )


        # 4 Initialize cluster state
        np.copyto( self.center, self._array_centers[0,:] )
        np.copyto( self.radii, self._array_radii[0,:] )


## -------------------------------------------------------------------------------------------------
    def _update_cluster_state(self):

        # 1 Update cluster state
        if self._transition_counter == 0:
            np.copyto( self.center, self._array_centers[self._current_phase,:] )
            np.copyto( self.radii, self._array_radii[self._current_phase,:] )


        # 2 Update process control variables
        if self._transition_steps is not None:
            self._transition_counter += 1

            if self._transition_counter >= self._transition_steps[self._current_phase]:
                self._current_phase   = ( self._current_phase + 1 ) % len( self._transition_steps )
                self._transition_counter = 0


        # 3 Update center, radii, and size
        np.add( self.center, self._array_velocities[self._current_phase,:], out=self.center )
        np.add( self.radii, self._array_roc_of_radii[self._current_phase,:], out=self.radii )
        self.size += 1


## -------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:

        # 1 Prepare a new instance
        feature_data = Element( p_set = self.get_feature_space() )
        new_inst     = Instance( p_feature_data = feature_data, p_tstamp = self.tstamp )
        

        # 2 Generate a random point within the cluster
        v  = np.random.normal(size=self._num_dim)
        v /= np.linalg.norm(v)
        r  = np.random.rand() ** (1.0 / self._num_dim)

        feature_data.set_values( self.center + v * r * self.radii )


        # 3 Update cluster parameters
        self._update_cluster_state()


        # 4 Return new instance
        return new_inst
    




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MultiStreamGenCluster (MultiStreamGenerator):
    """
    Multi-stream generator combining multiple StreamGenCluster instances. Additionally, it provides
    functionality for rescaling the feature space boundaries and for injecting outliers.

    Parameters
    ----------
    p_num_dim : int
        Number of dimensions (features) of the generated instances.
    p_name : str = ''
        Name of the multi-stream generator. Default is ''.
    p_seed : int = 0
        Random seed for reproducibility.
    p_num_instances : int = 0
        Number of instances to generate per sub-stream. If set to 0, the streams are infinite. 
        Default is 0.
    p_sampler : Sampler = None
        Optional sampler object for sampling instances. Default is None.
    p_boundaries_rescale : list = None
        List of tuples specifying different (min, max) boundaries for each dimension to rescale the 
        generated values. If None, no rescaling is applied. Default is None.
    p_outlier_rate : float = 0.0
        Probability of generating an outlier instance (between 0.0 and 1.0). Default is 0.0 (no outliers).
    p_dtype : type, default: np.float32
        Data type of the feature values (np.float32 or np.float64). Default is np.float32. 
    p_logging : int, default: Log.C_LOG_NOTHING
        Logging level. See :class:`mlpro.bf.Log` for details. Default is Log.C_LOG_NOTHING. 

    Attributes    
    ----------
    num_outliers : int
        Number of outlier instances generated so far.
    cluster_statistics : ClusterStatistics
        Statistics about the generated clusters.
    """

    C_NAME = 'Cluster'    

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_num_dim : int,
                  p_name : str = '',
                  p_seed : int = 0,
                  p_num_instances : int = 0,
                  p_sampler : Sampler = None,
                  p_boundaries_rescale : list = None,
                  p_outlier_rate : float = 0.0,
                  p_dtype : type = np.float32,
                  p_logging = Log.C_LOG_ALL ):

        super().__init__( p_num_dim = p_num_dim,
                          p_name = p_name,
                          p_seed = p_seed,
                          p_num_instances = p_num_instances,
                          p_sampler = p_sampler,
                          p_boundaries_rescale = p_boundaries_rescale,
                          p_outlier_rate = p_outlier_rate,
                          p_dtype = p_dtype,
                          p_logging = p_logging )

        self._cluster_id   = 0
        self.cluster_statistics = ClusterStatistics( feature_boundaries = self._boundaries_rescale,
                                                     feature_rescale_params = self._rescaling_params )  


## -------------------------------------------------------------------------------------------------
    def add_stream( self, 
                    p_stream : Stream, 
                    p_batch_size : int = 1,
                    p_start_instance : int = 0 ):
        
        super().add_stream( p_stream = p_stream, 
                            p_batch_size = p_batch_size, 
                            p_start_instance = p_start_instance )  

        if isinstance(p_stream, StreamGenCluster):
            self.cluster_statistics.clusters[self._cluster_id] = p_stream
            self._cluster_id += 1
