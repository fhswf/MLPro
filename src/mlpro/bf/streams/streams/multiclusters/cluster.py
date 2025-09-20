## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.streams.streams.clusters
## -- Module  : cluster.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-09-20  1.0.0     DA       Creation 
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-09-20)

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
from dataclasses import dataclass
import random

import numpy as np

from mlpro.bf import Log, Mode
from mlpro.bf.exceptions import ParamError
from mlpro.bf.math import Element, MSpace, ESpace
from mlpro.bf.streams import Feature, Instance, Stream



# Export list for public API
__all__ = [ 'ClusterState', 'StreamCluster' ]



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
class StreamCluster(Stream):
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
        defined by p_durations. If p_durations is None, the cluster will stay in the first state for all instances.
    p_durations : List[int], default: None
        List of durations (number of instances) for each state transition. The length of p_durations must be
        equal to len(p_states) - 1. If None, the cluster will stay in the first state for all instances.
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
    """

    C_NAME              = 'Cluster'
    C_RADII_RND_FACTOR  = 0.2           # Factor for random radii generation within boundary range
    C_BOUNDARIES        = [-1000, 1000] # Boundaries of the feature space
  
## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_num_dim : int,
                  p_id : int = 0,
                  p_seed : int = 0,
                  p_num_instances : int = 0,
                  p_states : List[ClusterState] = [ ClusterState() ],
                  p_durations : List[int] = None,
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
            if p_durations is None or ( not isinstance(p_durations, list) or len(p_durations) != len(p_states) - 1 ):
                raise ParamError('If multiple states are provided in p_states, p_durations must be a list of length len(p_states) - 1!')
        
        
        # 2 Init all attributes  
        self._num_dim                         = p_num_dim
        self._states                          = p_states
        self._durations                       = p_durations
        self._seed                            = p_seed
        self._dtype                           = p_dtype
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
        super().__init__( p_id = p_id,
                          p_num_instances = p_num_instances,
                          p_feature_space = self._setup_feature_space(),
                          p_mode = Mode.C_MODE_SIM,
                          p_logging = p_logging )


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
    def set_random_seed(self, p_seed=None):
        self._rgen = np.random.default_rng( seed = p_seed )
        

## -------------------------------------------------------------------------------------------------
    def _gen_rnd_array(self, p_low, p_high, p_size) -> np.array:
        return self._rgen.uniform( low=p_low, high=p_high, size=p_size ).astype( self._dtype )
    

## -------------------------------------------------------------------------------------------------
    def _reset(self):

        # 1 Cluster size and process variables
        self.size                     = 0
        self._current_phase : int  = 0
        self._transition_counter      = 0

        self.set_random_seed( self._seed )


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
        if self._durations is not None:
            if self._array_velocities is None:
                self._array_velocities    = np.empty( shape = ( len(self._durations), self._num_dim ), dtype=self._dtype )
                self._array_roc_of_radii  = np.empty( shape = ( len(self._durations), self._num_dim ), dtype=self._dtype )

            for i, duration in enumerate(self._durations):
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
        if self._durations is not None:
            self._transition_counter += 1

            if self._transition_counter >= self._durations[self._current_phase]:
                self._current_phase   = ( self._current_phase + 1 ) % len( self._durations )
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