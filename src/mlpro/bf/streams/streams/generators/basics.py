## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.streams.streams.generators
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-09-21  1.0.0     DA       Creation 
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-09-21)

This module provides template classes for single and multi-stream data generation in a d-dimensional 
feature space.

"""

import numpy as np

from mlpro.bf import Log, Mode
from mlpro.bf.events import Event, EventManager
from mlpro.bf.exceptions import ParamError
from mlpro.bf.math import Element, MSpace, ESpace
from mlpro.bf.streams import Feature, Instance, Stream, Sampler, MultiStream



# Export list for public API
__all__ = [ 'StreamGenerator', 'MultiStreamGenerator' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamGenerator (Stream, EventManager):
    """
    Template class for stream data generation. It introduces a d-dimensional feature space 
    [-1000, 1000]^d provides methods for generating random instances. Further features are:
    - Optional rescaling of the generated feature values to different boundaries per dimension.
    - Optional generation of outlier instances with a given probability.
    - Event generation when an outlier instance is created.

    Parameters
    ----------
    p_num_dim : int
        Number of dimensions (features) of the generated instances.
    p_id : int = 0
        ID of the stream generator.
    p_seed : int = 0
        Random seed for reproducibility.
    p_num_instances : int = 0
        Number of instances to generate. If set to 0, the stream is infinite. Default is 0.
    p_boundaries_rescale : list = None
        List of tuples specifying different (min, max) boundaries for each dimension to rescale the 
        generated values. If None, no rescaling is applied. Default is None.
    p_outlier_rate : float = 0.0
        Probability of generating an outlier instance (between 0.0 and 1.0). Default is 0.0 (no outliers).
    p_sampler : Sampler = None
        Optional sampler object for sampling instances. Default is None.
    p_dtype : type, default: np.float32
        Data type of the feature values (np.float32 or np.float64).
    p_logging : int, default: Log.C_LOG_NOTHING
        Logging level. See :class:`mlpro.bf.Log` for details.
    **p_kwargs : dict
        Additional keyword arguments for child classes.

    Attributes
    ----------
    num_outliers : int
        Number of outlier instances generated so far.
    """

    C_TYPE              = 'Stream Generator'
    C_BOUNDARIES        = [-1000, 1000]        # Default boundaries of the feature space
    C_EVENT_ID_OUTLIER  = 'Outlier'
  
## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_num_dim : int,
                  p_id = 0,
                  p_seed : int = 0,
                  p_num_instances : int = 0,
                  p_boundaries_rescale : list = None,
                  p_outlier_rate : float = 0.0,
                  p_sampler : Sampler = None,
                  p_dtype : type = np.float32,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):

        self._num_dim = p_num_dim
        if self._num_dim < 1:
            raise ParamError("Number of dimensions must be at least 1.")

        self._seed  = p_seed
        self._dtype = p_dtype
        

        # Optional explicit boundaries per dimension and resulting rescaling parameters
        if p_boundaries_rescale is not None:
            if len(p_boundaries_rescale) != self._num_dim:
                raise ParamError(f"Expected {self._num_dim} dimensions for feature boundaries.")
            self._boundaries_rescale    = p_boundaries_rescale
        else:
            self._boundaries_rescale    = [self.C_BOUNDARIES]*self._num_dim

        self._rescaling_params = self._get_rescaling_params(self._boundaries_rescale)


        # Outlier generation
        self._outlier_appearance = p_outlier_rate > 0.0
        self._outlier_rate       = p_outlier_rate
        self.num_outliers        = 0


        EventManager.__init__( self, p_logging = Log.C_LOG_NOTHING)

        Stream.__init__( self,
                         p_id = p_id,
                         p_num_instances = p_num_instances,
                         p_feature_space = None,
                         p_label_space = None,
                         p_sampler = p_sampler,
                         p_mode = Mode.C_MODE_SIM,
                         p_logging = p_logging,
                         **p_kwargs )        
        
        self.get_feature_space()
        self.get_label_space()


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
    def set_random_seed(self, p_seed=None):
        self._rgen = np.random.default_rng( seed = p_seed )
        

## -------------------------------------------------------------------------------------------------
    def _gen_rnd_array(self, p_low, p_high, p_size) -> np.array:
        return self._rgen.uniform( low=p_low, high=p_high, size=p_size ).astype( self._dtype )


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
    def _reset(self):
        self.set_random_seed( self._seed )
        self.num_outliers = 0
    

## -------------------------------------------------------------------------------------------------
    def __next__(self) -> Instance:

        raise_outlier = False

        if self._outlier_appearance and np.random.rand() < self._outlier_rate:
            # 1 Generate outlier instance
            outlier_data        = Element( p_set = self.get_feature_space() )
            outlier_values      = self._gen_rnd_array( self.C_BOUNDARIES[0], self.C_BOUNDARIES[1], self._num_dim )
            outlier_data.set_values( outlier_values )
            new_inst            = Instance( p_feature_data = outlier_data, p_tstamp = self.tstamp)
            new_inst.id         = self._next_inst_id
            self._next_inst_id += 1
       
            raise_outlier       = True
            self.num_outliers  += 1

        else:
            # 2 Generate normal instance
            new_inst = super().__next__()


        # 3 Rescale feature values if required
        if self._rescaling_params is not None:
            feature_values = new_inst.feature_values
            new_inst.feature_values = feature_values * self._rescaling_params[:,0] + self._rescaling_params[:,1]


        # 4 Optionally raise an outlier
        if raise_outlier:
            self._raise_event(p_event_id = self.C_EVENT_ID_OUTLIER,
                              p_event_object = Event(p_raising_object= self,
                                                     p_tstamp = new_inst.tstamp,
                                                     p_instance = new_inst ))

        return new_inst
    




# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
class MultiStreamGenerator (MultiStream, StreamGenerator):
    """
    Template class for multi-stream data generation. It combines multiple single StreamGenerator 
    instances to provide a multi-stream interface.

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
    """

    C_TYPE = 'Multi-stream Generator'

# -------------------------------------------------------------------------------------------------
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
        
        MultiStream.__init__( self,
                              p_id = 0,
                              p_name = p_name,
                              p_num_instances = p_num_instances,
                              p_sampler = p_sampler,
                              p_mode = Mode.C_MODE_SIM,
                              p_logging = Log.C_LOG_NOTHING )
        
        StreamGenerator.__init__( self,
                                  p_num_dim = p_num_dim,
                                  p_id = 0,
                                  p_seed = p_seed,
                                  p_num_instances = p_num_instances,
                                  p_boundaries_rescale = p_boundaries_rescale,
                                  p_outlier_rate = p_outlier_rate,
                                  p_sampler = p_sampler,
                                  p_dtype = p_dtype,
                                  p_logging = p_logging )
        

# -------------------------------------------------------------------------------------------------
    def _setup_feature_space(self):
        return StreamGenerator._setup_feature_space(self)
    

# -------------------------------------------------------------------------------------------------
    def _setup_label_space(self):
        return StreamGenerator._setup_label_space(self)
    

# -------------------------------------------------------------------------------------------------
    def _reset(self):
        self.set_random_seed(self._seed)
        MultiStream._reset(self)


# -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        StreamGenerator.set_random_seed( self, p_seed = p_seed )