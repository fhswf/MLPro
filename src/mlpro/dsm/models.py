## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.dsm
## -- Module  : models.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-01-06  0.0.0     DA       Creation
## -- 2022-mm-dd  1.0.0     DA       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2022-01-06)

Model classes for efficient online adaptive data stream processing.
"""


#from time import CLOCK_THREAD_CPUTIME_ID
#from itertools import combinations_with_replacement
from mlpro.bf.various import *
from mlpro.bf.ml import *
from mlpro.bf.math import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Feature (Dimension): pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Instance (Element): pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamReference (ScientificObject):

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_id) -> None:
        self._id = p_id


## -------------------------------------------------------------------------------------------------
    def get_id(self) -> str:
        return self._id


## -------------------------------------------------------------------------------------------------
    def get_name(self) -> str:
        return self.C_SCIREF_TITLE


## -------------------------------------------------------------------------------------------------
    def get_url(self) -> str:
        return self.C_SCIREF_URL





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Stream (Mode, LoadSave, ScientificObject):

    """
    Template class for data streams.

    Parameters
    ----------
    p_mode
        Operation mode. Valid values are stored in constant C_VALID_MODES.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL

    """

    C_TYPE          = 'Stream'
    C_NAME          = '????'

## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                 p_mode=Mode.C_MODE_SIM, 
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs ):

        super().__init__(p_mode=p_mode, p_logging=p_logging)
        self._kwargs = p_kwargs.copy()
        self._feature_space = self.setup()


## -------------------------------------------------------------------------------------------------
    def setup(self) -> MSpace:
        """
        Sets up the data stream and specially the related feature space by calling custom method
        _setup().

        Returns
        -------
        feature_space : MSpace
            The feature space of the data stream.

        """

        return self._setup()


## -------------------------------------------------------------------------------------------------
    def _setup(self) -> MSpace:
        """
        Custom method to set up the data stream and related feature space. See method setup() for
        more details. Use class Feature to define the feature space.

        Returns
        -------
        feature_space : MSpace
            The feature space of the data stream.

        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_feature_space(self):
        return self._feature_space


## -------------------------------------------------------------------------------------------------
    def reset(self, p_seed=None):
        """
        Resets stream generator and initializes an internal random generator with the given seed
        value by calling the custom method _reset().

        Parameters
        ----------
        p_seed : int
            Seed value for random generator.

        """

        self._reset(p_seed=p_seed)


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):
        """
        Custom reset method for data stream. See method reset() for more details.

        Parameters
        ----------
        p_seed : int
            Seed value for random generator.

        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_next(self) -> Instance:
        """
        Returns next data stream instance or None at the end of the stream. The next instance is
        determined by calling the custom method _get_next().

        Returns
        -------
        instance : Instance
            Next instance of data stream or None.

        """

        return self._get_next()


## -------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:
        """
        Custom method to determine the next data stream instance. See method get_next() for more
        details.

        Returns
        -------
        instance : Instance
            Next instance of data stream or None.
            
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamProvider (Log, ScientificObject):
    """
    Template class for stream providers.
    """

    C_TYPE          = 'Stream Provider'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def get_stream_list(self, **p_kwargs) -> list:
        """
        Gets a list of provided streams by calling custom method _get_stream_list().

        Returns
        -------
        stream_list : list
            List of provided streams.

        """

        self.log(self.C_LOG_TYPE_I, 'Getting list of streams...')
        stream_list = self._get_stream_list(p_kwargs)
        self.log(self.C_LOG_TYPE_I, 'Stream found:', len(stream_list))
        return stream_list


## -------------------------------------------------------------------------------------------------
    def _get_stream_list(self, **p_kwargs) -> list:
        """
        Custom method to get the list of provided streams. See method get_stream_list() for further
        details.

        Returns
        -------
        stream_list : list
            List of provided streams.

        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_stream(self, p_id) -> Stream:
        """
        Returns stream with the specified id by calling custom method _get_stream().

        Parameters
        ----------
        p_id : str
            Id of the requested stream.

        Returns
        -------
        s : Stream
            Stream object or None in case of an error.

        """

        self.log(self.C_LOG_TYPE_I, 'Requested stream:', str(p_id)) 
        s = self._get_stream(p_id)
        if s is None:
            self.log(self.C_LOG_TYPE_E, 'Stream', str(p_id), 'not found') 

        return s


## -------------------------------------------------------------------------------------------------
    def _get_stream(self, p_id) -> Stream:
        """
        Custom method to get the specified stream. See method get_stream() for further details.

        Parameters
        ----------
        p_id : str
            Id of the requested stream.

        Returns
        -------
        s : Stream
            Stream object or None in case of an error.

        """

        raise NotImplementedError 





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PreProStep (Model):
    """
    Template class for an adaptive data stream preprocessing step.

    Parameters
    ----------
    p_buffer_size : int
        Initial size of internal data buffer. Defaut = 0 (no buffering).
    p_input_space : MSpace
        Optional fixed input feature space. Default = None.
    p_output_space : MSpace
        Optional fixed output feature space. Default = None.
    p_ada : bool
        Boolean switch for adaptivitiy. Default = True.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : Dict
        Futher model specific parameters (to be defined in chhild class).

    """

    C_TYPE          = 'Prepro Step'

## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                 p_buffer_size=0,
                 p_input_space:MSpace=None,
                 p_output_space:MSpace=None,
                 p_ada=True,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        super().__init__( p_buffer_size=p_buffer_size,
                          p_ada=p_ada,
                          p_logging=p_logging,
                          p_par=p_kwargs )

        self._input_space   = p_input_space
        self._output_space  = p_output_space


## -------------------------------------------------------------------------------------------------
    def get_input_space(self) -> MSpace:
        return self._input_space


## -------------------------------------------------------------------------------------------------
    def get_output_space(self) -> MSpace:
        return self._output_space


## -------------------------------------------------------------------------------------------------
    def process(self, p_in_add, p_in_del): 
        """
        Processes new/obsolete instances of a data stream by calling the custom methods 
        _process_before(), adapt() and _process_after(). 

        Parameters
        ----------
        p_in_add : list     
            List of new instances.
        p_in_del : list     
            List of obsolete instances.

        """

        self.log(self.C_LOG_TYPE_I, 'Start processing:', len(p_in_add), 'new and', len(p_in_del), 'obsolete instaces')
        self._process_before(p_in_add, p_in_del)
        self.adapt(p_in_add, p_in_del)
        self._process_after(p_in_add, p_in_del)
        self.log(self.C_LOG_TYPE_I, 'End processing')


## -------------------------------------------------------------------------------------------------
    def _process_before(self, p_in_add, p_in_del):
        """
        Custom method for process steps before adaptation. See method process() for further details.

        Parameters
        ----------
        p_in_add : list     
            List of new instances.
        p_in_del : list     
            List of obsolete instances.

        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_in_add, p_in_del) -> bool:
        """
        Custom adaptation method. See method process() for further details.

        Parameters
        ----------
        p_in_add : list     
            List of new instances.
        p_in_del : list     
            List of obsolete instances.

        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _process_after(self, p_in_add, p_in_del):
        """
        Custom method for process steps after adaptation. See method process() for further details.

        Parameters
        ----------
        p_in_add : list     
            List of new instances.
        p_in_del : list     
            List of obsolete instances.

        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Preprocessor (Model):
    """
    Adaptive data stream preprocessor.

    Parameters
    ----------
    p_ada : bool
        Boolean switch for adaptivitiy. Default = True.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    
    """

    C_TYPE          = 'Preprocessor'
    C_NAME          = ''

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_ada=True, p_logging=Log.C_LOG_ALL):
        super().__init__(p_buffer_size=0, p_ada=p_ada, p_logging=p_logging)
        self._prepro_steps  = []
        self._input_space   = None
        self._output_space  = None


## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging):
        super().switch_logging(p_logging)
        for step in self._prepro_steps:
            step.switch_logging(p_logging)


## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada: bool):
        super().switch_adaptivity(p_ada)
        for step in self._prepro_steps:
            step.switch_adaptivity(p_ada)


## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        for step in self._prepro_steps:
            step.set_random_seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def get_adapted(self) -> bool:
        adapted = False
        for step in self._prepro_steps:
            if step.get_adapted():
                adapted = True
                break

        return adapted
        

## -------------------------------------------------------------------------------------------------
    def add_prepro_step(self, p_step:PreProStep):
        if len(self._prepro_steps) == 0: self._input_space = p_step.get_input_space()
        self._output_space = p_step.get_output_space()
        self._prepro_steps.append(p_step)


## -------------------------------------------------------------------------------------------------
    def process(self, p_in_add, p_in_del):
        self.log(self.C_LOG_TYPE_I, 'Start preprocessing')

        for step in self._prepro_steps:
            self.log(self.C_LOG_TYPE_I, 'Start processing step', step.C_TYPE, self.C_NAME)
            step.process( p_in_add, p_in_del )

        self.log(self.C_LOG_TYPE_I, 'End preprocessing')


# -------------------------------------------------------------------------------------------------
    def adapt(self, *p_args) -> bool: 
        """
        No adaptation steps at this level.
        """

        return False


# -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
       for step in self._prepro_steps:
            step.clear_buffer()


# -------------------------------------------------------------------------------------------------
    def get_maturity(self) -> float:
        """
        Calculates the maturity as mean value of the maturities of all preprocessing steps.

        Returns
        -------
        mean_maturity : float
            Mean maturity of preprocessing steps.

        """

        if len(self._prepro_steps) == 0: return 0

        maturity = 0
        for step in self._prepro_steps:
            maturity += step.get_maturity()

        return maturity / len(self._prepro_steps)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DSMApp: pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Cluster: pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterAnalyzer (DSMApp): pass





# ## -------------------------------------------------------------------------------------------------
# ## -------------------------------------------------------------------------------------------------
# class StreamProcessor(ProcessingStep):
#     """
#     Model class for sequential adaptive stream processing with optional data preprocessing. Owhn 
#     policy adaption and processing steps can be implemented by redefining methods adapt_policy() and
#     process_custom().
#     """

#     C_TYPE          = 'Stream Processor'

# ## -------------------------------------------------------------------------------------------------
#     def __init__(self, p_ada=True, p_logging=Log.C_LOG_ALL):
#         super().__init__(p_buffer_size=0, p_ada=p_ada, p_logging=p_logging)
#         self.prepro_steps   = []
 

# ## -------------------------------------------------------------------------------------------------
#     def switch_logging(self, p_logging):
#         Log.switch_logging(self, p_logging=p_logging)
#         for step in self.prepro_steps:
#             step.switch_logging(p_logging=p_logging)


# ## -------------------------------------------------------------------------------------------------
#     def add_prepro_step(self, p_step:ProcessingStep):
#         """
#         Adds a preprocessing step.

#         Parameters:
#             p_step      Preprocessing step object to be added
 
#         Returns: 
#             Nothing
#         """

#         p_step.set_adaptivity(self.adaptivity)
#         p_step.switch_logging(p_logging=self.logging)
#         self.prepro_steps.append(p_step)


# ## -------------------------------------------------------------------------------------------------
#     def process(self, p_x):
#         """
#         Processes input in three phases: at first all preprocessing steps will be executed. After that
#         the own policy will be adapted and at last the own process steps will be executed.

#         Parameters:
#             p_x         Input vector x

#         Returns: 
#             Nothing
#         """

#         # 0 Intro
#         x_add   = []
#         x_del   = []
#         x_add.append(p_x)
#         self.log(self.C_LOG_TYPE_I, 'Start processing of input ', p_x)

#         # 1 Preprocessing
#         if len(self.prepro_steps) > 0:
#             self.log(self.C_LOG_TYPE_I, 'Start of preprocessing')

#             for step_id, step in enumerate(self.prepro_steps): 
#                 self.log(self.C_LOG_TYPE_I, 'Preprocessing step ' + str(step_id) + ': ' + step.C_TYPE + ' ' + step.C_NAME)
#                 step.process_step(x_add, x_del)

#             self.log('End of preprocessing')

        
#         # 2 Adaption of own policy and main processing
#         self.process_step(x_add, x_del)





# ## -------------------------------------------------------------------------------------------------
# ## -------------------------------------------------------------------------------------------------
# class StreamProcess(Log):
#     """
#     Stream process, consisting of stream and stream processor object.
#     """

#     C_TYPE      = 'Stream Process'
#     C_NAME      = ''

# ## -------------------------------------------------------------------------------------------------
#     def __init__(self, p_stream:Stream, p_sproc:StreamProcessor, p_logging=True):
#         """
#         Parameters:
#             p_stream        Stream object
#             p_sproc         Stream processor object
#             p_logging       Boolean switch for logging
#         """

#         super().__init__(p_logging=p_logging)
#         self.stream = p_stream
#         self.sproc  = p_sproc

#         self.log(self.C_LOG_TYPE_I, 'Stream', self.stream.C_NAME, ' registered')
#         self.log(self.C_LOG_TYPE_I, self.sproc.C_TYPE, self.sproc.C_NAME, ' registered')


# ## -------------------------------------------------------------------------------------------------
#     def run(self, p_inst_limit=0, p_feature_ids=None):
#         """
#         Reads and processes all/limited number of stream instances.

#         Parameters:
#             p_inst_limit    Optional limitation of instances.
#             p_feature_ids   Optional list of ids of features to be processed

#         Returns:
#             Number of processed instances.
#         """

#         # 1 Intro
#         self.log(self.C_LOG_TYPE_I, 'Start of stream processing (limit='+ str(p_inst_limit) + ')')
#         num_inst = 0


#         # 2 Main processing loop
#         while True:
#             inst = self.stream.get_next()
#             if inst == None:
#                 self.log(self.C_LOG_TYPE_I, 'Stream limit reached') 
#                 break

#             num_inst += 1
#             self.sproc.process(inst)
#             self.log(self.C_LOG_TYPE_I, 'Instance', inst, 'processed')
#             if ( p_inst_limit > 0 ) and ( num_inst == p_inst_limit ): break          


#         # 3 Outro
#         self.log(self.C_LOG_TYPE_I, 'End of stream processing (' + str(num_inst) + ' instances)')
#         return num_inst





# ## -------------------------------------------------------------------------------------------------
# ## -- Class Group: Special types of preprocessing steps
# ## -------------------------------------------------------------------------------------------------


# ## -------------------------------------------------------------------------------------------------
# ## -------------------------------------------------------------------------------------------------
# class DataWindow (ProcessingStep):
#     """
#     Model class for data windows that can be used to deal with concept drift. 
#     """

#     C_TYPE      = 'Data Window'





# ## -------------------------------------------------------------------------------------------------
# ## -------------------------------------------------------------------------------------------------
# class Normalization (ProcessingStep):
#     """
#     Model class for adaptive normalization algorithms. 
#     """

#     C_TYPE      = 'Normalization'

# ## -------------------------------------------------------------------------------------------------
#     def process_before(self, p_x_add, p_x_del):
#         self.backup_policy()


# ## -------------------------------------------------------------------------------------------------
#     def process_after(self, p_x_add, p_x_del):
#         for x in p_x_add:
#             x = self.normalize(x)


# ## -------------------------------------------------------------------------------------------------
#     def normalize(self, p_x):
#         """
#         Normalizes an input vector - either by using the recent or the backup policy. To be redefined.

#         Parameters:
#             p_x         Input vector x to be denormalized
        
#         Returns:
#             Normalized input vector.
#         """
        
#         pass   


# ## -------------------------------------------------------------------------------------------------
#     def denormalize(self, p_x, p_backup=True):
#         """
#         Denormalizes an input - either by using the recent or the backup policy. To be redefined.

#         Parameters:
#             p_x         Input vector x to be denormalized
#             p_backup    If True, the backup policy shall be used. Recent policy otherwise.

#         Returns:
#             Denormalized input vector.    
#         """
        
#         pass


# ## -------------------------------------------------------------------------------------------------
#     def renormalize(self, p_x):
#         """
#         Reormalizes an input vector by denormalizing it with the backup policy and normalizing it
#         with the recent policy after that.

#         Parameters:
#             p_x         Input vector x to be denormalized

#         Returns:
#             Reormalized input vector.    
#         """

#         return self.normalize(self.denormalize(p_x, p_backup=True))



# ## -------------------------------------------------------------------------------------------------
#     def backup_policy(self):
#         """
#         Backups the recent policy. To be redefined.
#         """

#         pass




# ## -------------------------------------------------------------------------------------------------
# ## -- Class Group: Special types of stream processing applications
# ## -------------------------------------------------------------------------------------------------


