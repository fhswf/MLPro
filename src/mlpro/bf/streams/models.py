## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.streams
## -- Module  : models.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-01-06  0.0.0     DA       Creation
## -- 2022-05-25  0.0.1     LSB      Minor bug fix
## -- 2022-06-02  0.1.0     LSB      Refactoring for list of stream objects in get stream list
## -- 2022-06-04  0.1.1     DA       Specialization in stream providers and streams
## -- 2022-06-09  0.1.2     LSB      Additional attributes to stream object
## -- 2022-06-14  0.1.3     LSB      Enhancement
## -- 2022-06-18  0.1.4     LSB      Logging of stream list based on p_display_list parameter
## -- 2022-06-19  0.1.5     DA       - Class Stream: internal use of self.C_NAME instead of self._name
## --                                - Check/completion of doc strings
## -- 2022-06-25  0.2.0     LSB      New Label class with modified instance class
## -- 2022-10-24  0.3.0     DA       Class Instance: new method copy()
## -- 2022-10-25  0.4.0     DA       New classes StreamTask, StreamWorkfllow, StreamScenario
## -- 2022-10-29  0.4.1     DA       Refactoring after introduction of module bf.ops
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.4.1 (2022-10-29)

Model classes for stream providers, streams, stream-based tasks/workflows/scenarios.
"""


from mlpro.bf.various import *
from mlpro.bf.ops import Mode, ScenarioBase
from mlpro.bf.plot import Plottable, PlotSettings
from mlpro.bf.math import *
from mlpro.bf.mt import Task, Workflow, Shared
from datetime import datetime
from matplotlib.figure import Figure




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Feature (Dimension): pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Label (Dimension): pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Instance:
    """
    Instance class to store the current instance and the corresponding labels of the stream

    Parameters
    ----------
    p_feature_data : Element
        feature data of the instance
    p_label_data : Element
        label data of the corresponding instance

    """

    C_TYPE          = 'Instance'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_feature_data:Element, p_label_data:Element = None, **p_kwargs):

        self._feature_data = p_feature_data
        self._label_data = p_label_data
        self._time_stamp = datetime.now()
        self._kwargs = p_kwargs.copy()


## -------------------------------------------------------------------------------------------------
    def get_feature_data(self) -> Element:
        return self._feature_data


## -------------------------------------------------------------------------------------------------
    def get_label_data(self) -> Element:
        return self._label_data


## -------------------------------------------------------------------------------------------------
    def get_time_stamp(self):
        return self._time_stamp


## -------------------------------------------------------------------------------------------------
    def get_kwargs(self):
        return self._kwargs


## -------------------------------------------------------------------------------------------------
    def copy(self):
        duplicate = self.__class__( p_feature_data=self._feature_data.copy(),
                                    p_label_data=self._label_data,
                                    p_kwargs=self._kwargs )
        duplicate._time_stamp = self._time_stamp
        return duplicate





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Stream (Mode, LoadSave, ScientificObject):
    """
    Template class for data streams.

    Parameters
    ----------
    p_id
        id of the stream
    p_name : str
        name of the stream
    p_num_instances : int
        Number of instances in the stream
    p_version : str
        Version of the stream
    p_mode
        Operation mode. Valid values are stored in constant C_VALID_MODES.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs
        Further stream specific parameters

    """

    C_TYPE          = 'Stream'

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_id=0,
                  p_name:str='',
                  p_num_instances:int=0,
                  p_version:str='',
                  p_mode=Mode.C_MODE_SIM,
                  p_logging=Log.C_LOG_ALL,
                  **p_kwargs):

        super().__init__(p_mode=p_mode, p_logging=p_logging)
        self._id = p_id
        self.C_NAME = self.C_SCIREF_TITLE = p_name
        self._num_instances = p_num_instances
        self._version = p_version
        self._kwargs = p_kwargs.copy()


## -------------------------------------------------------------------------------------------------
    def get_id(self) -> str:
        return self._id


## -------------------------------------------------------------------------------------------------
    def get_name(self) -> str:
        return self.C_NAME


## -------------------------------------------------------------------------------------------------
    def get_url(self) -> str:
        return self.C_SCIREF_URL


## -------------------------------------------------------------------------------------------------
    def get_num_features(self) -> int:
        return self._num_instances


## -------------------------------------------------------------------------------------------------
    def get_feature_space(self):
        return self.get_feature_space()


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
        self.log(self.C_LOG_TYPE_W, "\n\n")
        self.log(self.C_LOG_TYPE_W, "Resetting the stream")


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

    Parameters
    ----------
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL

    """

    C_TYPE          = 'Stream Provider'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def get_stream_list(self, p_logging = Log.C_LOG_ALL, **p_kwargs) -> list:
        """
        Gets a list of provided streams by calling custom method _get_stream_list().

        Parameters
        ----------
        p_display_list:bool
            boolean value to log the list of streams

        Returns
        -------
        stream_list : list
            List of provided streams.

        """
        stream_list = self._get_stream_list(p_logging = p_logging ,**p_kwargs)
        self.log(self.C_LOG_TYPE_I, "\n\n\n")
        self.log(self.C_LOG_TYPE_W, 'Getting list of streams...')
        for stream in stream_list:
            self.log(self.C_LOG_TYPE_I, "Stream ID: {:<15} Stream Name: {:<30}".format(stream.C_ID, stream.C_NAME))
        self.log(self.C_LOG_TYPE_I, 'Number of streams found:', len(stream_list),'\n\n\n')
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
            self.log(self.C_LOG_TYPE_E, 'Stream', str(p_id), 'not found\n')

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
class StreamTask (Task, Plottable):
    """
    Template class for stream-based tasks.

    Parameters
    ----------
    p_name : str
        Optional name of the task. Default is None.
    p_range_max : int
        Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_PROCESS.
    p_duplicate_data : bool     
        If True the incoming data are copied before processing. Otherwise the origin incoming data
        are modified.        
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : dict
        Further optional named parameters.
    """

    C_TYPE              = 'Stream-Task'

    C_PLOT_ACTIVE       = True
    C_PLOT_STANDALONE   = True
    C_PLOT_VALID_VIEWS  = [ PlotSettings.C_VIEW_2D, PlotSettings.C_VIEW_3D, PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW = PlotSettings.C_VIEW_ND

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name: str = None, 
                  p_range_max=Task.C_RANGE_THREAD, 
                  p_duplicate_data:bool=False,
                  p_logging=Log.C_LOG_ALL, 
                  **p_kwargs ):

        super().__init__( p_name=p_name, 
                          p_range_max=p_range_max, 
                          p_autorun=Task.C_AUTORUN_NONE, 
                          p_class_shared=None, 
                          p_logging=p_logging, 
                          **p_kwargs )

        self._duplicate_data = p_duplicate_data


## -------------------------------------------------------------------------------------------------
    def run(self, p_inst_new:list, p_inst_del:list, p_range:int = None, p_wait: bool = False):
        """
        Executes the task specific actions implemented in custom method _run(). At the end event
        C_EVENT_FINISHED is raised to start subsequent actions (p_wait=True).

        Parameters
        ----------
        p_inst_new : list
            List of new stream instances to be processed.
        p_inst_del : list
            List of obsolete stream instances to be removed.
        p_range : int
            Optional deviating range of asynchonicity. See class Range. Default is None what means that the maximum
            range defined during instantiation is taken. Oterwise the minimum range of both is taken.
        p_wait : bool
            If True, the method waits until all (a)synchronous tasks are finished.
        p_kwargs : dict
            Further parameters handed over to custom method _run().
        """

        if self._duplicate_data:
            inst_new = [ inst.copy() for inst in p_inst_new ] 
            inst_del = [ inst.copy() for inst in p_inst_del ]
        else:
            inst_new = p_inst_new
            inst_del = p_inst_del

        super().run(p_range=p_range, p_wait=p_wait, p_inst_new=inst_new, p_inst_del=inst_del)


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new:list, p_inst_del:list):
        """
        Custom method that is called by method run(). 

        Parameters
        ----------
        p_inst_new : list
            List of new stream instances to be processed.
        p_inst_del : list
            List of obsolete stream instances to be removed.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Default implementation for online adaptive tasks. See class mlpro.bf.plot.Plottable for more
        details.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Default implementation for online adaptive tasks. See class mlpro.bf.plot.Plottable for more
        details.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Default implementation for online adaptive tasks. See class mlpro.bf.plot.Plottable for more
        details.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def update_plot(self, p_inst_new:list, p_inst_del:list, **p_kwargs):
        """
        Specialized definition of method update_plot() of class mlpro.bf.plot.Plottable.

        Parameters
        ----------
        p_inst_new : list
            List of new stream instances to be plotted.
        p_inst_del : list
            List of obsolete stream instances to be removed.
        p_kwargs : dict
            Further optional plot parameters.
        """

        return super().update_plot(p_inst_new=p_inst_new, p_inst_del=p_inst_del, **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_output: bool, p_settings: PlotSettings, p_inst_new:list, p_inst_del:list, **p_kwargs):
        """
        Default implementation for online adaptive tasks. See class mlpro.bf.plot.Plottable for more
        details.

        Parameters
        ----------
        p_output : bool
            If True, the plot output shall be carried out.  
        p_settings : PlotSettings
            Object with further plot settings.
        p_inst_new : list
            List of new stream instances to be plotted.
        p_inst_del : list
            List of obsolete stream instances to be removed.
        p_kwargs : dict
            Further optional plot parameters.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_output: bool, p_settings: PlotSettings, p_inst_new:list, p_inst_del:list, **p_kwargs):
        """
        Default implementation for online adaptive tasks. See class mlpro.bf.plot.Plottable for more
        details.

        Parameters
        ----------
        p_output : bool
            If True, the plot output shall be carried out.  
        p_settings : PlotSettings
            Object with further plot settings.
        p_inst_new : list
            List of new stream instances to be plotted.
        p_inst_del : list
            List of obsolete stream instances to be removed.
        p_kwargs : dict
            Further optional plot parameters.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_output: bool, p_settings: PlotSettings, p_inst_new:list, p_inst_del:list, **p_kwargs):
        """
        Default implementation for online adaptive tasks. See class mlpro.bf.plot.Plottable for more
        details.

        Parameters
        ----------
        p_output : bool
            If True, the plot output shall be carried out.  
        p_settings : PlotSettings
            Object with further plot settings.
        p_inst_new : list
            List of new stream instances to be plotted.
        p_inst_del : list
            List of obsolete stream instances to be removed.
        p_kwargs : dict
            Further optional plot parameters.
        """

        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamWorkflow (Workflow, Plottable):
    """
    Workflow for stream processing. See class bf.mt.Workflow for further details.

    Parameters
    ----------
    p_name : str
        Optional name of the task. Default is None.
    p_range_max : int
        Range of asynchonicity. See class Range. Default is Range.C_RANGE_THREAD.
    p_class_shared
        Optional class for a shared object (class Shared or a child class of Shared)
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : dict
        Further optional named parameters handed over to every task within.
    """

    C_TYPE      = 'Stream-Workflow'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name: str = None, 
                  p_range_max=Workflow.C_RANGE_THREAD, 
                  p_class_shared=Shared, 
                  p_logging=Log.C_LOG_ALL, 
                  **p_kwargs ):

        Workflow.__init__( self,
                           p_name=p_name, 
                           p_range_max=p_range_max, 
                           p_class_shared=p_class_shared, 
                           p_logging=p_logging, 
                           **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def run( self, p_inst:Instance, p_range: int = None, p_wait: bool = False ):
        """
        Runs all stream tasks according to their predecessor relations.

        Parameters
        ----------
        p_inst : Instance
            Single stream instance to process.
        p_range : int
            Optional deviating range of asynchonicity. See class Range. Default is None what means that 
            the maximum range defined during instantiation is taken. Oterwise the minimum range of both 
            is taken.
        p_wait : bool
            If True, the method waits until all (a)synchronous tasks are finished.
        """

        super().run(p_range=p_range, p_wait=p_wait, p_inst=p_inst)                          


## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Default implementation for stream tasks. See class mlpro.bf.plot.Plottable for more
        details.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Default implementation for stream tasks. See class mlpro.bf.plot.Plottable for more
        details.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Default implementation for stream tasks. See class mlpro.bf.plot.Plottable for more
        details.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def update_plot(self, p_inst_new:list, p_inst_del:list, **p_kwargs):
        """
        Specialized definition of method update_plot() of class mlpro.bf.plot.Plottable.

        Parameters
        ----------
        p_inst_new : list
            List of new stream instances to be plotted.
        p_inst_del : list
            List of obsolete stream instances to be removed.
        p_kwargs : dict
            Further optional plot parameters.
        """

        return super().update_plot(p_inst_new=p_inst_new, p_inst_del=p_inst_del, **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings: PlotSettings, p_inst_new:list, p_inst_del:list, **p_kwargs):
        """
        Default implementation for stream tasks. See class mlpro.bf.plot.Plottable for more
        details.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        p_inst_new : list
            List of new stream instances to be plotted.
        p_inst_del : list
            List of obsolete stream instances to be removed.
        p_kwargs : dict
            Further optional plot parameters.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings: PlotSettings, p_inst_new:list, p_inst_del:list, **p_kwargs):
        """
        Default implementation for stream tasks. See class mlpro.bf.plot.Plottable for more
        details.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        p_inst_new : list
            List of new stream instances to be plotted.
        p_inst_del : list
            List of obsolete stream instances to be removed.
        p_kwargs : dict
            Further optional plot parameters.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings: PlotSettings, p_inst_new:list, p_inst_del:list, **p_kwargs):
        """
        Default implementation for stream tasks. See class mlpro.bf.plot.Plottable for more
        details.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        p_inst_new : list
            List of new stream instances to be plotted.
        p_inst_del : list
            List of obsolete stream instances to be removed.
        p_kwargs : dict
            Further optional plot parameters.
        """

        pass





# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
class StreamScenario (ScenarioBase): 
    """
    Template class for stream based scenarios.

    Parameters
    ----------
    p_mode
        Operation mode. See Mode.C_VALID_MODES for valid values. Default = Mode.C_MODE_SIM.
    p_cycle_limit : int
        Maximum number of cycles. Default = 0 (no limit).
    p_visualize 
        Boolean switch for env/agent visualisation. Default = True.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.  
    """
    
    C_TYPE      = 'Stream-Scenario'

# -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_mode, 
                  p_cycle_limit=0, 
                  p_visualize: bool = True, 
                  p_logging=Log.C_LOG_ALL ):

        self._stream : Stream           = None
        self._workflow : StreamWorkflow = None

        super().__init__( p_mode, 
                          p_cycle_limit=p_cycle_limit, 
                          p_auto_setup=True, 
                          p_visualize=p_visualize, 
                          p_logging=p_logging )


# -------------------------------------------------------------------------------------------------
    def setup(self, p_mode, p_logging=Log.C_LOG_ALL):
        """
        Specialized method to set up a stream scenario. It is automatically called by the constructor
        and calls in turn the custom method _setup().

        Parameters
        ----------
        p_mode
            Operation mode. See Mode.C_VALID_MODES for valid values. Default = Mode.C_MODE_SIM.
        p_logging
            Log level (see constants of class Log). Default: Log.C_LOG_ALL.  
        """

        self._stream, self._workflow = self._setup(p_mode=p_mode, p_logging=Log.C_LOG_ALL)


# -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_logging):
        """
        Custom method to set up a stream scenario consisting of a stream and a processing stream
        worflow.

        Parameters
        ----------
        p_mode
            Operation mode. See Mode.C_VALID_MODES for valid values. Default = Mode.C_MODE_SIM.
        p_logging
            Log level (see constants of class Log). Default: Log.C_LOG_ALL.  

        Returns
        -------
        stream : Stream
            A stream object.
        workflow : StreamWorkflow
            A stream workflow object.
        """

        raise NotImplementedError


# -------------------------------------------------------------------------------------------------
    def _set_mode(self, p_mode):
        self._stream.set_mode(p_mode=p_mode)


# -------------------------------------------------------------------------------------------------
    def _run_cycle(self):
        """
        Gets next instance from the stream and lets process it by the stream workflow.

        Returns
        -------
        success : bool
            True on success. False otherwise.
        error : bool
            True on error. False otherwise.
        adapted : bool
            True, if something within the scenario has adapted something in this cycle. False otherwise.
        """

        self._workflow.run( p_inst=self._stream.get_next() )
        return True, False, False