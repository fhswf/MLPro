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
## -- 2022-10-31  0.4.2     DA       Refactoring after changes on bf.mt
## -- 2022-11-03  0.5.0     DA       - Class Instance: completion of constructor
## --                                - Class Stream: extensions and corrections
## --                                - Completion of doc strings 
## -- 2022-11-04  0.6.0     DA       Classes StreamProvider, Stream: refactoring
## -- 2022-11-05  0.7.0     DA       Class Stream: refactoring to make it iterable
## -- 2022-11-07  0.7.1     DA       Class StreamScenario: refactoring 
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.7.1 (2022-11-07)

This module provides classes for standardized stream processing. 
"""


from mlpro.bf.math.basics import MSpace
from mlpro.bf.various import *
from mlpro.bf.ops import Mode, ScenarioBase
from mlpro.bf.plot import PlotSettings
from mlpro.bf.math import Dimension, Element
from mlpro.bf.mt import Task, Workflow, Shared
from datetime import datetime
from matplotlib.figure import Figure
import random




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
        Feature data of the instance.
    p_label_data : Element
        Optional label data of the instance.
    p_time_stamp : datetime
        Optional time stamp of the instance.
    p_kwargs : dict
        Further optional named parameters.
    """

    C_TYPE          = 'Instance'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_feature_data : Element, 
                  p_label_data : Element = None, 
                  p_time_stamp : datetime = None,
                  **p_kwargs ):

        self._feature_data = p_feature_data
        self._label_data = p_label_data
        self._time_stamp = p_time_stamp
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
    Template class for data streams. Objects of this type can be used as iterators.

    Parameters
    ----------
    p_id
        Optional id of the stream. Default = 0.
    p_name : str
        Optional name of the stream. Default = ''.
    p_num_instances : int
        Optional number of instances in the stream. Default = 0.
    p_version : str
        Optional version of the stream. Default = ''.
    p_feature_space : MSpace
        Optional feature space. Default = None.
    p_label_space : MSpace
        Optional label space. Default = None.
    p_mode
        Operation mode. Default: Mode.C_MODE_SIM.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.
    p_kwargs : dict
        Further stream specific parameters.
    """

    C_TYPE          = 'Stream'

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_id = 0,
                  p_name : str = '',
                  p_num_instances : int = 0,
                  p_version : str = '',
                  p_feature_space : MSpace = None,
                  p_label_space : MSpace = None,
                  p_mode = Mode.C_MODE_SIM,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):

        self._id            = p_id
        self.C_NAME         = self.C_SCIREF_TITLE = p_name
        self._num_instances = p_num_instances
        self._version       = p_version
        self._feature_space = p_feature_space
        self._label_space   = p_label_space
        self._kwargs        = p_kwargs.copy()
        Mode.__init__(self, p_mode=p_mode, p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def get_id(self):
        """
        Returns the id of the stream.
        """

        return self._id


## -------------------------------------------------------------------------------------------------
    def get_name(self) -> str:
        """
        Returns the name of the stream.

        Returns
        -------
        stream_name : str
            Name of the stream.
        """

        return self.C_NAME


## -------------------------------------------------------------------------------------------------
    def get_url(self) -> str:
        """
        Returns the URL of the scientific source/reference.

        Returns
        -------
        url : str
            URL of the scientific source/reference.
        """

        return self.C_SCIREF_URL


## -------------------------------------------------------------------------------------------------
    def get_num_instances(self) -> int:
        """
        Returns the number of instances of the stream.

        Returns
        -------
        num_inst : int
            Number of instances of the stream. If 0 the number is unknown.
        """

        return self._num_instances


## -------------------------------------------------------------------------------------------------
    def get_feature_space(self) -> MSpace:
        """
        Returns the feature space of the stream. 

        Returns
        -------
        feature_space : MSpace
            Feature space of the stream.
        """

        if self._feature_space is None:
            self._feature_space = self._setup_feature_space()

        return self._feature_space


## -------------------------------------------------------------------------------------------------
    def _setup_feature_space(self) -> MSpace:
        """
        Custom method to set up the feature space of the stream. It is called by method get_feature_space().

        Returns
        -------
        feature_space : MSpace
            Feature space of the stream.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_label_space(self) -> MSpace:
        """
        Returns the label space of the stream. 

        Returns
        -------
        label_space : MSpace
            Label space of the stream.
        """

        if self._label_space is None:
            self._label_space = self._setup_label_space()

        return self._label_space


## -------------------------------------------------------------------------------------------------
    def _setup_label_space(self) -> MSpace:
        """
        Custom method to set up the label space of the stream. It is called by method get_label_space().

        Returns
        -------
        label_space : MSpace
            Label space of the stream.
        """

        return None


## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        """
        Resets the internal random generator using the given seed.
        """

        random.seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def __iter__(self):
        """
        Resets the stream by calling custom method _reset().

        Returns
        -------
        iter
            Iterable stream object
        """

        self.log(self.C_LOG_TYPE_I, 'Reset')
        self._reset()
        return self


## -------------------------------------------------------------------------------------------------
    def _reset(self):
        """
        Custom reset method for data stream. See method __iter__() for more details.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def __next__(self) -> Instance:
        """
        Returns next data stream instance by calling the custom method _get_next(). 

        Returns
        -------
        instance : Instance
            Next instance of data stream or None.
        """

        return self._get_next()


## -------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:
        """
        Custom method to determine the next data stream instance. At the end of the stream exception
        StopIteration is to be raised. See method __next__() for more
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
        Log.__init__(self, p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def get_stream_list( self, p_mode = Mode.C_MODE_SIM, p_logging = Log.C_LOG_ALL, **p_kwargs ) -> list:
        """
        Gets a list of provided streams by calling custom method _get_stream_list().

        Parameters
        ----------
        p_mode
            Operation mode. Default: Mode.C_MODE_SIM.
        p_logging
            Log level of stream objects (see constants of class Log). Default: Log.C_LOG_ALL.
        p_kwargs : dict
            Further stream specific parameters.

        Returns
        -------
        stream_list : list
            List of provided streams.
        """

        self.log(self.C_LOG_TYPE_I, 'Getting list of streams...')
        stream_list = self._get_stream_list( p_mode=p_mode, p_logging=p_logging, **p_kwargs )
#        for stream in stream_list:
#            self.log(self.C_LOG_TYPE_I, 'Stream [' + str(stream.get_id()) + '] ' + stream.get_name())

        self.log(self.C_LOG_TYPE_I, 'Number of streams found:', len(stream_list))
        return stream_list


## -------------------------------------------------------------------------------------------------
    def _get_stream_list( self, p_mode = Mode.C_MODE_SIM, p_logging = Log.C_LOG_ALL, **p_kwargs ) -> list:
        """
        Custom method to get the list of provided streams. See method get_stream_list() for further
        details.

        Parameters
        ----------
        p_mode
            Operation mode. Default: Mode.C_MODE_SIM.
        p_logging
            Log level of stream objects (see constants of class Log). Default: Log.C_LOG_ALL.
        p_kwargs : dict
            Further stream specific parameters.

        Returns
        -------
        stream_list : list
            List of provided streams.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_stream( self, p_id, p_mode = Mode.C_MODE_SIM, p_logging = Log.C_LOG_ALL, **p_kwargs ) -> Stream:
        """
        Returns stream with the specified id by calling custom method _get_stream().

        Parameters
        ----------
        p_id : str
            Id of the requested stream.
        p_mode
            Operation mode. Default: Mode.C_MODE_SIM.
        p_logging
            Log level of stream object (see constants of class Log). Default: Log.C_LOG_ALL.
        p_kwargs : dict
            Further stream specific parameters.

        Returns
        -------
        s : Stream
            Stream object or None in case of an error.
        """

        self.log(self.C_LOG_TYPE_I, 'Requested stream:', str(p_id))
        s = self._get_stream(p_id=p_id, p_mode=p_mode, p_logging=p_logging, **p_kwargs)
        if s is None:
            self.log(self.C_LOG_TYPE_E, 'Stream', str(p_id), 'not found')

        return s


## -------------------------------------------------------------------------------------------------
    def _get_stream( self, p_id, p_mode = Mode.C_MODE_SIM, p_logging = Log.C_LOG_ALL, **p_kwargs ) -> Stream:
        """
        Custom method to get the specified stream. See method get_stream() for further details.

        Parameters
        ----------
        p_id : str
            Id of the requested stream.
        p_mode
            Operation mode. Default: Mode.C_MODE_SIM.
        p_logging
            Log level of stream object (see constants of class Log). Default: Log.C_LOG_ALL.
        p_kwargs : dict
            Further stream specific parameters.

        Returns
        -------
        s : Stream
            Stream object or None in case of an error.
        """

        raise NotImplementedError 





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamTask (Task):
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
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
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
                  p_visualize:bool=False,
                  p_logging=Log.C_LOG_ALL, 
                  **p_kwargs ):

        super().__init__( p_name=p_name, 
                          p_range_max=p_range_max, 
                          p_autorun=Task.C_AUTORUN_NONE, 
                          p_class_shared=None, 
                          p_visualize=p_visualize,
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
    def _update_plot_2d(self, p_settings:PlotSettings, p_inst_new:list, p_inst_del:list, **p_kwargs):
        """
        Default implementation for online adaptive tasks. See class mlpro.bf.plot.Plottable for more
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
    def _update_plot_3d(self, p_settings:PlotSettings, p_inst_new:list, p_inst_del:list, **p_kwargs):
        """
        Default implementation for online adaptive tasks. See class mlpro.bf.plot.Plottable for more
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
    def _update_plot_nd(self, p_settings:PlotSettings, p_inst_new:list, p_inst_del:list, **p_kwargs):
        """
        Default implementation for online adaptive tasks. See class mlpro.bf.plot.Plottable for more
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
## -------------------------------------------------------------------------------------------------
class StreamWorkflow (StreamTask, Workflow):
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
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : dict
        Further optional named parameters handed over to every task within.
    """

    C_TYPE              = 'Stream-Workflow'
    C_PLOT_ACTIVE       = True

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name: str = None, 
                  p_range_max=Workflow.C_RANGE_THREAD, 
                  p_class_shared=Shared, 
                  p_visualize:bool=False,
                  p_logging=Log.C_LOG_ALL, 
                  **p_kwargs ):

        StreamTask.__init__( self,
                             p_name=p_name,
                             p_range_max=p_range_max,
                             p_duplicate_data=False,
                             p_visualize=p_visualize,
                             p_logging=p_logging,
                             **p_kwargs )
                             
        Workflow.__init__( self,
                           p_name=p_name, 
                           p_range_max=p_range_max, 
                           p_class_shared=p_class_shared, 
                           p_visualize=p_visualize,
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





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamScenario (ScenarioBase): 
    """
    Template class for stream based scenarios.

    Parameters
    ----------
    p_mode
        Operation mode. See Mode.C_VALID_MODES for valid values. Default = Mode.C_MODE_SIM.
    p_cycle_limit : int
        Maximum number of cycles. Default = 0 (no limit).
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.  
    """
    
    C_TYPE              = 'Stream-Scenario'

    C_PLOT_ACTIVE       = True

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_mode, 
                  p_cycle_limit=0, 
                  p_visualize:bool=False, 
                  p_logging=Log.C_LOG_ALL ):

        self._stream : Stream           = None
        self._iterator : Stream         = None
        self._workflow : StreamWorkflow = None

        super().__init__( p_mode, 
                          p_cycle_limit=p_cycle_limit, 
                          p_auto_setup=True, 
                          p_visualize=p_visualize, 
                          p_logging=p_logging )


## -------------------------------------------------------------------------------------------------
    def setup(self):
        """
        Specialized method to set up a stream scenario. It is automatically called by the constructor
        and calls in turn the custom method _setup().
        """

        self._stream, self._workflow = self._setup( p_mode=self.get_mode(), 
                                                    p_visualize=self.get_visualization(),
                                                    p_logging=self.get_log_level() )


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize:bool, p_logging):
        """
        Custom method to set up a stream scenario consisting of a stream and a processing stream
        workflow.

        Parameters
        ----------
        p_mode
            Operation mode. See Mode.C_VALID_MODES for valid values. Default = Mode.C_MODE_SIM.
        p_visualize : bool
            Boolean switch for visualisation.
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


## -------------------------------------------------------------------------------------------------
    def _set_mode(self, p_mode):
        self._stream.set_mode(p_mode=p_mode)


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):
        self._iterator = iter(self._stream)
        self._iterator.set_random_seed(p_seed=p_seed)


## -------------------------------------------------------------------------------------------------
    def get_latency(self) -> timedelta:
        return None


## -------------------------------------------------------------------------------------------------
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
        end_of_data : bool
            True, if the end of the related data source has been reached. False otherwise.
        """

        try:
            self._workflow.run( p_inst=iter(self._iterator) )
            end_of_data = False
            adapted     = self._workflow.get_adapted()
        except:
            adapted     = False
            end_of_data = True

        return True, False, adapted, end_of_data


## -------------------------------------------------------------------------------------------------
    def _init_figure(self) -> Figure:
        """
        Custom method to initialize a suitable standalone Matplotlib figure.

        Returns
        -------
        figure : Matplotlib.figure.Figure
            Matplotlib figure object to host the subplot(s)
        """

        return Figure()            


## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure:Figure, p_settings:PlotSettings):
        """
        Custom method to initialize a 2D plot. If attribute p_settings.axes is not None the 
        initialization shall be done there. Otherwise a new MatPlotLib Axes object shall be 
        created in the given figure and stored in p_settings.axes.

        Parameters
        ----------
        p_figure : Matplotlib.figure.Figure
            Matplotlib figure object to host the subplot(s).
        p_settings : PlotSettings
            Object with further plot settings.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure:Figure, p_settings:PlotSettings):
        """
        Custom method to initialize a 3D plot. If attribute p_settings.axes is not None the 
        initialization shall be done there. Otherwise a new MatPlotLib Axes object shall be 
        created in the given figure and stored in p_settings.axes.

        Parameters
        ----------
        p_figure : Matplotlib.figure.Figure
            Matplotlib figure object to host the subplot(s).
        p_settings : PlotSettings
            Object with further plot settings.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure:Figure, p_settings:PlotSettings):
        """
        Custom method to initialize a nD plot. If attribute p_settings.axes is not None the 
        initialization shall be done there. Otherwise a new MatPlotLib Axes object shall be 
        created in the given figure and stored in p_settings.axes.

        Parameters
        ----------
        p_figure : Matplotlib.figure.Figure
            Matplotlib figure object to host the subplot(s).
        p_settings : PlotSettings
            Object with further plot settings.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings:PlotSettings, **p_kwargs):
        """
        Custom method to update the 2d plot. The related MatPlotLib Axes object is stored in p_settings.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        **p_kwargs 
            Implementation-specific data and parameters.             
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings:PlotSettings, **p_kwargs):
        """
        Custom method to update the 3d plot. The related MatPlotLib Axes object is stored in p_settings.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        **p_kwargs 
            Implementation-specific data and parameters.             
        """

        pass


## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings:PlotSettings, **p_kwargs):
        """
        Custom method to update the nd plot. The related MatPlotLib Axes object is stored in p_settings.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        **p_kwargs 
            Implementation-specific data and parameters.             
        """

        pass
