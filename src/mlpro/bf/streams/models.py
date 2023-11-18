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
## -- 2022-10-24  0.2.1     DA       Class Instance: new method copy()
## -- 2022-10-25  0.3.0     DA       New classes StreamTask, StreamWorkfllow, StreamScenario
## -- 2022-10-29  0.3.1     DA       Refactoring after introduction of module bf.ops
## -- 2022-10-31  0.3.2     DA       Refactoring after changes on bf.mt
## -- 2022-11-03  0.4.0     DA       - Class Instance: completion of constructor
## --                                - Class Stream: extensions and corrections
## --                                - Completion of doc strings 
## -- 2022-11-04  0.4.1     DA       Classes StreamProvider, Stream: refactoring
## -- 2022-11-05  0.5.0     DA       Class Stream: refactoring to make it iterable
## -- 2022-11-07  0.5.1     DA       Class StreamScenario: refactoring 
## -- 2022-11-13  0.6.0     DA       - Class Stream: new custom method set_options()
## --                                - New class StreamShared
## -- 2022-11-18  0.6.1     DA       Refactoring of try/except statements
## -- 2022-11-19  0.6.2     DA       Class Stream: new parameter p_name for methods *get_stream()
## -- 2022-11-22  0.7.0     DA       Classes StreamWorkflow, StreamScenario: plot functionality
## -- 2022-12-08  0.7.1     DA       Classes StreamTask, StreamWorkflow: bugfixes on plotting
## -- 2022-12-16  0.7.2     DA       Class StreamTask: new method _run_wrapper()
## -- 2022-12-18  0.7.3     LSB      Removing obsolete instances from plot data
## -- 2022-12-19  0.7.4     DA       Class StreamTask: new parameter p_duplicate_data
## -- 2022-12-28  0.8.0     DA       Class StreamTask: default visualization 2D, 3D
## -- 2022-12-29  0.9.0     DA       - Refactoring of plot settings
## --                                - Bugfixes in methods StreamTask.update_plot2d/3d
## -- 2022-12-30  0.9.1     DA/LSB   - Class Instance: new parameter p_id, new metod get_id()
## --                                - Class StreamTask: optimized removal of deleted instances from 
## --                                  plots
## -- 2023-01-04  1.0.0     DA       Class Instance: new method set_id()
## --                                Class Stream: automatic instance id generation and assignment
## --                                Class StreamTask: 
## --                                - Refactoring of plotting
## --                                - incorporation of new plot parameter p_horizon
## -- 2023-01-05  1.0.1     DA       Refactoring of method StreamShared.get_instances()
## -- 2023-02-12  1.1.0     DA       Class StreamTask: implementation of plot parameter view_autoselect
## -- 2023-04-10  1.2.0     SY       Introduce class Sampler and update class Stream accordingly
## -- 2023-04-14  1.2.1     SY       Refactoring class Sampler and class Stream 
## -- 2023-04-15  1.2.2     DA       Class Stream: 
## --                                - replaced parent Persistent by Id
## --                                - removed own method get_id()
## --                                - constructor: keep value of internal attribute C_NAME if p_name = ''
## -- 2023-04-16  1.2.3     DA       Method StreamTask._run(): completed parameter types
## -- 2023-11-17  1.3.0     DA       Refactoring class Instance: 
## --                                - removed individual implementation of time stamp functionality
## --                                - added parent class bf.various.TStamp
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.0 (2023-11-17)

This module provides classes for standardized stream processing. 
"""


from mlpro.bf.math.basics import *
from mlpro.bf.various import *
from mlpro.bf.ops import Mode, ScenarioBase
from mlpro.bf.plot import PlotSettings
from mlpro.bf.math import Dimension, Element
from mlpro.bf.mt import *
from datetime import datetime
from matplotlib.figure import Figure
import random
import uuid

from typing import List



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Feature (Dimension): pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Label (Dimension): pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Instance (TStamp):
    """
    Instance class to store the current instance and the corresponding labels of the stream

    Parameters
    ----------
    p_feature_data : Element
        Feature data of the instance.
    p_label_data : Element
        Optional label data of the instance.
    p_tstamp : datetime
        Optional time stamp of the instance.
    p_kwargs : dict
        Further optional named parameters.
    """

    C_TYPE          = 'Instance'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_feature_data : Element, 
                  p_label_data : Element = None, 
                  p_tstamp : datetime = None,
                  **p_kwargs ):

        self._feature_data = p_feature_data
        self._label_data   = p_label_data
        TStamp.__init__(self, p_tstamp=p_tstamp)
        self._kwargs       = p_kwargs.copy()


## -------------------------------------------------------------------------------------------------
    def get_id(self):
        return self._id


## -------------------------------------------------------------------------------------------------
    def set_id(self, p_id:int):
        self._id = p_id


## -------------------------------------------------------------------------------------------------
    def get_feature_data(self) -> Element:
        return self._feature_data


## -------------------------------------------------------------------------------------------------
    def set_feature_data(self, p_feature_data:Element):
        self._feature_data = p_feature_data


## -------------------------------------------------------------------------------------------------
    def get_label_data(self) -> Element:
        return self._label_data


## -------------------------------------------------------------------------------------------------
    def set_label_data(self, p_label_data:Element):
        self._label_data = p_label_data


## -------------------------------------------------------------------------------------------------
    def get_kwargs(self):
        return self._kwargs


## -------------------------------------------------------------------------------------------------
    def copy(self):
        duplicate = self.__class__( p_feature_data=self.get_feature_data().copy(),
                                    p_label_data=self.get_label_data(),
                                    p_tstamp=self.get_tstamp(),
                                    p_kwargs=self._kwargs )
        duplicate.set_id(self.get_id())
        return duplicate





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamShared (Shared):
    """
    Template class for shared objects in the context of stream processing.

    Attributes
    ----------
    _instances : dict        
        Dictionary of new/deleted instances per task. At the beginning of a cycle it contains the incoming
        instance of a stream. The dictionalry evolves due to the manipulations of the stream tasks.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_range: int = Range.C_RANGE_PROCESS):
        Shared.__init__(self, p_range=p_range)
        self._instances = {}
    

## -------------------------------------------------------------------------------------------------
    def reset(self, p_inst_new:List[Instance]):
        """
        Resets the shared object and prepares the processing of the given set of new instances.

        Parameters
        ----------
        p_inst_new : List[Instance]
            List of new instances to be processed.
        """

        self._instances.clear()
        self._instances['wf'] = ( p_inst_new, [] )


## -------------------------------------------------------------------------------------------------
    def get_instances(self, p_task_ids:list):
        """
        Provides the result instances of all given task ids.

        Parameters
        ----------
        p_task_ids : list
            List of task ids.

        Returns
        -------
        inst_new : list
            List of new instances of all given task ids.
        inst_del : list
            List of instances to be deleted of all given task ids.
        """

        len_task_ids = len(p_task_ids)

        if len_task_ids == 1:
            # Most likely case: result instances of one predecessor or requested
            try:
                instances = self._instances[p_task_ids[0]]
            except KeyError:
                # Predecessor is the workflow
                instances = self._instances['wf']

            inst_new = instances[0].copy()
            inst_del = instances[1].copy()

        elif len_task_ids > 1:
            # Result instances of more than one predecessors are requested
            dict_inst_new = {}
            dict_inst_del = {}

            for task_id in p_task_ids:
                try:
                    instances = self._instances[task_id]
                except KeyError:
                    instances = self._instances['wf']

                for inst in instances[0]:
                    dict_inst_new[inst.get_id()] = inst

                for inst in instances[1]:
                    dict_inst_del[inst.get_id()] = inst

            sorted_ids = list( dict_inst_new.keys() )
            sorted_ids.sort()
            inst_new = [ dict_inst_new[i] for i in sorted_ids ]

            sorted_ids = list( dict_inst_del.keys() )
            sorted_ids.sort()
            inst_del = [ dict_inst_del[i] for i in sorted_ids ]

        else:
            # No predecessor task id -> origin incoming instances on workflow level are forwarded
            instances = self._instances['wf']
            inst_new = instances[0].copy()
            inst_del = instances[1].copy()

        return inst_new, inst_del


## -------------------------------------------------------------------------------------------------
    def set_instances(self, p_task_id, p_inst_new:List[Instance], p_inst_del:List[Instance]):
        """
        Stores result instances of a task in the shared object.

        Parameters
        ----------
        p_task_id
            Id of related task.
        p_inst_new : list
            List of new instances.
        p_inst_del : list
            List of instances to be deleted.
        """

        self._instances[p_task_id] = ( p_inst_new, p_inst_del ) 





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Sampler (ScientificObject):
    """
    Template class for data streams sampler. This object can be used in Stream.

    Parameters
    ----------
    p_num_instances : int
        number of instances.
    p_kwargs : dict
        Further sampler specific parameters.
    """

    C_TYPE = 'Sampler'


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_num_instances:int=0, **p_kwargs):
        
        self._num_instances = p_num_instances
        self._kwargs        = p_kwargs


## -------------------------------------------------------------------------------------------------
    def reset(self):
        """
        A method to reset the sampler's settings. Please redefine this method!
        """
        
        raise NotImplementedError
        

## -------------------------------------------------------------------------------------------------
    def get_num_instances(self) -> int:
        """
        A method to get the number of instances that is being processed by the sampler.

        Returns
        -------
        int
            Number of instances.

        """
    
        return self._num_instances
        

## -------------------------------------------------------------------------------------------------
    def set_num_instances(self, p_num_instances:int):
        """
        A method to set the number of instances that is going to be processed by the sampler.

        Parameters
        ----------
        p_num_instances : int
            Number of instances.

        """
    
        self._num_instances = p_num_instances
        

## -------------------------------------------------------------------------------------------------
    def omit_instance(self, p_inst:Instance) -> bool:
        """
        A method to filter any incoming instances.

        Parameters
        ----------
        p_inst : Instance
            An input instance to be filtered.

        Returns
        -------
        bool
            False means the input instance is not omitted, otherwise True.

        """
        
        return self._omit_instance(p_inst)
        

## -------------------------------------------------------------------------------------------------
    def _omit_instance(self, p_inst:Instance) -> bool:
        """
        A custom method to filter any incoming instances, which is being called by omit_instance()
        method. Please redefine this method!

        Parameters
        ----------
        p_inst : Instance
            An input instance to be filtered.

        Returns
        -------
        bool
            False means the input instance is not omitted, otherwise True.

        """
        
        raise NotImplementedError
        




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Stream (Mode, Id, ScientificObject):
    """
    Template class for data streams. Objects of this type can be used as iterators.

    Parameters
    ----------
    p_id
        Optional id of the stream. Default = None.
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
    p_sampler
        Optional sampler. Default: None.
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
                  p_id = None,
                  p_name : str = '',
                  p_num_instances : int = 0,
                  p_version : str = '',
                  p_feature_space : MSpace = None,
                  p_label_space : MSpace = None,
                  p_sampler : Sampler = None,
                  p_mode = Mode.C_MODE_SIM,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):

        if p_name != '':
            self.C_NAME         = self.C_SCIREF_TITLE = p_name

        self._num_instances = p_num_instances
        self._version       = p_version
        self._feature_space = p_feature_space
        self._label_space   = p_label_space

        Id.__init__(self, p_id = p_id)
        Mode.__init__(self, p_mode=p_mode, p_logging=p_logging)

        self.set_options(**p_kwargs)
        
        if p_sampler is None:
            try:
                self._sampler = self.setup_sampler()
            except:
                self._sampler = None
        else:
            self._sampler   = p_sampler
 

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
    def set_options(self, **p_kwargs):
        """
        Method to set specific options for the stream. The possible options depend on the 
        stream provider and stream itself.
        """

        self._kwargs        = p_kwargs.copy()


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
        self._next_inst_id = 0
        self._reset()
        
        if self._sampler is not None:
            self._sampler.reset()
            
        return self


## -------------------------------------------------------------------------------------------------
    def _reset(self):
        """
        Custom reset method for data stream. See method __iter__() for more details.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def setup_sampler(self) -> Sampler:
        """
        A static method to set up a sampler, which allows to set a sampler after instantiation of
        a stream. 

        Returns
        -------
        Sampler
            An instantiated sampler.
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
        
        if self._sampler is not None:
            ret = True
            while ret:
                inst = self._get_next()
                ret = self._sampler.omit_instance(inst)
        else:
            inst = self._get_next()
            
        inst.set_id(self._next_inst_id)
        self._next_inst_id += 1
        return inst


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
    def get_stream( self, p_id:str=None, p_name:str=None, p_mode = Mode.C_MODE_SIM, p_logging = Log.C_LOG_ALL, **p_kwargs ) -> Stream:
        """
        Returns stream with the specified id by calling custom method _get_stream().

        Parameters
        ----------
        p_id : str
            Optional Id of the requested stream. Default = None.
        p_name : str
            Optional name of the requested stream. Default = None.
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

        if p_id is not None:
            self.log(self.C_LOG_TYPE_I, 'Id of requested stream:', p_id)
        elif p_name is not None:
            self.log(self.C_LOG_TYPE_I, 'Name of requested stream:', p_name)
        else:
            raise ParamError('Please specify the requested stream by id or name')

        s = self._get_stream(p_id=p_id, p_name=p_name, p_mode=p_mode, p_logging=p_logging, **p_kwargs)
        if s is None:
            self.log(self.C_LOG_TYPE_E, 'Stream', str(p_id), 'not found')

        return s


## -------------------------------------------------------------------------------------------------
    def _get_stream( self, p_id:str=None, p_name:str=None, p_mode = Mode.C_MODE_SIM, p_logging = Log.C_LOG_ALL, **p_kwargs ) -> Stream:
        """
        Custom method to get the specified stream. See method get_stream() for further details.

        Parameters
        ----------
        p_id : str
            Optional Id of the requested stream. Default = None.
        p_name : str
            Optional name of the requested stream. Default = None.
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
        If True, instances will be duplicated before processing. Default = False.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : dict
        Further optional named parameters.
    """

    C_TYPE                  = 'Stream-Task'

    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = True
    C_PLOT_VALID_VIEWS      = [ PlotSettings.C_VIEW_2D, PlotSettings.C_VIEW_3D, PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW     = PlotSettings.C_VIEW_ND

    C_PLOT_ND_XLABEL_INST   = 'Instance index'
    C_PLOT_ND_XLABEL_TIME   = 'Time index'
    C_PLOT_ND_YLABEL        = 'Feature Data'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name: str = None, 
                  p_range_max = Task.C_RANGE_THREAD, 
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL, 
                  **p_kwargs ):

        Task.__init__( self, 
                       p_name=p_name, 
                       p_range_max=p_range_max, 
                       p_autorun=Task.C_AUTORUN_NONE, 
                       p_class_shared=None, 
                       p_visualize=p_visualize,
                       p_logging=p_logging, 
                       **p_kwargs )

        self._duplicate_data      = p_duplicate_data


## -------------------------------------------------------------------------------------------------
    def _get_custom_run_method(self):
        return self._run_wrapper


## -------------------------------------------------------------------------------------------------
    def run( self, 
             p_range : int = None, 
             p_wait: bool = False, 
             p_inst_new : List[Instance] = None, 
             p_inst_del : List[Instance] = None ):
        """
        Executes the specific actions of the task implemented in custom method _run(). At the end event
        C_EVENT_FINISHED is raised to start subsequent actions.

        Parameters
        ----------
        p_range : int
            Optional deviating range of asynchonicity. See class Range. Default is None what means that the maximum
            range defined during instantiation is taken. Oterwise the minimum range of both is taken.
        p_wait : bool
            If True, the method waits until all (a)synchronous tasks are finished.
        p_inst_new : list
            Optional list of new stream instances to be processed. If None, the list of the shared object
            is used instead. Default = None.
        p_inst_del : list
            List of obsolete stream instances to be removed. If None, the list of the shared object
            is used instead. Default = None.
        """

        so : StreamShared = self.get_so()

        if p_inst_new is not None:
            inst_new = p_inst_new
            inst_del = p_inst_del
        else:
            if so is None:
                raise ImplementationError('Class StreamTask needs instance data as parameters or from a shared object')

            try: 
                inst_new, inst_del = so.get_instances(p_task_ids=self._predecessor_ids)
            except AttributeError:
                raise ImplementationError('Shared object not compatible to class StreamShared')
        
        if ( len(inst_new) + len(inst_del) ) == 0: 
            self.log(Log.C_LOG_TYPE_S, 'No inputs -> SKIP')

        if self._duplicate_data:
            inst_new_copy = []
            inst_del_copy = []
            for inst in inst_new:
                inst_new_copy.append(inst.copy())
            for inst in inst_del:
                inst_del_copy.append(inst.copy())

            inst_new = inst_new_copy
            inst_del = inst_del_copy

        Task.run(self, p_range=p_range, p_wait=p_wait, p_inst_new=inst_new, p_inst_del=inst_del)


## -------------------------------------------------------------------------------------------------
    def _run_wrapper( self, 
                      p_inst_new : List[Instance], 
                      p_inst_del : List[Instance] ):
        """
        Internal use.
        """

        self._run( p_inst_new = p_inst_new, p_inst_del=p_inst_del )
        self.get_so().set_instances( p_task_id = self.get_tid(), p_inst_new=p_inst_new, p_inst_del=p_inst_del )


## -------------------------------------------------------------------------------------------------
    def _run( self, 
              p_inst_new : List[Instance], 
              p_inst_del : List[Instance] ):
        """
        Custom method that is called by method run(). 

        Parameters
        ----------
        p_inst_new : set
            Set of new stream instances to be processed.
        p_inst_del : set
            Set of obsolete stream instances to be removed.
        """

        raise NotImplementedError
  

## -------------------------------------------------------------------------------------------------
    def init_plot( self, 
                   p_figure: Figure = None, 
                   p_plot_settings : PlotSettings = None ):

        try:
            if ( not self.C_PLOT_ACTIVE ) or ( not self._visualize ): return
        except:
            return

        self._plot_num_inst = 0

        Task.init_plot( self,
                        p_figure=p_figure, 
                        p_plot_settings=p_plot_settings )


## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Default implementation for stream tasks. See class mlpro.bf.plot.Plottable for more
        details.
        """

        Task._init_plot_2d( self, p_figure=p_figure, p_settings=p_settings )

        self._plot_2d_plot   = None
        self._plot_inst_ids  = []
        self._plot_2d_xdata  = []
        self._plot_2d_ydata  = []
        self._plot_2d_xmin   = None
        self._plot_2d_xmax   = None
        self._plot_2d_ymin   = None
        self._plot_2d_ymax   = None
 

## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Default implementation for stream tasks. See class mlpro.bf.plot.Plottable for more
        details.
        """

        Task._init_plot_3d( self, p_figure=p_figure, p_settings=p_settings )

        self._plot_3d_plot   = None
        self._plot_inst_ids  = []
        self._plot_3d_xdata  = []
        self._plot_3d_ydata  = []
        self._plot_3d_zdata  = []
        self._plot_3d_xmin   = None
        self._plot_3d_xmax   = None
        self._plot_3d_ymin   = None
        self._plot_3d_ymax   = None
        self._plot_3d_zmin   = None
        self._plot_3d_zmax   = None


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Default implementation for stream tasks. See class mlpro.bf.plot.Plottable for more
        details.
        """

        Task._init_plot_nd( self, p_figure=p_figure, p_settings=p_settings )

        self._plot_nd_xlabel = self.C_PLOT_ND_XLABEL_INST
        p_settings.axes.set_xlabel(self.C_PLOT_ND_XLABEL_INST)
        p_settings.axes.set_ylabel(self.C_PLOT_ND_YLABEL)
        p_settings.axes.grid(visible=True)
        p_settings.axes.set_xlim(0,1)
        p_settings.axes.set_ylim(-1,1)

        self._plot_inst_ids  = []
        self._plot_nd_xdata  = []
        self._plot_nd_plots  = None
        self._plot_nd_ymin   = None
        self._plot_nd_ymax   = None


## -------------------------------------------------------------------------------------------------
    def _finalize_plot_view(self, p_inst_ref : Instance ):
        num_dim = p_inst_ref.get_feature_data().get_related_set().get_num_dim()

        if num_dim == 2:
            view_new = PlotSettings.C_VIEW_2D
        elif num_dim == 3:
            view_new = PlotSettings.C_VIEW_3D
        else:
            view_new = PlotSettings.C_VIEW_ND

        if view_new not in self.C_PLOT_VALID_VIEWS: return

        view_current = self._plot_settings.view
        if view_new == view_current: return

        ps = self._plot_settings
        ps.view = view_new

        if self.C_PLOT_STANDALONE and ( ps.axes is not None ):
            try:
                ps.axes.clear()
                self._figure.clear()
            except:
                pass
            ps.axes = None   

        plot_settings_new      = ps
        self._plot_initialized = False

        self.init_plot( p_figure = self._figure, 
                        p_plot_settings = plot_settings_new )     


## -------------------------------------------------------------------------------------------------
    def update_plot( self, 
                     p_inst_new : List[Instance] = None, 
                     p_inst_del : List[Instance] = None, 
                     **p_kwargs ):
        """
        Specialized definition of method update_plot() of class mlpro.bf.plot.Plottable.

        Parameters
        ----------
        p_inst_new : List[Instance]
            List of new stream instances to be plotted.
        p_inst_del : List[Instance]
            List of obsolete stream instances to be removed.
        p_kwargs : dict
            Further optional plot parameters.
        """

        try:
            if ( not self.C_PLOT_ACTIVE ) or ( not self._visualize ): return
        except:
            return

        if p_inst_new is None:
            inst_new, inst_del = self.get_so().get_instances(p_task_ids=[self.get_tid()])
        else:
            inst_new = p_inst_new
            inst_del = p_inst_del


        if len(inst_new) > 0:
            inst_ref = inst_new[0]
        elif len(inst_del) > 0:
            inst_ref = inst_del[0]
        else:
            return

        try:
            self._plot_view_finalized
        except:
            if self._plot_settings.view_autoselect:
                self._finalize_plot_view(p_inst_ref=inst_ref)

            self._plot_view_finalized = True

        Task.update_plot(self, p_inst_new=inst_new, p_inst_del=inst_del, **p_kwargs)

        self._plot_num_inst += len(inst_new)


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d( self, 
                         p_settings : PlotSettings, 
                         p_inst_new : list, 
                         p_inst_del : list, 
                         **p_kwargs ):
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

        # 1 Check for new instances to be plotted and deleted instances to be removed
        if len(p_inst_new) > 0: inst_ref = p_inst_new[0]
        elif len(p_inst_del) > 0: inst_ref = p_inst_del[0]
        else: return


        # 2 Update of plot data and ax limits
        ax_limits_changed = False

        if len(p_inst_new) > 0:
            xdata_new = []
            ydata_new = []

            for inst in p_inst_new:
                feature_values = inst.get_feature_data().get_values()
                xdata_new.append(feature_values[0])
                ydata_new.append(feature_values[1])
                self._plot_inst_ids.append( inst.get_id() )

            self._plot_2d_xdata.extend(xdata_new)
            self._plot_2d_ydata.extend(ydata_new)
            xmin = min(xdata_new)
            xmax = max(xdata_new)
            ymin = min(ydata_new)
            ymax = max(ydata_new)

            try:
                if xmin < self._plot_2d_xmin:
                    self._plot_2d_xmin = xmin
                    ax_limits_changed = True
            except:
                self._plot_2d_xmin = xmin
                ax_limits_changed = True

            try:
                if xmax > self._plot_2d_xmax:
                    self._plot_2d_xmax = xmax
                    ax_limits_changed = True
            except:
                self._plot_2d_xmax = xmax
                ax_limits_changed = True

            try:
                if ymin < self._plot_2d_ymin:
                    self._plot_2d_ymin = ymin
                    ax_limits_changed = True
            except:
                self._plot_2d_ymin = ymin
                ax_limits_changed = True

            try:
                if ymax > self._plot_2d_ymax:
                    self._plot_2d_ymax = ymax
                    ax_limits_changed = True
            except:
                self._plot_2d_ymax = ymax
                ax_limits_changed = True
            

        if len(p_inst_del) > 0:
            for inst in p_inst_del:
                try:
                    idx = self._plot_inst_ids.index(inst.get_id())
                    del self._plot_2d_xdata[idx]
                    del self._plot_2d_ydata[idx]
                    del self._plot_inst_ids[idx]
                except:
                    pass

            self._plot_2d_xmin = min(self._plot_2d_xdata)
            self._plot_2d_xmax = max(self._plot_2d_xdata)
            self._plot_2d_ymin = min(self._plot_2d_ydata)
            self._plot_2d_ymax = max(self._plot_2d_ydata)

            ax_limits_changed = True
            

        # 3 Plot current data
        if self._plot_2d_plot is None:            
            # 3.1 Late initialization of plot object
            feature_dim = inst_ref.get_feature_data().get_related_set().get_dims()
            p_settings.axes.set_xlabel(feature_dim[0].get_name_short() )
            p_settings.axes.set_ylabel(feature_dim[1].get_name_short() )

            self._plot_2d_plot,  = p_settings.axes.plot( self._plot_2d_xdata, 
                                                         self._plot_2d_ydata, 
                                                         marker='+', 
                                                         color='blue', 
                                                         linestyle='',
                                                         markersize=3 )

        else:
            # 3.2 Update of existing plot object
            self._plot_2d_plot.set_xdata(self._plot_2d_xdata)
            self._plot_2d_plot.set_ydata(self._plot_2d_ydata)


        # 4 Update of ax limits
        if ax_limits_changed:
            p_settings.axes.set_xlim( self._plot_2d_xmin, self._plot_2d_xmax )
            p_settings.axes.set_ylim( self._plot_2d_ymin, self._plot_2d_ymax )


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d( self, 
                         p_settings : PlotSettings, 
                         p_inst_new : list, 
                         p_inst_del : list, 
                         **p_kwargs ):
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

        # 1 Check for new instances to be plotted and deleted instances to be removed
        if len(p_inst_new) > 0: inst_ref = p_inst_new[0]
        elif len(p_inst_del) > 0: inst_ref = p_inst_del[0]
        else: return


        # 2 Update of plot data and ax limits
        ax_limits_changed = False

        if len(p_inst_new) > 0:
            xdata_new = []
            ydata_new = []
            zdata_new = []

            for inst in p_inst_new:
                feature_values = inst.get_feature_data().get_values()
                xdata_new.append(feature_values[0])
                ydata_new.append(feature_values[1])
                zdata_new.append(feature_values[2])
                self._plot_inst_ids.append( inst.get_id() )

            self._plot_3d_xdata.extend(xdata_new)
            self._plot_3d_ydata.extend(ydata_new)
            self._plot_3d_zdata.extend(zdata_new)

            xmin = min(xdata_new)
            xmax = max(xdata_new)
            ymin = min(ydata_new)
            ymax = max(ydata_new)
            zmin = min(zdata_new)
            zmax = max(zdata_new)

            try:
                if xmin < self._plot_3d_xmin:
                    self._plot_3d_xmin = xmin
                    ax_limits_changed = True
            except:
                self._plot_3d_xmin = xmin
                ax_limits_changed = True

            try:
                if xmax > self._plot_3d_xmax:
                    self._plot_3d_xmax = xmax
                    ax_limits_changed = True
            except:
                self._plot_3d_xmax = xmax
                ax_limits_changed = True

            try:
                if ymin < self._plot_3d_ymin:
                    self._plot_3d_ymin = ymin
                    ax_limits_changed = True
            except:
                self._plot_3d_ymin = ymin
                ax_limits_changed = True

            try:
                if ymax > self._plot_3d_ymax:
                    self._plot_3d_ymax = ymax
                    ax_limits_changed = True
            except:
                self._plot_3d_ymax = ymax
                ax_limits_changed = True

            try:
                if zmin < self._plot_3d_zmin:
                    self._plot_3d_zmin = zmin
                    ax_limits_changed = True
            except:
                self._plot_3d_zmin = zmin
                ax_limits_changed = True

            try:
                if zmax > self._plot_3d_zmax:
                    self._plot_3d_zmax = zmax
                    ax_limits_changed = True
            except:
                self._plot_3d_zmax = zmax
                ax_limits_changed = True


        if len(p_inst_del) > 0:
            for inst in p_inst_del:
                try:
                    idx = self._plot_inst_ids.index(inst.get_id())
                    del self._plot_3d_xdata[idx]
                    del self._plot_3d_ydata[idx]
                    del self._plot_3d_zdata[idx]
                    del self._plot_inst_ids[idx]
                except:
                    pass

            self._plot_3d_xmin = min(self._plot_3d_xdata)
            self._plot_3d_xmax = max(self._plot_3d_xdata)
            self._plot_3d_ymin = min(self._plot_3d_ydata)
            self._plot_3d_ymax = max(self._plot_3d_ydata)
            self._plot_3d_zmin = min(self._plot_3d_zdata)
            self._plot_3d_zmax = max(self._plot_3d_zdata)

            ax_limits_changed = True
            

        # 3 Plot current data
        if self._plot_3d_plot is None:            
            # 3.1 Late initialization of plot object
            feature_dim = inst_ref.get_feature_data().get_related_set().get_dims()
            p_settings.axes.set_xlabel(feature_dim[0].get_name_short() )
            p_settings.axes.set_ylabel(feature_dim[1].get_name_short() )
            p_settings.axes.set_zlabel(feature_dim[2].get_name_short() )

        else:
            self._plot_3d_plot.remove()

        self._plot_3d_plot,  = p_settings.axes.plot( self._plot_3d_xdata, 
                                                     self._plot_3d_ydata, 
                                                     self._plot_3d_zdata,
                                                     marker='+', 
                                                     color='blue',
                                                     linestyle='',
                                                     markersize=4 )                                                        


        # 4 Update of ax limits
        if ax_limits_changed:
            p_settings.axes.set_xlim( self._plot_3d_xmin, self._plot_3d_xmax )
            p_settings.axes.set_ylim( self._plot_3d_ymin, self._plot_3d_ymax )
            p_settings.axes.set_zlim( self._plot_3d_zmin, self._plot_3d_zmax )


## -------------------------------------------------------------------------------------------------

    def _update_plot_nd( self, 
                         p_settings : PlotSettings, 
                         p_inst_new : list, 
                         p_inst_del : list, 
                         **p_kwargs ):
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

        # 1 Check for new instances to be plotted and deleted instances to be removed
        if len(p_inst_new) > 0: inst_ref = p_inst_new[0]
        elif len(p_inst_del) > 0: inst_ref = p_inst_del[0]
        else: return


        # 2 Late initialization of plot object
        if self._plot_nd_plots is None:

            # 2.1 Check whether x label needs to be changed to time index
            if ( self._plot_nd_xlabel == self.C_PLOT_ND_XLABEL_INST ) and ( inst_ref.get_tstamp() is not None ):
                p_settings.axes.set_xlabel(self.C_PLOT_ND_XLABEL_TIME)

            # 2.2 Add plot for each feature
            self._plot_nd_plots = []
            feature_space       = inst_ref.get_feature_data().get_related_set()

            for feature in feature_space.get_dims():
                if feature.get_base_set() in [ Dimension.C_BASE_SET_R, Dimension.C_BASE_SET_N, Dimension.C_BASE_SET_Z ]:
                    feature_xdata = self._plot_nd_xdata
                    feature_ydata = []
                    feature_plot, = p_settings.axes.plot( feature_xdata, 
                                                          feature_ydata, 
                                                          lw=1 )

                    self._plot_nd_plots.append( [feature_ydata, feature_plot] )


        # 3 Add data of new instances to plot objects
        for inst in p_inst_new:
            inst_id = inst.get_id()
            self._plot_inst_ids.append(inst_id)
            self._plot_nd_xdata.append(inst_id)

            feature_data = inst.get_feature_data().get_values()

            for i, fplot in enumerate(self._plot_nd_plots):
                feature_value = feature_data[i]

                if ( self._plot_nd_ymin is None ) or ( self._plot_nd_ymin > feature_value ):
                    self._plot_nd_ymin = feature_value

                if ( self._plot_nd_ymax is None ) or ( self._plot_nd_ymax < feature_value ):
                    self._plot_nd_ymax = feature_value

                fplot[0].append(feature_value)


        # 4 Removing obsolete data from the plots
        for inst in p_inst_del:
            try:
                idx = self._plot_inst_ids.index(inst.get_id())
                del self._plot_inst_ids[idx]
                del self._plot_nd_xdata[idx]
                for fplot in self._plot_nd_plots: del fplot[0][idx]
            except:
                pass


        # 5 Set new plot data of all feature plots
        for fplot in self._plot_nd_plots:
            fplot[1].set_xdata(self._plot_nd_xdata)
            fplot[1].set_ydata(fplot[0])


        # 6 Update ax limits
        p_settings.axes.set_xlim(self._plot_nd_xdata[0], self._plot_nd_xdata[len(self._plot_nd_xdata)-1])
        p_settings.axes.set_ylim(self._plot_nd_ymin, self._plot_nd_ymax)
                    




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
        Optional class for a shared object (class StreamShared or a child class of StreamShared).
        Default = StreamShared
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
                  p_range_max = Workflow.C_RANGE_THREAD, 
                  p_class_shared = StreamShared, 
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL, 
                  **p_kwargs ):

        Workflow.__init__( self,
                           p_name=p_name, 
                           p_range_max=p_range_max, 
                           p_class_shared=p_class_shared, 
                           p_visualize=p_visualize,
                           p_logging=p_logging, 
                           **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def run( self, 
             p_range : int = None, 
             p_wait: bool = False, 
             p_inst_new : list = None, 
             p_inst_del : list = None ):
        """
        Runs all stream tasks according to their predecessor relations.

        Parameters
        ----------
        p_range : int
            Optional deviating range of asynchonicity. See class Range. Default is None what means that 
            the maximum range defined during instantiation is taken. Oterwise the minimum range of both 
            is taken.
        p_wait : bool
            If True, the method waits until all (a)synchronous tasks are finished.
        p_inst_new : list
            Optional list of new stream instances to be processed. If None, the list of the shared object
            is used instead. Default = None.
        p_inst_del : list
            List of obsolete stream instances to be removed. If None, the list of the shared object
            is used instead. Default = None.
        """

        if p_inst_new is not None:
            # This workflow is the leading workflow and opens a new process cycle based on external instances
            try:
                self.get_so().reset( p_inst_new )
            except AttributeError:
                raise ImplementationError('Stream workflows need a shared object of type StreamShared (or inherited)')

        Workflow.run(self, p_range=p_range, p_wait=p_wait)                          


## -------------------------------------------------------------------------------------------------
    def init_plot( self, 
                   p_figure: Figure = None, 
                   p_plot_settings : PlotSettings = None ):

        try:
            if ( not self.C_PLOT_ACTIVE ) or ( not self._visualize ): return
        except:
            return

        self._plot_num_inst = 0

        return Workflow.init_plot( self, 
                                   p_figure=p_figure, 
                                   p_plot_settings=p_plot_settings )


## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Default implementation for stream tasks. See class mlpro.bf.plot.Plottable for more
        details.
        """

        StreamTask._init_plot_2d( self, p_figure=p_figure, p_settings=p_settings)


## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Default implementation for stream tasks. See class mlpro.bf.plot.Plottable for more
        details.
        """

        StreamTask._init_plot_3d( self, p_figure=p_figure, p_settings=p_settings)


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Default implementation for stream tasks. See class mlpro.bf.plot.Plottable for more
        details.
        """

        StreamTask._init_plot_nd( self, p_figure=p_figure, p_settings=p_settings)


## -------------------------------------------------------------------------------------------------
    def update_plot( self, 
                     p_inst_new : list = None, 
                     p_inst_del : list = None, 
                     **p_kwargs ):
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

        # Update of workflow master plot by using the StreamTask default implementation
        StreamTask.update_plot(self, p_inst_new=p_inst_new, p_inst_del=p_inst_del, **p_kwargs)





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

        ScenarioBase.__init__( self,
                               p_mode, 
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
            inst_new = [ next(self._iterator) ]
            self._workflow.run( p_inst_new = inst_new, p_inst_del=[] )
            end_of_data = False
        except StopIteration:
            end_of_data = True

        return False, False, False, end_of_data


## -------------------------------------------------------------------------------------------------
    def _init_figure(self) -> Figure:
        """
        Custom method to initialize a suitable standalone Matplotlib figure.

        Returns
        -------
        figure : Matplotlib.figure.Figure
            Matplotlib figure object to host the subplot(s)
        """

        return None


## -------------------------------------------------------------------------------------------------
    def init_plot( self, 
                   p_figure: Figure = None, 
                   p_plot_settings : PlotSettings = None ):

        self._workflow.init_plot( p_figure=p_figure, 
                                  p_plot_settings=p_plot_settings )


## -------------------------------------------------------------------------------------------------
    def update_plot(self, **p_kwargs):
        """
        Plot updates take place during workflow/task processing and are disabled here...
        """
        pass


## -------------------------------------------------------------------------------------------------
    def get_stream(self) -> Stream:
        return self._stream


## -------------------------------------------------------------------------------------------------
    def get_workflow(self) -> StreamWorkflow:
        return self._workflow