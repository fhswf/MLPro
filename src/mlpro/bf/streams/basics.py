## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.streams
## -- Module  : basics.py
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
## -- 2024-05-19  1.4.0     DA       Class Instance: 
## --                                - new parent class Id
## --                                - method set_id(): if tstamp is None, it is initialized with id
## -- 2024-05-21  2.0.0     DA       Classes StreamShared, StreamTask, StreamWorkflow, StreamScenario: 
## --                                - refactoring of instance handling
## --                                Class StreamTask:
## --                                - optimization of code for plotting
## -- 2024-05-23  2.0.1     DA       Bugfix in method StreamTask.run()
## -- 2024-06-07  2.0.2     LSB      Fixing timedelta handling in ND plotting
## -- 2024-07-19  2.0.3     DA       Class StreamTask: excluded non-numeric feature data from default
## --                                visualization 2D,3D,ND
## -- 2024-09-11  2.1.0     DA       Class Instance: new parent KWArgs
## -- 2024-10-01  2.1.1     DA       Method StreamScenario.__init__(): simplification
## -- 2024-10-29  2.2.0     DA       Changed definiton of InstType, InstTypeNew, InstTypeDel
## -- 2024-10-30  2.3.0     DA       Refactoring of StreamTask.update_plot()
## -- 2024-11-10  2.4.0     DA       Refactoring of StreamWorkflow.init_plot()
## -- 2024-12-11  2.4.1     DA       Pseudo class Figure if matplotlib is not installed
## -- 2025-04-03  2.5.0     DA       - New class MultiStream
## --                                - Class Stream: new parent class TStamp 
## --                                - Class StreamTask: new method _get_tstamp()
## --                                - Class StreamShared: new parent TStamp, new methods 
## --                                  assign_stream(), get_tstamp() 
## -- 2025-04-25  2.5.1     DA       Method Stream._get_tstamp_real(): 
## --                                - replaced datetime.now() by time.perf_counter()
## -- 2025-06-06  2.6.0     DA       Refactoring: p_inst -> p_instance/s
## -- 2025-06-08  2.7.0     DA       Refactoring of StreamTask._update_plot*: new return parameter 
## -- 2025-06-24  2.8.0     DA       Class StreamTask: tuning of plot updates
## -- 2025-07-01  2.9.0     DA       Class StreamTask: replaced the exception by pass in _run() to
## --                                make the class be usable as a plot host.
## -- 2025-07-12  2.9.1     DA       Method StreamTask._update_plot_3d(): added explicite ax scaling
## -- 2025-07-16  3.0.0     DA       New classes StreamHelper, StreamTaskHelper
## -- 2025-07-18  3.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 3.1.0 (2025-07-18)

This module provides classes for standardized data stream processing. 

"""

import random
from typing import Dict, Tuple
from collections.abc import Iterator
from itertools import cycle
import time
from datetime import timedelta, datetime

try:
    from matplotlib.figure import Figure
except:
    class Figure : pass

from mlpro.bf.various import Id, TStampType, TStamp, KWArgs, ScientificObject, Log
from mlpro.bf.exceptions import Error, ImplementationError, ParamError
from mlpro.bf.events import Event
from mlpro.bf.ops import Mode, ScenarioBase
from mlpro.bf.mt import Range, Shared, Task, Workflow
from mlpro.bf.plot import Plottable, PlotSettings
from mlpro.bf.math import Dimension, Element, MSpace



# Export list for public API
__all__ = [ 'Feature',
            'Label',
            'InstId',
            'Instance',
            'InstType',
            'InstTypeNew',
            'InstTypeDel',
            'InstDict',
            'Sampler',
            'Stream', 
            'StreamShared', 
            'MultiStream',
            'StreamProvider',
            'StreamTask', 
            'StreamHelper',
            'StreamTaskHelper',
            'StreamWorkflow', 
            'StreamScenario' ]



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Feature (Dimension): pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Label (Dimension): pass





# Type alias for instance ids
InstId = int


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Instance (Id, TStamp, KWArgs):
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
                  p_tstamp : TStampType = None,
                  **p_kwargs ):

        self._feature_data = p_feature_data
        self._label_data   = p_label_data
        TStamp.__init__(self, p_tstamp=p_tstamp)
        KWArgs.__init__(self, **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def set_id(self, p_id:InstId):
        Id.set_id(self, p_id)
        if self.tstamp is None: self.tstamp = p_id


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
        return self._get_kwargs()


## -------------------------------------------------------------------------------------------------
    def copy(self):
        duplicate = self.__class__( p_feature_data=self.get_feature_data().copy(),
                                    p_label_data=self.get_label_data(),
                                    p_tstamp=self.get_tstamp(),
                                    p_kwargs=self._get_kwargs() )
        duplicate.id = self.id
        return duplicate


## -------------------------------------------------------------------------------------------------
    id = property( fget=Id.get_id, fset=set_id )





## -------------------------------------------------------------------------------------------------
## -- Type aliases for instance handling
## -------------------------------------------------------------------------------------------------
InstType    = str
InstTypeNew = '+'
InstTypeDel = '-'
InstDict    = Dict[InstId, Tuple[InstType, Instance]]





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
    def set_num_instances(self, p_num_instances : int):
        """
        A method to set the number of instances that is going to be processed by the sampler.

        Parameters
        ----------
        p_num_instances : int
            Number of instances.

        """
    
        self._num_instances = p_num_instances
        

## -------------------------------------------------------------------------------------------------
    def omit_instance(self, p_instance : Instance) -> bool:
        """
        A method to filter any incoming instances.

        Parameters
        ----------
        p_instance : Instance
            An input instance to be filtered.

        Returns
        -------
        bool
            False means the input instance is not omitted, otherwise True.

        """
        
        return self._omit_instance(p_instance)
        

## -------------------------------------------------------------------------------------------------
    def _omit_instance(self, p_instance : Instance) -> bool:
        """
        A custom method to filter any incoming instances, which is being called by omit_instance()
        method. Please redefine this method!

        Parameters
        ----------
        p_instance : Instance
            An input instance to be filtered.

        Returns
        -------
        bool
            False means the input instance is not omitted, otherwise True.

        """
        
        raise NotImplementedError
        




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Stream (Mode, Id, TStamp, ScientificObject):
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
            self.C_NAME = self.C_SCIREF_TITLE = p_name

        self._num_instances = p_num_instances
        self._version       = p_version
        self._feature_space = p_feature_space
        self._label_space   = p_label_space

        Id.__init__(self, p_id = p_id)
        Mode.__init__(self, p_mode=p_mode, p_logging=p_logging)
        TStamp.__init__(self)

        self.set_options(**p_kwargs)
        
        if p_sampler is None:
            try:
                self._sampler = self.setup_sampler()
            except:
                self._sampler = None
        else:
            self._sampler = p_sampler
 

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
    def get_version(self) -> str:
        return self._version


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
    def get_tstamp(self) -> TStampType:
        """
        Returns the current streaming time. Depending on the stream mode (sim/real) the method
        _get_tstamp_sim() or the custom method _get_tstamp_real() are called.

        Returns
        -------
        TStampType
            Current streaming time.
        """
        
        if self.get_mode() == Mode.C_MODE_REAL:
            return self._get_tstamp_real()
        else:
            return self._get_tstamp_sim()


## -------------------------------------------------------------------------------------------------
    def _get_tstamp_sim(self) -> TStampType:
        """
        Custom method to determine the current time stamp of the running simulation.

        Returns
        -------
        TStampType
            Current streaming time.
        """

        return self._next_inst_id 


## -------------------------------------------------------------------------------------------------
    def _get_tstamp_real(self) -> TStampType:
        return time.perf_counter() - self._perf_counter0


## -------------------------------------------------------------------------------------------------
    def __iter__(self) -> Iterator[Instance]:
        """
        Resets the stream by calling custom method _reset().

        Returns
        -------
        Iterator
            Iterator object
        """

        self.log(self.C_LOG_TYPE_I, 'Reset')
        self._next_inst_id  = 0
        self._perf_counter0 = time.perf_counter()
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

        if ( self._num_instances > 0) and ( self._next_inst_id == self._num_instances ):
            raise StopIteration

        
        if self._sampler is not None:
            ret = True
            while ret:
                inst = self._get_next()
                ret = self._sampler.omit_instance(inst)
        else:
            inst = self._get_next()
            

        if inst.tstamp is None:
            inst.tstamp = self.get_tstamp()

        inst.id = self._next_inst_id
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
    tstamp = property( fget = get_tstamp )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamShared (Shared, TStamp):
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
        TStamp.__init__(self)
        self._instances = {}


## -------------------------------------------------------------------------------------------------
    def assign_stream(self, p_stream : Stream):
        self._stream : Stream = p_stream


## -------------------------------------------------------------------------------------------------
    def get_tstamp(self) -> TStampType:
        try:
            return self._stream.tstamp
        except:
            return datetime.now()
    

## -------------------------------------------------------------------------------------------------
    def reset(self, p_instances : InstDict):
        """
        Resets the shared object and prepares the processing of the given set of new instances.

        Parameters
        ----------
        p_instances : InstDict
            List of new instances to be processed.
        """

        self._instances.clear()
        self._instances['wf'] = p_instances


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
        inst : InstDict
            Instances of all given task ids.
        """

        len_task_ids = len(p_task_ids)

        if len_task_ids == 1:
            # Most likely case: result instances of one predecessor are requested
            try:
                instances = self._instances[p_task_ids[0]]
            except KeyError:
                # Predecessor is the workflow
                instances = self._instances['wf']

            inst = instances.copy()

        elif len_task_ids > 1:
            # Result instances of more than one predecessors are requested
            inst : InstDict = {}

            for task_id in p_task_ids:
                try:
                    inst_task = self._instances[task_id]
                except KeyError:
                    inst_task = self._instances['wf']

                inst.update(inst_task)

        else:
            # No predecessor task id -> origin incoming instances on workflow level are forwarded
            inst = self._instances['wf'].copy()

        return inst


## -------------------------------------------------------------------------------------------------
    def set_instances(self, p_task_id, p_instances : InstDict):
        """
        Stores result instances of a task in the shared object.

        Parameters
        ----------
        p_task_id
            Id of related task.
        p_instances : InstDict
            Instances of related task.
        """

        self._instances[p_task_id] = p_instances


## -------------------------------------------------------------------------------------------------
    tstamp = property( fget = get_tstamp )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MultiStream (Stream):
    """
    Container class for multiple stream objects that can be streamed in sequence or parallel. Just
    instantiate and use the add_stream method to add streams.

    Pararmeters
    -----------
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

    C_TYPE = 'Multi-Stream'

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_id = None,
                  p_name : str = '',
                  p_num_instances : int = 0,
                  p_version : str = '',
                  p_sampler : Sampler = None,
                  p_mode = Mode.C_MODE_SIM,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):

        super().__init__( p_id = p_id,
                          p_name = p_name,
                          p_num_instances = p_num_instances,
                          p_version = p_version,
                          p_sampler = p_sampler,
                          p_mode = p_mode,
                          p_logging = p_logging )

        self.streams               = {}
        self._iterables            = {}
        self._num_streams          = 0
        self._stream_cycle : cycle = None


## -------------------------------------------------------------------------------------------------
    def add_stream( self, p_stream : Stream, p_batch_size : int = 1 ):
        """
        Adds a stream object to the multi-stream.

        Parameters
        ----------
        p_stream : stream
            Stream object to be added.
        p_batch_size : int = 1
            Number of instances to be taken from the stream in sequence, before moving on to the next
            stream. Default = 1. A value of 0 causes the entire stream to be read before moving on to 
            the next stream.
        """

        self.streams[self._num_streams] = [p_stream, p_batch_size, None]
        self._num_streams += 1


## -------------------------------------------------------------------------------------------------
    def _reset(self):
        for stream_entry in self.streams.values():
            stream_entry[2] = iter( stream_entry[0] )

        self._stream_cycle = cycle( self.streams.values() )

        self._switch_stream()
        

## -------------------------------------------------------------------------------------------------
    def _switch_stream(self):

        for i in range(self._num_streams):
            self._current_stream = next( self._stream_cycle )
            self._current_iterable = self._current_stream[2]
            if self._current_iterable is None: continue
            self._batch_counter = 0
            self._batch_size    = self._current_stream[1]
            return

        raise StopIteration


## -------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:

        if ( self._batch_size > 0 ) and ( self._batch_counter >= self._batch_size ):
            self._switch_stream()

        while True:
            try:
                next_inst = next( self._current_iterable )
                self._batch_counter += 1
                break
            except StopIteration:
                self._current_stream[2] = None
                self._switch_stream()

        return next_inst

        
## -------------------------------------------------------------------------------------------------
    def get_tstamp( self ) -> TStampType:
        return self._streams[0][0].tstamp


## -------------------------------------------------------------------------------------------------
    def set_random_seed( self, p_seed = None ):
        for stream_entry in self.streams.values():
            stream_entry[0].set_random_seed( p_seed = p_seed )


## -------------------------------------------------------------------------------------------------
    def get_feature_space( self ) -> MSpace:
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_label_space( self ) -> MSpace:
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    tstamp = property( fget = get_tstamp )



       

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

    C_PLOT_ND_XLABEL_TIME   = 'Time index'
    C_PLOT_ND_YLABEL        = 'Feature Data'

    C_EPSILON               = 0.01

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
    def _get_tstamp(self):
        return self.get_so().tstamp


## -------------------------------------------------------------------------------------------------
    def run( self, 
             p_range : int = None, 
             p_wait: bool = False, 
             p_instances : InstDict = None ):
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
        p_instances : InstDict
            Instances to be processed. If None, the instances are taken from the shared object. Default = None.
        """

        so : StreamShared = self.get_so()

        if p_instances is not None:
            instances = p_instances
        else:
            if so is None:
                raise ImplementationError('Class StreamTask needs instance data as parameters or from a shared object')

            try: 
                instances = so.get_instances(p_task_ids=self._predecessor_ids)
            except AttributeError:
                raise ImplementationError('Shared object not compatible to class StreamShared')
        
        if len(instances) == 0: 
            self.log(Log.C_LOG_TYPE_S, 'No inputs -> SKIP')

        if self._duplicate_data:
            inst_copy : InstDict = {}

            for inst_id, (inst_type, inst) in instances.items():
                inst_copy[inst_id] = ( inst_type, inst.copy() )

            instances = inst_copy

        Task.run( self, 
                  p_range = p_range, 
                  p_wait = p_wait, 
                  p_instances = instances )


## -------------------------------------------------------------------------------------------------
    def _run_wrapper( self, p_instances : InstDict ):
        """
        Internal use.
        """

        self._run( p_instances = p_instances )
        self.get_so().set_instances( p_task_id = self.get_tid(), p_instances = p_instances )


## -------------------------------------------------------------------------------------------------
    def _run( self, p_instances : InstDict ):
        """
        Custom method that is called by method run(). 

        Parameters
        ----------
        p_instances: InstDict
            Instances to be processed.
        """

        pass
  

## -------------------------------------------------------------------------------------------------
    def init_plot( self, 
                   p_figure: Figure = None, 
                   p_plot_settings : PlotSettings = None ):

        try:
            if ( not self.C_PLOT_ACTIVE ) or ( not self._visualize ): return
        except:
            return

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

        self._plot_2d_plot      = None
        self._plot_feature_ids  = []
        self._plot_inst_ids     = []
        self._plot_2d_xdata     = []
        self._plot_2d_ydata     = []
        self._plot_2d_xmin      = None
        self._plot_2d_xmax      = None
        self._plot_2d_ymin      = None
        self._plot_2d_ymax      = None

        self._update_ax_limits  = False
        self._recalc_ax_limits  = False

        p_settings.axes.margins(x=0, y=0)
 

## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Default implementation for stream tasks. See class mlpro.bf.plot.Plottable for more
        details.
        """

        Task._init_plot_3d( self, p_figure=p_figure, p_settings=p_settings )

        self._plot_3d_plot      = None
        self._plot_feature_ids  = []
        self._plot_inst_ids     = []
        self._plot_3d_xdata     = []
        self._plot_3d_ydata     = []
        self._plot_3d_zdata     = []
        self._plot_3d_xmin      = None
        self._plot_3d_xmax      = None
        self._plot_3d_ymin      = None
        self._plot_3d_ymax      = None
        self._plot_3d_zmin      = None
        self._plot_3d_zmax      = None

        self._update_ax_limits  = False
        self._recalc_ax_limits  = False


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure: Figure, p_settings: PlotSettings):
        """
        Default implementation for stream tasks. See class mlpro.bf.plot.Plottable for more
        details.
        """

        Task._init_plot_nd( self, p_figure=p_figure, p_settings=p_settings )

        p_settings.axes.set_xlabel(self.C_PLOT_ND_XLABEL_TIME)
        p_settings.axes.set_ylabel(self.C_PLOT_ND_YLABEL)
        p_settings.axes.grid(visible=True)

        self._plot_inst_ids    = []
        self._plot_nd_xdata    = []
        self._plot_nd_plots    = None
        self._plot_nd_ymin     = None
        self._plot_nd_ymax     = None

        self._update_ax_limits = False
        self._recalc_ax_limits = False


## -------------------------------------------------------------------------------------------------
    def _finalize_plot_view(self, p_inst_ref : Instance ):

        # Determine the number of numeric(!) dimensions of the feature space
        num_dim = 0 
        for dim in p_inst_ref.get_feature_data().get_related_set().get_dims():
            if dim.get_base_set() in [ Dimension.C_BASE_SET_R, Dimension.C_BASE_SET_N, Dimension.C_BASE_SET_Z ]:
                num_dim += 1

        if num_dim == 0:
            raise Error('The stream does not provide numeric data')

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
                     p_instances : InstDict = None, 
                     **p_kwargs ):
        """
        Specialized definition of method update_plot() of class mlpro.bf.plot.Plottable.

        Parameters
        ----------
        p_instances : InstDict
            Instances to be plotted.
        p_kwargs : dict
            Further optional plot parameters.
        """

        try:
            if ( not self.C_PLOT_ACTIVE ) or ( not self._visualize ): return
        except:
            return

        if p_instances is None:
            instances = self.get_so().get_instances(p_task_ids=[self.get_tid()])
        else:
            instances = p_instances

        try:
            self._plot_view_finalized
        except:
            if self._plot_settings.view_autoselect and ( len(instances) > 0 ):
                self._finalize_plot_view(p_inst_ref=next(iter(instances.values()))[1])
                self._plot_view_finalized = True

        Task.update_plot(self, p_instances = instances, **p_kwargs)

            
## -------------------------------------------------------------------------------------------------
    def _update_plot_2d( self, 
                         p_settings : PlotSettings, 
                         p_instances : InstDict, 
                         **p_kwargs ) -> bool:
        """
        Default implementation for stream tasks. See class mlpro.bf.plot.Plottable for more
        details.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        p_instances : InstDict
            Instances to be plotted.
        p_kwargs : dict
            Further optional plot parameters.

        Returns
        -------
        bool   
            True, if changes on the plot require a refresh of the figure. False otherwise.          
        """

        # 1 Check: something to do?
        if len(p_instances) == 0: return False


        # 2 Determine the feature ids to be plotted
        if not self._plot_feature_ids:
            inst_ref     = next(iter(p_instances.values()))[1]
            feature_data = inst_ref.get_feature_data()

            for feature_id, feature in enumerate(feature_data.get_related_set().get_dims()):
                if feature.get_base_set() in [ Dimension.C_BASE_SET_R, Dimension.C_BASE_SET_N, Dimension.C_BASE_SET_Z ]:
                    self._plot_feature_ids.append(feature_id)

            if len(self._plot_feature_ids) < 2:
                raise Error('Data stream does not provide two numeric features')


        # 3 Update plot data
        for inst_id, (inst_type, inst) in p_instances.items():
               
            feature_data   = inst.get_feature_data()
            feature_values = feature_data.get_values()
            x              = feature_values[self._plot_feature_ids[0]]
            y              = feature_values[self._plot_feature_ids[1]]

            if inst_type == InstTypeNew:

                if not self._plot_feature_ids:
                    for feature_id, feature in enumerate(feature_data.get_related_set().get_dims()):
                        if feature.get_base_set() in [ Dimension.C_BASE_SET_R, Dimension.C_BASE_SET_N, Dimension.C_BASE_SET_Z ]:
                            self._plot_feature_ids.append(feature_id)

                    if len(self._plot_feature_ids) < 2:
                        raise Error('Data stream does not provide two numeric features')

                self._plot_2d_xdata.append(x)
                self._plot_2d_ydata.append(y)
                self._plot_inst_ids.append(inst_id)

                if self._plot_2d_xmin is None:
                    self._plot_2d_xmin = x
                    self._plot_2d_xmax = x
                    self._plot_2d_ymin = y
                    self._plot_2d_ymax = y
                    self._update_ax_limits   = True
                else:
                    if x < self._plot_2d_xmin: 
                        self._plot_2d_xmin = x
                        self._update_ax_limits   = True
                    elif x > self._plot_2d_xmax: 
                        self._plot_2d_xmax = x
                        self._update_ax_limits   = True

                    if y < self._plot_2d_ymin: 
                        self._plot_2d_ymin = y
                        self._update_ax_limits   = True
                    elif y > self._plot_2d_ymax: 
                        self._plot_2d_ymax = y
                        self._update_ax_limits   = True

            else:
                if inst_id == self._plot_inst_ids[0]:
                    self._plot_inst_ids = self._plot_inst_ids[1:]
                    self._plot_2d_xdata = self._plot_2d_xdata[1:]
                    self._plot_2d_ydata = self._plot_2d_ydata[1:]

                else:
                    idx = self._plot_inst_ids.index(inst_id)
                    del self._plot_inst_ids[idx]
                    del self._plot_2d_xdata[idx]
                    del self._plot_2d_ydata[idx]

                if not self._recalc_ax_limits:
                    tol_x = ( self._plot_2d_xmax - self._plot_2d_xmin ) * self.C_EPSILON
                    tol_y = ( self._plot_2d_ymax - self._plot_2d_ymin ) * self.C_EPSILON

                    if ( not ( ( self._plot_2d_xmin + tol_x ) <= x <= ( self._plot_2d_xmax - tol_x ) ) ) or \
                       ( not ( ( self._plot_2d_ymin + tol_y ) <= y <= ( self._plot_2d_ymax - tol_y ) ) ):
                        self._update_ax_limits = True
                        self._recalc_ax_limits = True


        # 4 If buffer size is limited, remove obsolete data
        if p_settings.data_horizon > 0:
            num_del = max(0, len(self._plot_inst_ids) - p_settings.data_horizon )
            if num_del > 0:
                self._plot_inst_ids = self._plot_inst_ids[num_del:]
                self._plot_2d_xdata = self._plot_2d_xdata[num_del:]
                self._plot_2d_ydata = self._plot_2d_ydata[num_del:]


        # 5 Plot current data
        if self._plot_2d_plot is None:            
            # 5.1 First plot
            inst_ref    = next(iter(p_instances.values()))[1]
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
            # 5.2 Update of existing plot object
            self._plot_2d_plot.set_xdata(self._plot_2d_xdata)
            self._plot_2d_plot.set_ydata(self._plot_2d_ydata)


        # 6 Update of ax limits
        if self._update_ax_limits:
            if self._recalc_ax_limits:
                self._plot_2d_xmin = min(self._plot_2d_xdata)
                self._plot_2d_xmax = max(self._plot_2d_xdata)
                self._plot_2d_ymin = min(self._plot_2d_ydata)
                self._plot_2d_ymax = max(self._plot_2d_ydata)
                self._recalc_ax_limits = False

            p_settings.axes.set_xlim( [ self._plot_2d_xmin, self._plot_2d_xmax ] )
            p_settings.axes.set_ylim( [ self._plot_2d_ymin, self._plot_2d_ymax ] )
            self._update_ax_limits = False


        return True


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d( self, 
                         p_settings : PlotSettings, 
                         p_instances : InstDict, 
                         **p_kwargs ) -> bool:
        """
        Default implementation for stream tasks. See class mlpro.bf.plot.Plottable for more
        details.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        p_instances : InstDict
            Instances to be plotted.
        p_kwargs : dict
            Further optional plot parameters.

        Returns
        -------
        bool   
            True, if changes on the plot require a refresh of the figure. False otherwise.          
        """

        # 1 Check: something to do?
        if len(p_instances) == 0: return False


        # 2 Determine the feature ids to be plotted
        if not self._plot_feature_ids:
            inst_ref     = next(iter(p_instances.values()))[1]
            feature_data = inst_ref.get_feature_data()

            for feature_id, feature in enumerate(feature_data.get_related_set().get_dims()):
                if feature.get_base_set() in [ Dimension.C_BASE_SET_R, Dimension.C_BASE_SET_N, Dimension.C_BASE_SET_Z ]:
                    self._plot_feature_ids.append(feature_id)

            if len(self._plot_feature_ids) < 3:
                raise Error('Data stream does not provide two numeric features')


        # 3 Update plot data
        for inst_id, (inst_type, inst) in p_instances.items():
               
            feature_data   = inst.get_feature_data()
            feature_values = feature_data.get_values()
            x              = feature_values[self._plot_feature_ids[0]]
            y              = feature_values[self._plot_feature_ids[1]]
            z              = feature_values[self._plot_feature_ids[2]]

            if inst_type == InstTypeNew:

                if not self._plot_feature_ids:
                    for feature_id, feature in enumerate(feature_data.get_related_set().get_dims()):
                        if feature.get_base_set() in [ Dimension.C_BASE_SET_R, Dimension.C_BASE_SET_N, Dimension.C_BASE_SET_Z ]:
                            self._plot_feature_ids.append(feature_id)

                    if len(self._plot_feature_ids) < 2:
                        raise Error('Data stream does not provide two numeric features')

                self._plot_3d_xdata.append(x)
                self._plot_3d_ydata.append(y)
                self._plot_3d_zdata.append(z)
                self._plot_inst_ids.append(inst_id)

                if self._plot_3d_xmin is None:
                    self._plot_3d_xmin = x
                    self._plot_3d_xmax = x
                    self._plot_3d_ymin = y
                    self._plot_3d_ymax = y
                    self._plot_3d_zmin = z
                    self._plot_3d_zmax = z
                    self._update_ax_limits   = True
                else:
                    if x < self._plot_3d_xmin: 
                        self._plot_3d_xmin = x
                        self._update_ax_limits   = True
                    elif x > self._plot_3d_xmax: 
                        self._plot_3d_xmax = x
                        self._update_ax_limits   = True

                    if y < self._plot_3d_ymin: 
                        self._plot_3d_ymin = y
                        self._update_ax_limits   = True
                    elif y > self._plot_3d_ymax: 
                        self._plot_3d_ymax = y
                        self._update_ax_limits   = True

                    if z < self._plot_3d_zmin: 
                        self._plot_3d_zmin = z
                        self._update_ax_limits   = True
                    elif z > self._plot_3d_zmax: 
                        self._plot_3d_zmax = z
                        self._update_ax_limits   = True

            else:
                if inst_id == self._plot_inst_ids[0]:
                    self._plot_inst_ids = self._plot_inst_ids[1:]
                    self._plot_3d_xdata = self._plot_3d_xdata[1:]
                    self._plot_3d_ydata = self._plot_3d_ydata[1:]
                    self._plot_3d_zdata = self._plot_3d_zdata[1:]

                else:
                    idx = self._plot_inst_ids.index(inst_id)
                    del self._plot_inst_ids[idx]
                    del self._plot_3d_xdata[idx]
                    del self._plot_3d_ydata[idx]
                    del self._plot_3d_zdata[idx]

                if not self._recalc_ax_limits:
                    tol_x = ( self._plot_3d_xmax - self._plot_3d_xmin ) * self.C_EPSILON
                    tol_y = ( self._plot_3d_ymax - self._plot_3d_ymin ) * self.C_EPSILON
                    tol_z = ( self._plot_3d_zmax - self._plot_3d_zmin ) * self.C_EPSILON

                    if ( not ( ( self._plot_3d_xmin + tol_x ) <= x <= ( self._plot_3d_xmax - tol_x ) ) ) or \
                       ( not ( ( self._plot_3d_ymin + tol_y ) <= y <= ( self._plot_3d_ymax - tol_y ) ) ) or \
                       ( not ( ( self._plot_3d_ymin + tol_z ) <= z <= ( self._plot_3d_zmax - tol_z ) ) ):
                        self._update_ax_limits = True
                        self._recalc_ax_limits = True


        # 4 If buffer size is limited, remove obsolete data
        if p_settings.data_horizon > 0:
            num_del = max(0, len(self._plot_inst_ids) - p_settings.data_horizon )
            if num_del > 0:
                self._plot_inst_ids = self._plot_inst_ids[num_del:]
                self._plot_3d_xdata = self._plot_3d_xdata[num_del:]
                self._plot_3d_ydata = self._plot_3d_ydata[num_del:]
                self._plot_3d_zdata = self._plot_3d_zdata[num_del:]

        # 5 Plot current data
        if self._plot_3d_plot is None:            
            # 5.1 First plot
            inst_ref    = next(iter(p_instances.values()))[1]
            feature_dim = inst_ref.get_feature_data().get_related_set().get_dims()
            p_settings.axes.set_xlabel(feature_dim[self._plot_feature_ids[0]].get_name_short() )
            p_settings.axes.set_ylabel(feature_dim[self._plot_feature_ids[1]].get_name_short() )
            p_settings.axes.set_zlabel(feature_dim[self._plot_feature_ids[2]].get_name_short() )

        else:
            self._plot_3d_plot.remove()

        self._plot_3d_plot,  = p_settings.axes.plot( self._plot_3d_xdata, 
                                                     self._plot_3d_ydata, 
                                                     self._plot_3d_zdata,
                                                     marker='+', 
                                                     color='blue',
                                                     linestyle='',
                                                     markersize=4 )   


        # 6 Update of ax limits
        if self._update_ax_limits:
            if self._recalc_ax_limits:
                self._plot_3d_xmin = min(self._plot_3d_xdata)
                self._plot_3d_xmax = max(self._plot_3d_xdata)
                self._plot_3d_ymin = min(self._plot_3d_ydata)
                self._plot_3d_ymax = max(self._plot_3d_ydata)
                self._plot_3d_zmin = min(self._plot_3d_zdata)
                self._plot_3d_zmax = max(self._plot_3d_zdata)
                self._recalc_ax_limits = False

            p_settings.axes.set_xlim( [ self._plot_3d_xmin, self._plot_3d_xmax ] )
            p_settings.axes.set_ylim( [ self._plot_3d_ymin, self._plot_3d_ymax ] )
            p_settings.axes.set_zlim( [ self._plot_3d_zmin, self._plot_3d_zmax ] )
            self._update_ax_limits = False


        return True


## -------------------------------------------------------------------------------------------------
    def _update_plot_nd( self, 
                         p_settings : PlotSettings, 
                         p_instances : InstDict, 
                         **p_kwargs ) -> bool:
        """
        Default implementation for stream tasks. See class mlpro.bf.plot.Plottable for more
        details.

        Parameters
        ----------
        p_settings : PlotSettings
            Object with further plot settings.
        p_instances : InstDict
            Instances to be plotted.
        p_kwargs : dict
            Further optional plot parameters.

        Returns
        -------
        bool   
            True, if changes on the plot require a refresh of the figure. False otherwise.          
        """

        # 1 Check: something to do?
        if not p_instances: return False


        # 2 Late initialization of plot object
        if self._plot_nd_plots is None:

            inst_ref = next(iter(p_instances.values()))[1]

            # 2.1 Add plot for each feature
            feature_space        = inst_ref.get_feature_data().get_related_set()
            self._plot_nd_plots  = []
            self._plot_y_min     = None
            self._plot_y_max     = None

            for feature in feature_space.get_dims():
                if feature.get_base_set() in [ Dimension.C_BASE_SET_R, Dimension.C_BASE_SET_N, Dimension.C_BASE_SET_Z ]:
                    feature_ydata = []
                    feature_plot, = p_settings.axes.plot([], [], lw=1, label = feature.get_name_short() )
                    self._plot_nd_plots.append( [feature_ydata, feature_plot] )

            p_settings.axes.legend(title='Features', alignment='left', loc='upper right', draggable=True)


        # 3 Update plot data
        for inst_id, (inst_type, inst) in sorted(p_instances.items()):

            if inst_type == InstTypeNew:
                self._plot_inst_ids.append(inst_id)
                # Handling if the tstamps are timedeltas
                try:
                    self._plot_nd_xdata.append(inst.tstamp.total_seconds())
                except:
                    self._plot_nd_xdata.append(inst.tstamp)

                feature_data = inst.get_feature_data().get_values()

                for i, fplot in enumerate(self._plot_nd_plots):
                    feature_value = feature_data[i]
                    fplot[0].append(feature_value)
                    
                    if self._plot_y_min is None:
                            self._plot_y_min = feature_value
                            self._plot_y_max = feature_value
                            self._update_ax_limits = True
                    elif feature_value < self._plot_y_min:
                            self._plot_y_min = feature_value
                            self._update_ax_limits = True
                    elif feature_value > self._plot_y_max:
                            self._plot_y_max = feature_value
                            self._update_ax_limits = True

            else:
                if inst_id == self._plot_inst_ids[0]:
                    self._plot_inst_ids = self._plot_inst_ids[1:]
                    self._plot_nd_xdata = self._plot_nd_xdata[1:]
                    for fplot in self._plot_nd_plots:
                        fplot[0] = fplot[0][1:]
                else:
                    idx = self._plot_inst_ids.index(inst_id)
                    del self._plot_inst_ids[idx]
                    del self._plot_nd_xdata[idx]
                    for fplot in self._plot_nd_plots:
                        del fplot[0][idx]

                self._update_ax_limits = True   
                self._recalc_ax_limits    = True


        # 4 If buffer size is limited, remove obsolete data
        if p_settings.data_horizon > 0:
            num_del = max(0, len(self._plot_nd_xdata) - p_settings.data_horizon )
            if num_del > 0:
                self._plot_inst_ids = self._plot_inst_ids[num_del:]
                self._plot_nd_xdata = self._plot_nd_xdata[num_del:]
                for fplot in self._plot_nd_plots: fplot[0] = fplot[0][num_del:]

                self._update_ax_limits   = True
                self._recalc_ax_limits      = True


        # 5 Set new plot data of all feature plots
        for fplot in self._plot_nd_plots:
            fplot[1].set_data(self._plot_nd_xdata, fplot[0])


        # 6 Update axes limits
        if p_settings.plot_horizon > 0:
            xlim_id = max(0, len(self._plot_nd_xdata) - p_settings.plot_horizon)
        else:
            xlim_id = 0

        if isinstance(self._plot_nd_xdata[xlim_id], timedelta):
            x_min = self._plot_nd_xdata[xlim_id].total_seconds()
            x_max = self._plot_nd_xdata[-1].total_seconds()
        else:
            x_min = self._plot_nd_xdata[xlim_id]
            x_max = self._plot_nd_xdata[-1]

        if x_min != x_max:
            p_settings.axes.set_xlim(x_min, x_max)

        if self._update_ax_limits:
            if self._recalc_ax_limits:
                self._plot_y_min = None
                self._plot_y_max = None
                for fplot in self._plot_nd_plots:
                    try:
                        self._plot_y_min = min( self._plot_y_min, min( fplot[0]) )
                        self._plot_y_max = max( self._plot_y_max, max( fplot[0]) )
                    except:
                        self._plot_y_min = min(fplot[0])
                        self._plot_y_max = max(fplot[0])

                self._recalc_ax_limits = False

            if self._plot_y_min < self._plot_y_max:
                p_settings.axes.set_ylim(self._plot_y_min, self._plot_y_max)

            self._update_ax_limits = False

        return True





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamHelper (Plottable, Log, KWArgs):
    """
    Template class for own helpers to be added to a stream workflow

    Parameters
    ----------
    p_window_title : str = None
        Optional window title for the helper. If None, a default title is generated.
    p_visualize : bool
        Boolean switch for visualisation. Default = True.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    **p_kwargs
        Further keyword arguments.
    """

    C_TYPE           = 'Stream Helper'
    C_PLOT_ACTIVE    = False

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_window_title: str = None,
                  p_visualize : bool = True,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        Plottable.__init__(self, p_visualize = p_visualize)
        Log.__init__(self, p_logging = p_logging)
        KWArgs.__init__(self, **p_kwargs)
    
        self._window_title = p_window_title


## -------------------------------------------------------------------------------------------------
    def _init_figure(self, p_window_title : str = None):

        window_title = self._window_title if self._window_title is not None else p_window_title
        return super()._init_figure( p_window_title = window_title )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamTaskHelper (StreamHelper):
    """
    Extended task-related template class for own helpers to be added to a stream workflow. It reacts
    automatically on events raised by the related task. 

    Parameters
    ----------
    p_related_task : StreamTask
        Related stream task that raises events.
    p_event_ids : list
        List of event IDs that this helper should react on. Default is an empty list.
    p_no_per_task : int = 0
        Helper number of the task. This is used to distinguish between multiple helpers for the same task.
    p_annotation : str = None
        Optional annotation for the helper.
    p_window_title: str = None
        Optional window title for the helper. If None, a default title is generated. 
    p_visualize : bool = True
        Boolean switch for visualisation. 
    p_logging = Log.C_LOG_ALL
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    **p_kwargs
        Further keyword arguments.
    """

    C_TYPE           = 'Stream Task Helper'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_related_task : StreamTask,
                  p_event_ids : list = [],
                  p_no_per_task : int = 0,
                  p_annotation : str = None,
                  p_window_title: str = None,
                  p_visualize : bool = True,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        self._related_task : StreamTask = p_related_task
        self._no_per_task  : int        = p_no_per_task

        if p_window_title is not None:
            window_title = p_window_title
        else:
            window_title = self.C_NAME + ' for Task "' + self._related_task.get_name() + '"'
            if self._no_per_task > 0:
                window_title += ' (' + str(self._no_per_task) + ')'

            if p_annotation is not None:
                window_title += ' - ' + p_annotation
        
        StreamHelper.__init__( self, 
                               p_window_title = window_title, 
                               p_visualize = p_visualize, 
                               p_logging = p_logging, 
                               **p_kwargs )
        

        for event_id in p_event_ids:
            self._related_task.register_event_handler( p_event_id = event_id, p_event_handler = self._event_handler )


## -------------------------------------------------------------------------------------------------
    def _event_handler(self, p_event_id, p_event_object : Event ):
        """
        Template event handler for task related events. This method is called by the task
        when an event is raised and can be redefined in a child class.

        Parameters
        ----------
        p_event_id 
            ID of the event.
        p_event_object : Event
            Event object that contains the raising task and further information.
        """

        # 0 Intro
        self.log( Log.C_LOG_TYPE_W, 'Task "' + p_event_object.get_raising_object().get_name() + '" raised an event of type "' + str(p_event_id) + '"' )


        # 1 Update plot
        self.update_plot( p_event_object = p_event_object )





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
        
        self._helpers = []

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
             p_instances : InstDict = None ):
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
        p_instances : InstDict
            Optional list of stream instances to be processed. If None, the list of the shared object
            is used instead. Default = None.
        """

        if p_instances is not None:
            # This workflow is the leading workflow and opens a new process cycle based on external instances
            try:
                self.get_so().reset( p_instances = p_instances )
            except AttributeError:
                raise ImplementationError('Stream workflows need a shared object of type StreamShared (or inherited)')

        Workflow.run(self, p_range=p_range, p_wait=p_wait)   


## -------------------------------------------------------------------------------------------------
    def add_helper(self, p_helper : StreamHelper):
        self._helpers.append(p_helper)                       


## -------------------------------------------------------------------------------------------------
    def init_plot( self, 
                   p_figure: Figure = None, 
                   p_plot_settings : PlotSettings = None ):

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
                     p_instances : InstDict = None, 
                     **p_kwargs ):
        """
        Specialized definition of method update_plot() of class mlpro.bf.plot.Plottable.

        Parameters
        ----------
        p_instances : InstDict
            Stream instances to be plotted.
        p_kwargs : dict
            Further optional plot parameters.
        """

        # Update of workflow master plot by using the StreamTask default implementation
        StreamTask.update_plot(self, p_instances = p_instances, **p_kwargs)





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
    **p_kwargs 
        Custom keyword arguments handed over to custom method setup().
    """
    
    C_TYPE              = 'Stream-Scenario'
    C_PLOT_ACTIVE       = True

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_mode, 
                  p_cycle_limit=0, 
                  p_visualize:bool=False, 
                  p_logging=Log.C_LOG_ALL,
                  **p_kwargs ):

        ScenarioBase.__init__( self,
                               p_mode = p_mode, 
                               p_cycle_limit = p_cycle_limit, 
                               p_auto_setup = True, 
                               p_visualize = p_visualize, 
                               p_logging = p_logging,
                               **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def setup(self, **p_kwargs):
        """
        Specialized method to set up a stream scenario. It is automatically called by the constructor
        and calls in turn the custom method _setup().

        Parameters
        ----------
        p_kwargs : dict
            Custom keyword parameters
        """

        self._stream, self._workflow = self._setup( p_mode=self.get_mode(), 
                                                    p_visualize=self.get_visualization(),
                                                    p_logging=self.get_log_level(),
                                                    **p_kwargs )

        self._workflow.get_so().assign_stream( p_stream = self._stream )


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize:bool, p_logging, **p_kwargs):
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
        p_kwargs : dict
            Custom keyword parameters

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
            inst_new = next(self._iterator)
            inst     = { inst_new.id : (InstTypeNew, inst_new) }
            self._workflow.run( p_instances = inst )
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
    def remove_plot(self, p_refresh = True):
        self._workflow.remove_plot(p_refresh=p_refresh)


## -------------------------------------------------------------------------------------------------
    def get_stream(self) -> Stream:
        return self._stream


## -------------------------------------------------------------------------------------------------
    def get_workflow(self) -> StreamWorkflow:
        return self._workflow