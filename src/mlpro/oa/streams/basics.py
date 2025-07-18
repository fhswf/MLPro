## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-06-04  0.0.0     DA       Creation
## -- 2022-10-09  0.1.0     DA       Initial class definitions
## -- 2022-10-26  0.2.0     DA       Refactoring
## -- 2022-10-29  0.3.0     DA       Refactoring
## -- 2022-11-30  0.4.0     DA       Refactoring after changes on bf.streams design
## -- 2022-12-09  0.4.1     DA       Corrections
## -- 2022-12-20  0.5.0     DA       Refactoring
## -- 2023-01-01  0.6.0     DA       Refactoring
## -- 2023-02-23  0.6.1     DA       Removed class OAFunction
## -- 2023-03-27  0.6.1     DA       Refactoring
## -- 2023-04-09  0.7.0     DA       Class OATask: new methods adapt(), _adapt(), adapt_reverse()
## -- 2023-05-15  0.7.1     DA       Class OATask: new parameter p_buffer_size
## -- 2023-12-20  0.8.0     DA       Class OATask: new methods for renormalization
## -- 2024-05-18  0.9.0     DA       - Class OATask: new methods _adapt_pre(), _adapt_post()
## --                                - Classes OATrainingResults, OATraining removed
## -- 2024-05-22  1.0.0     DA       Initial design finished
## -- 2024-05-29  1.0.1     DA       Correction in method OATask.adapt()
## -- 2024-06-18  1.0.2     DA       Litte code cleanup
## -- 2024-11-30  1.1.0     DA       Renaming OA... to OAStream...
## -- 2025-05-28  1.2.0     DA       - New class OASreamAdaptation
## --                                - Class OAStreamTask
## --                                  - Refactoring of method adapt()
## --                                  - Redefinition of method _set_adapted()
## --                                  - Refactoring of method renormalize_on_event()
## -- 2025-06-02  1.3.0     DA       New class OAStreamAdaptationType
## -- 2025-06-06  1.4.0     DA       Refactoring: p_inst -> p_instances
## -- 2025-06-15  1.4.1     DA       Class OAStreamScenario: added parameter **p_kwargs to
## --                                - __init__()
## --                                - setup() and _setup()   
## -- 2025-07-15  1.4.2     DA       Class OAStreamAdaptationType: new parent class AdaptationType   
## -- 2025-07-16  1.4.3     DA       Refactoring 
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.4.3 (2025-07-16)

Core classes for online-adaptive data stream processing (OADSP).

"""

from mlpro.bf.mt import Event
from mlpro.bf.various import Log, TStampType
from mlpro.bf.plot import PlotSettings
from mlpro.bf.mt import Task
from mlpro.bf.ops import Mode
from mlpro.bf.streams import InstDict, Instance, InstTypeNew, StreamShared, StreamTask, StreamWorkflow, StreamScenario
from mlpro.bf.math.normalizers import Normalizer
from mlpro.bf.ml import AdaptationType, Adaptation, Model, AWorkflow


# Export list for public API
__all__ = [ 'OAStreamShared', 
            'OAStreamAdaptationType',
            'OAStreamAdaptation',
            'OAStreamTask',
            'OAStreamWorkflow',
            'OAStreamScenario' ]



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAStreamShared (StreamShared):
    """
    Template class for shared objects in the context of online-adaptive data stream processing.
    """ 
    
    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAStreamAdaptationType (AdaptationType):
    """
    Extended types of adaptation, specific to online-adaptive data stream processing.
    """

    REVERSE     = 'Reverse'
    RENORM      = 'Renorm'
    
    
    
    

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAStreamAdaptation (Adaptation):
    """
    Class for extended adaptation events raised by the OAStreamTask class.
    
    Parameters
    ----------
    p_raising_object
        Raising object.
    p_subtype : OAStreamAdaptationType
        Type of adaptation.
    p_tstamp : TStampType = None
        Time stamp of adaptation.
    **p_kwargs
        Further keyword arguments to be transported by the event.
    """

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_raising_object, 
                  p_subtype : OAStreamAdaptationType, 
                  p_tstamp = None, 
                  p_num_inst = 1,
                  **p_kwargs ):
        
        super().__init__( p_raising_object = p_raising_object, 
                          p_subtype = p_subtype, 
                          p_tstamp = p_tstamp, 
                          **p_kwargs )
        
        self.num_inst = p_num_inst





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAStreamTask (StreamTask, Model):
    """
    Template class for online adaptive ML tasks.

    Parameters
    ----------
    p_name : str
        Optional name of the task. Default is None.
    p_range_max : int
        Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_PROCESS.
    p_ada : bool
        Boolean switch for adaptivitiy. Default = True.
    p_buffer_size : int
        Initial size of internal data buffer. Defaut = 0 (no buffering).
    p_duplicate_data : bool
        If True, instances will be duplicated before processing. Default = False.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    **p_kwargs 
        Further optional keyword arguments.
    """

    C_TYPE              = 'OA Stream-Task'

    C_EVENT_CLS         = OAStreamAdaptation

    C_PLOT_ACTIVE       = True
    C_PLOT_STANDALONE   = True
    C_PLOT_VALID_VIEWS  = [ PlotSettings.C_VIEW_2D, PlotSettings.C_VIEW_3D, PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW = PlotSettings.C_VIEW_ND

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name: str = None, 
                  p_range_max = StreamTask.C_RANGE_THREAD, 
                  p_ada : bool = True, 
                  p_buffer_size : int = 0,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL, 
                  **p_kwargs ):

        Model.__init__( self,
                        p_ada = p_ada,
                        p_name = p_name,
                        p_range_max = p_range_max,
                        p_autorun = Task.C_AUTORUN_NONE,
                        p_class_shared = None,
                        p_buffer_size = p_buffer_size,
                        p_visualize = p_visualize,
                        p_logging = p_logging )    

        StreamTask.__init__( self,
                             p_name = p_name,
                             p_range_max = p_range_max,
                             p_duplicate_data = p_duplicate_data,
                             p_visualize = p_visualize,
                             p_logging = p_logging,
                             **p_kwargs )                             


## -------------------------------------------------------------------------------------------------
    def _set_adapted( self, 
                      p_adapted : bool, 
                      p_subtype : OAStreamAdaptationType = OAStreamAdaptationType.NONE, 
                      p_tstamp : TStampType= None,
                      p_num_inst = 1,
                      **p_kwargs ):
        """
        
        """
        
        if p_tstamp is None:
            try:
                tstamp = self.get_so().tstamp
            except:
                tstamp = None
        else:
            tstamp = p_tstamp

        return super()._set_adapted( p_adapted  = p_adapted, 
                                     p_subtype  = p_subtype, 
                                     p_tstamp   = tstamp,
                                     p_num_inst = p_num_inst,
                                     **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def adapt(self, p_instances : InstDict) -> bool:

        # 0 Intro
        if not self._adaptivity: return False
        self.log(self.C_LOG_TYPE_S, 'Adaptation started')

        adapted_forward  = False
        adapted_reverse  = False
        num_inst_forward = 0
        num_inst_reverse = 0
        

        # 1 Preprocessing 
        try:
            atype_pre = self._adapt_pre()

            if atype_pre != OAStreamAdaptationType.NONE:
                if atype_pre == OAStreamAdaptationType.FORWARD: adapted_forward = True
                elif atype_pre == OAStreamAdaptationType.REVERSE: adapted_reverse = True

                self.log(self.C_LOG_TYPE_S, 'Preprocessing done')

        except NotImplementedError:
            pass


        # 2 Main adaptation loop
        for inst_id, (inst_type, inst) in sorted(p_instances.items()):

            if inst_type == InstTypeNew:
                # 2.1 Adaptation on a new stream instance
                self.log(self.C_LOG_TYPE_S, 'Adaptation on new instance', inst_id)
                if self._adapt( p_instance_new=inst):
                    adapted_forward      = True
                    num_inst_forward    += 1
                    self.log(self.C_LOG_TYPE_S, 'Policy adapted')
                else:
                    self.log(self.C_LOG_TYPE_S, 'Policy not adapted')

            else:
                # 2.2 Reverse adaptation on an obsolete stream instance
                self.log(self.C_LOG_TYPE_S, 'Reverse adaptation on obsolete instance', inst_id)
                try:
                    if self._adapt_reverse( p_instance_del=inst ):
                        adapted_reverse      = True
                        num_inst_reverse    += 1
                        self.log(self.C_LOG_TYPE_S, 'Policy adapted')
                    else:
                        self.log(self.C_LOG_TYPE_S, 'Policy not adapted')
                except NotImplementedError:
                    self.log(self.C_LOG_TYPE_W, 'Reverse adaptation not implemented', inst_id)


        # 3 Postprocessing
        try:
            atype_post = self._adapt_post()

            if atype_post != OAStreamAdaptationType.NONE:
                if atype_post == OAStreamAdaptationType.FORWARD: adapted_forward = True
                elif atype_post == OAStreamAdaptationType.REVERSE: adapted_reverse = True

                self.log(self.C_LOG_TYPE_S, 'Postprocessing done')

        except NotImplementedError:
            pass


        # 4 Outro: Logging and adaptation events
        if adapted_forward or adapted_reverse:
            self.log(self.C_LOG_TYPE_S, 'Adaptation done with changes')
            tstamp = self.get_so().tstamp

            if adapted_reverse:
                self._set_adapted( p_adapted = True,
                                   p_subtype = OAStreamAdaptationType.REVERSE,
                                   p_tstamp = tstamp,
                                   p_num_inst = num_inst_reverse )

            if adapted_forward:
                self._set_adapted( p_adapted = True,
                                   p_subtype = OAStreamAdaptationType.FORWARD,
                                   p_tstamp = tstamp,
                                   p_num_inst = num_inst_forward )
                
            return True
        
        else:
            self.log(self.C_LOG_TYPE_S, 'Adaptation done without changes')
            self._set_adapted( p_adapted = False )
            return False


## -------------------------------------------------------------------------------------------------
    def _adapt_pre(self) -> OAStreamAdaptationType:
        """
        Optional custom method for preprocessing steps of adaption.

        Returns
        -------
        OAStreamAdaptationType
            Type of adaptation carried out.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_instance_new:Instance) -> bool:
        """
        Obligatory custom method for adaptations on a new instance during regular operation. 

        Parameters
        ----------
        p_instance_new : Instance
            New stream instances to be processed.

        Returns
        -------
        bool
            True, if something has been adapted. False otherwise.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _adapt_reverse(self, p_instance_del:Instance) -> bool:
        """
        Optional custom method for reverse adaptation on an obsolete instance during regular operation. 

        Parameters
        ----------
        p_instance_del : Instance
            Obsolete stream instances to be removed.

        Returns
        -------
        adapted : bool
            True, if something has been adapted. False otherwise.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _adapt_post(self) -> OAStreamAdaptationType:
        """
        Optional custom method for postprocessing steps of adaption.

        Returns
        -------
        OAStreamAdaptationType
            Type of adaptation carried out.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _renormalize(self, p_normalizer:Normalizer):
        """
        Custom method to renormalize internally buffered data using the given normalizer object. 
        This is necessary after an adaptation of a related predecessor normalizer task. See method
        renormalize_on_event() for further details.
        
        Parameters
        ----------
        p_normalizer : Normalizer
            Normalizer object to be applied on task-specific 
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def renormalize_on_event(self, p_event_id: str, p_event_object: Event):
        """
        Event handler method to be registered on event Model.C_EVENT_ADAPTED of an online adaptive
        normalizer task. It carries out the task-specific renormalization of internally buffered
        data by calling the custom method _renormalize().

        Parameters
        ----------
        p_event_id : str
            Unique event id
        p_event_object : Event
            Event object with further context informations
        """

        self.log(Log.C_LOG_TYPE_I, 'Renormalization triggered')
        try:
            self._renormalize( p_normalizer=p_event_object.get_raising_object() )
            self.log(Log.C_LOG_TYPE_I, 'Renormalization completed')

        except NotImplementedError:
            self.log(Log.C_LOG_TYPE_W, 'Renormalization not implemented')
            return


        self._set_adapted( p_adapted = True,
                           p_subtype = OAStreamAdaptationType.RENORM )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAStreamWorkflow (StreamWorkflow, AWorkflow):
    """
    Online adaptive workflow based on a stream-workflow and an adaptive workflow.

    Parameters
    ----------
    p_name : str
        Optional name of the workflow. Default is None.
    p_range_max : int
        Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_PROCESS.
    p_class_shared
        Optional class for a shared object (class OAShared or a child class of OAShared)
    p_ada : bool
        Boolean switch for adaptivitiy. Default = True.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : dict
        Further optional named parameters.
     """

    C_TYPE      = 'OA Stream-Workflow'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name: str = None, 
                  p_range_max = StreamWorkflow.C_RANGE_THREAD, 
                  p_class_shared = OAStreamShared, 
                  p_ada : bool = True, 
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL, 
                  **p_kwargs ):

        StreamWorkflow.__init__( self, 
                                 p_name = p_name,
                                 p_range_max = p_range_max,
                                 p_class_shared = p_class_shared,
                                 p_visualize = p_visualize,
                                 p_logging = p_logging,
                                 **p_kwargs )
        
        AWorkflow.__init__( self,
                            p_name = p_name,
                            p_range_max = p_range_max,
                            p_class_shared = p_class_shared,
                            p_ada = p_ada,
                            p_visualize = p_visualize,
                            p_logging = p_logging,
                            **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def add_task(self, p_task : StreamTask, p_pred_tasks: list = None):
        AWorkflow.add_task( self, p_task=p_task, p_pred_tasks=p_pred_tasks )


## -------------------------------------------------------------------------------------------------
    def init_plot( self, 
                   p_figure = None, 
                   p_plot_settings = None ):
        
        super().init_plot( p_figure = p_figure, 
                           p_plot_settings = p_plot_settings )
        
        for helper in self._helpers:
            helper.init_plot()


## -------------------------------------------------------------------------------------------------
    def remove_plot(self, p_refresh=True):
        super().remove_plot(p_refresh)

        for helper in self._helpers:
            helper.remove_plot( p_refresh = p_refresh )





# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
class OAStreamScenario (StreamScenario): 
    """
    Template class for stream based scenarios with online adaptive workflows. 

    Parameters
    ----------
    p_mode
        Operation mode. See bf.ops.Mode.C_VALID_MODES for valid values. Default = Mode.C_MODE_SIM.
    p_ada : bool
        Boolean switch for adaptivitiy. Default = True.
    p_cycle_limit : int
        Maximum number of cycles (0=no limit, -1=get from env). Default = 0.
    p_visualize : bool
        Boolean switch for env/agent visualisation. Default = False.
    p_logging
        Log level (see constants of class mlpro.bf.various.Log). Default = Log.C_LOG_WE.
    **p_kwargs
        Custom keyword arguments handed over to the custom method setup().
    """
    
    C_TYPE      = 'OA Stream-Scenario'

# -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_mode = Mode.C_MODE_SIM,  
                  p_ada : bool = True,  
                  p_cycle_limit = 0, 
                  p_visualize : bool = False, 
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        self._ada = p_ada

        super().__init__( p_mode = p_mode, 
                          p_cycle_limit = p_cycle_limit, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging,
                          **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def setup(self, **p_kwargs):
        """
        Specialized method to set up an oa stream scenario. It is automatically called by the 
        constructor and calls in turn the custom method _setup().

        Parameters
        ----------
        p_kwargs : dict
            Custom keyword parameters
        """

        self._stream, self._workflow = self._setup( p_mode = self.get_mode(),
                                                    p_ada = self._ada, 
                                                    p_visualize = self.get_visualization(),
                                                    p_logging = self.get_log_level(),
                                                    **p_kwargs )

        self._workflow.get_so().assign_stream( p_stream = self._stream )


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging, **p_kwargs):
        """
        Custom method to set up a stream scenario consisting of a stream and a processing stream
        workflow.

        Parameters
        ----------
        p_mode
            Operation mode. See Mode.C_VALID_MODES for valid values. Default = Mode.C_MODE_SIM.
        p_ada : bool
            Boolean switch for adaptivitiy. Default = True.
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
        workflow : OAWorkflow
            An online adaptive stream workflow object.
        """

        raise NotImplementedError
