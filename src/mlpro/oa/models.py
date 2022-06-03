## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.dsm
## -- Module  : models
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-01-06  0.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2022-01-06)

Model classes for efficient online adaptive data stream processing.
"""


from time import CLOCK_THREAD_CPUTIME_ID
from mlpro.bf.various import *
from mlpro.bf.ml import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ProcessingStep(Model):
    """
    Model class for an adaptive data stream (pre-)processing step.
    """

    C_TYPE          = 'Prepro Step'

## -------------------------------------------------------------------------------------------------
    def process_step(self, p_x_add, p_x_del): 
        """
        Processes step.

        Parameters:
            p_x_add     List of input vectors to be added
            p_x_del     List of input vectors to be deleted

        Returns: 
            Nothing
        """

        # 1 Processing steps before policy adaption
        self.log('Process custom step after adaption')
        self.process_before(p_x_add, p_x_del)

        # 2 Policy adaption
#        self.log('Policy adaption')
#        self.policy_adapted = self.adapt_policy(p_x_add, p_x_del)
#        self.log('Policy adapted = ' + self.policy_adapted)

        # 3 Processing steps after policy adaption
        self.log('Process custom step after adaption')
        self.process_after(p_x_add, p_x_del)


## -------------------------------------------------------------------------------------------------
    def process_before(self, p_x_add, p_x_del): 
        """
        Processes custom steps before policy adaption. To be redefined.

        Parameters:
            p_x_add     List of input vectors to be added
            p_x_del     List of input vectors to be deleted

        Returns: 
            Nothing
        """

        pass


## -------------------------------------------------------------------------------------------------
    def process_after(self, p_x_add, p_x_del): 
        """
        Processes custom steps after policy adaption. To be redefined.

        Parameters:
            p_x_add     List of input vectors to be added
            p_x_del     List of input vectors to be deleted

        Returns: 
            Nothing
        """

        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamProcessor(ProcessingStep):
    """
    Model class for sequential adaptive stream processing with optional data preprocessing. Owhn 
    policy adaption and processing steps can be implemented by redefining methods adapt_policy() and
    process_custom().
    """

    C_TYPE          = 'Stream Processor'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_ada=True, p_logging=Log.C_LOG_ALL):
        super().__init__(p_buffer_size=0, p_ada=p_ada, p_logging=p_logging)
        self.prepro_steps   = []
 

## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging):
        Log.switch_logging(self, p_logging=p_logging)
        for step in self.prepro_steps:
            step.switch_logging(p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def add_prepro_step(self, p_step:ProcessingStep):
        """
        Adds a preprocessing step.

        Parameters:
            p_step      Preprocessing step object to be added
 
        Returns: 
            Nothing
        """

        p_step.set_adaptivity(self.adaptivity)
        p_step.switch_logging(p_logging=self.logging)
        self.prepro_steps.append(p_step)


## -------------------------------------------------------------------------------------------------
    def process(self, p_x):
        """
        Processes input in three phases: at first all preprocessing steps will be executed. After that
        the own policy will be adapted and at last the own process steps will be executed.

        Parameters:
            p_x         Input vector x

        Returns: 
            Nothing
        """

        # 0 Intro
        x_add   = []
        x_del   = []
        x_add.append(p_x)
        self.log(self.C_LOG_TYPE_I, 'Start processing of input ', p_x)

        # 1 Preprocessing
        if len(self.prepro_steps) > 0:
            self.log(self.C_LOG_TYPE_I, 'Start of preprocessing')

            for step_id, step in enumerate(self.prepro_steps): 
                self.log(self.C_LOG_TYPE_I, 'Preprocessing step ' + str(step_id) + ': ' + step.C_TYPE + ' ' + step.C_NAME)
                step.process_step(x_add, x_del)

            self.log('End of preprocessing')

        
        # 2 Adaption of own policy and main processing
        self.process_step(x_add, x_del)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Stream:

    C_NAME          = '????'
    C_DESCRIPTION   = '...'
    C_URL           = ''
    C_CITATION      = ''
    C_DOI           = ''
    C_FEATURES      = 0
    C_INSTANCES     = 0

## -------------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        self.def_space()
        self.reset()


## -------------------------------------------------------------------------------------------------
    def def_space(self):
        """
        Defines the internal feature space and it's dimensions. To be redefined. Please bind a well
        defined space object to the internal attribute self.space. 
        """
        
        # Example implementation using the Euclidian space...
        self.space = ESpace()
        self.space.add_dim(Dimension(0, 'X1', '', '', 'm', '', [-20,20]))
        self.space.add_dim(Dimension(1, 'X2', '', '', 'm/s', '', [-10,10]))
        self.space.add_dim(Dimension(2, 'X3', '', '', 'm/s²', '', [-5,5]))


## -------------------------------------------------------------------------------------------------
    def get_space(self):
        return self.space


## -------------------------------------------------------------------------------------------------
    def reset(self):
        """
        Resets stream generator. To be redefined.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def get_next(self):
        """
        Returns next data stream instance or None at the end of the stream. To be redefined.
        """

        return None





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamProcess(Log):
    """
    Stream process, consisting of stream and stream processor object.
    """

    C_TYPE      = 'Stream Process'
    C_NAME      = ''

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_stream:Stream, p_sproc:StreamProcessor, p_logging=True):
        """
        Parameters:
            p_stream        Stream object
            p_sproc         Stream processor object
            p_logging       Boolean switch for logging
        """

        super().__init__(p_logging=p_logging)
        self.stream = p_stream
        self.sproc  = p_sproc

        self.log(self.C_LOG_TYPE_I, 'Stream', self.stream.C_NAME, ' registered')
        self.log(self.C_LOG_TYPE_I, self.sproc.C_TYPE, self.sproc.C_NAME, ' registered')


## -------------------------------------------------------------------------------------------------
    def run(self, p_inst_limit=0, p_feature_ids=None):
        """
        Reads and processes all/limited number of stream instances.

        Parameters:
            p_inst_limit    Optional limitation of instances.
            p_feature_ids   Optional list of ids of features to be processed

        Returns:
            Number of processed instances.
        """

        # 1 Intro
        self.log(self.C_LOG_TYPE_I, 'Start of stream processing (limit='+ str(p_inst_limit) + ')')
        num_inst = 0


        # 2 Main processing loop
        while True:
            inst = self.stream.get_next()
            if inst == None:
                self.log(self.C_LOG_TYPE_I, 'Stream limit reached') 
                break

            num_inst += 1
            self.sproc.process(inst)
            self.log(self.C_LOG_TYPE_I, 'Instance', inst, 'processed')
            if ( p_inst_limit > 0 ) and ( num_inst == p_inst_limit ): break          


        # 3 Outro
        self.log(self.C_LOG_TYPE_I, 'End of stream processing (' + str(num_inst) + ' instances)')
        return num_inst





## -------------------------------------------------------------------------------------------------
## -- Class Group: Special types of preprocessing steps
## -------------------------------------------------------------------------------------------------


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DataWindow (ProcessingStep):
    """
    Model class for data windows that can be used to deal with concept drift. 
    """

    C_TYPE      = 'Data Window'





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Normalization (ProcessingStep):
    """
    Model class for adaptive normalization algorithms. 
    """

    C_TYPE      = 'Normalization'

## -------------------------------------------------------------------------------------------------
    def process_before(self, p_x_add, p_x_del):
        self.backup_policy()


## -------------------------------------------------------------------------------------------------
    def process_after(self, p_x_add, p_x_del):
        for x in p_x_add:
            x = self.normalize(x)


## -------------------------------------------------------------------------------------------------
    def normalize(self, p_x):
        """
        Normalizes an input vector - either by using the recent or the backup policy. To be redefined.

        Parameters:
            p_x         Input vector x to be denormalized
        
        Returns:
            Normalized input vector.
        """
        
        pass   


## -------------------------------------------------------------------------------------------------
    def denormalize(self, p_x, p_backup=True):
        """
        Denormalizes an input - either by using the recent or the backup policy. To be redefined.

        Parameters:
            p_x         Input vector x to be denormalized
            p_backup    If True, the backup policy shall be used. Recent policy otherwise.

        Returns:
            Denormalized input vector.    
        """
        
        pass


## -------------------------------------------------------------------------------------------------
    def renormalize(self, p_x):
        """
        Reormalizes an input vector by denormalizing it with the backup policy and normalizing it
        with the recent policy after that.

        Parameters:
            p_x         Input vector x to be denormalized

        Returns:
            Reormalized input vector.    
        """

        return self.normalize(self.denormalize(p_x, p_backup=True))



## -------------------------------------------------------------------------------------------------
    def backup_policy(self):
        """
        Backups the recent policy. To be redefined.
        """

        pass




## -------------------------------------------------------------------------------------------------
## -- Class Group: Special types of stream processing applications
## -------------------------------------------------------------------------------------------------


