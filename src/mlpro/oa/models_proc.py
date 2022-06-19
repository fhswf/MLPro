## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa
## -- Module  : models_proc.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-06-04  0.0.0     DA       Creation
## -- 2022-06-19  0.1.0     DA       Initial implemetation of classes OAStep, OAProcessor
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2022-06-19)

Template classes for serial processing of stream data.
"""


from mlpro.bf.various import Log
from mlpro.bf.math import MSpace
from mlpro.bf.ml import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SharedMemory:
    """
    Template class for a shared memory. 
    """ 
    
    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAStep (Model):
    """
    Template class for an online adaptive processing step. Internal and external adaptatation is 
    supported. For internal adaptation (=unsupervised learning) please implement custom method _adapt_u().
    For external adaptation (=supervised/reinforcement learning) custom method _adapt() can be implemented.

    Parameters
    ----------
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

    C_TYPE          = 'OAStep'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_input_space:MSpace=None,
                  p_output_space:MSpace=None,
                  p_ada=True,
                  p_logging=Log.C_LOG_ALL,
                  **p_kwargs):

        super().__init__( p_buffer_size=0,
                          p_ada=p_ada,
                          p_logging=p_logging,
                          p_par=p_kwargs )

        self._smem          = None
        self._input_space   = p_input_space
        self._output_space  = p_output_space


## -------------------------------------------------------------------------------------------------
    def set_shared_memory(self, p_smem:SharedMemory):
        self._smem = p_smem


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

        if self._adaptivity: 
            self._adapted = self._adapt_int(p_in_add, p_in_del)
        else:
            self._adapted = False

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
    def _adapt_int(self, p_in_add, p_in_del) -> bool:
        """
        Custom method for internal adaptation (=unsupervised learning). See method process() for 
        further details.

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
class OAProcessor (Model):
    """
    Ready to use class for serial data processing. 

    Parameters
    ----------
    p_ada : bool
        Boolean switch for adaptivitiy. Default = True.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    
    """

    C_TYPE          = 'OAProcessor'
    C_NAME          = ''

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_ada=True, 
                  p_logging=Log.C_LOG_ALL,
                  p_cls_smem=SharedMemory ):

        super().__init__(p_buffer_size=0, p_ada=p_ada, p_logging=p_logging)
        self._oasteps  = []
        self._smem     = p_cls_smem()


## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging):
        super().switch_logging(p_logging)
        for step in self._oasteps:
            step.switch_logging(p_logging)


## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada: bool):
        super().switch_adaptivity(p_ada)
        for step in self._oa_steps:
            step.switch_adaptivity(p_ada)


## -------------------------------------------------------------------------------------------------
    def get_hyperparam(self) -> HyperParamTuple:
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        for step in self._oa_steps:
            step.set_random_seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def get_adapted(self) -> bool:
        adapted = False
        for step in self._oa_steps:
            if step.get_adapted():
                adapted = True
                break

        return adapted
        

## -------------------------------------------------------------------------------------------------
    def add_step(self, p_step:OAStep):
        self._oasteps.append(p_step)


## -------------------------------------------------------------------------------------------------
    def process(self, p_in_add, p_in_del):
        self.log(self.C_LOG_TYPE_I, 'Start of processing')

        for step in self._oasteps:
            self.log(self.C_LOG_TYPE_I, 'Start processing step', step.C_TYPE, step.C_NAME)
            step.process( p_in_add, p_in_del )

        self.log(self.C_LOG_TYPE_I, 'End of processing')


# -------------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool: 
        for step in self._oasteps: step.adapt( p_args )


# -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
       for step in self._oasteps:
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

        if len(self._oasteps) == 0: return 0

        maturity = 0
        for step in self._oasteps:
            maturity += step.get_maturity()

        return maturity / len(self._oasteps)





# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
class OAScenario (Scenario): pass





# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
class OATrainingResults (TrainingResults): pass





# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
class OATraining (Training): pass
