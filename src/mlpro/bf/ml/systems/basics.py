## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.systems
## -- Module  : adaptive_systems.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-02-09  0.0.0     LSB      Creation
## -- 2023-04-03  0.1.0     LSB      Moved Adaptive Functions from RL to BF-ML-AdaptiveSystems
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2023-04-03)

This module provides models and templates for adaptive state based systems.
"""



from mlpro.bf.ml.basics import HyperParamTuner
from mlpro.bf.systems import *
from mlpro.bf.ml import Model, Mode
from mlpro.bf.math import *
from typing import Union
from mlpro.sl.basics import SLAdaptiveFunction





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AFctBase (Model):
    """
    Base class for all special adaptive functions (state transition, reward, success, broken). 

    Parameters
    ----------
    p_afct_cls 
        Adaptive function class (compatible to class mlpro.sl.SLAdaptiveFunction)
    p_state_space : MSpace
        State space of an environment or observation space of an agent
    p_action_space : MSpace
        Action space of an environment or agent
    p_input_space_cls
        Space class that is used for the generated input space of the embedded adaptive function (compatible to class
        MSpace)
    p_output_space_cls
        Space class that is used for the generated output space of the embedded adaptive function (compatible to class
        MSpace)
    p_output_elem_cls 
        Output element class (compatible to/inherited from class Element)
    p_threshold : float
        Threshold for the difference between a set point and a computed output. Computed outputs with
        a difference less than this threshold will be assessed as 'good' outputs. Default = 0.
    p_buffer_size : int
        Initial size of internal data buffer. Default = 0 (no buffering).
    p_ada : bool
        Boolean switch for adaptivity. Default = True.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : Dict
        Further model specific parameters (to be specified in child class).

    Attributes
    ----------
    _state_space : MSpace
        State space
    _action_space : MSpace
        Action space
    _input_space : MSpace
        Input space of embedded adaptive function
    _output_space : MSpace
        Output space oof embedded adaptive function
    _afct : SLAdaptiveFunction
        Embedded adaptive function
    """

    C_TYPE = 'AFct Base'

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_afct_cls,
                  p_state_space : MSpace,
                  p_action_space : MSpace,
                  p_input_space_cls = ESpace,
                  p_output_space_cls = ESpace,
                  p_output_elem_cls = Element,
                  p_threshold = 0,
                  p_buffer_size = 0,
                  p_ada : bool = True,
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):

        self._state_space = p_state_space
        self._action_space = p_action_space
        self._input_space = p_input_space_cls()
        self._output_space = p_output_space_cls()

        self._setup_spaces(self._state_space, self._action_space, self._input_space, self._output_space)

        try:
            self._afct = p_afct_cls( p_input_space=self._input_space,
                                     p_output_space=self._output_space,
                                     p_output_elem_cls=p_output_elem_cls,
                                     p_threshold=p_threshold,
                                     p_buffer_size=p_buffer_size,
                                     p_ada=p_ada,
                                     p_visualize=p_visualize,
                                     p_logging=p_logging,
                                     **p_kwargs )
        except:
            raise ParamError('Class ' + str(p_afct_cls) + ' is not compatible to class mlpro.sl.SLAdaptiveFunction')

        super().__init__(p_buffer_size=0, p_ada=p_ada, p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self, p_state_space: MSpace, p_action_space: MSpace, p_input_space: MSpace,
                      p_output_space: MSpace):
        """
        Custom method to set up the input and output space of the embedded adaptive function. Use the
        method add_dimension() of the empty spaces p_input_space and p_output_space to enrich them
        with suitable dimensions.

        Parameters
        ----------
        p_state_space : MSpace
            State space of an environment respectively observation space of an agent.
        p_action_space : MSpace
            Action space of an environment or agent.
        p_input_space : MSpace
            Empty input space of embedded adaptive function to be enriched with dimension.
        p_output_space : MSpace
            Empty output space of embedded adaptive function to be enriched with dimension.

        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_afct(self) -> SLAdaptiveFunction:
        return self._afct


## -------------------------------------------------------------------------------------------------
    def get_state_space(self) -> MSpace:
        return self._state_space


## -------------------------------------------------------------------------------------------------
    def get_action_space(self) -> MSpace:
        return self._action_space


## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self, **p_par):
        pass


## -------------------------------------------------------------------------------------------------
    def get_hyperparam(self) -> HyperParamTuner:
        return self._afct.get_hyperparam()


## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada: bool):
        super().switch_adaptivity(p_ada)
        self._afct.switch_adaptivity(p_ada)


## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging):
        super().switch_logging(p_logging)
        if self._afct is not None:
            self._afct.switch_logging(p_logging)


## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        self._afct.set_random_seed(p_seed=p_seed)


## -------------------------------------------------------------------------------------------------
    def get_adapted(self) -> bool:
        return self._afct.get_adapted()


## -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
        self._afct.clear_buffer()


## -------------------------------------------------------------------------------------------------
    def get_accuracy(self):
        return self._afct.get_accuracy()


## -------------------------------------------------------------------------------------------------
    def init_plot( self, 
                   p_figure: Figure = None, 
                   p_plot_settings: list = [], 
                   p_plot_depth: int = 0, 
                   p_detail_level: int = 0, 
                   p_step_rate: int = 0, 
                   **p_kwargs ):

        self._afct.init_plot( p_figure=p_figure,
                              p_plot_settings=p_plot_settings,
                              p_plot_depth=p_plot_depth,
                              p_detail_level=p_detail_level,
                              p_step_rate=p_step_rate,
                              **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def update_plot(self, **p_kwargs):
        self._afct.update_plot( **p_kwargs )





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AFctSTrans (AFctBase, FctSTrans):
    """
    Online adaptive version of a state transition function. See parent classes for further details.
    """

    C_TYPE = 'AFct STrans'

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_afct_cls,
                  p_state_space: MSpace,
                  p_action_space: MSpace,
                  p_input_space_cls=ESpace,
                  p_output_space_cls=ESpace,
                  p_output_elem_cls=State,  # Specific output element type
                  p_threshold=0,
                  p_buffer_size=0,
                  p_ada:bool=True,
                  p_visualize:bool=False,
                  p_logging=Log.C_LOG_ALL,
                  **p_par):

        super().__init__(p_afct_cls,
                         p_state_space,
                         p_action_space,
                         p_input_space_cls=p_input_space_cls,
                         p_output_space_cls=p_output_space_cls,
                         p_output_elem_cls=p_output_elem_cls,
                         p_threshold=p_threshold,
                         p_buffer_size=p_buffer_size,
                         p_ada=p_ada,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         **p_par)


## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self, p_state_space: MSpace, p_action_space: MSpace, p_input_space: MSpace,
                      p_output_space: MSpace):
        # 1 Setup input space
        p_input_space.append( p_set=p_state_space )
        p_input_space.append( p_set=p_action_space, p_ignore_duplicates=True)

        # 2 Setup output space
        p_output_space.append(p_state_space)


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
        # 1 Create input vector from given state and action
        input_values = p_state.get_values().copy()
        if isinstance(input_values, np.ndarray):
            input_values = np.append(input_values, p_action.get_sorted_values())
        else:
            input_values.extend(p_action.get_sorted_values())
        input = Element(self._input_space)
        input.set_values(input_values)

        # 2 Compute and return new state
        return self._afct.map(input)


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_state:State, p_action:Action, p_state_new:State) -> bool:
        """
        Triggers adaptation of the embedded adaptive function.

        Parameters
        ----------
        p_state : State
            State.
        p_action : Action
            Action
        p_state_new : State
            New state

        Returns
        -------
        bool
            True, if something was adapted. False otherwise.
        """

        input_values = p_state.get_values().copy()
        if isinstance(input_values, np.ndarray):
            input_values = np.append(input_values, p_action.get_sorted_values())
        else:
            input_values.extend(p_action.get_sorted_values())
        input = Element(self._input_space)
        input.set_values(input_values)

        return self._afct.adapt(p_input=input, p_output=p_state_new)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AFctSuccess (AFctBase, FctSuccess):
    """
    Online adaptive version of a function that determine whether or not a state is a success state.
    See parent classes for further details.
    """

    C_TYPE = 'AFct Success'

## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self, p_state_space: MSpace, p_action_space: MSpace, p_input_space: MSpace,
                      p_output_space: MSpace):

        # 1 Setup input space
        p_input_space.append(p_state_space)

        # 2 Setup output space
        p_output_space.add_dim(
            Dimension(p_name_short='Success', p_base_set=Dimension.C_BASE_SET_R, p_name_long='Success',
                      p_boundaries=[0, 1]))


## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state: State) -> bool:
        output = self._afct.map(p_state)

        if output.get_values()[0] >= 0.5:
            return True
        return False


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_state:State) -> bool:
        output = Element(self._output_space)
        ids_ = output.get_dim_ids()
        if p_state.get_success():
            output.set_value(ids_[0], 1)
        else:
            output.set_value(ids_[0], 0)

        return self._afct.adapt(p_input=p_state, p_output=output)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AFctBroken (AFctBase, FctBroken):
    """
    Online adaptive version of a function that determine whether or not a state is a broken state.
    See parent classes for further details.
    """

    C_TYPE = 'AFct Broken'

## -------------------------------------------------------------------------------------------------
    def _setup_spaces( self, 
                       p_state_space:MSpace, 
                       p_action_space:MSpace, 
                       p_input_space:MSpace,
                       p_output_space: MSpace ):

        # 1 Setup input space
        p_input_space.append(p_state_space)

        # 2 Setup output space
        p_output_space.add_dim(
            Dimension(p_name_short='Success', p_base_set=Dimension.C_BASE_SET_R, p_name_long='Success',
                      p_boundaries=[0, 1]))


## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state:State) -> bool:
        output = self._afct.map(p_state)

        if output.get_values()[0] >= 0.5:
            return True
        return False


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_state:State) -> bool:
        output = Element(self._output_space)
        ids_ = output.get_dim_ids()
        if p_state.get_success():
            output.set_value(ids_[0], 1)
        else:
            output.set_value(ids_[0], 0)

        return self._afct.adapt(p_input=p_state, p_output=output)






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ASystem(System, Model):
    """
    This is a template class for Adaptive Systems

    Parameters
    ----------
    p_mode
    p_latency
    p_fct_strans
    p_fct_success
    p_fct_broken
    p_visualize
    p_logging
    """

    C_NAME = 'Adaptive Systems'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_mode = Mode.C_MODE_SIM,
                 p_latency = None,
                 p_fct_strans: Union[FctSTrans, AFctSTrans] = None,
                 p_fct_success: Union[FctSuccess, AFctSuccess] = None,
                 p_fct_broken: Union[FctBroken, AFctBroken] = None,
                 p_adaptivity: bool = True,
                 p_visualize: bool = False,
                 p_logging =Log.C_LOG_ALL):


        System.__init__(self,
                        p_mode = p_mode,
                        p_latency = p_latency,
                        p_fct_strans = p_fct_strans,
                        p_fct_success = p_fct_success,
                        p_fct_broken = p_fct_broken,
                        p_visualize = p_visualize,
                        p_logging = p_logging)


        self._fct_strans  = p_fct_strans
        self._fct_broken  = p_fct_broken
        self._fct_success = p_fct_success
        self.switch_adaptivity(p_ada = p_adaptivity)


## -------------------------------------------------------------------------------------------------
    def _set_adapted(self, p_adapted:bool):
        """

        Parameters
        ----------
        p_adapted

        Returns
        -------

        """

        Model._set_adapted(self, p_adapted=p_adapted)


## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada:bool):
        """

        Parameters
        ----------
        p_ada

        Returns
        -------

        """
        self.log(Log.C_LOG_TYPE_I, 'Switched Adaptivity')
        self._adaptivity = p_ada

        try: self._fct_strans.switch_adaptivity(p_ada=p_ada)
        except: pass

        try: self._fct_broken.switch_adaptivity(p_ada=p_ada)
        except:pass

        try: self._fct_success.switch_adaptivity(p_ada=p_ada)
        except: pass


## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_kwargs) -> bool:
        """

        Parameters
        ----------
        p_kwargs

        Returns
        -------

        """

        adapted = False

        try:
            self._fct_strans.adapt()
            if self._fct_strans.get_adapted():
                adapted = True

        except: pass



        try:
            self._fct_broken.adapt()
            if self._fct_broken.get_adapted():
                adapted = True

        except: pass



        try:
            self._fct_success.adapt()
            if self._fct_success.adapt():
                adapted = True

        except: pass

        return adapted