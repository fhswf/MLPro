from mlpro.bf.control.controllers.pid_controller import PIDController
from mlpro.bf.ml.basics import *
from mlpro.rl import Policy,SARSElement
from mlpro.bf.systems import Action, State


class RLPID(Policy):

    def __init__(self, p_observation_space: MSpace, p_action_space: MSpace,pid_controller:PIDController ,policy:Policy,p_id=None, p_buffer_size: int = 1, p_ada: bool = True, p_visualize: bool = False, p_logging=Log.C_LOG_ALL ):
        super().__init__(p_observation_space, p_action_space, p_id, p_buffer_size, p_ada, p_visualize, p_logging)

        self._pid_controller = pid_controller
        self._policy = policy
        self._old_action = None #None
        self._action_space = p_action_space


    ## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self, **p_par):

        # 1 Create a dispatcher hyperparameter tuple for the RLPID policy
        self._hyperparam_tuple = HyperParamDispatcher(p_set=self._hyperparam_space)

        # 2 Extend RLPID policy's hp space and tuple from policy
        try:
            self._hyperparam_space.append( self._policy.get_hyperparam().get_related_set(), p_new_dim_ids=False)
            self._hyperparam_tuple.add_hp_tuple(self._policy.get_hyperparam())
        except:
            pass

    ## -------------------------------------------------------------------------------------------------
    def get_hyperparam(self) -> HyperParamTuple:
       return self._policy.get_hyperparam()
    
    ## -------------------------------------------------------------------------------------------------
    def _update_hyperparameters(self) -> bool:
       return self._policy._update_hyperparameters()    

    ## -------------------------------------------------------------------------------------------------
    
    def _adapt(self, p_sars_elem: SARSElement) -> bool:

        is_adapted = False

        #compute new action with new error value (second s of Sars element)
        self._old_action = self._policy.compute_action(p_obs=p_state_new) 
        
        if self._old_action is not None:

            #get SARS Elements 
            p_state,p_action,p_reward,p_state_new=tuple(p_sars_elem.get_data().values())


           # create a new SARS
            p_sars_elem_new = SARSElement(p_state=p_state,
                                        p_action=self._old_action,
                                        p_reward=p_reward, 
                                        p_state_new=p_state_new)
            
            #adapt own policy
            is_adapted = self._policy._adapt(p_sars_elem_new)        
                
            # compute new action with new error value (second s of Sars element)
            self._old_action=self._policy.compute_action(p_obs=p_state_new)

            #get the pid paramter values 
            pid_values = self._old_action.get_feature_data().get_values()

            #set paramter pid
            self._pid_controller.set_parameter(p_param={"Kp":pid_values[0],
                                                    "Ti":pid_values[1],
                                                    "Tv":pid_values[2]})
        return is_adapted 
    
    ## -------------------------------------------------------------------------------------------------

    def compute_action(self, p_obs: State) -> Action:  

        #get action 
        action=self._pid_controller.compute_action(p_ctrl_error=p_obs)

        #return action
        return action 
    
class RLPIDOffPolicy(Policy):

    def __init__(self, p_observation_space: MSpace, p_action_space: MSpace,pid_controller:PIDController ,p_id=None, p_buffer_size: int = 1, p_ada: bool = True, p_visualize: bool = False, p_logging=Log.C_LOG_ALL ):
        super().__init__(p_observation_space, p_action_space, p_id, p_buffer_size, p_ada, p_visualize, p_logging)

        self._pid_controller = pid_controller 
        self._action_space = p_action_space.get_dim(p_id=0)
        
        
    def _init_hyperparam(self, **p_par):

        # create hp
        # 1- add dim (Kp,Tn,Tv) in hp space 
        # 2- create hp tuple from hp space
        # 3- set hp tuple values


        # 1 
        self._hyperparam_space.add_dim( self._action_space.get_dim(p_id=0))
        self._hyperparam_space.add_dim(self._action_space.get_dim(p_id=1))
        self._hyperparam_space.add_dim(self._action_space.get_dim(p_id=2))

        # 2 
        self._hyperparam_tuple = HyperParamTuple( p_set=self._hyperparam_space )

        #3
        self._hyperparam_tuple.set_values(self._pid_controller.get_parameter_values())

    
    ## -------------------------------------------------------------------------------------------------

    
    def _adapt(self, p_sars_elem: SARSElement) -> bool:
       return False
       
    ## -------------------------------------------------------------------------------------------------


    def compute_action(self, p_obs: State) -> Action:  

        #get action 
        action=self._pid_controller.compute_action(p_ctrl_error=p_obs)

        #return action
        return action   

