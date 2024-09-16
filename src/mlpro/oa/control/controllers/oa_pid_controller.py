from mlpro.bf.control.controllers.pid_controller import PIDController
from mlpro.bf.streams import InstDict
from mlpro.rl import Policy, FctReward
from mlpro.oa.control.basics import OAController
from mlpro.bf.math.basics import Log,Set,MSpace
from mlpro.bf.mt import Log, Task
from mlpro.bf.systems import Action
from mlpro.bf.control.basics import CTRLError, ControlError, Controller, SetPoint
from mlpro.bf.systems.basics import ActionElement, State
from mlpro.bf.various import Log
from mlpro.bf.streams import InstDict, Instance





class RLController(OAController):

    C_TYPE= 'RL PID-Controller'

    def __init__(self, p_name: str = None, p_range_max=Task.C_RANGE_THREAD, p_duplicate_data: bool = False, p_visualize: bool = False, 
                 p_logging=Log.C_LOG_ALL,p_error_id:int = 0,p_cls_policy:type = None, p_param_policy = None,p_fct_reward:FctReward = None ,**p_kwargs):
        super().__init__(p_name, p_range_max, p_duplicate_data, p_visualize, p_logging, **p_kwargs)



class RLPIDController(RLController):


    def __init__(self, p_name: str = None, p_range_max=Task.C_RANGE_THREAD, p_duplicate_data: bool = False, p_visualize: bool = False,
                  p_logging=Log.C_LOG_ALL, p_error_id: int = 0, p_cls_policy: type = None, p_param_policy=None, p_fct_reward: FctReward = None,pid_controller:PIDController = None,**p_kwargs):
        super().__init__(p_name, p_range_max, p_duplicate_data, p_visualize, p_logging, p_error_id, p_cls_policy, p_param_policy, p_fct_reward,**p_kwargs)

        self._policy = self._setup_policy(p_param_policy)
        
        self._pid_controller = pid_controller

    def _setup_policy_action_space(self) -> MSpace:
        pass


    def _setup_policy(self,p_param_policy:dict)-> Policy:
        pass



    def _run(self, p_inst:InstDict):
        pass 


    def compute_action(self, p_ctrl_error: ControlError) -> Action:
        return self._pid_controller.compute_action(p_ctrl_error)
    
    def _adapt(self, p_setpoint: SetPoint, p_ctrl_error: ControlError, p_state: State, p_action: Action,p_reward:float) -> bool:
        
        """
        Specialized custom method for online adaptation in closed-loop control scenarios.

        Parameters
        ----------
        p_ctrl_error : ControlError
            Control error.
        p_state : State
            State of control system.
        p_setpoint : SetPoint
            Setpoint.        
        p_Action : Action
            control variable          
        p_reward : float
            Output valaue of the reward function
        """
        
        pass




        

        



    

    

    

    

