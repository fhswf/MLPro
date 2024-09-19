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
from mlpro.rl.models_env_ada import SARSElement
from mlpro_int_sb3.wrappers.basics import WrPolicySB32MLPro
from stable_baselines3 import A2C, PPO, DQN, DDPG




class RLPID(Policy):

    def __init__(self, p_observation_space: MSpace, p_action_space: MSpace,pid_controller:PIDController ,p_id=None, p_buffer_size: int = 1, p_ada: bool = True, p_visualize: bool = False, p_logging=Log.C_LOG_ALL ):
        super().__init__(p_observation_space, p_action_space, p_id, p_buffer_size, p_ada, p_visualize, p_logging)

        self._pid_controller = pid_controller


    
    def _adapt(self, p_sars_elem: SARSElement) -> bool:

        """
        

        policy_sb3 = PPO(
            policy="MlpPolicy",
            n_steps=5,
            env=None,
            _init_setup_model=False,
            device="cpu")

        sb3_policy =WrPolicySB32MLPro()
        sb3_policy._adapt_on_policy(p_sars_elem)
        sb3_policy._compute_action_on_policy()
        




        p_param={}
        self._pid_controller.set_parameter(p_param)

        """
        pass
    

  

    

    def compute_action(self, p_obs: State) -> Action:

        #create control error from p_obs
        crtl_error = ControlError(p_obs.get_feature_data(),p_obs.get_label_data(),p_obs.get_tstamp())

        #get action 
        action=self._pid_controller.compute_action(crtl_error)

        #return action
        return action 
        



    
        

        



    


    

    

