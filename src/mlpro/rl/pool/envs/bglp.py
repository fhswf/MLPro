 ## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : BGLP
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-06-07  0.0.0     SY       Creation
## -- 2021-08-31  1.0.0     SY       Release of first version
## -- 2021-09-01  1.0.1     SY       Minor improvements, code cleaning, add descriptions
## -- 2021-09-06  1.0.2     SY       Minor improvements, combine bglp and BGLP classes
## -- 2021-09-11  1.0.3     MRD      Change Header information to match our new library name
## -- 2021-10-02  1.0.4     SY       Minor change
## -- 2021-10-05  1.0.5     SY       Update following new attributes done and broken in State
## -- 2021-10-07  2.0.0     SY       Enable batch production scenario
## -- 2021-10-25  2.0.1     SY       Add scientific references related to the Environment
## -- 2021-11-15  2.1.0     DA       Refactoring
## -- 2021-11-16  2.1.1     SY       Update following model improvements
## -- 2021-11-16  2.1.2     SY       Add data storing for overflow, demand, energy
## -- 2021-11-17  2.1.3     SY       Random initial states
## -- 2021-11-21  2.1.4     SY       Remove dependency from torch
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.1.4 (2021-11-21)

This module provides an environment of Bulk Good Laboratory Plant (BGLP).
"""

from mlpro.rl.models import *
from mlpro.bf.various import *
import numpy as np
import random
        
        
        
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class Actuator:
    reg_a               = []
    idx_a               = 0
    power_max           = 0
    power_min           = 0
    power_coeff         = 0
    action_max          = 0
    action_min          = 0
    mass_coeff          = 0
    t_activated         = 0
    t_end               = 0
    status              = False
    cur_mass_transport  = 0
    cur_power           = 0
    cur_action          = 0
    cur_speed           = 0
    type_               = ""
    
    def __init__(self, minpower, maxpower, minaction, maxaction, masscoeff):
        """
        

        Parameters
        ----------
        maxpower : Numeric Types
            the maximum power of an actuator
        minaction : Numeric Types
            the minimum action of an actuator
        maxaction : Numeric Types
            the maximum action of an actuator
        masscoeff : Numeric Types
            the mass transport coefficient of an actuator.

        Returns
        -------
        None.

        """
        self.idx_a              = len(self.reg_a)
        self.reg_a.append(self)
        self.power_max          = maxpower
        self.power_min          = minpower
        self.action_min         = minaction
        self.action_max         = maxaction
        self.mass_coeff         = masscoeff
        self.cur_mass_transport = 0
        self.cur_power          = 0
        self.cur_action         = 0
        self.cur_speed          = 0
        
        
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
    
class VacuumPump(Actuator):
    reg_v       = []
    idx_v       = 0
    name        = ""
    t_end_max   = 0
    
    def __init__(self, name, minpower, maxpower, minaction, maxaction, masscoeff):    
        """
        

        Parameters
        ----------
        name : Text Type
            label for an actuator (e.g. Belt_A, Vac_A, etc.)
        maxpower : Numeric Types
            the maximum power of an actuator
        minaction : Numeric Types
            the minimum action of an actuator
        maxaction : Numeric Types
            the maximum action of an actuator
        masscoeff : Numeric Types
            the mass transport coefficient of an actuator.

        Returns
        -------
        None.

        """
        Actuator.__init__(self, minpower, maxpower, minaction, maxaction, masscoeff)
        self.idx_v  = len(self.reg_v)
        self.reg_v.append(self)
        self.name   = name
        self.type_  = "VAC"
        
    def start_t(self, now, duration, overwrite = False): 
        """
        

        Parameters
        ----------
        now : Numeric Types
            current_simulation time
        duration : Numeric Types (0 to 1)
            the selected action by an agent
        overwrite : Boolean, optional
            to select whether the actuator is already activated or not

        Returns
        -------
        None.

        """
        if self.status == False:
            self.t_activated    = now
            self.t_end          = now + duration * self.action_max
            self.t_end_max      = now + self.action_max
            self.cur_action     = duration * self.action_max
            self.status         = True
        else:
            if overwrite:
                self.cur_action = duration * self.action_max
                self.t_end      = now + duration * self.action_max    
                self.status     = True
                
    def calc_mass(self, now):
        """
        

        Parameters
        ----------
        now : Numeric Types
            current_simulation time

        Returns
        -------
        current transported mass

        """
        if self.status == True:
            t_diff = now - (self.t_activated + self.action_min)
            if t_diff >= 0:
                self.cur_mass_transport = (2*self.mass_coeff[1]) + self.mass_coeff[0]
        return self.cur_mass_transport        
        
    def calc_energy(self):
        """
        

        Returns
        -------
        current power consumption

        """
        if self.status == True:
            self.cur_power = self.power_max
        return self.cur_power / 1000.0
        
    def update(self, now):
        """
        

        Parameters
        ----------
        now : Numeric Types
            current_simulation time

        Returns
        -------
        None.

        """
        if self.status == True:
            if self.t_end < now or self.t_end_max < now:
                self.deactivate()
    
    def deactivate(self):
        self.status             = False
        self.cur_mass_transport = 0
        self.cur_power          = 0
        
        
        
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
        
class Belt(Actuator):
    reg_b       = []
    idx_b       = 0
    name        = ""
    actiontype  = ""
    speed       = 0
    
    def __init__(self, name, actiontype, minpower, maxpower, minaction, maxaction, masscoeff):
        """
        

        Parameters
        ----------
        name : Text Type
            label for an actuator (e.g. Belt_A, Vac_A, etc.)
        actiontype : Text Type
            "C" for continuous action, "B" for binary action
        maxpower : Numeric Types
            the maximum power of an actuator
        minaction : Numeric Types
            the minimum action of an actuator
        maxaction : Numeric Types
            the maximum action of an actuator
        masscoeff : Numeric Types
            the mass transport coefficient of an actuator.

        Returns
        -------
        None.

        """
        Actuator.__init__(self, minpower, maxpower, minaction, maxaction, masscoeff)
        self.idx_b = len(self.reg_b)
        self.reg_b.append(self)
        self.name = name
        self.actiontype = actiontype
        self.speed = 0
        self.type_ = "BLT"
        
    def start_t(self, now, duration, speed = 0, overwrite = False):
        """
        

        Parameters
        ----------
        
        now : Numeric Types
            current_simulation time
        duration : Numeric Types
            time set parameter.
        speed : Numeric Types (0 to 1)
            the selected action by an agent
        overwrite : Boolean, optional
            to select whether the actuator is already activated or not

        Returns
        -------
        None.

        """
        if speed > 1.0:
            self.speed = 1.0
        else:
            self.speed = speed
        if self.status == False or overwrite:
            self.t_activated = now
            
            if self.actiontype == "C":
                self.t_end = now + duration
                self.cur_action = self.speed
            elif self.actiontype == "B":
                self.t_end = now + duration * self.action_max
                self.cur_action = duration * self.action_max
            self.status = True
            
    def calc_mass(self, now = 0):
        """


        Parameters
        ----------
        now : Numeric Types
            current_simulation time

        Returns
        -------
        current transported mass

        """
        if self.status == True:
            if self.actiontype == "C":
                self.cur_mass_transport = self.mass_coeff*(self.speed*(self.action_max-self.action_min)+self.action_min)
            elif self.actiontype == "B":
               self.cur_mass_transport = self.mass_coeff*self.speed
        return self.cur_mass_transport
    
    def calc_energy(self):
        """


        Returns
        -------
        current power consumption

        """
        if self.status == True:
            if self.actiontype == "C":
                self.cur_power  = self.speed*(self.power_max-self.power_min)+self.power_min
            elif self.actiontype == "B":
                self.cur_power = self.power_max
        return self.cur_power / 1000.0
        
    def update(self, now):
        """


        Parameters
        ----------
        now : Numeric Types
            current_simulation time

        Returns
        -------
        None.

        """
        if self.status == True:
            if self.t_end < now:
                self.deactivate()
    
    def deactivate(self):
        self.status             == False
        self.cur_mass_transport = 0
        self.cur_power          = 0
        
    
        
        
 

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class Reservoir:
    reg_r           = []
    idx_r           = []
    vol_max         = 0
    vol_init_abs    = 0
    vol_cur_abs     = 0
    vol_cur_rel     = 0
    change          = 0

    def __init__(self, vol_max, vol_init_abs=0):
        """
        

        Parameters
        ----------
        vol_max : Numeric Types
            maximum volume of a reservoir
        vol_init_abs : Numeric Types
            absolute initial volume of a reservoir

        Returns
        -------
        None.

        """
        self.idx_r          = len(self.reg_r)
        self.reg_r.append(self)
        self.vol_max        = vol_max
        self.vol_init_abs   = vol_init_abs
        self.vol_cur_abs    = vol_init_abs
        self.vol_cur_rel    = self.vol_cur_abs / self.vol_max
        self.change         = 0
        
    def set_change(self, vol_change):
        """
        

        Parameters
        ----------
        vol_change : Numeric Types
            volume change for each time step

        Returns
        -------
        None.

        """
        self.change = vol_change
            
    def update(self):
        self.vol_cur_abs += self.change
        self.vol_cur_rel = self.vol_cur_abs / self.vol_max
        
    
    
    
    
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
        
class Silo(Reservoir):
    reg_s   = []
    idx_s   = []
    name    = ""
    type_   = ""
    
    def __init__(self, name, vol_max, vol_cur = 0, mode = "abs"):
        """
        

        Parameters
        ----------
        nname : Text Type
            label for an actuator (e.g. Hopper_A, Silo_A, etc.)
        vol_max : Numeric Types
            maximum volume of an reservoir
        vol_cur : Numeric Types
            current volume of an reservoir
        mode : Text Type
            The default is "abs".

        Returns
        -------
        None.

        """
        if mode == "abs":
            vol_cur = vol_cur
        elif mode == "rel":
            vol_cur = vol_max * vol_cur
        Reservoir.__init__(self, vol_max, vol_cur)
        self.name   = name
        self.idx_s  = len(self.reg_s)
        self.reg_s.append(self)
        self.type_  = "SIL"




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
    
class Hopper(Reservoir):
    reg_h   = []
    idx_h   = []
    name    = ""
    type_   = ""
    
    def __init__(self, name, vol_max, vol_cur = 0, mode = "abs"):
        """
        

        Parameters
        ----------
        nname : Text Type
            label for an actuator (e.g. Hopper_A, Silo_A, etc.)
        vol_max : Numeric Types
            maximum volume of an reservoir
        vol_cur : Numeric Types
            current volume of an reservoir
        mode : Text Type
            The default is "abs".

        Returns
        -------
        None.

        """
        if mode == "abs":
            vol_cur = vol_cur
        elif mode == "rel":
            vol_cur = vol_max * vol_cur
        Reservoir.__init__(self, vol_max, vol_cur)
        self.name   = name
        self.idx_h  = len(self.reg_h)
        self.reg_h.append(self)
        self.type_  = "HOP" 






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class BGLP(Environment):
    C_NAME              = "BGLP"
    C_LATENCY           = timedelta(0,1,0)    
    C_INFINITY          = np.finfo(np.float32).max
    sils                = []
    hops                = []
    ress                = []
    blts                = []
    vacs                = []
    acts                = []
    con_act_to_res      = []
    m_param             = []
    _demand             = 0
    t                   = 0
    t_step              = 0
    lr_margin           = 0
    lr_demand           = 0
    lr_energy           = 0
    overflow            = []
    demand              = []
    energy              = []
    transport           = []
    reward              = []
    levels_init         = 0
    reset_levels        = 0
    overflow_t          = 0
    demand_t            = 0
    energy_t            = 0
    transport_t         = 0
    margin_t            = 0
    prod_reached        = 0
    prod_target         = 0
    prod_scenario       = 0

    def __init__(self, p_reward_type=Reward.C_TYPE_OVERALL, p_logging=True,
                 t_step=0.5, t_set=10.0, demand=0.1, lr_margin=1.0, lr_demand=4.0,
                 lr_energy=0.0010, margin_p=[0.2,0.8,4], prod_target=10000,
                 prod_scenario='continuous'):
        """
        Parameters:
            p_reward_type   Reward type to be computed
            p_logging       Boolean switch for logging
            t_step          Time per step
            t_set           A set of time to update action
            demand          Production demand (L/s)
            lr_margin       Learning rate for margin (rewarding)
            lr_demand       Learning rate for demand (rewarding)
            lr_energy       Learning rate for energy (rewarding)
            margin_p        Margin parameters
            prod_target     Production target in one episode (L)
            prod_scenario   Production scenarion ('continuous'/'batch')
        """
        
        self.num_envs       = 5                                                 # Number of internal sub-environments
        self.reward_type    = p_reward_type
        super().__init__(p_mode=Environment.C_MODE_SIM, p_logging=p_logging)
        
        self.C_SCIREF_TYPE    = self.C_SCIREF_TYPE_ARTICLE
        self.C_SCIREF_AUTHOR  = "Dorothea Schwung, Steve Yuwono, Andreas Schwung, Steven X. Ding"
        self.C_SCIREF_TITLE   = "Decentralized learning of energy optimal production policies using PLC-informed reinforcement learning"
        self.C_SCIREF_JOURNAL = "Computers & Chemical Engineering"
        self.C_SCIREF_YEAR    = "2021"
        self.C_SCIREF_MONTH   = "05"
        self.C_SCIREF_DAY     = "28"
        self.C_SCIREF_VOLUME  = "152"
        self.C_SCIREF_DOI     = "10.1016/j.compchemeng.2021.107382"
        
        self.t              = 0
        self.t_step         = t_step
        self.t_set          = t_set
        self._demand        = demand
        self.lr_margin      = lr_margin
        self.lr_demand      = lr_demand
        self.lr_energy      = lr_energy
        self.prod_target    = prod_target
        self.prod_scenario  = prod_scenario
        self.levels_init    = np.ones(6,1)*0.5
        self.sils           = []
        self.hops           = []
        self.ress           = []
        self.blts           = []
        self.vacs           = []
        self.acts           = []
        
        belt_a              = Belt("Belt_A", "C", 40.0, 50.5, 450, 1850, 0.01/60)
        belt_b              = Belt("Belt_B", "B", 0, 26.9, 0, self.t_set, 0.4)
        belt_c              = Belt("Belt_C", "C", 114.828, 370, 450, 1450, 0.01249/60)
        vac_b               = VacuumPump("Vac_B", 0, 305, 0.567, 4.575*int(self.t_set//4.575), [0.464, 0.0332])
        vac_c               = VacuumPump("Vac_C", 0, 456, 0.979, 9.5*int(self.t_set//9.5), [0.3535, 0.0096])
        
        sil_a               = Silo("Silo_A", 17.42, random.uniform(0,1), mode="rel")                
        sil_b               = Silo("Silo_B", 17.42, random.uniform(0,1), mode="rel")
        sil_c               = Silo("Silo_C", 17.42, random.uniform(0,1), mode="rel")
        hop_a               = Hopper("Hopper_A", 9.1, random.uniform(0,1), mode="rel")                
        hop_b               = Hopper("Hopper_B", 9.1, random.uniform(0,1), mode="rel")
        hop_c               = Hopper("Hopper_C", 9.1, random.uniform(0,1), mode="rel")
        
        self.sils.append(sil_a)
        self.sils.append(sil_b)
        self.sils.append(sil_c)
        self.hops.append(hop_a)
        self.hops.append(hop_b)
        self.hops.append(hop_c)
        self.ress.append(sil_a)
        self.ress.append(hop_a)
        self.ress.append(sil_b)
        self.ress.append(hop_b)
        self.ress.append(sil_c)
        self.ress.append(hop_c)
        
        self.blts.append(belt_a)
        self.blts.append(belt_b)
        self.blts.append(belt_c)
        self.vacs.append(vac_b)
        self.vacs.append(vac_c)
        self.acts.append(belt_a)
        self.acts.append(vac_b)
        self.acts.append(belt_b)
        self.acts.append(vac_c)
        self.acts.append(belt_c)
        
        self.margin_p           = margin_p
        self.margin             = np.zeros((len(self.ress),1))
        self.overflow           = np.zeros((len(self.ress),1))
        self.demand             = np.zeros((len(self.ress),1))
        self.energy             = np.zeros((len(self.acts),1))
        self.transport          = np.zeros((len(self.acts),1))
        self.overflow_t         = np.zeros((len(self.ress),1))
        self.demand_t           = np.zeros((len(self.ress),1))
        self.energy_t           = np.zeros((len(self.acts),1))
        self.transport_t        = np.zeros((len(self.acts),1))
        self.margin_t           = np.zeros((len(self.ress),1))
        self.reward             = np.zeros((len(self.acts),1))
        self.con_res_to_act     = [[-1,0],[0,1],[1,2],[2,3],[3,4],[4,-1]]
        
        self.data_lists         = ["time","overflow","energy","demand"]
        self.data_storing       = DataStoring(self.data_lists)
        self.data_frame         = None
        
        self.reset()
            
    def _setup_spaces(self):
        """
        To enrich the state and action space with specific dimensions. 
        """
        levels_max = [17.42, 9.10, 17.42, 9.10, 17.42, 9.10]
        self._state_space.add_dim(Dimension(0, 'E-0 LvlSiloA', 'R', 'Env-0 Level of Silo A', '', 'L', 'L',[0, levels_max[0]]))
        self._state_space.add_dim(Dimension(1, 'E-0 LvlHopperA', 'R', 'Env-0 Level of Hopper A', '', 'L', 'L',[0, levels_max[1]]))
        self._state_space.add_dim(Dimension(2, 'E-0 LvlSiloB', 'R', 'Env-0 Level of Silo B', '', 'L', 'L',[0, levels_max[2]]))
        self._state_space.add_dim(Dimension(3, 'E-0 LvlHopperB', 'R', 'Env-0 Level of Hopper B', '', 'L', 'L',[0, levels_max[3]]))
        self._state_space.add_dim(Dimension(4, 'E-0 LvlSiloC', 'R', 'Env-0 Level of Silo C', '', 'L', 'L',[0, levels_max[4]]))
        self._state_space.add_dim(Dimension(5, 'E-0 LvlHopperC', 'R', 'Env-0 Level of Hopper C', '', 'L', 'L',[0, levels_max[5]]))
        
        self._action_space.add_dim(Dimension(0, 'E-0 Act', 'R', 'Env-0 Actuator Control', '', '', '', [0,1]))
        self._action_space.add_dim(Dimension(1, 'E-1 Act', 'R', 'Env-1 Actuator Control', '', '', '', [0,1]))
        self._action_space.add_dim(Dimension(2, 'E-2 Act', 'Z', 'Env-2 Actuator Control', '', '', '', [0,1]))
        self._action_space.add_dim(Dimension(3, 'E-3 Act', 'R', 'Env-3 Actuator Control', '', '', '', [0,1]))
        self._action_space.add_dim(Dimension(4, 'E-4 Act', 'R', 'Env-4 Actuator Control', '', '', '', [0,1]))

    def collect_substates(self) -> State:
        """
        To provide a method that set the value of a state, which will be used
        for reset method
        """
        state = State(self._state_space)
        sub_state_val = self.calc_state()
        for i in range(len(sub_state_val)):
            state.set_value(i, sub_state_val[i])
        return state
    
    def reset(self, p_seed=None) -> None:
        """
        To reset environment
        """
        self.set_random_seed(p_seed)
        self.reset_levels()
        self.reset_actuators()
        obs                 = self.calc_state()
        self.t              = 0
        self.prod_reached   = 0
        self._state         = self.collect_substates()
        self.get_state().set_done(False)
        if self.data_frame == None:
            self.data_frame = 0
        else:
            self.data_frame += 1
        self.data_storing.add_frame(str(self.data_frame))

    def simulate_reaction(self, p_state:State, p_action:Action) -> State:
        """
        To simulate the environment reacting selected actions
        """
        action          = []
        for agent_id in p_action.get_agent_ids():
            action_elem = p_action.get_elem(agent_id)
            for action_id in action_elem.get_dim_ids():
                action_elem_env = ActionElement(self.get_action_space())
                action_elem_env.set_value(action_id, action_elem.get_value(action_id))
                action_env      = Action()
                action_env.add_elem(agent_id, action_elem_env)
                action.append(action_elem_env.get_value(action_id))
        
        self.overflow_t         = np.zeros((len(self.ress),1))
        self.demand_t           = np.zeros((len(self.ress),1))
        self.energy_t           = np.zeros((len(self.acts),1))
        self.transport_t        = np.zeros((len(self.acts),1))
        self.margin_t           = np.zeros((len(self.ress),1))
        
        x = 0
        while x < (self.t_set//self.t_step):
            overflow_diff, demand_diff, energy_diff, transport_diff, margin_diff = self.get_status(self.t, self._demand)
            self.overflow_t     += overflow_diff
            self.demand_t       += demand_diff
            self.energy_t       += energy_diff
            self.transport_t    += transport_diff
            self.margin_t       += margin_diff
            self.t              += self.t_step
            x += 1
            
        self.set_actions(action)
        self._state.set_done(False)
        self._state.set_broken(False)
        self._state = self.collect_substates()
        
        self.data_storing.memorize("time",str(self.data_frame),self.t)
        self.data_storing.memorize("overflow",str(self.data_frame), (sum(self.overflow_t[:])/self.t_set).item())
        self.data_storing.memorize("energy",str(self.data_frame), (sum(self.energy_t[:])/self.t_set).item())
        self.data_storing.memorize("demand",str(self.data_frame), (self.demand_t[-1]/self.t_set).item())

        return self._state

    def compute_done(self, p_state:State) -> bool:
        """
        Updates the goal achievement value in [0,1] and the flags done
        based on the current state. Please redefine.
   
        Returns:
          -
        """

        
        if self.prod_scenario == 'continuous':
            return False
        else:
            if self.prod_reached >= self.prod_target:
                return True
            else:
                return False
    
    def compute_broken(self, p_state:State) -> bool:
        """
        Updates the goal achievement value in [0,1] and the flags broken
        based on the current state. Please redefine.
   
        Returns:
          -
        """
        return False
    
    def _compute_goal_achievement(self, p_state:State=None):
        """
        Optional goal achievement computation.

        Parameters:
            p_state         Optional external state. If none, please use internal state.

        Returns:
            Goal avievement value in interval [0,1].
        """

        if self.prod_scenario == 'continuous':
            self.goal_achievement = 0.0
        else:
            if self.prod_reached >= self.prod_target:
                self.goal_achievement = 1.0
            else:
                self.goal_achievement = self.prod_reached/self.prod_target

    def compute_reward(self) -> Reward:
        """
        To calculate reward (can be redifined)
        """
        reward = Reward(self.reward_type)

        if self.reward_type == Reward.C_TYPE_OVERALL:
            r_overall = 0
            r_overall = r_overall + sum(self.calc_reward()).item()
            reward.set_overall_reward(r_overall)

        else:
           for agent_id in self.last_action.get_agent_ids():
                agent_action_elem   = self.last_action.get_elem(agent_id)
                agent_action_ids    = agent_action_elem.get_dim_ids()
                r_agent             = 0
                r_reward            = self.calc_reward()
                for action_id in agent_action_ids:
                    r_action        = r_reward[action_id]
                    if self.reward_type == Reward.C_TYPE_EVERY_ACTION:
                        reward.add_action_reward(agent_id, action_id, r_action)
                    elif self.reward_type == Reward.C_TYPE_EVERY_AGENT:
                        r_agent = r_agent + r_action
                if self.reward_type == Reward.C_TYPE_EVERY_AGENT:
                    r_agent = r_agent / len(agent_action_ids)
                    reward.add_agent_reward(agent_id, r_agent)
        return reward

    def visualize(self) -> None:
        """
        Updates the visualization of the environment. Please redefine.
        """
        
        pass
  
    def calc_mass_flows(self):
        """
        To calculate mass flow transported by actuators
        """
        for act_num in range(len(self.acts)):
            self.transport[act_num] = self.acts[act_num].calc_mass(self.t)*self.t_step

    def calc_energy(self):
        """
        To calculate power consumptions per actuator
        """
        for act_num in range(len(self.acts)):
            self.energy[act_num] = self.acts[act_num].calc_energy()*self.t_step
                
    def calc_margin(self):   
        """
        To calculate margin level for every reservoir
        """             
        for i in range(len(self.ress)):
            vol_rel = self.ress[i].vol_cur_rel
            if vol_rel < self.margin_p[0]:
                self.margin[i] = (0-self.margin_p[2])/(self.margin_p[0])*(vol_rel-self.margin_p[0])*self.t_step
            elif vol_rel > self.margin_p[1]:
                self.margin[i] = self.margin_p[2]/(1-self.margin_p[1])*(vol_rel-self.margin_p[1])*self.t_step
            else:
                self.margin[i] = 0.0
                
    def set_volume_changes(self, demandval): 
        """
        To set up volume changes for every reservoir
        """
        for resnum in range(len(self.ress)):
            res = self.ress[resnum]
            current_vol = res.vol_cur_abs
            if resnum == 0:
                ins = 0
                outs = self.transport[self.con_res_to_act[resnum][1]]
            elif resnum == len(self.ress)-1:
                ins = self.transport[self.con_res_to_act[resnum][0]]
                outs = demandval*self.t_step
            else:
                ins = self.transport[self.con_res_to_act[resnum][0]]
                outs = self.transport[self.con_res_to_act[resnum][1]]
                
            if ins > self.ress[resnum-1].vol_cur_abs and resnum != 0:
                ins = self.ress[resnum-1].vol_cur_abs
            demand = current_vol+ins-outs
            if outs > current_vol:
                outs = current_vol
            if demand > 0:
                demand = 0
            overflow = current_vol+ins-outs-res.vol_max
            if overflow < 0:
                overflow = 0
            self.overflow[resnum] = overflow
            self.demand[resnum] = demand
            res.set_change(ins-outs-overflow)
            if resnum == len(self.ress)-1:
                self.prod_reached += outs

    def update_levels(self):
        """
        To update current reservoirs' level'
        """
        for resnum in range(len(self.ress)):
            res = self.ress[resnum]
            res.update()
            if resnum == 0:
                if res.vol_cur_rel <= self.margin_p[0]:
                    res.vol_cur_abs = self.levels_init[resnum]*res.vol_max
                    res.vol_cur_rel = self.levels_init[resnum]

    def reset_levels(self):
        """
        To reset reservoirs
        """
        self.levels_init = np.random.rand(6,1)
        for resnum in range(len(self.ress)):
            res = self.ress[resnum]
            res.vol_cur_abs = self.levels_init[resnum]*res.vol_max
            res.vol_cur_rel = self.levels_init[resnum]
    
    def reset_actuators(self):
        """
        To reset actuators
        """
        for act in self.acts:
            act.deactivate()

    def update_actuators(self):
        """
        To update actuators
        """
        for act in self.acts:
            act.update(self.t)
                
    def update(self, demandval):
        """
        To set up volume changes, update reservoirs' level, and update actuators'
        """
        self.set_volume_changes(demandval)
        self.update_levels()
        self.update_actuators()
        
    def get_status(self, t, demandval):
        """
        To calculate overflow, demand, energy, transport, and margin.
        This function will be called every time step.
        """
        self.t = t       
        self.calc_mass_flows()
        self.calc_energy()
        self.calc_margin()
        self.update(demandval)        
        return self.overflow, self.demand, self.energy, self.transport, self.margin
            
    def set_actions(self, actions):   
        """
        To set up actions for actuators.
        This function will be called every time set.
        """
        t_set = self.t_set-2*self.t_step  
        for actnum in range(len(self.acts)):
            act = self.acts[actnum]
            if act.type_ == "VAC":
                act.start_t(self.t, actions[actnum])
            elif act.type_ == "BLT":
                if act.actiontype == "B":
                    if actions[actnum] >= 0.5:
                        actions[actnum] = 1
                    else:
                        actions[actnum] = 0
                act.start_t(self.t, t_set, actions[actnum])
    
    def calc_state(self):
        """
        To get current levels of reservoirs
        """
        levels = np.zeros((len(self.ress),1))
        for resnum in range(len(self.ress)):
            res = self.ress[resnum]
            levels[resnum] = res.vol_cur_rel
        return levels
            
    def calc_reward(self):
        """
        To calculate the utility/reward
        """
        for actnum in range(len(self.acts)):
            acts = self.acts[actnum]
            self.reward[actnum] = 1/(1+self.lr_margin*self.margin_t[actnum])+1/(1+self.lr_energy*self.energy_t[actnum]/(acts.power_max/1000.0))
            if actnum == len(self.acts)-1:
                self.reward[actnum] += 1/(1-self.lr_demand*self.demand_t[-1])
            else:
                self.reward[actnum] += 1/(1+self.lr_margin*self.margin_t[actnum+1])
        return self.reward[:]
