## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.rl.envs
## -- Module  : bglp.py
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
## -- 2021-11-26  2.1.5     SY       Update reward type
## -- 2021-12-03  2.1.6     DA       Refactoring
## -- 2021-12-09  2.1.7     SY       Clean code assurance
## -- 2021-12-19  2.1.8     DA       Replaced 'done' by 'success'
## -- 2021-12-21  2.1.9     DA       Class BGLP: renamed method reset() to _reset()
## -- 2022-01-21  2.2.0     SY       Add cycle_limit as an input parameter
## -- 2022-01-24  2.2.1     SY       Update seeding procedure, refactoring _reset()
## -- 2022-02-25  2.2.2     SY       Refactoring due to auto generated ID in class Dimension
## -- 2022-05-23  2.2.3     SY       Bug fixing: Reward computation
## -- 2022-05-30  2.2.4     SY       Replace 'energy' related parameters to 'power'
## -- 2022-06-14  2.2.5     SY       Add termination condition for batch production scenario
## -- 2022-08-24  2.2.6     SY       Update state calculation function
## -- 2022-11-09  2.2.7     DA       Refactoring due to changes on plot systematics
## -- 2023-02-22  2.3.0     SY       Refactoring
## -- 2023-03-27  2.3.1     DA       Method BGLP._compute_reward(): refactoring of reward type
## --                                Reward.C_TYPE_EVERY_AGENT
## -- 2023-08-22  2.3.2     SY       Storing power consumption per actuator in data storing
## -- 2025-07-17  2.4.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.4.0 (2025-07-17) 

This module provides an RL environment of Bulk Good Laboratory Plant (BGLP).
"""

from datetime import timedelta
import random

import numpy as np

from mlpro.bf.various import *
from mlpro.bf.data import DataStoring
from mlpro.bf.math import ESpace, Dimension
from mlpro.bf.systems import State, Action

from mlpro.rl.models import *



# Export list for public API
__all__ = [ 'Actuator',
            'VacuumPump',   
            'Belt',
            'Reservoir',
            'Silo',
            'Hopper',
            'BGLP' ]



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Actuator:
    """
    This class serves as a parent class of different types of actuators, which provides the main 
    attributes of an actuator in the BGLP environment.

    Parameters
    ----------
    minpower : float
        minimum power of an actuator.
    maxpower : float
        maximum power of an actuator.
    minaction : float
        minimum action of an actuator.
    maxaction : float
        maximum action of an actuator.
    masscoeff : float
        mass transport coefficient of an actuator.
        
    Attributes
    ----------
    reg_a : list of objects
        list of existing actuators in the environment.
    idx_a : int
        length of reg_a.
    power_max : float
        maximum power of an actuator.
    power_min : float
        minimum power of an actuator.
    power_coeff : float
        power coefficient of an actuator, if necessary.
    action_max : float
        maximum action of an actuator.
    action_min : float
        minimum action of an actuator.
    mass_coeff : float
        mass transport coefficient of an actuator.
    t_activated : float
        a time indicator about an actuator is activated.
    t_end : float
        a time indicator about the end of an activation sequence of the actuator.
    status : bool
        status of an actuator, false means inactive and true means active.
    cur_mass_transport : float
        current transported mass of an actuator.
    cur_power : float
        current power consumption of an actuator.
    cur_action : float
        current taken action of an actuator in RL context.
    cur_speed : float
        current speed of an actuator.
    type_ : str
        a short name for an actuator, usually 3 capital letters (e.g VAC, BLT).
    """
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
    
## -------------------------------------------------------------------------------------------------
    def __init__(self, minpower, maxpower, minaction, maxaction, masscoeff):
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
class VacuumPump (Actuator):
    """
    This class inherits Actuator class and serves as a child class of Actuator.
    This class represents a type of actuators in the BGLP environment, namely Vacuum Pump.
    Vacuum Pumps are mostly used to transport material from mini hoppers to silos.
    However, the parameter of each vacuum pump can be dissimilar to each other based on their settings.

    Parameters
    ----------
    name : str
        specific name or id of a vacuum pump (e.g. Vac_A, etc.).
    minpower : float
        minimum power of a vacuum pump.
    maxpower : float
        maximum power of a vacuum pump.
    minaction : float
        minimum action of a vacuum pump.
    maxaction : float
        maximum action of a vacuum pump.
    masscoeff : float
        mass transport coefficient of a vacuum pump.
        
    Attributes
    ----------
    reg_v : list of objects
        list of existing vacuum pumps.
    idx_v : int
        length of reg_v.
    name : str
        specific name or id of a vacuum pump.
    t_end_max : float
        maximum end of activation time of a vacuum pump with respect to current time.
    """
    reg_v       = []
    idx_v       = 0
    name        = ""
    t_end_max   = 0
    
## -------------------------------------------------------------------------------------------------
    def __init__(self, name, minpower, maxpower, minaction, maxaction, masscoeff):    
        Actuator.__init__(self, minpower, maxpower, minaction, maxaction, masscoeff)
        self.idx_v  = len(self.reg_v)
        self.reg_v.append(self)
        self.name   = name
        self.type_  = "VAC"
        

## -------------------------------------------------------------------------------------------------
    def start_t(self, now, duration, overwrite = False): 
        """
        This method calculates the activation time and the end of activation time of the vacuum pump.
        This method is called, if the vacuum pump would like to be activated or updated.

        Parameters
        ----------
        now : float
            current time of the system.
        duration : float
            duration of the vacuum pump being activated or the action by an agent in RL context.
        overwrite : bool, optional
            To indicate whether the current operation can be overwritten or not.
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
                

## -------------------------------------------------------------------------------------------------
    def calc_mass(self, now):
        """
        This method calculates the transported mass flow by the vacuum pump for a time step.

        Parameters
        ----------
        now : float
            current time of the system.

        Returns
        -------
        cur_mass_transport : float
            current transported mass.

        """
        if self.status == True:
            t_diff = now - (self.t_activated + self.action_min)
            if t_diff >= 0:
                self.cur_mass_transport = (2*self.mass_coeff[1]) + self.mass_coeff[0]
        return self.cur_mass_transport        
        

## -------------------------------------------------------------------------------------------------
    def calc_power(self):
        """
        This method calculates the power consumption of a vacuum pump.

        Returns
        -------
        cur_power : float
            current power consumption.

        """
        if self.status == True:
            self.cur_power = self.power_max
        return self.cur_power / 1000.0
        

## -------------------------------------------------------------------------------------------------
    def update(self, now):
        """
        This method calculates whether a vacuum pump must be deactived or not.

        Parameters
        ----------
        now : float
            current time of the system.

        """
        if self.status == True:
            if self.t_end < now or self.t_end_max < now:
                self.deactivate()
    

## -------------------------------------------------------------------------------------------------
    def deactivate(self):
        """
        This method is used to deactivate a vacuum pump.
        """
        self.status             = False
        self.cur_mass_transport = 0
        self.cur_power          = 0




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Belt(Actuator):
    """
    This class inherits Actuator class and serves as a child class of Actuator.
    This class represents a type of actuators in the BGLP environment, namely Belt.
    This class can be used for Conveyor Belt, Rotary Feeder, Vibratory Conveyor, or similar type of actuators.
    Belts are mostly used to transport material from silos to hoppers.
    However, the parameter of each actuator can be dissimilar to each other based on their settings.

    Parameters
    ----------
    name : str
        specific name or id of an actuator (e.g. Belt_A, etc.).
    actiontype : str
        "C" for continuous action, "B" for binary action.
    minpower : float
        minimum power of an actuator.
    maxpower : float
        maximum power of an actuator.
    minaction : float
        minimum action of an actuator.
    maxaction : float
        maximum action of an actuator.
    masscoeff : float
        mass transport coefficient of an actuator.
        
    Attributes
    ----------
    reg_b : list of objects
        list of existing actuators.
    idx_b : int
        length of reg_b.
    name : str
        specific name or id of an actuator.
    actiontype : str
        "C" for continuous action, "B" for binary action.
    speed : float
        speed of an actuator.
    """
    reg_b       = []
    idx_b       = 0
    name        = ""
    actiontype  = ""
    speed       = 0
    
## -------------------------------------------------------------------------------------------------
    def __init__(self, name, actiontype, minpower, maxpower, minaction, maxaction, masscoeff):
        Actuator.__init__(self, minpower, maxpower, minaction, maxaction, masscoeff)
        self.idx_b = len(self.reg_b)
        self.reg_b.append(self)
        self.name = name
        self.actiontype = actiontype
        self.speed = 0
        self.type_ = "BLT"
        

## -------------------------------------------------------------------------------------------------
    def start_t(self, now, duration, speed, overwrite = False):
        """
        This method calculates the activation time and the end of activation time of the belt.
        This method is called, if the belt would like to be activated or updated.

        Parameters
        ----------
        now : float
            current time of the system.
        duration : float
            duration of the belt being activated, which being defined by the time set.
        speed : float
            speed of the belt or the action by an agent in RL context.
        overwrite : bool, optional
            To indicate whether the current operation can be overwritten or not.
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
            

## -------------------------------------------------------------------------------------------------
    def calc_mass(self, now):
        """
        This method calculates the transported mass flow by the belt for a time step.

        Parameters
        ----------
        now : float
            current time of the system.

        Returns
        -------
        cur_mass_transport : float
            current transported mass.

        """
        if self.status == True:
            if self.actiontype == "C":
                self.cur_mass_transport = self.mass_coeff*(self.speed*(self.action_max-self.action_min)+self.action_min)
            elif self.actiontype == "B":
               self.cur_mass_transport = self.mass_coeff*self.speed
        return self.cur_mass_transport
    

## -------------------------------------------------------------------------------------------------
    def calc_power(self):
        """
        This method calculates the power consumption of a belt.

        Returns
        -------
        cur_power : float
            current power consumption.

        """
        if self.status == True:
            if self.actiontype == "C":
                self.cur_power  = self.speed*(self.power_max-self.power_min)+self.power_min
            elif self.actiontype == "B":
                self.cur_power = self.power_max
        return self.cur_power / 1000.0
        

## -------------------------------------------------------------------------------------------------
    def update(self, now):
        """
        This method calculates whether a belt must be deactived or not.

        Parameters
        ----------
        now : float
            current time of the system.

        """
        if self.status == True:
            if self.t_end < now:
                self.deactivate()
    

## -------------------------------------------------------------------------------------------------
    def deactivate(self):
        """
        This method is used to deactivate a belt.
        """
        self.status             == False
        self.cur_mass_transport = 0
        self.cur_power          = 0




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Reservoir:
    """
    This class serves as a parent class of different types of reservoirs, which provides the main 
    attributes of a buffer in the BGLP environment.

    Parameters
    ----------
    vol_max : float
        maximum volume of a reservoir.
    vol_init_abs : float, optional
        initial volume of a reservoir. The default is 0.
        
    Attributes
    ----------
    reg_r : list of objects
        list of existing reservoirs in the environment.
    idx_r : int
        length of reg_r.
    vol_max : float
        maximum volume of a reservoir.
    vol_init_abs : float
        initial volume of a reservoir.
    vol_cur_abs : float
        current volume of a reservoir.
    vol_cur_rel : float
        current volume of a reservoir in percentage.
    change : float
        volume change of a reservoir in a time step.
    """
    reg_r           = []
    idx_r           = []
    vol_max         = 0
    vol_init_abs    = 0
    vol_cur_abs     = 0
    vol_cur_rel     = 0
    change          = 0

## -------------------------------------------------------------------------------------------------
    def __init__(self, vol_max, vol_init_abs=0):
        self.idx_r          = len(self.reg_r)
        self.reg_r.append(self)
        self.vol_max        = vol_max
        self.vol_init_abs   = vol_init_abs
        self.vol_cur_abs    = vol_init_abs
        self.vol_cur_rel    = self.vol_cur_abs / self.vol_max
        self.change         = 0
        

## -------------------------------------------------------------------------------------------------
    def set_change(self, vol_change):
        """
        This method sets up a volume change of a reservoir.

        Parameters
        ----------
        vol_change : float
            volume change of a reservoir in a time step.
        """
        self.change = vol_change
            

## -------------------------------------------------------------------------------------------------
    def update(self):
        """
        This method calculates the current volume of a reservoir after volume changes are made.
        
        """
        self.vol_cur_abs += self.change
        self.vol_cur_rel = self.vol_cur_abs / self.vol_max




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Silo(Reservoir):
    """
    This class inherits Reservoir class and serves as a child class of Reservoir.
    This class represents a type of buffers in the BGLP environment, namely Silo.
    Silos are used to temporary stored the transported materials.
    However, the parameter of each silo can be dissimilar to each other based on their settings.

    Parameters
    ----------
    name : str
        specific name or id of a silo (e.g. Silo_A, etc.).
    vol_max : float
        maximum capacity of a silo.
    vol_cur : float
        current volume of a silo.
    mode : str, optional
        mode of measuring the current volume.
        "abs" means absolute value, "rel" means percentage. The default is "abs".
    
    Attributes
    ----------
    reg_s : list of objects
        list of existing silos.
    idx_s : int
        length of reg_s.
    name : str
        specific name or id of a silo.
    type_ : str
        a short name for a silo, usually 3 capital letters (e.g SIL).
    """
    reg_s   = []
    idx_s   = []
    name    = ""
    type_   = ""
    
## -------------------------------------------------------------------------------------------------
    def __init__(self, name, vol_max, vol_cur = 0, mode = "abs"):
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
class Hopper (Reservoir):
    """
    This class inherits Reservoir class and serves as a child class of Reservoir.
    This class represents a type of buffers in the BGLP environment, namely Hopper.
    Hoppers are used to temporary stored the transported materials.
    However, the parameter of each hopper can be dissimilar to each other based on their settings.

    Parameters
    ----------
    name : str
        specific name or id of a hopper (e.g. Hop_A, etc.).
    vol_max : float
        maximum capacity of a hopper.
    vol_cur : float
        current volume of a hopper.
    mode : str, optional
        mode of measuring the current volume.
        "abs" means absolute value, "rel" means percentage. The default is "abs".
    
    Attributes
    ----------
    reg_h : list of objects
        list of existing silos.
    idx_h : int
        length of reg_h.
    name : str
        specific name or id of a hopper.
    type_ : str
        a short name for a hopper, usually 3 capital letters (e.g HOP).
    """
    reg_h   = []
    idx_h   = []
    name    = ""
    type_   = ""
    
## -------------------------------------------------------------------------------------------------
    def __init__(self, name, vol_max, vol_cur = 0, mode = "abs"):
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
class BGLP (Environment):
    """
    This is the main class of BGLP environment that inherits Environment class from MLPro.
    
    Parameters
    ----------
    p_reward_type : Reward, optional
        rewarding type. The default is Reward.C_TYPE_EVERY_AGENT.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging : Log, optional
        logging functionalities. The default is Log.C_LOG_ALL.
    t_step : float, optional
        time for each time step (in seconds). The default is 0.5.
    t_set : float, optional
        time set for one horizon (in seconds). The default is 10.0.
    demand : float, optional
        the constant output flow from the system (in L/s). The default is 0.1.
    lr_margin : float, optional
        the learning rate for margin parameter, related to implemented reward function. The default is 1.0.
    lr_demand : float, optional
        the learning rate for production demand, related to implemented reward function. The default is 4.0.
    lr_power : float, optional
        the learning rate for power consumption, related to implemented reward function. The default is 0.0010.
    margin_p : list of floats, optional
        the margin parameter of reservoirs [low, high, multplicator]. The default is [0.2,0.8,4].
    prod_target : float, optional
        the production target for batch operation (in L). The default is 10000.
    prod_scenario : str, optional
        'batch' means batch production scenario and 'continuous' means continuous production scenario. The default is 'continuous'.

    Attributes
    ----------
    C_NAME : str
        name of the environment.
    C_LATENCY : timedelta()
        latency.
    C_INFINITY : np.finfo()
        infinity.
    C_REWARD_TYPE : Reward
        rewarding type.
    sils : list of objects
        list of existing silos.
    hops : list of objects
        list of existing hoppers.
    ress : list of objects
        list of existing reservoirs.
    blts : list of objects
        list of existing belts.
    acts : list of objects
        list of existing actuators.
    vacs : list of objects
        list of existing vacuum pumps.
    con_act_to_res : list of lists of int
        connection between actuators and reservoirs and setup the sequence.
    m_param : list of floats
        the margin parameter of reservoirs [low, high, multplicator].
    _demand : float
        the constant output flow from the system (in L/s).
    t : float
        current time of the system (in seconds).
    t_step : float
        time for each time step (in seconds).
    lr_margin : float
        the learning rate for margin parameter.
    lr_demand : float
        the learning rate for production demand.
    lr_power : float
        the learning rate for power consumption.
    overflow : list of floats
        current overflow.
    power : list of floats
        current power consumption.
    transport : list of floats
        current transported mass flow.
    reward : list of floats
        current rewards.
    levels_init : float
        initial level of reservoirs.
    overflow_t : float
        current overflow for a specific buffer in a specific time.
    demand_t : float
        current demand for a specific buffer in a specific time.
    power_t : float
        current power consumption for a specific actuator in a specific time.
    transport_t : float
        current transported material by a specific actuaor in a specific time.
    margin_t : float
        current margin for a specific buffer in a specific time.
    prod_reached : float
        current production reached in L for batch operation.
    prod_target : float
        the production target for batch operation (in L).
    prod_scenario : str
        'batch' means batch production scenario and 'continuous' means continuous production scenario.
    cycle_limit : int
        the number of cycle limit.
    
    """
    C_NAME              = "BGLP"
    C_LATENCY           = timedelta(0,1,0)    
    C_INFINITY          = np.finfo(np.float32).max
    C_REWARD_TYPE       = Reward.C_TYPE_EVERY_AGENT
    C_CYCLE_LIMIT       = 0
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
    lr_power            = 0
    overflow            = []
    demand              = []
    power               = []
    transport           = []
    reward              = []
    levels_init         = 0
    reset_levels        = 0
    overflow_t          = 0
    demand_t            = 0
    power_t             = 0
    transport_t         = 0
    margin_t            = 0
    prod_reached        = 0
    prod_target         = 0
    prod_scenario       = 0

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_reward_type=Reward.C_TYPE_EVERY_AGENT, p_visualize:bool=False, p_logging=Log.C_LOG_ALL,
                 t_step=0.5, t_set=10.0, demand=0.1, lr_margin=1.0, lr_demand=4.0,
                 lr_power=0.0010, margin_p=[0.2,0.8,4], prod_target=10000,
                 prod_scenario='continuous', cycle_limit=0):
        self.num_envs       = 5                                                 # Number of internal sub-environments
        self.reward_type    = p_reward_type
        super().__init__(p_mode=Environment.C_MODE_SIM, p_visualize=p_visualize, p_logging=p_logging)
        
        self.C_SCIREF_TYPE    = self.C_SCIREF_TYPE_ARTICLE
        self.C_SCIREF_AUTHOR  = "Dorothea Schwung, Steve Yuwono, Andreas Schwung, Steven X. Ding"
        self.C_SCIREF_TITLE   = "Decentralized learning of energy optimal production policies using PLC-informed reinforcement learning"
        self.C_SCIREF_JOURNAL = "Computers & Chemical Engineering"
        self.C_SCIREF_YEAR    = "2021"
        self.C_SCIREF_MONTH   = "05"
        self.C_SCIREF_DAY     = "28"
        self.C_SCIREF_VOLUME  = "152"
        self.C_SCIREF_DOI     = "10.1016/j.compchemeng.2021.107382"
        
        self.C_CYCLE_LIMIT  = cycle_limit
        self.t              = 0
        self.t_step         = t_step
        self.t_set          = t_set
        self._demand        = demand
        self.lr_margin      = lr_margin
        self.lr_demand      = lr_demand
        self.lr_power       = lr_power
        self.prod_target    = prod_target
        self.prod_scenario  = prod_scenario
        self.levels_init    = np.ones((6,1))*0.5
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
        self.power              = np.zeros((len(self.acts),1))
        self.transport          = np.zeros((len(self.acts),1))
        self.overflow_t         = np.zeros((len(self.ress),1))
        self.demand_t           = np.zeros((len(self.ress),1))
        self.power_t            = np.zeros((len(self.acts),1))
        self.transport_t        = np.zeros((len(self.acts),1))
        self.margin_t           = np.zeros((len(self.ress),1))
        self.reward             = np.zeros((len(self.acts),1))
        self.con_res_to_act     = [[-1,0],[0,1],[1,2],[2,3],[3,4],[4,-1]]
        
        self.data_lists         = ["time",
                                   "total overflow",
                                   "total power",
                                   "demand",
                                   "power act1",
                                   "power act2",
                                   "power act3",
                                   "power act4",
                                   "power act5"]
        
        self.data_storing       = DataStoring(self.data_lists)
        self.data_frame         = None
        
        self.reset()
            

## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        """
        This method is used to setup action and state spaces of the system.
        
        The actions and states are normalized between 0 to 1.
        For the actions, 0 means minimum action and 1 means maximum action.
        Meanwhile, for the states, 0 means minimum capacity (empty) and 1 means maximum capacity (full)

        Returns
        -------
        state_space : ESpace()
            state space of the system.
        action_space : ESpace()
            action space of the system.

        """
        state_space     = ESpace()
        action_space    = ESpace()

        state_space.add_dim(Dimension('E-0 LvlSiloA', 'R', 'Res-1 Level of Silo A', '', '', '', [0, 1]))
        state_space.add_dim(Dimension('E-0 LvlHopperA', 'R', 'Res-2 Level of Hopper A', '', '', '', [0, 1]))
        state_space.add_dim(Dimension('E-0 LvlSiloB', 'R', 'Res-3 Level of Silo B', '', '', '', [0, 1]))
        state_space.add_dim(Dimension('E-0 LvlHopperB', 'R', 'Res-4 Level of Hopper B', '', '', '', [0, 1]))
        state_space.add_dim(Dimension('E-0 LvlSiloC', 'R', 'Res-5 Level of Silo C', '', '', '', [0, 1]))
        state_space.add_dim(Dimension('E-0 LvlHopperC', 'R', 'Res-6 Level of Hopper C', '', '', '', [0, 1]))
        
        action_space.add_dim(Dimension('E-0 Act', 'R', 'Act-0 Belt Conveyor A', '', '', '', [0,1]))
        action_space.add_dim(Dimension('E-1 Act', 'R', 'Act-1 Vacuum Pump B', '', '', '', [0,1]))
        action_space.add_dim(Dimension('E-2 Act', 'Z', 'Act-2 Vibratory Conveyor B', '', '', '', [0,1]))
        action_space.add_dim(Dimension('E-3 Act', 'R', 'Act-3 Vacuum Pump C', '', '', '', [0,1]))
        action_space.add_dim(Dimension('E-4 Act', 'R', 'Act-4 Rotary Feeder C', '', '', '', [0,1]))

        return state_space, action_space


## -------------------------------------------------------------------------------------------------
    def collect_substates(self) -> State:
        """
        This method is called during resetting the environment.

        Returns
        -------
        state : State
            current states.

        """
        state = State(self._state_space)
        sub_state_val = self.calc_state()
        for i in range(len(sub_state_val)):
            state.set_value(state.get_dim_ids()[i], sub_state_val[i])
        return state
    

## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:
        """
        This method is used to reset the environment.

        Parameters
        ----------
        p_seed : int
            Seed parameter for an internal random generator.

        """
        np.random.seed(p_seed)
        self.levels_init = np.random.rand(6,1)
        self.reset_levels()
        self.reset_actuators()
        obs                 = self.calc_state()
        self.t              = 0
        self.prod_reached   = 0
        self._state         = self.collect_substates()
        self.get_state().set_success(False)
        self.get_state().set_broken(False)
        if self.data_frame == None:
            self.data_frame = 0
        else:
            self.data_frame += 1
        self.data_storing.add_frame(str(self.data_frame))


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state:State, p_action:Action) -> State:
        """
        This method is used to calculate the next states of the system after a set of actions.

        Parameters
        ----------
        p_state : State
            State.
        p_action : Action
            ACtion.

        Returns
        -------
        _state : State
            current states.

        """
        action          = []
        for agent_id in p_action.get_agent_ids():
            action_elem = p_action.get_elem(agent_id)
            for action_id in action_elem.get_dim_ids():
                action.append(action_elem.get_value(action_id))
        
        self.overflow_t         = np.zeros((len(self.ress),1))
        self.demand_t           = np.zeros((len(self.ress),1))
        self.power_t            = np.zeros((len(self.acts),1))
        self.transport_t        = np.zeros((len(self.acts),1))
        self.margin_t           = np.zeros((len(self.ress),1))
            
        self.set_actions(action)
        x = 0
        while x < (self.t_set//self.t_step):
            overflow_diff, demand_diff, power_diff, transport_diff, margin_diff = self.get_status(self.t, self._demand)
            self.overflow_t     += overflow_diff
            self.demand_t       += demand_diff
            self.power_t        += power_diff
            self.transport_t    += transport_diff
            self.margin_t       += margin_diff
            self.t              += self.t_step
            x += 1
            
        self._state.set_success(False)
        self._state.set_broken(False)
        self._state = self.collect_substates()
        
        self.data_storing.memorize("time",str(self.data_frame),self.t)
        self.data_storing.memorize("total overflow",str(self.data_frame), (sum(self.overflow_t[:])/self.t_set).item())
        self.data_storing.memorize("total power",str(self.data_frame), (sum(self.power_t[:])/self.t_set).item())
        self.data_storing.memorize("demand",str(self.data_frame), (self.demand_t[-1]/self.t_set).item())
        self.data_storing.memorize("power act1",str(self.data_frame), (self.power_t[0]/self.t_set).item())
        self.data_storing.memorize("power act2",str(self.data_frame), (self.power_t[1]/self.t_set).item())
        self.data_storing.memorize("power act3",str(self.data_frame), (self.power_t[2]/self.t_set).item())
        self.data_storing.memorize("power act4",str(self.data_frame), (self.power_t[3]/self.t_set).item())
        self.data_storing.memorize("power act5",str(self.data_frame), (self.power_t[4]/self.t_set).item())

        return self._state


## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state:State) -> bool:
        """
        This method computes the success flag. This method can be redefined.

        Parameters
        ----------
        p_state : State
            state.

        Returns
        -------
        bool
            success or not success.

        """        
        if self.prod_scenario == 'continuous':
            return False
        else:
            if self.prod_reached >= self.prod_target:
                self._state.set_terminal(True)
                return True
            else:
                return False
    

## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state:State) -> bool:
        """
        This method computes the broken flag. This method can be redefined.

        Parameters
        ----------
        p_state : State
            state.

        Returns
        -------
        bool
            broken or not.

        """ 
        return False
    

## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old:State, p_state_new:State) -> Reward:
        """
        This method calculates the reward for different reward types.

        Parameters
        ----------
        p_state_old : State
            previous state.
        p_state_new : State
            new state.

        Returns
        -------
        reward : Reward
            reward values.

        """
        reward = Reward(self.reward_type)

        if self.reward_type == Reward.C_TYPE_OVERALL:
            r_overall = 0
            r_overall = r_overall + sum(self.calc_reward()).item()
            reward.set_overall_reward(r_overall)
        
        elif self.reward_type == Reward.C_TYPE_EVERY_AGENT:
           for i, agent_id in enumerate( self._last_action.get_agent_ids() ):
               r_reward = self.calc_reward()
               reward.add_agent_reward(agent_id, r_reward[i])
               
        else:
           for agent_id in self._last_action.get_agent_ids():
                agent_action_elem = self._last_action.get_elem(agent_id)
                agent_action_ids = agent_action_elem.get_dim_ids()
                r_agent = 0
                r_reward = self.calc_reward()
                action_idx = 0
                for action_id in agent_action_ids:
                    r_action = r_reward[action_idx]
                    action_idx += 1
                    reward.add_action_reward(agent_id, action_id, r_action)
                    
        return reward

  
## -------------------------------------------------------------------------------------------------
    def calc_mass_flows(self):
        """
        This method calculates the mass flow transported by actuators.
        
        """
        for act_num in range(len(self.acts)):
            self.transport[act_num] = self.acts[act_num].calc_mass(self.t)*self.t_step


## -------------------------------------------------------------------------------------------------
    def calc_power(self):
        """
        This method calculates the power consumptions per actuator.
        
        """
        for act_num in range(len(self.acts)):
            self.power[act_num] = self.acts[act_num].calc_power()*self.t_step
                

## -------------------------------------------------------------------------------------------------
    def calc_margin(self):   
        """
        This method calculates margin level for every reservoir.
        
        """             
        for i in range(len(self.ress)):
            vol_rel = self.ress[i].vol_cur_rel
            if vol_rel < self.margin_p[0]:
                self.margin[i] = (0-self.margin_p[2])/(self.margin_p[0])*(vol_rel-self.margin_p[0])*self.t_step
            elif vol_rel > self.margin_p[1]:
                self.margin[i] = self.margin_p[2]/(1-self.margin_p[1])*(vol_rel-self.margin_p[1])*self.t_step
            else:
                self.margin[i] = 0.0
                

## -------------------------------------------------------------------------------------------------
    def set_volume_changes(self, demandval): 
        """
        This method sets up volume changes for every reservoir.
        
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


## -------------------------------------------------------------------------------------------------
    def update_levels(self):
        """
        This method updates the current level of reservoirs.
        
        """
        for resnum in range(len(self.ress)):
            res = self.ress[resnum]
            res.update()
            if resnum == 0:
                if res.vol_cur_rel <= self.margin_p[0]:
                    res.vol_cur_abs = self.levels_init[resnum]*res.vol_max
                    res.vol_cur_rel = self.levels_init[resnum]


## -------------------------------------------------------------------------------------------------
    def reset_levels(self):
        """
        This method resets reservoirs.
        
        """
        for resnum in range(len(self.ress)):
            res = self.ress[resnum]
            res.vol_cur_abs = self.levels_init[resnum]*res.vol_max
            res.vol_cur_rel = self.levels_init[resnum]
    

## -------------------------------------------------------------------------------------------------
    def reset_actuators(self):
        """
        This method resets actuators.
        
        """
        for act in self.acts:
            act.deactivate()


## -------------------------------------------------------------------------------------------------
    def update_actuators(self):
        """
        This method updates actuators.
        
        """
        for act in self.acts:
            act.update(self.t)
                

## -------------------------------------------------------------------------------------------------
    def update(self, demandval):
        """
        This method sets up volume changes, updates reservoirs' level, and updates actuators.
        
        Parameters
        ----------
        demandval : float
            production outflow target in L/s.
            
        """
        self.set_volume_changes(demandval)
        self.update_levels()
        self.update_actuators()
        

## -------------------------------------------------------------------------------------------------
    def get_status(self, t, demandval):
        """
        This method calculates overflow, demand, power, transport, and margin.
        This function will be called every time step.
        
        Parameters
        ----------
        t : float
            current time in sec.
        demandval : float
            production outflow target in L/s.

        Returns
        -------
        overflow : list of floats
            overflow levels.
        demand : list of floats
            demand fulfilled.
        power : list of floats
            power consumptions.
        transport : list of floats
            transported materials.
        margin : list of floats
            margin levels.

        """
        self.t = t       
        self.calc_mass_flows()
        self.calc_power()
        self.calc_margin()
        self.update(demandval)        
        return self.overflow, self.demand, self.power, self.transport, self.margin
            

## -------------------------------------------------------------------------------------------------
    def set_actions(self, actions):   
        """
        This method sets up actions for actuators. This function will be called every time set.
        
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
    

## -------------------------------------------------------------------------------------------------
    def calc_state(self):
        """
        This method obtains current levels of reservoirs.

        Returns
        -------
        levels : list of floats
            level of each reservoir.
        
        """
        levels = np.zeros((len(self.ress),1))
        for resnum in range(len(self.ress)):
            res = self.ress[resnum]
            levels[resnum] = res.vol_cur_rel
        return levels
            

## -------------------------------------------------------------------------------------------------
    def calc_reward(self):
        """
        This method calculates the reward. This method can be redifined!
        The current reward function is suitable for continuous operation and scalar reward for individual agents.

        Returns
        -------
        reward : list of floats
            reward for each agent.
        """
        for actnum in range(len(self.acts)):
            acts = self.acts[actnum]
            self.reward[actnum] = 1/(1+self.lr_margin*self.margin_t[actnum])+1/(1+self.lr_power*self.power_t[actnum]/(acts.power_max/1000.0))
            if actnum == len(self.acts)-1:
                self.reward[actnum] += 1/(1-self.lr_demand*self.demand_t[-1])
            else:
                self.reward[actnum] += 1/(1+self.lr_margin*self.margin_t[actnum+1])
        return self.reward[:]


## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        pass


## -------------------------------------------------------------------------------------------------
    def update_plot(self):
        pass