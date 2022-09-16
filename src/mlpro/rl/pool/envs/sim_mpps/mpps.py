## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.envs.sim_mpps
## -- Module  : mpps.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-08-24  0.0.0     SY/ML    Creation
## -- 2022-??-??  1.0.0     SY/ML    Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2022-08-24)

This module provides a multi-purpose environment of a continuous and batch production systems with
modular settings and high-flexibility.

The users are able to develop and simulate their own production systems including setting up own
actuators, reservoirs, modules/stations, production sequences and many more. We also provide the
default implementations of actuators, reservoirs, and modules, which can be found in the pool of
objects.

To be noted, the usage of this simulation is not limited to RL tasks, but it also can be as a
testing environment for GT tasks, evolutionary algorithms, supervised learning, model predictive
control, and many more.
"""


from mlpro.rl.models import *
from mlpro.bf.various import *
import numpy as np
import random
import uuid




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class Actuator(ScientificObject, Log):
    """
    This class serves as a base class of actuators, which provides the main attributes of an actuator.
    
    Parameters
    ----------
    
        
    Attributes
    ----------
    

    """

    C_TYPE = 'Actuator'
    C_NAME = ''

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str,
                 p_status:bool=False,
                 p_id:int=None,
                 p_logging=Log.C_LOG_ALL):
        
        if p_name != '':
            self.set_name(p_name)
        else:
            self.set_name(self.C_NAME)
        self.set_id(p_id)
        self.set_status(p_status)
        
        Log.__init__(self, p_logging=p_logging)
        self._process = None
        self.setup_process()
        self.reset()
        

## -------------------------------------------------------------------------------------------------
    def set_id(self, p_id:int=None):
        if p_id is None:
            self._id = str(uuid.uuid4())
        else:
            self._id = str(p_id)


## -------------------------------------------------------------------------------------------------
    def get_id(self) -> str:
        return self._id


## -------------------------------------------------------------------------------------------------
    def set_name(self, p_name):
        self._name = p_name
        self.C_NAME = p_name


## -------------------------------------------------------------------------------------------------
    def get_name(self) -> str:
        return self._name
        

## -------------------------------------------------------------------------------------------------
    def activate(self, p_max_actuation, **p_args) -> bool:
        if not self.get_status():
            self.set_status(True)
            self._actuation_time = 0
            self._max_actuation_time = p_max_actuation
            self.log(self.C_LOG_TYPE_I, 'Actuator ' + self.get_name() + ' is turned on.')
            raise NotImplementedError('Turning on procedure is missing, please redefine this function!')
        else:
            self.log(self.C_LOG_TYPE_E, 'Actuator ' + self.get_name() + ' is still on.')
            raise NotImplementedError('Please redefine this function! You can either overwrite the current process or continue with the current process')


## -------------------------------------------------------------------------------------------------    
    def deactivate(self, **p_args) -> bool:
        self._actuation_time = 0
        try:
            if p_args['force_stop']:
                self.set_status(False)
                raise NotImplementedError('Turning off procedure due to force stop is missing, please redefine this function!')
        except:
            pass
        
        try:
            if p_args['emergency_stop']:
                self.set_status(False)
                raise NotImplementedError('Turning off procedure due to emergency stop is missing, please redefine this function!')
        except:
            pass
        
        if self.get_status():
            self.set_status(False)
            self.log(self.C_LOG_TYPE_I, 'Actuator ' + self.get_name() + ' is turned off.')
            raise NotImplementedError('Turning off procedure is missing, please redefine this function!')
        else:
            self.log(self.C_LOG_TYPE_E, 'Actuator ' + self.get_name() + ' is already off.')
            return True
  
    
## -------------------------------------------------------------------------------------------------      
    def set_status(self, p_status:bool=False):
        self.status = p_status
  
    
## -------------------------------------------------------------------------------------------------      
    def get_status(self) -> bool:
        return self.status


## -------------------------------------------------------------------------------------------------        
    def reset(self):
        if self.get_status():
            self.force_stop()
        
        self.log(self.C_LOG_TYPE_I, 'Actuator ' + self.get_name() + ' is succesfully reset.')
        
        raise NotImplementedError('Please redefine this function!')


## -------------------------------------------------------------------------------------------------    
    def emergency_stop(self):
        self.deactivate(emergency_stop=True)
    
        self.log(self.C_LOG_TYPE_W, 'Actuator ' + self.get_name() + ' is stopped due to emergeny.')


## -------------------------------------------------------------------------------------------------    
    def force_stop(self):
        self.deactivate(force_stop=True)
    
        self.log(self.C_LOG_TYPE_I, 'Actuator ' + self.get_name() + ' is forcely stopped.')


## -------------------------------------------------------------------------------------------------    
    def setup_process(self):
        if self._process() is None:
            self._process = Process(self.get_name())
            
        # self._process.add(p_name, p_id, p_type, p_param_1=.., p_param_2=.., .....)
        # self._process.add(p_name, p_id, p_type, p_param_1=.., p_param_2=.., .....)

        raise NotImplementedError('Please redefine this function and setup processes!')


## -------------------------------------------------------------------------------------------------    
    def run_process(self,
                    p_time_step:float,
                    p_max_actuation:float=None,
                    p_activate:bool=True,
                    p_stop:bool=False,
                    **p_args) -> dict:
        if p_activate:
            self.activate(p_max_actuation, p_args)
            
        if p_stop:
            self.deactivate()
            p_time_step = 0
            
        if (self._max_actuation_time is not None) and (self._actuation_time >= self._max_actuation_time):
            self.deactivate()
            p_time_step = 0
            
        output = self._process.run(self._actuation_time, p_time_step)
        
        if self.get_status():
            self._actuation_time += p_time_step
        
        return output




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class Reservoir(ScientificObject, Log):
    """
    This class serves as a base class of reservoirs, which provides the main attributes of a reservoir.
    
    Parameters
    ----------
    
        
    Attributes
    ----------
    

    """

    C_TYPE = 'Reservoir'
    C_NAME = ''
    C_RES_TYPE_CONT = 0
    C_RES_TYPE_2POS = 1
    C_2POS_SENSORS_LEVELS_1 = 'Low'
    C_2POS_SENSORS_LEVELS_2 = 'Medium'
    C_2POS_SENSORS_LEVELS_3 = 'High'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str,
                 p_max_capacity:float,
                 p_sensor:int=None,
                 p_id:int=None,
                 p_logging=Log.C_LOG_ALL,
                 p_init:float=None,
                 p_sensor_low:float=None,
                 p_sensor_high:float=None):
        
        if p_name != '':
            self.set_name(p_name)
        else:
            self.set_name(self.C_NAME)
        self.set_id(p_id)

        Log.__init__(self, p_logging=p_logging)
        if p_sensor is None:
            raise ParamError('sensor type is missing.')
        else:
            self.sensor_type = p_sensor
        if self.sensor_type == self.C_RES_TYPE_2POS:
            try:
                self.sensor_low = p_sensor_low
                self.sensor_high = p_sensor_high
            except:
                raise ParamError('sensor_low and sensor_high parameters are missing.')
        self.set_maximum_capacity(p_max_capacity)
        self.set_initial_level(p_init)
        self.reset()
        self.overflow = 0
        

## -------------------------------------------------------------------------------------------------
    def set_id(self, p_id:int=None):
        if p_id is None:
            self._id = str(uuid.uuid4())
        else:
            self._id = str(p_id)


## -------------------------------------------------------------------------------------------------
    def get_id(self) -> str:
        return self._id


## -------------------------------------------------------------------------------------------------
    def set_name(self, p_name):
        self._name = p_name
        self.C_NAME = p_name


## -------------------------------------------------------------------------------------------------
    def get_name(self) -> str:
        return self._name
        

## -------------------------------------------------------------------------------------------------
    def update(self, p_in:float, p_out:float):
        self._volume = self._volume + p_in - p_out
        self.overflow = 0
        if self._volume < 0:
            self._volume = 0
        elif self._volume > self.max_capacity:
            self.overflow = self._volume - self.max_capacity
            self._volume = self.max_capacity
        self.log(self.C_LOG_TYPE_I, 'Reservoir ' + self.get_name() + ' is updated.')
        

## -------------------------------------------------------------------------------------------------
    def get_volume(self):
        if self.sensor_type == self.C_RES_TYPE_CONT:
            return self._volume
        elif self.sensor_type == self.C_RES_TYPE_2POS:
            if self._volume < self.sensor_low:
                return self.C_2POS_SENSORS_LEVELS_1
            elif self._volume > self.sensor_high:
                return self.C_2POS_SENSORS_LEVELS_3
            else:
                return self.C_2POS_SENSORS_LEVELS_2

## -------------------------------------------------------------------------------------------------
    def set_maximum_capacity(self, p_max_capacity:float):
        self.max_capacity = p_max_capacity
        

## -------------------------------------------------------------------------------------------------
    def get_maximum_capacity(self) -> float:
        return self.max_capacity
        

## -------------------------------------------------------------------------------------------------
    def get_overflow(self) -> float:
        return self.overflow
        

## -------------------------------------------------------------------------------------------------
    def reset(self):
        self._volume = self.init_level
        self.log(self.C_LOG_TYPE_I, 'Reservoir ' + self.get_name() + ' is reset.')
        

## -------------------------------------------------------------------------------------------------
    def set_initial_level(self, p_init:float=None):
        if p_init is None:
            self.init_level = random.uniform(0, self.get_maximum_capacity())
        else:
            self.init_level = p_init
        

## -------------------------------------------------------------------------------------------------
    def get_initial_level(self) -> float:
        return self.init_level




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class ManufacturingProcess:


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name, p_max_capacity, p_prod_rate, **p_param):
        ...
        ...


## -------------------------------------------------------------------------------------------------
    def set_id(self):
        ...


## -------------------------------------------------------------------------------------------------
    def get_id(self):
        ...


## -------------------------------------------------------------------------------------------------
    def set_name(self):
        ...


## -------------------------------------------------------------------------------------------------
    def get_name(self):
        ...
        

## -------------------------------------------------------------------------------------------------
    def start_process(self):
        ...
        

## -------------------------------------------------------------------------------------------------
    def stop_process(self):
        ...
        

## -------------------------------------------------------------------------------------------------
    def get_status(self):
        ...
        

## -------------------------------------------------------------------------------------------------
    def get_maximum_capacity(self):
        ...
        

## -------------------------------------------------------------------------------------------------
    def get_prod_rate(self):
        ...
        

## -------------------------------------------------------------------------------------------------
    def get_waiting_materials(self):
        ...
        

## -------------------------------------------------------------------------------------------------
    def get_finished_products(self):
        ...
        

## -------------------------------------------------------------------------------------------------
    def reset(self):
        ...
        

## -------------------------------------------------------------------------------------------------
    def update(self):
        #update waiting materials
        #update current process
        #update finished products
        ...




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class Module:


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name, **p_param):
        ...


## -------------------------------------------------------------------------------------------------
    def set_id(self):
        ...


## -------------------------------------------------------------------------------------------------
    def get_id(self):
        ...


## -------------------------------------------------------------------------------------------------
    def set_name(self):
        ...


## -------------------------------------------------------------------------------------------------
    def get_name(self):
        ...


## -------------------------------------------------------------------------------------------------
    def add_actuator(self, p_pos):
        ...


## -------------------------------------------------------------------------------------------------
    def add_reservoir(self, p_pos):
        ...


## -------------------------------------------------------------------------------------------------
    def setup_sequence(self):
        ...


## -------------------------------------------------------------------------------------------------
    def get_information(self):
        ... # information regarding the actuators, reservoirs, and sequences


## -------------------------------------------------------------------------------------------------
    def reset(self):
        ...




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class TransferFunction:


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name : str, p_id=None, **p_args) -> None:

        if p_name != '':
            self.set_name(p_name)
        else:
            self.set_name(self.C_NAME)

        self.args = p_args

        self.set_id(p_id)


## -------------------------------------------------------------------------------------------------
    def set_id(self, p_id:int=None):
        if p_id is None:
            self._id = str(uuid.uuid4())
        else:
            self._id = str(p_id)


    ## -------------------------------------------------------------------------------------------------
    def get_id(self) -> str:
        return self._id


## -------------------------------------------------------------------------------------------------
    def set_name(self, p_name):
        self._name = p_name
        self.C_NAME = p_name


## -------------------------------------------------------------------------------------------------
    def get_name(self) -> str:
        return self._name


## -------------------------------------------------------------------------------------------------
    def call(self, *p_value):

        function = getattr(self, self._name)

        return function(*p_value)


## -------------------------------------------------------------------------------------------------
    def linear(self, *p_value):
        
        return self.args["arg0"] * p_value[0] + self.args["arg1"]


    ## -------------------------------------------------------------------------------------------------
    def cosinus(self, *p_value):
        return math.cos(self.args["arg0"]) * p_value[0]


    ## -------------------------------------------------------------------------------------------------
    def sinus(self, *p_value):
        return math.cos(self.args["arg0"]) * p_value[0]


    ## -------------------------------------------------------------------------------------------------
    def my_function(self, *p_value):
        """
        This function represents the template to cereate your own function and must be reinitalisied.
        Hereby are p_value[] changing values and self.args[] fix values.

        For example: 
        I(t) = I(0) * e^(-(1/(RC)) * t)
        return self.args["arg0"] * math.exp(-(1/(self.args["arg1"]*self.args["arg2"]))*p_value[0])
        """
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def plot(self, p_lim:int):
    
        x_value = range(p_lim)
        y_value = []

        for para in x_value:
            # function is limited of functions with one input value
            y_value.append(self.call(para))

        
        fig, ax = plt.subplots()
        ax.plot(x_value, y_value, linewidth=2.0)
        plt.show()
        
##### def FunctionApproximation? #####




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class Process(Log):
    """
    This class serves as a base class of an actuation process for a specific actuator, which provides
    the main attributes of an actuation process.
    
    Parameters
    ----------
    
        
    Attributes
    ----------
    

    """

    C_TYPE = 'Process'
    C_NAME = ''


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str,
                 p_id:int=None,
                 p_logging=Log.C_LOG_ALL):
        
        if p_name != '':
            self.set_name(p_name)
        else:
            self.set_name(self.C_NAME)
        self.set_id(p_id)
        
        self.output = {}
        self.all_processes = None
        

## -------------------------------------------------------------------------------------------------
    def set_id(self, p_id:int=None):
        if p_id is None:
            self._id = str(uuid.uuid4())
        else:
            self._id = str(p_id)


## -------------------------------------------------------------------------------------------------
    def get_id(self) -> str:
        return self._id


## -------------------------------------------------------------------------------------------------
    def set_name(self, p_name):
        self._name = p_name
        self.C_NAME = p_name


## -------------------------------------------------------------------------------------------------
    def get_name(self) -> str:
        return self._name


## -------------------------------------------------------------------------------------------------
    def add(self, p_name, p_id, p_type, **p_args):
        if self.all_processes is None:
            self.all_processes = []
        
        self.all_processes.append(TransferFunction(p_name, p_id, p_type, **p_args))


## -------------------------------------------------------------------------------------------------
    def run(self, p_time, p_time_step):
        for proc in range(len(self.all_processes)):
            proc_name = self.all_processes[proc].get_name()
            proc_output = self.all_processes[proc].call(p_time+p_time_step)
            self.output[proc_name] = proc_output
        return self.output




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class Sim_MPPS(Environment):


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name, **p_param):
        ...


## -------------------------------------------------------------------------------------------------
    def set_id(self):
        ...


## -------------------------------------------------------------------------------------------------
    def get_id(self):
        ...


## -------------------------------------------------------------------------------------------------
    def set_name(self):
        ...


## -------------------------------------------------------------------------------------------------
    def get_name(self):
        ...


## -------------------------------------------------------------------------------------------------
    def add_module(self):
        ...


## -------------------------------------------------------------------------------------------------
    def setup_modules(self):
        ...


## -------------------------------------------------------------------------------------------------
    def to_be_added(self):
        ... # to be added later
        





        



    
    
