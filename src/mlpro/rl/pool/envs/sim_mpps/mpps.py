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

This module provides a multi-purpose environment for continuous and batch production systems with
modular setting and high-flexibility.

The users are able to develop and simulate their own production systems including setting up own
actuators, reservoirs, processes, modules/stations, production sequences and many more.
We also provide the default implementations of actuators, reservoirs, and modules, which can be
found in the pool of objects.

To be noted, the usage of this simulation is not limited to RL tasks, but it also can be as a
testing environment for GT tasks, evolutionary algorithms, supervised learning, model predictive
control, domain learning, transfer learning, and many more.
"""


from mlpro.rl.models import *
from mlpro.bf.various import *
import numpy as np
import random
import uuid
import math
import matplotlib.pyplot as plt




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class Actuator(ScientificObject, Log):
    """
    This class serves as a base class of actuators, which provides the main attributes of an actuator.
    An actuator is a component of a machine that is responsible for moving and controlling a mechanism
    or system.
    
    Parameters
    ----------
    p_name : str
        name of the actuator.
    p_status : bool
        status of the actuator, either on or off. Default: False.
    p_id : int
        unique id of the actuator. Default: None.
    p_logging : int
        logging level. Default: Log.C_LOG_ALL.
        
    Attributes
    ----------
    C_TYPE : str
        Type of the base class. Default: 'Actuator'.
    C_NAME : str
        Name of the actuator. Default:''.
        
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
        """
        This method provides a functionality to set an unique ID.

        Parameters
        ----------
        p_id : int, optional
            An unique ID. Default: None.

        """
        if p_id is None:
            self._id = str(uuid.uuid4())
        else:
            self._id = str(p_id)


## -------------------------------------------------------------------------------------------------
    def get_id(self) -> str:
        """
        This method provides a functionality to get the defined unique ID.

        Returns
        -------
        str
            The unique ID.

        """
        return self._id


## -------------------------------------------------------------------------------------------------
    def set_name(self, p_name:str):
        """
        This method provides a functionality to set an unique name of the related component.

        Parameters
        ----------
        p_name : str
            An unique name of the related component.

        """
        self._name = p_name
        self.C_NAME = p_name


## -------------------------------------------------------------------------------------------------
    def get_name(self) -> str:
        """
        This method provides a functionality to get the unique name of the related component.

        Returns
        -------
        str
            The unique name of the related component.

        """
        return self._name
        

## -------------------------------------------------------------------------------------------------
    def activate(self, p_max_actuation:float=None, **p_args) -> bool:
        """
        This method provides a functionality to activate the actuator.
        This method needs to be redefined because the behavior of each actuator can be different.

        Parameters
        ----------
        p_max_actuation : float
            maximum actuation time. Default: None.
        **p_args :
            extra parameters for activation, if required.

        Returns
        -------
        bool
            True means succesfully activated and  False means failed to activate.

        """
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
        """
        This method provides a functionality to deactivate the actuator.
        This method needs to be redefined because the behavior of each actuator can be different.

        Parameters
        ----------
        **p_args :
            extra parameters for activation, if required.

        Returns
        -------
        bool
            True means succesfully deactivated and  False means failed to deactivate..

        """
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
        """
        This method provides a functionality to set the status of the related components.

        Parameters
        ----------
        p_status : bool, optional
            Status is on/off. True means on, false means off. Default: False.
            
        """
        self.status = p_status
  
    
## -------------------------------------------------------------------------------------------------      
    def get_status(self) -> bool:
        """
        This method provides a functionality to get the status of the related components.

        Returns
        -------
        bool
            Status is on/off. True means on, false means off.

        """
        return self.status


## -------------------------------------------------------------------------------------------------        
    def reset(self):
        """
        This method provides a functionality to reset the actuator.
        This method needs to be redefined because the behavior of each actuator can be different.

        """
        if self.get_status():
            self.force_stop()
        
        self.log(self.C_LOG_TYPE_I, 'Actuator ' + self.get_name() + ' is succesfully reset.')
        
        raise NotImplementedError('Please redefine this function!')


## -------------------------------------------------------------------------------------------------    
    def emergency_stop(self):
        """
        This method provides a functionality to stop the actuator in an emergency situation.

        """
        self.deactivate(emergency_stop=True)
    
        self.log(self.C_LOG_TYPE_W, 'Actuator ' + self.get_name() + ' is stopped due to emergeny.')


## -------------------------------------------------------------------------------------------------    
    def force_stop(self):
        """
        This method provides a functionality to forcely stop the actuator.

        """
        self.deactivate(force_stop=True)
    
        self.log(self.C_LOG_TYPE_I, 'Actuator ' + self.get_name() + ' is forcely stopped.')


## -------------------------------------------------------------------------------------------------    
    def setup_process(self):
        """
        This method provides a functionality to setup the proceses of the actuator.
        This method needs to be redefined because the behavior of each actuator can be different.

        """
        if self._process() is None:
            self._process = Process(self.get_name())

        # define Function
        # p_function_1 = TransferFunction(p_name, p_id, p_type, p_param_1=.., p_param_2=.., .....)
        # p_function_2 = TransferFunction(p_name, p_id, p_type, p_param_1=.., p_param_2=.., .....)
        
        # add Function
        # self._process.add(p_function_1)
        # self._process.add(p_function_2)

        raise NotImplementedError('Please redefine this function and setup processes!')


## -------------------------------------------------------------------------------------------------    
    def run_process(self,
                    p_time_step:float,
                    p_max_actuation:float=None,
                    p_activate:bool=True,
                    p_stop:bool=False,
                    **p_args) -> dict:
        """
        This method provides a functionality to run all the proceseses of the actuator.

        Parameters
        ----------
        p_time_step : float
            time step.
        p_max_actuation : float, optional
            maximum actuation time. The default is None.
        p_activate : bool, optional
            activating the actuator. The default is True.
        p_stop : bool, optional
            stop the actuator. The default is False.
        **p_args : 
            extra parameter, if required.

        Returns
        -------
        dict
            a dictionary of the name of all processes and their ouput values.

        """
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
    A reservoir is a storage space to temporary store the materials up to the maximum capacity.
    
    There are two possible types of sensors attached to a reservoir,
    (1) C_RES_TYPE_CONT: A sensor that can detect the continuous value of the fill level of a reservoir.
    (2) C_RES_TYPE_2POS: A two position sensor that can only detect whether the current fill level
    has passed the sensors. Thus it can only detect whether the current level is 'low', 'medium', or 'high'.
    
    Parameters
    ----------
    p_name : str
        name of a reservoir.
    p_max_capacity : float
        the maximum capacity of the reservoir, which can be in any units.
    p_sensor : int
        type of the sensor, either C_RES_TYPE_CONT or C_RES_TYPE_2POS. Default: None.
    p_id : int
        an unique id. Default: None.
    p_logging :
        logging level. Default: Log.C_LOG_ALL.
    p_init : float
        the initial level of the reservoir. Default: None.
    p_sensor_low : float
        the location of the first (low) sensor. Rhis is specifically for C_RES_TYPE_2POS type
        of reservoir. Default: None.
    p_sensor_high : float
        the location of the second (high) sensor. Rhis is specifically for C_RES_TYPE_2POS type
        of reservoir. Default: None.
        
    Attributes
    ----------
    C_TYPE : str
        name of the base class. Default: 'Reservoir'
    C_NAME : str
        name of the reservoir. Default: ''
    C_RES_TYPE_CONT : int
        first type of the sensor (continuous). Default: 0
    C_RES_TYPE_2POS : int
        second type of the sensor (2-position). Default: 1
    C_2POS_SENSORS_LEVELS_1 : str
        first output type of 2-position sensor. Default: 'Low'
    C_2POS_SENSORS_LEVELS_2 : str
        second output type of 2-position sensor. Default: 'Medium'
    C_2POS_SENSORS_LEVELS_3 : str
        third output type of 2-position sensor. Default: 'High'
        
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
        

## -------------------------------------------------------------------------------------------------
    def set_id(self, p_id:int=None):
        """
        This method provides a functionality to set an unique ID.

        Parameters
        ----------
        p_id : int, optional
            An unique ID. Default: None.

        """
        if p_id is None:
            self._id = str(uuid.uuid4())
        else:
            self._id = str(p_id)


## -------------------------------------------------------------------------------------------------
    def get_id(self) -> str:
        """
        This method provides a functionality to get the defined unique ID.

        Returns
        -------
        str
            The unique ID.

        """
        return self._id


## -------------------------------------------------------------------------------------------------
    def set_name(self, p_name:str):
        """
        This method provides a functionality to set an unique name of the related component.

        Parameters
        ----------
        p_name : str
            An unique name of the related component.

        """
        self._name = p_name
        self.C_NAME = p_name


## -------------------------------------------------------------------------------------------------
    def get_name(self) -> str:
        """
        This method provides a functionality to get the unique name of the related component.

        Returns
        -------
        str
            The unique name of the related component.

        """
        return self._name
        

## -------------------------------------------------------------------------------------------------
    def update(self, p_in:float, p_out:float):
        """
        This method provides a functionality to update the current fill-level of the reservoir.

        Parameters
        ----------
        p_in : float
            the transported material going in to the reservoir.
        p_out : float
            the transported material going out to the reservoir.

        """
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
        """
        This method provides a functionality to get the current volume of the reservoir depending on
        the type of the installed sensor. If C_RES_TYPE_CONT, then the output is the continuous value
        of the fill-level. If C_RES_TYPE_2POS, then either 'low', 'medium', or 'high'.

        Returns
        -------
        float/str
            the current volume of the reservoir.

        """
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
        """
        This method provides a functionality to set the maximum capacity of the reservoir.

        Parameters
        ----------
        p_max_capacity : float
            the maximum capacity of the reservoir.

        """
        self.max_capacity = p_max_capacity
        

## -------------------------------------------------------------------------------------------------
    def get_maximum_capacity(self) -> float:
        """
        This method provides a functionality to get the maximum capacity of the reservoir.

        Returns
        -------
        float
            the maximum capacity of the reservoir.

        """
        return self.max_capacity
        

## -------------------------------------------------------------------------------------------------
    def get_overflow(self) -> float:
        """
        This method provides a functionality to get the overflow level for current itteration.

        Returns
        -------
        float
            overflow level.

        """
        return self.overflow
        

## -------------------------------------------------------------------------------------------------
    def reset(self):
        """
        This method provides a functionality to reset the reservoir.

        """
        self._volume = self.init_level
        self.overflow = 0
        self.log(self.C_LOG_TYPE_I, 'Reservoir ' + self.get_name() + ' is reset.')
        

## -------------------------------------------------------------------------------------------------
    def set_initial_level(self, p_init:float=None):
        """
        This method provides a functionality to set the initial level of the reservoir, once it is
        reset.

        Parameters
        ----------
        p_init : float, optional
            the initial level of the reservoir. Default: None.

        """
        if p_init is None:
            self.init_level = random.uniform(0, self.get_maximum_capacity())
        else:
            self.init_level = p_init
        

## -------------------------------------------------------------------------------------------------
    def get_initial_level(self) -> float:
        """
        This method provides a functionality to get the initial level of the reservoir.

        Returns
        -------
        float
            the initial level of the reservoir.

        """
        return self.init_level




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class ManufacturingProcess(ScientificObject, Log):
    """
    This class serves as a base class of manufacturing processes, e.g. packaging, weighing, palletizing,
    etc., which provides the main attributes of a manufacturing process.
    
    Parameters
    ----------
    p_name : str
        name of a manufacturing process.
    p_input_unit : str
        the unit of the input material before being processed, e.g. L, Kg, etc.
    p_output_unit : str
        the unit of the output material after being processed, e.g. L, Kg, etc.
    p_processing_time : float
        the processing time for each cycle.
    p_prod_rate_per_time : float
        the production rate per time step.
    p_id : int
        an unique id. Default: None.
    p_max_buffer_input : float, optional
        the maximum capacity of the buffer that is located before the process.
        If none means no buffer is applied. Default: None.
    p_max_buffer_output : float, optional
        the maximum capacity of the buffer that is located after the process.
        If none means no buffer is applied. Default: None.
    p_status : bool
        to check whether the process is still going on. Default: False.
    p_init_buffer_input : float, optional
        the initial capacity of the buffer that is located before the process.
        If none means the initial capacity is 0. Default: None.
    p_init_buffer_output : float, optional
        the initial capacity of the buffer that is located after the process.
        If none means the initial capacity is 0. Default: None.
    p_logging :
        logging level. Default: Log.C_LOG_ALL.
    **p_param :
        extra parameters.
    
        
    Attributes
    ----------
    C_TYPE : str
        Type of the base class. Default: 'ManufacturingProcess'.
    C_NAME : str
        Name of the manufacturing process. Default:''.
    
    """

    C_TYPE = 'ManufacturingProcess'
    C_NAME = ''


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str,
                 p_input_unit:str,
                 p_output_unit:str,
                 p_processing_time:float,
                 p_prod_rate_per_time:float,
                 p_id:int=None,
                 p_max_buffer_input:float=None,
                 p_max_buffer_output:float=None,
                 p_status:bool=False,
                 p_init_buffer_input:float=None,
                 p_init_buffer_output:float=None,
                 p_logging=Log.C_LOG_ALL):
        
        if p_name != '':
            self.set_name(p_name)
        else:
            self.set_name(self.C_NAME)
        self.set_id(p_id)
        self.set_status(p_status)
        self.set_unit(p_input_unit, p_output_unit)
        self.set_buffer_capacity(p_max_buffer_input, p_max_buffer_output)
        self.set_processing_time(p_processing_time)
        self.set_prod_rate(p_prod_rate_per_time)
        self.set_initial_level(p_init_buffer_input, p_init_buffer_output)

        Log.__init__(self, p_logging=p_logging)
        self.reset()
        

## -------------------------------------------------------------------------------------------------
    def set_id(self, p_id:int=None):
        """
        This method provides a functionality to set an unique ID.

        Parameters
        ----------
        p_id : int, optional
            An unique ID. Default: None.

        """
        if p_id is None:
            self._id = str(uuid.uuid4())
        else:
            self._id = str(p_id)


## -------------------------------------------------------------------------------------------------
    def get_id(self) -> str:
        """
        This method provides a functionality to get the defined unique ID.

        Returns
        -------
        str
            The unique ID.

        """
        return self._id


## -------------------------------------------------------------------------------------------------
    def set_name(self, p_name:str):
        """
        This method provides a functionality to set an unique name of the related component.

        Parameters
        ----------
        p_name : str
            An unique name of the related component.

        """
        self._name = p_name
        self.C_NAME = p_name


## -------------------------------------------------------------------------------------------------
    def get_name(self) -> str:
        """
        This method provides a functionality to get the unique name of the related component.

        Returns
        -------
        str
            The unique name of the related component.

        """
        return self._name
        

## -------------------------------------------------------------------------------------------------
    def start_process(self):
        """
        This method provides a functionality to start the process.

        """
        if not self.get_status():
            self.set_status(True)
            self.actual_process_time = 0
        self.log(self.C_LOG_TYPE_I, 'Process ' + self.get_name() + ' is started.')
        

## -------------------------------------------------------------------------------------------------
    def process(self, p_time_step:float):
        """
        This method provides a functionality to execute the process.

        Parameters
        ----------
        p_time_step : float
            time step.

        Returns
        -------
        proc_in : float
            material coming in to the process.
        proc_out : float
            produced material by the process.

        """
        if self.get_status():
            if self.actual_process_time == 0:
                proc_in = self.get_prod_rate()
                proc_out = 0
                self.in_process_prods = self.get_prod_rate()
                self.actual_process_time += p_time_step
            else:
                self.actual_process_time += p_time_step
                if self.actual_process_time >= self.get_processing_time():
                    proc_in = 0
                    proc_out = self.get_prod_rate()
                    self.in_process_prods = 0
                    self.stop_process()
                else:
                    proc_in = 0
                    proc_out = 0
                    self.in_process_prods = self.get_prod_rate()
        else:
            proc_in = 0
            proc_out = 0
            self.in_process_prods = 0
        return proc_in, proc_out
        

## -------------------------------------------------------------------------------------------------
    def stop_process(self):
        """
        This method provides a functionality to stop the process.

        """
        if self.get_status():
            self.set_status(False)
        self.log(self.C_LOG_TYPE_I, 'Process ' + self.get_name() + ' is stopeed.')
  
    
## -------------------------------------------------------------------------------------------------      
    def set_unit(self, p_input_unit:str, p_output_unit:str):
        """
        This method provides a functionality to set up the input and output materials' units.

        Parameters
        ----------
        p_input_unit : str
            the unit of the input material before being processed, e.g. L, Kg, etc..
        p_output_unit : str
            the unit of the output material before being processed, e.g. L, Kg, etc..

        """
        self._input_unit = p_input_unit
        self._output_unit = p_output_unit
  
    
## -------------------------------------------------------------------------------------------------      
    def get_unit(self):
        """
        This method provides a functionality to get the input and output materials' units.

        Returns
        -------
        str
            the unit of the input material before being processed, e.g. L, Kg, etc.
        str
            the unit of the output material before being processed, e.g. L, Kg, etc..

        """
        return self._input_unit, self._output_unit
  
    
## -------------------------------------------------------------------------------------------------      
    def set_buffer_capacity(self, p_max_buffer_input:float=None, p_max_buffer_output:float=None):
        """
        This method provides a functionality to set up the maximum capacity of the buffers.

        Parameters
        ----------
        p_max_buffer_input : float, optional
            the maximum capacity of the buffer that is located before the process.
            If none means no buffer is applied. Default: None.
        p_max_buffer_output : float, optional
            the maximum capacity of the buffer that is located after the process.
            If none means no buffer is applied. Default: None.

        """
        self._max_buffer_input = p_max_buffer_input
        self._max_buffer_output = p_max_buffer_output
  
    
## -------------------------------------------------------------------------------------------------      
    def get_buffer_capacity(self):
        """
        This method provides a functionality to get the maximum capacity of the buffers.

        Returns
        -------
        float
            the maximum capacity of the buffer that is located before the process.
            If none means no buffer is applied. 
        float
            the maximum capacity of the buffer that is located after the process.
            If none means no buffer is applied. 

        """
        return self._max_buffer_input, self._max_buffer_output
  
    
## -------------------------------------------------------------------------------------------------      
    def set_status(self, p_status:bool=False):
        """
        This method provides a functionality to set the status of the related components.

        Parameters
        ----------
        p_status : bool, optional
            Status is on/off. True means on, false means off. Default: False.
            
        """
        self.status = p_status
  
    
## -------------------------------------------------------------------------------------------------      
    def get_status(self) -> bool:
        """
        This method provides a functionality to get the status of the related components.

        Returns
        -------
        bool
            Status is on/off. True means on, false means off.

        """
        return self.status
        

## -------------------------------------------------------------------------------------------------
    def set_prod_rate(self, p_prod_rate_per_time:float):
        """
        This method provides a functionality to set up the production rate.
        
        Parameters
        ----------
        p_prod_rate_per_time : float
            the production rate per time step.

        """
        self._production_rate = p_prod_rate_per_time
        

## -------------------------------------------------------------------------------------------------
    def get_prod_rate(self) -> float:
        """
        This method provides a functionality to get the production rate.

        Returns
        -------
        float
            the production rate per time step.

        """
        return self._production_rate
        

## -------------------------------------------------------------------------------------------------
    def set_processing_time(self, p_processing_time:float):
        """
        This method provides a functionality to set up the processing time for each cycle.

        Parameters
        ----------
        p_processing_time : float
            the processing time for each cycle.

        """
        self._processing_time = p_processing_time
        

## -------------------------------------------------------------------------------------------------
    def get_processing_time(self) -> float:
        """
        This method provides a functionality to get the processing time for each cycle.

        Returns
        -------
        float
            the processing time for each cycle.

        """
        return self._processing_time
        

## -------------------------------------------------------------------------------------------------
    def update_waiting_materials(self, p_in:float, p_out:float):
        """
        This method provides a functionality to update the number of waiting materials in the
        input buffer.

        Parameters
        ----------
        p_in : float
            material in to the input buffer.
        p_out : float
            material out from the input buffer or to be processed.

        """
        self._buffer_input = self._buffer_input + p_in - p_out
        buffer_capacity, _ = self.get_buffer_capacity()
        self._buffer_input_overflow = 0
        if self._buffer_input < 0:
            self._buffer_input = 0
        if buffer_capacity is not None:
            if self._buffer_input > buffer_capacity:
                self._buffer_input_overflow = self._buffer_input - buffer_capacity
                self._buffer_input = buffer_capacity
        self.log(self.C_LOG_TYPE_I, 'Waiting Products in Process ' + self.get_name() + ' is calculated.')
        

## -------------------------------------------------------------------------------------------------
    def get_waiting_materials(self) -> float:
        """
        This method provides a functionality to get the number of waiting materials in the
        input buffer.

        Returns
        -------
        float
            the number of waiting materials in the input buffer.

        """
        return self._buffer_input
        

## -------------------------------------------------------------------------------------------------
    def update_finished_products(self, p_in:float, p_out:float):
        """
        This method provides a functionality to update the number of finished products in the
        output buffer.

        Parameters
        ----------
        p_in : float
            finished products after being processed or material in to the output buffer.
        p_out : float
            material out from the output buffer.

        """
        self._buffer_output = self._buffer_output + p_in - p_out
        _, buffer_capacity = self.get_buffer_capacity()
        self._buffer_output_overflow = 0
        if self._buffer_output < 0:
            self._buffer_output = 0
        if buffer_capacity is not None:
            if self._buffer_output > buffer_capacity:
                self._buffer_output_overflow = self._buffer_output - buffer_capacity
                self._buffer_output = buffer_capacity
        self.log(self.C_LOG_TYPE_I, 'Finished Products in Process ' + self.get_name() + ' is calculated.')
        

## -------------------------------------------------------------------------------------------------
    def get_finished_products(self) -> float:
        """
        This method provides a functionality to get the number of finished products in the
        output buffer.

        Returns
        -------
        float
            the number of finished products in the output buffer.

        """
        return self._buffer_output
        

## -------------------------------------------------------------------------------------------------
    def get_in_process_products(self) -> float:
        """
        This method provides a functionality to get the number of in-processed products in the
        process.

        Returns
        -------
        float
            the number of in-processed products in the process.

        """
        return self.in_process_prods
        

## -------------------------------------------------------------------------------------------------
    def set_initial_level(self, p_init_buffer_input:float=None, p_init_buffer_output:float=None):
        """
        This method provides a functionality to set up the initial levels of input and output
        buffers.

        Parameters
        ----------
        p_init_buffer_input : float, optional
            the initial capacity of the buffer that is located before the process.
            If none means the initial capacity is 0. Default: None.
        p_init_buffer_output : float, optional
            the initial capacity of the buffer that is located after the process.
            If none means the initial capacity is 0. Default: None.

        """
        capacity_input, capacity_output = self.get_buffer_capacity()
        
        if (p_init_buffer_input is None) and (capacity_input is None):
            raise NotImplementedError('Please define either p_init_buffer_input or p_max_buffer_input!')
        elif (p_init_buffer_input is None) and (capacity_input is not None):
            self.init_buffer_input = random.uniform(0, capacity_input)
        else:
            self.init_buffer_input = p_init_buffer_input
            
        if (p_init_buffer_output is None) and (capacity_output is None):
            raise NotImplementedError('Please define either p_init_buffer_output or p_max_buffer_output!')
        elif (p_init_buffer_output is None) and (capacity_output is not None):
            self.init_buffer_output = random.uniform(0, capacity_output)
        else:
            self.init_buffer_output = p_init_buffer_output
        

## -------------------------------------------------------------------------------------------------
    def reset(self):
        """
        This method provides a functionality to reset the process.

        """
        if self.get_status():
            self.stop_process()
        self._buffer_input = self.init_buffer_input
        self._buffer_output = self.init_buffer_output
        self._buffer_input_overflow = 0
        self._buffer_output_overflow = 0
        self.in_process_prods = 0
        self.actual_process_time = 0
        
        self.log(self.C_LOG_TYPE_I, 'Process ' + self.get_name() + ' is succesfully reset.')
        

## -------------------------------------------------------------------------------------------------
    def update(self,
               p_time_step:float,
               p_in:float,
               p_out:float,
               p_start:bool=True,
               p_stop:bool=False):
        """
        This method provides a functionality to perform the process.

        Parameters
        ----------
        p_time_step : float
            time step.
        p_in : float
            material in to the input buffer.
        p_out : float
            material out from the output buffer.
        p_start : bool, optional
            start the process. The default is True.
        p_stop : bool, optional
            stop the process. The default is False.

        """
        if p_start:
            self.start_process(p_time_step)
        
        if p_stop:
            self.stop_process()
        
        proc_in, proc_out = self.process(p_time_step)
        self.update_waiting_materials(p_in, proc_in)
        self.update_finished_products(proc_out, p_out)




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class Module(ScientificObject, Log):
    """
    This class serves as a base class of modules or stations, which provides the main attributes of
    a module or station.
    
    Parameters
    ----------
    p_name : str
        name of a module.
    p_id : int
        an unique id. Default: None.
    p_logging :
        logging level. Default: Log.C_LOG_ALL.
    
        
    Attributes
    ----------
    C_TYPE : str
        Type of the base class. Default: 'Module'.
    C_NAME : str
        Name of the module. Default:''.
    
    """

    C_TYPE = 'Module'
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
        
        self._components = []
        self.setup_components()
        
        Log.__init__(self, p_logging=p_logging)
        self.reset()
        

## -------------------------------------------------------------------------------------------------
    def set_id(self, p_id:int=None):
        """
        This method provides a functionality to set an unique ID.

        Parameters
        ----------
        p_id : int, optional
            An unique ID. Default: None.

        """
        if p_id is None:
            self._id = str(uuid.uuid4())
        else:
            self._id = str(p_id)


## -------------------------------------------------------------------------------------------------
    def get_id(self) -> str:
        """
        This method provides a functionality to get the defined unique ID.

        Returns
        -------
        str
            The unique ID.

        """
        return self._id


## -------------------------------------------------------------------------------------------------
    def set_name(self, p_name:str):
        """
        This method provides a functionality to set an unique name of the related component.

        Parameters
        ----------
        p_name : str
            An unique name of the related component.

        """
        self._name = p_name
        self.C_NAME = p_name


## -------------------------------------------------------------------------------------------------
    def get_name(self) -> str:
        """
        This method provides a functionality to get the unique name of the related component.

        Returns
        -------
        str
            The unique name of the related component.

        """
        return self._name


## -------------------------------------------------------------------------------------------------
    def add_component(self, p_component):
        """
        This method provides a functionality to add a component to the module.

        Parameters
        ----------
        p_component :
            a component, which is build based on Actuator / ManufacturingProcess / Reservoir.

        """
        self._components.append(p_component)


## -------------------------------------------------------------------------------------------------
    def setup_components(self):
        """
        This method provides a functionality to setup components to the module.
        
        There are some requirements to be satisfied:
        1) The components are installed in a serial process manner. If a parallel process is required,
        then it can be set up in different module.
        2) An actuator can not be connected to other actuators. Thus, it must be connected to either
        a reservoir or a manufacturing process.
        3) A manufacturing process or a reservoir can be connected with each other. Without any actuators,
        it is not possible to transfer material within reservoirs and/or manufacturing processes.
        4) If this is the first module of the production system, then it should always start with a
        reservoir or manufacturing process.
        5) If this is the last module of the production system, then it should always end with an
        actuator. Otherwise no material flow from the system can be executed.

        Parameters
        ----------
        p_component :
            a component, which is build based on Actuator / ManufacturingProcess / Reservoir.

        """
        
        # self.add_component(VacuumPump(...))
        # self.add_component(Silo(....))
        # self.add_component(ConveyorBelt(....))
        
        raise NotImplementedError('Please add components to your module')


## -------------------------------------------------------------------------------------------------
    def check(self, p_prev_component:str=None, p_last_module:bool=False) -> bool:
        """
        This method provides a functionality to check whether the module is correct.
        We check whether the components are installed in a correct sequence to assure that the
        production can be performed and material can flow from the first station to the last station.

        Parameters
        ----------
        p_prev_component : str, optional
            The last component of the previous module. If none, this is the first module.
            Default: None.
        p_last_module : bool, optional
            True means this is the last module. Default: False.

        Returns
        -------
        bool
            True means the module could pass the check.

        """
        
        prev_component = p_prev_component
        
        for com in self._components:
            if isinstance(com, Actuator):
                if prev_component == None:
                    raise ValueError('This is the first module. The first module should start with a reservoir or manufacturing process.')
                elif prev_component == 'Actuator':
                    raise ValueError('Actuator can not be connected with other actuators. Hint: you can merge both actuator in one class.')
            elif isinstance(com, ManufacturingProcess) or isinstance(com, Reservoir):
                if prev_component == 'ManufacturingProcess' or prev_component == 'Reservoir':
                    raise ValueError('ManufacturingProcess/Reservoir can not be connected with each other and can only be with Actuator.')
            prev_component = type(com).__name__
        
        if p_last_module:
            if prev_component == 'ManufacturingProcess' or prev_component == 'Reservoir':
                raise ValueError('The module has to be ended with Actuator to transfer the material out.')
    
        return True


## -------------------------------------------------------------------------------------------------
    def get_components(self):
        """
        This method provides a functionality to get all the components in the module.

        Returns
        -------
        list
            list of components.

        """
        return self._components


## -------------------------------------------------------------------------------------------------
    def reset(self):
        """
        This method provides a functionality to reset all the components in the module.

        """
        for com in self._components:
            com.reset()
        self.log(self.C_LOG_TYPE_I, 'Module ' + self.get_name() + ' is succesfully reset.')




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class TransferFunction(ScientificObject, Log):
    """
    This class serves as a base class of transfer functions, which provides the main attributes of
    a transfer function. By default, there are three ready-to-use transfer function types available,
    such as 'linear', 'cosinus', and 'sinus'. If none of them suits to your transfer function, then
    you can also select a 'custom' type of transfer function and design your own function. Another
    possibility is to use a function approximation functionality provided by MLPro.
    
    Parameters
    ----------
    p_name : str
        name of the transfer function.
    p_id : int
        unique id of the transfer function. Default: None.
    p_type : int
        type of the transfer function. Default: None.
    p_dt : float
        delta time. Default: 0.01.
    p_args :
        extra parameter for each specific transfer function.
        
    Attributes
    ----------
    C_TYPE : str
        type of the base class. Default: 'TransferFunction'.
    C_NAME : str
        name of the transfer function. Default: ''.
    C_TRF_FUNC_LINEAR : int
        linear function. Default: 0.
    C_TRF_FUNC_COS : int
        cosine function. Default: 1.
    C_TRF_FUNC_SIN : int
        sine function. Default: 2.
    C_TRF_FUNC_CUSTOM : int
        custom transfer function. Default: 3.
    C_TRF_FUNC_APPROX : int
        function approximation. Default: 4.
    
    """

    C_TYPE              = 'TransferFunction'
    C_NAME              = ''
    C_TRF_FUNC_LINEAR   = 0
    C_TRF_FUNC_COS      = 1
    C_TRF_FUNC_SIN      = 2
    C_TRF_FUNC_CUSTOM   = 3
    C_TRF_FUNC_APPROX   = 4


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str,
                 p_id:int=None,
                 p_type:int=None,
                 p_dt:float=0.01,
                 **p_args) -> None:

        if p_name != '':
            self.set_name(p_name)
        else:
            self.set_name(self.C_NAME)
        
        self.set_id(p_id)
        self.set_type(p_type)
        self.dt = p_dt
        
        if self.get_type() is not None:
            self.set_function_parameters(p_args)
        else:
            raise NotImplementedError('Please define p_type!')
            

## -------------------------------------------------------------------------------------------------
    def set_id(self, p_id:int=None):
        """
        This method provides a functionality to set an unique ID.

        Parameters
        ----------
        p_id : int, optional
            An unique ID. Default: None.

        """
        if p_id is None:
            self._id = str(uuid.uuid4())
        else:
            self._id = str(p_id)


## -------------------------------------------------------------------------------------------------
    def get_id(self) -> str:
        """
        This method provides a functionality to get the defined unique ID.

        Returns
        -------
        str
            The unique ID.

        """
        return self._id


## -------------------------------------------------------------------------------------------------
    def set_name(self, p_name:str):
        """
        This method provides a functionality to set an unique name of the related component.

        Parameters
        ----------
        p_name : str
            An unique name of the related component.

        """
        self._name = p_name
        self.C_NAME = p_name


## -------------------------------------------------------------------------------------------------
    def get_name(self) -> str:
        """
        This method provides a functionality to get the unique name of the related component.

        Returns
        -------
        str
            The unique name of the related component.

        """
        return self._name


## -------------------------------------------------------------------------------------------------
    def set_type(self, p_type:int):
        """
        This method provides a functionality to set the type of the transfer function.

        Parameters
        ----------
        p_type : int
            the type of the transfer function.

        """
        self._type = p_type


## -------------------------------------------------------------------------------------------------
    def get_type(self) -> int:
        """
        This method provides a functionality to get the type of the transfer function.

        Returns
        -------
        int
            the type of the transfer function.

        """
        return self._type


## -------------------------------------------------------------------------------------------------
    def call(self, p_input, p_range=None):
        """
        This method provides a functionality to call the transfer function by giving an input value.

        Parameters
        ----------
        p_input :
            input value.
        p_range :
            range of the calculation. None means 0. Default: None.

        Returns
        -------
        output :
            output value.

        """
        if self.get_type() == self.C_TRF_FUNC_LINEAR:
            output = self.linear(p_input, p_range)
        
        elif self.get_type() == self.C_TRF_FUNC_COS:
            output = self.cosine(p_input, p_range)
            
        elif self.get_type() == self.C_TRF_FUNC_SIN:
            output = self.sine(p_input, p_range)
        
        elif self.get_type() == self.C_TRF_FUNC_CUSTOM:
            output = self.custom_function(p_input, p_range)
        
        elif self.get_type() == self.C_TRF_FUNC_APPROX:
            output = self.function_approximation(p_input, p_range)
        
        return output


## -------------------------------------------------------------------------------------------------
    def set_function_parameters(self, p_args) -> bool:
        """
        This method provides a functionality to set the parameters of the transfer function.

        Parameters
        ----------
        p_args : dict
            set of parameters of the transfer function.

        Returns
        -------
        bool
            true means no parameters are missing.

        """
        if self.get_type() == self.C_TRF_FUNC_LINEAR:
            try:
                self.m = p_args['m']
            except:
                raise NotImplementedError('Parameter m for linear function is missing.')
            try:
                self.b = p_args['b']
            except:
                raise NotImplementedError('Parameter b for linear function is missing.')
        
        elif self.get_type() == self.C_TRF_FUNC_COS or self.get_type() == self.C_TRF_FUNC_SIN:
            try:
                self.A = p_args['A']
                self.B = p_args['B']
                self.C = p_args['C']
                self.D = p_args['D']
            except:
                self.A = 1
                self.B = 1
                self.C = 0
                self.D = 0
                self.log(self.C_LOG_TYPE_W, 'Function ' + self.get_name() + ' has been supplied with default parameters.')
        
        elif self.get_type() == self.C_TRF_FUNC_CUSTOM:
            for key, val in p_args.items():
                exec(key + '=val')
        
        elif self.get_type() == self.C_TRF_FUNC_APPROX:
            raise NotImplementedError('Function approximation is not yet available.')
                
        return True


## -------------------------------------------------------------------------------------------------
    def linear(self, p_input, p_range=None):
        """
        This method provides a functionality for linear transfer function.
        
        Formula --> y = mx+b
        y = output
        m = slope
        x = input
        b = y-intercept

        Parameters
        ----------
        p_input :
            input value.
        p_range :
            range of the calculation. None means 0. Default: None.

        Returns
        -------
        float
            output value.

        """
        
        if p_range is None:
            return self.m * p_input + self.b
        else:
            points = int(p_range/self.dt)
            output = 0
            for x in range(points+1):
                current_input = p_input + x * self.dt
                output += self.m * current_input + self.b
            return output


## -------------------------------------------------------------------------------------------------
    def cosine(self, p_input, p_range=None):
        """
        This method provides a functionality for cosine transfer function.
        
        Formula --> y = A cos(Bx+c) + D, by default: A=1, B=1, C=0, D=0
        A = amplitude
        B = 2phi/B (period)
        C/B = phase shift
        D = vertical shift

        Parameters
        ----------
        p_input :
            input value.
        p_range :
            range of the calculation. None means 0. Default: None.

        Returns
        -------
        float
            output value.

        """
        
        if p_range is None:
            return self.A * math.cos(self.B * p_input + self.C) + self.D
        else:
            points = int(p_range/self.dt)
            output = 0
            for x in range(points+1):
                current_input = p_input + x * self.dt
                output += self.A * math.cos(self.B * current_input + self.C) + self.D
            return output


## -------------------------------------------------------------------------------------------------
    def sine(self, p_input, p_range=None):
        """
        This method provides a functionality for sine transfer function.
        
        Formula --> y = A sin(Bx+c) + D, by default: A=1, B=1, C=0, D=0
        A = amplitude
        B = 2phi/B (period)
        C/B = phase shift
        D = vertical shift

        Parameters
        ----------
        p_input :
            input value.
        p_range :
            range of the calculation. None means 0. Default: None.

        Returns
        -------
        float
            output value.

        """
        
        if p_range is None:
            return self.A * math.sin(self.B * p_input + self.C) + self.D
        else:
            points = int(p_range/self.dt)
            output = 0
            for x in range(points+1):
                current_input = p_input + x * self.dt
                output += self.A * math.sin(self.B * current_input + self.C) + self.D
            return output
        

## -------------------------------------------------------------------------------------------------
    def custom_function(self, p_input, p_range=None):
        """
        This function represents the template to create a custom function and must be redefined.

        For example: 
        I(t) = I(0) * e^(-(1/(RC)) * t)
        return self.args["arg0"] * math.exp(-(1/(self.args["arg1"]*self.args["arg2"]))*p_value[0])

        Parameters
        ----------
        p_input :
            input value.
        p_range :
            range of the calculation. None means 0. Default: None.

        Returns
        -------
        float
            output value.
    
        """
        
        if p_range is None:
            raise NotImplementedError('This custom function is missing.')
        else:
            raise NotImplementedError('This custom function is missing.')
        

## -------------------------------------------------------------------------------------------------
    def plot(self, p_x_init, p_x_end):
        """
        This methods provides functionality to plot the defined function within a range.

        Parameters
        ----------
        p_x_init : float
            The initial value of the input (x-axis).
        p_x_end : float
            The initial value of the input (y-axis).

        """
        x_value = []
        output = []
        p_range = p_x_end-p_x_init
        points = int(p_range/self.dt)

        for x in range(points+1):
            current_input = p_x_init + x * self.dt
            x_value.append(current_input)
            output.append(self.call(current_input, p_range=None))
        
        fig, ax = plt.subplots()
        ax.plot(x_value, output, linewidth=2.0)
        plt.show()


## -------------------------------------------------------------------------------------------------
    def function_approximation(self, p_input, p_range=None):
        """
        ........................

        Parameters
        ----------
        p_input : TYPE
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.

        """
        raise NotImplementedError('Function approximation is not yet available in this version.')




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class Process(Log):
    """
    This class serves as a base class of an actuation process for a specific actuator, which provides
    the main attributes of an actuation process.
    The process can include, for example, current power consumption, transported material calculation,
    current temperature of the actuator, etc. 
    
    Parameters
    ----------
    p_name : str
        name of the acutation process.
    p_id : int
        unique id of the process. Default: None.
    p_logging : int
        logging level. Default: Log.C_LOG_ALL.
        
    Attributes
    ----------
    C_TYPE : str
        Type of the base class. Default: 'Process'.
    C_NAME : str
        Name of the process. Default:''.

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
        
        Log.__init__(self, p_logging=p_logging)
        self.output = {}
        self.all_processes = None
        

## -------------------------------------------------------------------------------------------------
    def set_id(self, p_id:int=None):
        """
        This method provides a functionality to set an unique ID.

        Parameters
        ----------
        p_id : int, optional
            An unique ID. Default: None.

        """
        if p_id is None:
            self._id = str(uuid.uuid4())
        else:
            self._id = str(p_id)


## -------------------------------------------------------------------------------------------------
    def get_id(self) -> str:
        """
        This method provides a functionality to get the defined unique ID.

        Returns
        -------
        str
            The unique ID.

        """
        return self._id


## -------------------------------------------------------------------------------------------------
    def set_name(self, p_name:str):
        """
        This method provides a functionality to set an unique name of the related component.

        Parameters
        ----------
        p_name : str
            An unique name of the related component.

        """
        self._name = p_name
        self.C_NAME = p_name


## -------------------------------------------------------------------------------------------------
    def get_name(self) -> str:
        """
        This method provides a functionality to get the unique name of the related component.

        Returns
        -------
        str
            The unique name of the related component.

        """
        return self._name


## -------------------------------------------------------------------------------------------------
    def add(self, p_function:TransferFunction):
        """
        This method provides a functionality to add a process to the all processes' list.

        Parameters
        ----------
        p_function : TransferFunction
            the transfer function.

        """
        if self.all_processes is None:
            self.all_processes = []
        
        self.all_processes.append(p_function)


## -------------------------------------------------------------------------------------------------
    def run(self, p_time:float, p_time_step:float):
        """
        This method provides a functionality to run the processes within a period of time.

        Parameters
        ----------
        p_time : float
            current production time.
        p_time_step : float
            a period of time for current time step.

        Returns
        -------
        dict
            the output of the processes in the form of dictionary, e.g. {'name': values, ...}.

        """
        for proc in range(len(self.all_processes)):
            proc_name = self.all_processes[proc].get_name()
            proc_output = self.all_processes[proc].call(p_time, p_time_step)
            self.output[proc_name] = proc_output
        return self.output




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class Sim_MPPS(HWControl):


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name, **p_param):
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




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class HWControl(Environment):


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_mode=Mode.C_MODE_REAL,
                 p_latency:timedelta=None,
                 p_afct_strans:AFctSTrans=None,
                 p_afct_reward:AFctReward=None,
                 p_afct_success:AFctSuccess=None,
                 p_afct_broken:AFctBroken=None,
                 p_visualize:bool=True,
                 p_logging=Log.C_LOG_ALL):
        
        super().__init__(p_mode=p_mode,
                        p_latency=p_latency,
                        p_afct_strans=p_afct_strans,
                        p_afct_reward=p_afct_reward,
                        p_afct_success=p_afct_success,
                        p_afct_broken=p_afct_broken,
                        p_visualize=p_visualize,
                        p_logging=p_logging)
        
        self.controller = {}


## -------------------------------------------------------------------------------------------------
    def add_controller(self, p_controller, p_id:str=None):
        """
        This method provides a functionality to add or replace a controller to the system.

        Parameters
        ----------
        p_controller :
            the class of a specific controller, e.g. ConMQTT(), ConOPCUA(), or else.
        p_id : str
            the controller ID. Default: None.

        Returns
        -------
        bool
            True, if action export was successful. False otherwise.

        """
        if p_id is None:
            self._id = str(uuid.uuid4())
        else:
            self._id = str(p_id)
        self.controller[p_id] = p_controller


## -------------------------------------------------------------------------------------------------
    def _export_action(self, p_action: Action) -> bool:
        """
        Mode C_MODE_REAL only: exports given action to be processed externally 
        (for instance by a real hardware). Please redefine. 

        Parameters
        ----------
        p_action : Action
            Action to be exported

        Returns
        -------
        bool
            True, if action export was successful. False otherwise.

        """

        # send the data to the actuator class of the controller(s)

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _import_state(self) -> bool:
        """
        Mode C_MODE_REAL only: imports state from an external system (for instance a real hardware). 
        Please redefine. Please use method set_state() for internal update.

        Returns
        -------
        bool
            True, if state import was successful. False otherwise.

        """

        # get the data from the sensor class of the controller(s)

        raise NotImplementedError



