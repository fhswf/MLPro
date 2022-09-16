## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.envs.sim_mpps.pool.acts
## -- Module  : motor.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-08-24  0.0.0     ML       Creation
## -- 2022-??-??  1.0.0     ML       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2022-08-26)
This module provides a motor subsystem as default implementation.
To be noted, the usage of this simulation is not limited to RL tasks, but it also can be as a
testing environment for GT tasks, evolutionary algorithms, supervised learning, model predictive
control, and many more.
"""

from ast import arg
from mpps import *

class Function():
    C_TYPE = 'Math Function'
    C_NAME = ''
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

    def plot(self, p_lim:int):
        
        x_value = range(p_lim)
        y_value = []

        for para in x_value:
            # function is limited of functions with one input value
            y_value.append(self.call(para))

        
        fig, ax = plt.subplots()
        ax.plot(x_value, y_value, linewidth=2.0)
        plt.show()

        



class Process():
    """
    This class serves as a base class of actuators, which provides the main attributes of an actuator.
    
    Parameters
    ----------
    
        
    Attributes
    ----------
    

    """

    C_TYPE = 'Basic'
    C_NAME = ''

    def __init__(self, p_name: str, p_id: int = None) -> None:
    
        if p_name != '':
                self.set_name(p_name)
        else:
            self.set_name(self.C_NAME)

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
    def set_name(self, p_name: str):
        self._name = p_name
        self.C_NAME = p_name

    #def add(self, p_name: str, p_function,  **p_args):


class Motor(Actuator):

    def __init__(self, p_name: str, p_status: bool = False, p_id: int = None, p_logging = Log.C_LOG_ALL):
        super().__init__(p_name, p_status, p_id, p_logging)

    
        

    ## -------------------------------------------------------------------------------------------------    
    def setup_process(self):
        if self._process() is None:
            self._process = Process(self.get_name())
            
        # self._process.add(p_param_1=.., p_param_2=.., .....)
        # self._process.add(p_param_1=.., p_param_2=.., .....)

        raise NotImplementedError('Please redfine this function!')


## -------------------------------------------------------------------------------------------------    
    def run_process(self, p_time:float, *p_args):
        if not self.get_status():
            self.activate(p_args)
        self._process.run(p_time)
        self._actuation_time += p_time

        raise NotImplementedError('Please redfine this function!')

if __name__ == "__main__":

    math_function = MSpace()

    
    math_function.add_dim(Function(p_name='linear', p_id=None, arg0=5, arg1=3))
    math_function.add_dim(Function(p_name="cosinus",  p_id=None, arg0=0.8))

    _ids = math_function.get_dim_ids()

    LINEAR = _ids[0]
    COSINE = _ids[1]

    print(math_function.get_dim(LINEAR).call(5))

    math_function.get_dim(LINEAR).plot(10)