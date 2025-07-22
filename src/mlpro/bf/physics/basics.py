## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.physics
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-02-04  0.0.0     SY       Creation 
## -- 2023-02-04  1.0.0     SY       First release 
## -- 2024-12-11  1.0.1     DA       Optional import of matplotlib
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2024-12-11)

This module provides models and templates for physics.
"""


from mlpro.bf.various import ScientificObject, PersonalisedStamp, Log

try:
    import matplotlib.pyplot as plt
except:
    pass



# Export list for public API
__all__ = [ 'TransferFunction' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class TransferFunction(ScientificObject, Log, PersonalisedStamp):
    """
    This class serves as a base class of transfer functions, which provides the main attributes of
    a transfer function. By default, there are several ready-to-use transfer function types
    available. If none of them suits to your transfer function, then you can also select a 'custom'
    type of transfer function and design your own function. Another possibility is to use a function
    approximation functionality provided by MLPro (coming soon).
    
    Parameters
    ----------
    p_name : str
        name of the transfer function.
    p_id : int
        unique id of the transfer function. Default: None.
    p_type : int
        type of the transfer function. Default: None.
    p_unit_in : str
        unit of the transfer function's input. Default: None.
    p_unit_out : str
        unit of the transfer function's output. Default: None.
    p_dt : float
        delta time. Default: 0.01.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.
    p_args : dict
        extra parameter for each specific transfer function.
        
    Attributes
    ----------
    C_TYPE : str
        type of the base class. Default: 'TransferFunction'.
    C_NAME : str
        name of the transfer function. Default: ''.
    C_TRF_FUNC_LINEAR : int
        linear function. Default: 0.
    C_TRF_FUNC_CUSTOM : int
        custom transfer function. Default: 1.
    C_TRF_FUNC_APPROX : int
        function approximation. Default: 2.
    """

    C_TYPE              = 'TransferFunction'
    C_NAME              = ''
    C_TRF_FUNC_LINEAR   = 0
    C_TRF_FUNC_CUSTOM   = 1
    C_TRF_FUNC_APPROX   = 2


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str,
                 p_id:int=None,
                 p_type:int=None,
                 p_unit_in:str=None,
                 p_unit_out:str=None,
                 p_dt:float=0.01,
                 p_logging=Log.C_LOG_ALL,
                 **p_args) -> None:

        self.C_NAME = p_name
        self._set_type(p_type)
        self.dt = p_dt
        self._unit_in = p_unit_in
        self._unit_out = p_unit_out

        Log.__init__(self, p_logging=p_logging)
        PersonalisedStamp.__init__(self, p_name, p_id)
        
        if self.get_type() is not None:
            self._set_function_parameters(p_args)
        else:
            raise NotImplementedError('Please define p_type!')


## -------------------------------------------------------------------------------------------------
    def get_units(self):
        """
        This method provides a functionality to get the SI units of the input and output data.

        Returns
        -------
        self._unit_in : str
            the SI unit of the input data.
        self._unit_out : str
            the SI unit of the output data.
        """
        return self._unit_in, self._unit_out


## -------------------------------------------------------------------------------------------------
    def _set_type(self, p_type:int):
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
    def __call__(self, p_input, p_range=None):
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
            output = self._linear(p_input, p_range)
        
        elif self.get_type() == self.C_TRF_FUNC_CUSTOM:
            output = self._custom_function(p_input, p_range)
        
        elif self.get_type() == self.C_TRF_FUNC_APPROX:
            output = self._function_approximation(p_input, p_range)
        
        return output


## -------------------------------------------------------------------------------------------------
    def _set_function_parameters(self, p_args:dict) -> bool:
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
        
        elif self.get_type() == self.C_TRF_FUNC_CUSTOM:
            for key, val in p_args.items():
                exec(key + '=val')
        
        elif self.get_type() == self.C_TRF_FUNC_APPROX:
            raise NotImplementedError('Function approximation is not yet available.')
                
        return True


## -------------------------------------------------------------------------------------------------
    def _linear(self, p_input:float, p_range=None) -> float:
        """
        This method provides a functionality for linear transfer function.
        
        Formula --> y = mx+b
        y = output
        m = slope
        x = input
        b = y-intercept

        Parameters
        ----------
        p_input : float
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
    def _custom_function(self, p_input, p_range=None):
        """
        This function represents the template to create a custom function and must be redefined.

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
    def plot(self, p_x_init:float, p_x_end:float):
        """
        This methods provides functionality to plot the defined function within a range.

        Parameters
        ----------
        p_x_init : float
            The initial value of the input (x-axis).
        p_x_end : float
            The end value of the input (x-axis).
        """
        x_value = []
        output = []
        p_range = p_x_end-p_x_init
        points = int(p_range/self.dt)

        for x in range(points+1):
            current_input = p_x_init + x * self.dt
            x_value.append(current_input)
            output.append(self(current_input, p_range=None))
        
        fig, ax = plt.subplots()
        ax.plot(x_value, output, linewidth=2.0)
        plt.show()


## -------------------------------------------------------------------------------------------------
    def _function_approximation(self, p_input, p_range=None):
        """
        The function approximation is not yet ready (coming soon).

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
