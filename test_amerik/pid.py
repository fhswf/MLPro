## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Module  : howto_bf_streams_101_basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-10-27  0.0.0     DA       Creation
## -- 2022-12-14  1.0.0     DA       First implementation
## -- 2024-02-06  1.1.0     DA       Replaced the native stream by Clouds3D8C2000Static
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2024-02-06)

This module demonstrates the principles of stream processing with MLPro. To this regard, a stream of
a stream provider is combined with a stream workflow to a stream scenario. The workflow consists of 
a custom task only. The stream scenario is used to process some instances.

You will learn:

1) How to implement an own custom stream task.

2) How to set up a stream workflow based on stream tasks.

3) How to set up a stream scenario based on a stream and a processing stream workflow.

4) How to run a stream scenario dark or with visualization.

"""


from mlpro.bf.math.basics import Log
from mlpro.bf.mt import Log, Task, Workflow
from mlpro.bf.streams import *

from mlpro.bf.math import *
from mlpro.bf.streams.basics import InstDict, StreamShared
from mlpro.bf.various import *
from datetime import datetime

from mlpro.bf.various import Log




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ComparatorTask (StreamTask):
    """
    Demo implementation of a stream task with custom method _run().
    """

    # needed for proper logging (see class mlpro.bf.various.Log)
    C_NAME      = 'Comperator'
    FIRST_RUN = True

    def __init__(self,set_point:float, p_name: str = None, p_range_max=Task.C_RANGE_THREAD, p_duplicate_data: bool = False, p_visualize: bool = False, p_logging=Log.C_LOG_ALL ,**p_kwargs):
        super().__init__(p_name, p_range_max, p_duplicate_data, p_visualize, p_logging, **p_kwargs)
        self.set_point = set_point
        self.error=0.0
        

## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict): 

        act_value = 0.0

        if self.FIRST_RUN:
            act_value = self.measure_act_value()
            self.FIRST_RUN = False
        else:
            act_value = p_inst[0][1].get_feature_data().get_values()[0]
        
        self.error = self.set_point- act_value

        
        # Erstellen einer Instanz von Set
        my_set = Set()

        # Erstellen einer Instanz von Element, indem das Set übergeben wird
        my_element = Element(my_set)

        # Setzen von Werten im Element
        my_element.set_values([self.error])

        # Abrufen der Werte aus dem Element
        values = my_element.get_values()
        print(values)  # Ausgabe: [1, 2, 3]
        inst = Instance(my_element,p_tstamp=datetime.now())
        print(inst)

        p_inst[0] = (0,inst)
        self.get_so().reset(p_inst)

        return p_inst
    
    def measure_act_value(self):
        return 24.0

        

"""
        print(f'current value: {current_value}')

        #calculate the control difference [e]
        difference = self.set_point - current_value
        Shared.set('difference', difference)
        out = dict()
        out['error']= difference

        #return e
        return out

"""

        






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PIDRegulatorTask (StreamTask):
    """
    Demo implementation of a stream task with custom method _run().
    """

    # needed for proper logging (see class mlpro.bf.various.Log)
    C_NAME      = 'PID'

    def __init__(self,kp: float, ki: float,kd: float,p_name: str = None, p_range_max=Task.C_RANGE_THREAD, p_duplicate_data: bool = False, p_visualize: bool = False, p_logging=Log.C_LOG_ALL, **p_kwargs):
        super().__init__(p_name, p_range_max, p_duplicate_data, p_visualize, p_logging, **p_kwargs)

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error =0.0
        self.control_signal = 0.0
       

## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict):

        
        error = p_inst[0][1].get_feature_data().get_values()[0]
        #extract the current value [e]
        #error = p_inst.get('error',0.0)
        

        print(f'current error: {error}')

        # PID Control algorithm
        self.integral += error
        derivative = error - self.prev_error
        self.control_signal = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        # store the current error for the next calc 
        self.prev_error = error

        # Erstellen einer Instanz von Set
        my_set = Set()

        # Erstellen einer Instanz von Element, indem das Set übergeben wird
        my_element = Element(my_set)

        # Setzen von Werten im Element
        my_element.set_values([self.control_signal])

        # Abrufen der Werte aus dem Element
        values = my_element.get_values()
        print(values)  # Ausgabe: [1, 2, 3]
        inst = Instance(my_element,p_tstamp=datetime.now())

        p_inst[0] = (0,inst)
        self.get_so().reset(p_inst)

        return p_inst
        

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class ProcessTask(StreamTask):
    """
    Demo implementation of a stream task with custom method _run().
    """

    # needed for proper logging (see class mlpro.bf.various.Log)
    C_NAME      = 'Regelstrecke'

    def __init__(self,current_value:float, p_name: str = None, p_range_max=Task.C_RANGE_THREAD, p_duplicate_data: bool = False, p_visualize: bool = False, p_logging=Log.C_LOG_ALL, **p_kwargs):
        super().__init__(p_name, p_range_max, p_duplicate_data, p_visualize, p_logging, **p_kwargs)

        self.current_value = current_value

## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict):
        
   
        control_signal=p_inst[0][1].get_feature_data().get_values()[0]
       
        print(f'control signal: {control_signal}')

        self.current_value = 3*control_signal+15
       
        print("current value",self.current_value)

      

        # Erstellen einer Instanz von Set
        my_set = Set()

        # Erstellen einer Instanz von Element, indem das Set übergeben wird
        my_element = Element(my_set)

        # Setzen von Werten im Element
        my_element.set_values([self.current_value])

        # Abrufen der Werte aus dem Element
        values = my_element.get_values()
        print(values)  # Ausgabe: [1, 2, 3]
        inst = Instance(my_element,p_tstamp=datetime.now())

        p_inst[0] = (0,inst)
        self.get_so().reset(p_inst)


        return p_inst

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class PIDWorkflow(StreamWorkflow):

    def __init__(self, p_name: str = None, p_range_max=Workflow.C_RANGE_THREAD, p_class_shared=StreamShared, p_visualize: bool = False, p_logging=Log.C_LOG_ALL, **p_kwargs):
        super().__init__(p_name, p_range_max, p_class_shared, p_visualize, p_logging, **p_kwargs)
    

    def run( self, 
            p_range : int = None, 
            p_wait: bool = False, 
            p_inst : InstDict = None ):
        
        """
        Runs all stream tasks according to their predecessor relations.

        Parameters
        ----------
        p_range : int
            Optional deviating range of asynchonicity. See class Range. Default is None what means that 
            the maximum range defined during instantiation is taken. Oterwise the minimum range of both 
            is taken.
        p_wait : bool
            If True, the method waits until all (a)synchronous tasks are finished.
        p_inst : InstDict
            Optional list of stream instances to be processed. If None, the list of the shared object
            is used instead. Default = None.
        """

        if p_inst is not None:
            # This workflow is the leading workflow and opens a new process cycle based on external instances
            try:
                # Erstellen einer Instanz von Set
                my_set = Set()

                # Erstellen einer Instanz von Element, indem das Set übergeben wird

                my_element = Element(my_set)

                # Setzen von Werten im Element
                my_element.set_values([12])

                # Abrufen der Werte aus dem Element
                values = my_element.get_values()
                print(values)  # Ausgabe: [1, 2, 3]
                inst = Instance(my_element,p_tstamp=datetime.now())
                print(type(inst))
                print(type(p_inst))
              
                p_inst[0] = (0,inst)
                a = {}
                a
                
                if len(self._so._instances)>0:
                    print(self._so._instances.values())
                    p_inst[0] =(0,list(self._so._instances.values())[-1][0][1])

             

                self.get_so().reset(p_inst)               
               

            except AttributeError:
                raise ImplementationError('Stream workflows need a shared object of type StreamShared (or inherited)')

        Workflow.run(self, p_range=p_range, p_wait=p_wait)    







class ControlScenario (StreamScenario):
    """
    Example of a custom stream scenario including a stream and a stream workflow. See class 
    mlpro.bf.streams.models.StreamScenario for further details and explanations.
    """

    C_NAME      = 'Demo'

    def __init__(self, p_mode, p_cycle_limit=0, p_visualize: bool = False, p_logging=Log.C_LOG_ALL):
        super().__init__(p_mode, p_cycle_limit, p_visualize, p_logging)



## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize: bool, p_logging):

        # 1 Import a stream from OpenML
        provider_mlpro = StreamProviderMLPro(p_logging=p_logging)
        stream = provider_mlpro.get_stream('Clouds3D8C2000Static', p_logging=p_logging)

        # 2 Set up a stream workflow 
        workflow = PIDWorkflow( p_name='wf1', 
                                   p_range_max=Task.C_RANGE_NONE, 
                                   p_visualize=p_visualize,
                                   p_logging=logging )
        
        self.comperator = ComparatorTask(set_point=22.0,p_name='t1', p_visualize=p_visualize, p_logging=logging)
        self.pid_regulator = PIDRegulatorTask(kp=20, ki=0.1, kd=0.01,p_name='t2', p_visualize=p_visualize, p_logging=logging)
        self.process = ProcessTask(current_value=20.0,p_name='t3', p_visualize=p_visualize, p_logging=logging)

        # 2.1 Set up and add an own custom task    
        workflow.add_task( self.comperator)
        workflow.add_task(self.pid_regulator,[self.comperator ])
        workflow.add_task(self.process,[self.pid_regulator])    
        

        # 3 Return stream and workflow
        return stream, workflow




# 1 Preparation of demo/unit test mode
if __name__ == '__main__':
    # 1.1 Parameters for demo mode
    cycle_limit = 500
    logging     = Log.C_LOG_ALL
    visualize   = False
  
else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 2
    logging     = Log.C_LOG_NOTHING
    visualize   = False


# 2 Instantiate the stream scenario
myscenario = ControlScenario( p_mode=Mode.C_MODE_SIM,
                         p_cycle_limit=cycle_limit,
                         p_visualize=visualize,
                         p_logging=logging )


# 3 Reset and run own stream scenario
myscenario.reset()


if __name__ == '__main__':
   # myscenario.init_plot()

    myscenario.run()

    input('Press ENTER to start stream processing...')

myscenario.run()

if __name__ == '__main__':
    input('Press ENTER to exit...')





