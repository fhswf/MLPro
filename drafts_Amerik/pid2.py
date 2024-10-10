
from mlpro.bf.math.basics import Log
from mlpro.bf.mt import Async, Log, Task, Workflow
from mlpro.bf.streams import *

from mlpro.bf.math import *
from mlpro.bf.streams.basics import InstDict, StreamShared
from mlpro.bf.various import *
from datetime import datetime
import matplotlib.pyplot as plt
from mlpro.bf.various import Log


class Link(Task):


    def __init__(self, p_id=None, p_name: str = None, p_range_max: int = Async.C_RANGE_THREAD, p_autorun=..., p_class_shared=None, p_visualize: bool = False, p_logging=Log.C_LOG_ALL, **p_kwargs):
        super().__init__(p_id, p_name, p_range_max, p_autorun, p_class_shared, p_visualize, p_logging, **p_kwargs)



    def _run(self, **p_kwargs):
        print('Task 1')
        #control_signal=self._so.get_setpoint(self.get_id())
        setpoint= self._so.get_setpoint(self.get_id())
        act_value= self._so.get_actual_value(self.get_id())
        self._so.set_error(setpoint-act_value,self.get_id())
        #print(setpoint-act_value)

class PIDTask(Task):

    def __init__(self,kp: float, ki: float,kd: float, p_id=None, p_name: str = None, p_range_max: int = Async.C_RANGE_THREAD, p_autorun=..., p_class_shared=None, p_visualize: bool = False, p_logging=Log.C_LOG_ALL, **p_kwargs):
        super().__init__(p_id, p_name, p_range_max, p_autorun, p_class_shared, p_visualize, p_logging, **p_kwargs)

        self.kp = kp
        self.Ti = ki
        self.Td = kd
        self.integral = 0.0
        self.prev_error =0.0      
        self.time_step = 1.0
    
    def _run(self, **p_kwargs):
        print('Task 2')
        error = self._so.get_error(self.get_id())
        self.integral += error * self.time_step
        derivative = (error - self.prev_error) / self.time_step    
        # PID-Regler Berechnung
        control_signal = self.kp * error + (self.kp/self.Ti)*self.integral + self.kp* derivative*self.Td
        # Begrenzung der Steuergröße und Normierung auf 0 bis 1
        control_signal = np.clip(control_signal, 0, 100) / 100
        self._so.set_control_signal(control_signal,self.get_id())
        #print(control_signal)
        self.prev_error = error

class ProcessTask(Task):
    
    def __init__(self, p_id=None, p_name: str = None, p_range_max: int = Async.C_RANGE_THREAD, p_autorun=..., p_class_shared=None, p_visualize: bool = False, p_logging=Log.C_LOG_ALL, **p_kwargs):
        super().__init__(p_id, p_name, p_range_max, p_autorun, p_class_shared, p_visualize, p_logging, **p_kwargs)

        # Heizwendel Parameter
        self.coil_mass = 10.0          # Masse der Heizwendel (kg)
        self.specific_heat_coil = 0.5  # Spezifische Wärmekapazität der Heizwendel (J/(kg*K))
        self.coil_temp = 15   # Anfangstemperatur der Heizwendel
        self.heat_transfer_coeff = 0.1  # Wärmeübertragungskoeffizient (W/K)
        self.ambient_temp = 15.0
        self.time_step=1

    def _run(self, **p_kwargs):

        print('Task 3')
        control_signal = self._so.get_control_signal(self.get_id())
        act_value=self._so.get_actual_value(self.get_id())




            # Heizwendel-Erwärmung
        power_input = control_signal * 100  # z.B. in Watt
        
        self.coil_temp += (power_input - self.heat_transfer_coeff * (self.coil_temp - self.ambient_temp)) / (self.coil_mass * self.specific_heat_coil) * self.time_step
        print(f"difference:{self.coil_temp-self.ambient_temp}",f"Power:{power_input}")

        # Wärmeübertragung zum Raum
        heating_power = self.heat_transfer_coeff * (self.coil_temp - act_value)
        
        # Temperaturänderung des Raums
        delta_temp = heating_power/ 60
        act_value+=delta_temp
        act_value +=  (self.ambient_temp-act_value)*0.01
        self._so.set_actual_value(act_value,self.get_id())
        #print(act_value)
      
   
class MasterShared(Shared):

    def __init__(self,p_range: int = Range.C_RANGE_PROCESS):
        super().__init__(p_range)

        self.setpoint = 0.0
        self.actual_value = 0.0
        self.control_signal = 0.0
        self.error= 0.0
        self.actual_values =[]

        self.SpShared = Shared()
        self.ActShared = Shared()
        self.CrtlShared= Shared()
        self.ErrShared = Shared() 
              

    
    def set_setpoint(self,setpoint,p_id):

        test=self.SpShared.lock(p_id,3)
        if test:
            self.SpShared.clear_results()
            self.SpShared.add_result(p_id,setpoint)      
        self.SpShared.unlock()


    def get_setpoint(self,p_id):
 
        test =self.SpShared.lock(p_id,3)
        if test:
            dummy = list(self.SpShared.get_results().values())
            if len(dummy)>0:
                self.setpoint = dummy[-1]
   
        self.SpShared.unlock()

        return self.setpoint
    
    def set_error(self,error,p_id):

        test=self.ErrShared.lock(p_id,3)
        if test:
            self.ErrShared.clear_results()
            self.ErrShared.add_result(p_id,error)      
        self.ErrShared.unlock()


    def get_error(self,p_id):
 
        test =self.ErrShared.lock(p_id,3)
        if test:
            dummy = list(self.ErrShared.get_results().values())
            if len(dummy)>0:
                self.error = dummy[-1]
   
        self.ErrShared.unlock()
        
        return self.error
    
    def set_control_signal(self,control_signal,p_id):

        test=self.CrtlShared.lock(p_id,3)
        if test:
            self.CrtlShared.clear_results()
            self.CrtlShared.add_result(p_id,control_signal)
                
        self.CrtlShared.unlock()


    def get_control_signal(self,p_id):
 
        test =self.CrtlShared.lock(p_id,3)
        if test:
            dummy = list(self.CrtlShared.get_results().values())
            if len(dummy)>0:
                self.control_signal = dummy[-1]   
        self.CrtlShared.unlock()
        
        return self.control_signal

    def set_actual_value(self,actual_value,p_id):

        test=self.ActShared.lock(p_id,3)
        if test:
            self.ActShared.clear_results()
            self.ActShared.add_result(p_id,actual_value) 
            self.actual_values.append(actual_value)       
        self.ActShared.unlock()


    def get_actual_value(self,p_id):
 
        test =self.ActShared.lock(p_id,3)
        if test:
            dummy = list(self.ActShared.get_results().values())
            if len(dummy)>0:
                self.actual_value = dummy[-1]
   
        self.ActShared.unlock()
        
        return self.actual_value

class RLWorkflow(Workflow):

    def __init__(self, p_name: str = None, p_range_max=Async.C_RANGE_THREAD, p_class_shared=None, p_visualize: bool = False, p_logging=Log.C_LOG_ALL, **p_kwargs):
        super().__init__(p_name, p_range_max, p_class_shared, p_visualize, p_logging, **p_kwargs)



class RLSecenario(ScenarioBase):

    def __init__(self, p_mode, p_id=None, p_cycle_limit=0, p_auto_setup: bool = True, p_visualize: bool = True, p_logging=Log.C_LOG_ALL):
        super().__init__(p_mode, p_id, p_cycle_limit, p_auto_setup, p_visualize, p_logging)
    

    def setup(self):

        """
        Specialized method to set up a stream scenario. It is automatically called by the constructor
        and calls in turn the custom method _setup().
        """

        self._workflow = self._setup( p_mode=self.get_mode(), 
                                    p_visualize=self.get_visualization(),
                                    p_logging=Log.C_LOG_NOTHING)#self.get_log_level() )

    
    def _setup(self, p_mode, p_visualize:bool, p_logging):


        # 2 Set up a stream workflow 
        wf = RLWorkflow(p_name="wf1",p_range_max=Workflow.C_RANGE_THREAD,p_class_shared=MasterShared)

        t1 = Link(p_name="t1",logging=Log.C_LOG_NOTHING)
        t3 = ProcessTask(p_name="t2",logging=Log.C_LOG_NOTHING)  
        t2 = PIDTask(10,100,250,p_name="t3",logging=Log.C_LOG_NOTHING)

        wf._so.set_actual_value(15.0,self.get_id())
        wf._so.set_setpoint(22.0,self.get_id())

        # 2.1 Set up and add an own custom task    
        wf.add_task( p_task=t1 )
        wf.add_task( p_task=t2)    
        wf.add_task( p_task=t3 )   

        # 3 Return stream and workflow
        return  wf
    
    def get_latency(self) -> timedelta:
        return None
    

    def _run_cycle(self):

        """
        Gets next instance from the stream and lets process it by the stream workflow.

        Returns
        -------
        success : bool
            True on success. False otherwise.
        error : bool
            True on error. False otherwise.
        adapted : bool
            True, if something within the scenario has adapted something in this cycle. False otherwise.
        end_of_data : bool
            True, if the end of the related data source has been reached. False otherwise.
        """

        try:                    
            self._workflow.run( p_range=Workflow.C_RANGE_THREAD, p_wait=True) #alt p_wait=True
            end_of_data = False
        except StopIteration:
            end_of_data = True

        return False, False, False, end_of_data
    

cycle_limit = 5000

logging = Log.C_LOG_NOTHING
visualize = False 

myscenario = RLSecenario( p_mode=Mode.C_MODE_SIM,
                         p_cycle_limit=cycle_limit,
                         p_visualize=visualize,
                         p_logging=logging )


myscenario.run()
temperature=myscenario._workflow._so.actual_values
# Plotten der Ergebnisse
plt.plot([i for i in range(len(temperature))], temperature, label='Raumtemperatur')
#plt.plot(time,setpoints, color='r', linestyle='--', label='Sollwert')
plt.xlabel('Zeit (Minuten)')
plt.ylabel('Temperatur (°C)')
plt.title('Temperaturregelung mit normiertem PID-Algorithmus')
plt.legend()
plt.grid(True)
plt.show()
input('Press ENTER to exist...')
    




        

        
















        
    

