## -------------------------------------------------------------------------------------------------
## Howtos.py - Sammlung von MLPro Howtos
## -------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# From: howto_bf_001_logging.py
# Tags: mlpro-bf, mlpro-bf-various
# --------------------------------------------------------------------------------------------------

"""
Ver. 1.1.2 (2023-03-02)
Dieses Howto demonstriert die Nutzung der Log-Klasse in MLPro.
"""

from mlpro.bf.various import Log

class MyClass(Log):
    C_TYPE  = 'Demo class'
    C_NAME  = 'MyClass'

    def __init__(self, p_logging=True):
        super().__init__(p_logging=p_logging)

    def my_method(self):
        self.log(self.C_LOG_TYPE_I, 'Let me tell you what\'s going on...')
        self.log(self.C_LOG_TYPE_W, 'Something is weird...')
        self.log(self.C_LOG_TYPE_E, 'And here something failed...')
        self.log(self.C_LOG_TYPE_I, 'But don\'t worry. Everything is fine. It\'s just a demo:)')
        self.log(self.C_LOG_TYPE_S, 'This method terminated successfully!\n')

if __name__ == "__main__":
    mc = MyClass(p_logging=Log.C_LOG_ALL)
    mc.my_method()
    mc.switch_logging(Log.C_LOG_NOTHING)
    mc.my_method()
    mc.switch_logging(Log.C_LOG_WE)
    mc.my_method()
    mc.switch_logging(Log.C_LOG_E)
    mc.my_method()
    mc.switch_logging(Log.C_LOG_ALL)
    mc.my_method()


# --------------------------------------------------------------------------------------------------
# From: howto_bf_002_timer.py
# Tags: mlpro-bf, mlpro-bf-various
# --------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2023-03-02)
Dieses Howto demonstriert die Nutzung der Timer-Klasse in MLPro.
"""

import time, random
from datetime import timedelta
from mlpro.bf.various import Timer, Log

class TimerDemo(Log):
    C_TYPE = 'Demo class'
    C_NAME = 'Timer'

    def __init__(self, p_timer: Timer):
        self.timer = p_timer
        super().__init__()

    def log(self, p_type, *p_args):
        super().log(p_type, 'Process time', self.timer.get_time(), ', Cycle', self.timer.get_lap_id(), 'Lap time', self.timer.get_lap_time(), '--', *p_args)

    def run_step(self, p_step_id):
        self.log(self.C_LOG_TYPE_I, 'Process step', p_step_id, 'started')
        time.sleep(0.6 * random.random())
        self.log(self.C_LOG_TYPE_I, 'Process step', p_step_id, 'ended')

    def run_cycle(self):
        self.run_step(1)
        self.run_step(2)
        self.run_step(3)
        if not self.timer.finish_lap():
            self.log(self.C_LOG_TYPE_W, 'Last process cycle timed out!!')

    def run(self):
        for _ in range(10):
            self.run_cycle()

if __name__ == "__main__":
    print('\nExample 1: Virtual timer with 1 day and 15 seconds')
    t = Timer(Timer.C_MODE_VIRTUAL, timedelta(1,15,0))
    d = TimerDemo(t)
    d.run()

    print('\nExample 2: Real timer with 1 second and forced timeouts')
    t = Timer(Timer.C_MODE_REAL, timedelta(0,1,0))
    d = TimerDemo(t)
    d.run()


# --------------------------------------------------------------------------------------------------
# From: howto_bf_003_store_plot_and_save_variables.py
# Tags: mlpro-bf, mlpro-bf-data, mlpro-bf-various
# --------------------------------------------------------------------------------------------------

"""
Ver. 1.2.5 (2025-01-17)
Storing, plotting, saving, and loading variables using MLPro.
"""

from mlpro.bf.various import *
from mlpro.bf.data import *
from mlpro.bf.plot import *
import random

if __name__ == "__main__":
    num_eps, num_cycles = 10, 10000
    data_names = ["reward","states_1","states_2","model_loss"]
    data_printing = {"reward":[True,0,10],"states_1":[True,0,4],"states_2":[True,0,4],"model_loss":[True,0,-1]}
    mem = DataStoring(data_names)
    for ep in range(num_eps):
        ep_id = f"ep. {ep+1}"
        mem.add_frame(ep_id)
        for _ in range(num_cycles):
            mem.memorize("reward",ep_id,random.uniform(0+(ep*0.5),5+(ep*0.5)))
            mem.memorize("states_1",ep_id,random.uniform(2-(ep*0.2),4-(ep*0.2)))
            mem.memorize("states_2",ep_id,random.uniform(0+(ep*0.2),2+(ep*0.2)))
            mem.memorize("model_loss",ep_id,random.uniform(0.25-(ep*0.02),1-(ep*0.07)))
    mem_plot = DataPlotting(mem, p_type=DataPlotting.C_PLOT_TYPE_EP, p_window=1000, p_showing=True, p_printing=data_printing, p_figsize=(7,7), p_color="darkblue")
    mem_plot.get_plots()


# --------------------------------------------------------------------------------------------------
# From: howto_bf_004_buffers.py
# Tags: mlpro-bf, mlpro-bf-data
# --------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2023-03-02)
Demonstrates Buffer and BufferElement usage in MLPro.
"""

from mlpro.bf.data import *
import random

if __name__ == "__main__":
    buffer = BufferRnd(p_size=100)
    for i in range(150):
        be = BufferElement({"reward":random.uniform(-10,10), "actions":[random.uniform(0,1),random.uniform(0,1)]})
        be.add_value_element({"accuracy":random.uniform(0,1)})
        buffer.add_element(be)
        print(f"Cycle {i+1}")
        if buffer.is_full():
            print('Buffer is full!')
            sample = buffer.get_sample(p_num=10)
        else:
            print('Buffer not full.')
    buffer.clear()


# --------------------------------------------------------------------------------------------------
# From: howto_bf_005_persistence.py
# Tags: mlpro-bf, mlpro-bf-various
# --------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-03-22)
Persistence functionalities using MLPro.
"""

from mlpro.bf.various import *
from datetime import datetime
from pathlib import Path
import os

class MyClass(Persistent):
    C_TYPE = 'Custom'
    C_NAME = 'Myclass'
    def __init__(self, p_id=None, p_logging=Log.C_LOG_ALL):
        super().__init__(p_id, p_logging)
        self._data, self._separate_data = {}, {}
    def set_data(self, **p_kwargs): self._data = p_kwargs.copy()
    def set_separate_data(self, **p_kwargs): self._separate_data = p_kwargs.copy()
    def log_data(self):
        self.log(Log.C_LOG_TYPE_I, 'Data:\n', self._data)
        self.log(Log.C_LOG_TYPE_I, 'Separate data:\n', self._separate_data)
    def _complete_state(self, p_path, p_os_sep, p_filename_stub):
        with open(p_path + p_os_sep + p_filename_stub + '.dat', 'r') as f:
            for line in f: key, val = line.strip().split('='); self._separate_data[key] = val
    def _reduce_state(self, p_state, p_path, p_os_sep, p_filename_stub):
        with open(p_path + p_os_sep + p_filename_stub + '.dat', 'w') as f:
            for k in p_state['_separate_data']: f.write(f"{k}={p_state['_separate_data'][k]}\n")
        del p_state['_separate_data']
    def __del__(self):
        try: self.log(Log.C_LOG_TYPE_W, 'Object deleted.')
        except: pass

if __name__ == '__main__':
    logging = Log.C_LOG_ALL
    now = datetime.now()
    path = f"{Path.home()}{os.sep}{now.strftime('%Y-%m-%d %H.%M.%S')} MLPro Persistence Test"
    mc = MyClass(p_logging=logging)
    mc.set_data(p1='Hello', p2='World')
    mc.set_separate_data(p1='How', p2='are', p3='you', p4='today?')
    mc.save(p_path=path)
    filename = mc.get_filename()
    mc = None
    mc = MyClass.load(p_path=path, p_filename=filename)
    mc.log_data()


# --------------------------------------------------------------------------------------------------
# From: howto_bf_006_time_stamps.py
# Tags: mlpro-bf, mlpro-bf-various
# --------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-05-21)
Time stamp handling using MLPro.
"""

from mlpro.bf.various import TStamp, Log
from datetime import datetime, timedelta
log = Log(p_logging=Log.C_LOG_ALL)
log.C_TYPE = 'Demo'
log.C_NAME = 'Time Stamps'
log.log(Log.C_LOG_TYPE_I, 'Integer:', TStamp(10).tstamp, TStamp(20).tstamp)
log.log(Log.C_LOG_TYPE_I, 'Float:', TStamp(10.5).tstamp, TStamp(20.7).tstamp)
log.log(Log.C_LOG_TYPE_I, 'Absolute:', TStamp(datetime.now()).tstamp, TStamp(datetime.now()+timedelta(seconds=1)).tstamp)
log.log(Log.C_LOG_TYPE_I, 'Relative:', TStamp(timedelta(seconds=10.5)).tstamp, TStamp(timedelta(seconds=11.5)).tstamp)
