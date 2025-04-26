## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_005_persistence.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-22  1.0.0     DA       Creation/Release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-03-22)

This module demonstrates the basic persistence functionalities of MLPro.

You will learn:

1. How to create an own persistent custom class

2. How to save an object of your custom class to a file

3. How to load an object of your custom class from a file

4. How to implement custom methods to separately save internal data that can not be pickled

"""


from mlpro.bf.various import *
from datetime import datetime
from pathlib import Path




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyClass (Persistent):
    """
    Own custom class that uses MLPro's persistence...
    """

    C_TYPE      = 'Custom'
    C_NAME      = 'Myclass'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_id=None, p_logging=Log.C_LOG_ALL):
        super().__init__(p_id, p_logging)
        self._data = {}
        self._separate_data = {}


## -------------------------------------------------------------------------------------------------
    def set_data(self, **p_kwargs):
        self._data = p_kwargs.copy()


## -------------------------------------------------------------------------------------------------
    def set_separate_data(self, **p_kwargs):
        self._separate_data = p_kwargs.copy()


## -------------------------------------------------------------------------------------------------
    def log_data(self):
        self.log(Log.C_LOG_TYPE_I, 'Data to be pickled:\n\n', self._data, '\n')
        self.log(Log.C_LOG_TYPE_I, 'Separate data:\n\n', self._separate_data, '\n')


## -------------------------------------------------------------------------------------------------
    def _complete_state(self, p_path: str, p_os_sep: str, p_filename_stub: str):

        # Complete object state from separate external data file
        self._separate_data = {}

        with open(p_path + p_os_sep + p_filename_stub + '.dat', 'r') as f:
            for line in f:
                (key, value) = line.split(sep='=')
                (value, nl) = value.split(sep='\n')
                self._separate_data[key] = value


## -------------------------------------------------------------------------------------------------
    def _reduce_state(self, p_state: dict, p_path: str, p_os_sep: str, p_filename_stub: str):

        # 1 Persist all separate data that can/shall not be pickled
        with open(p_path + p_os_sep + p_filename_stub + '.dat', 'w') as f:
            for key in p_state['_separate_data'].keys():
                f.write(key + '=' + p_state['_separate_data'][key] + '\n')

        # 2 Remove separate data from object state
        del p_state['_separate_data']
        

## -------------------------------------------------------------------------------------------------
    def __del__(self):
        try:
            self.log(Log.C_LOG_TYPE_W, 'I just died in our arms tonight...')
        except:
            pass




# 1 Preparation of demo/unit test mode
if __name__ == '__main__':
    # 1.1 Parameters for demo mode
    logging = Log.C_LOG_ALL

else:
    # 1.2 Parameters for internal unit test    
    logging = Log.C_LOG_NOTHING

now     = datetime.now()
path    = str(Path.home()) + os.sep + '%04d-%02d-%02d %02d.%02d.%02d ' % (now.year, now.month, now.day, now.hour, now.minute, now.second) + ' MLPro Persistence Test'


# 2 Instantiate the demo object
mc = MyClass(p_logging=logging)

# 2.1 Persistent classes in MLPro have unique id and filename...
mc.log(Log.C_LOG_TYPE_I, 'My unique Id:', str(mc.get_id()))
mc.log(Log.C_LOG_TYPE_I, 'My unique filename:', mc.get_filename())


# 3 Store data to demo object
mc.set_data( p1='Hello', p2='World!' )
mc.set_separate_data( p1='How', p2='are', p3='you', p4='today?' )
mc.log_data()


# 4 Save demo object to file
mc.save(p_path=path)
filename = mc.get_filename()
mc = None


# 5 Reload same demo object from file
mc = MyClass.load(p_path=path, p_filename=filename)
mc.log(Log.C_LOG_TYPE_I, 'My unique Id:', str(mc.get_id()))
mc.log(Log.C_LOG_TYPE_I, 'My unique filename:', mc.get_filename())
mc.log_data()

