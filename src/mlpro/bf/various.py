## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf
## -- Module  : various
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-04-16  0.0.0     DA       Creation
## -- 2021-05-29  1.0.0     DA       Release of first version
## -- 2021-06-16  1.1.0     SY       Adding the first version of data storing,
## --                                data plotting, and data saving classes
## -- 2021-06-17  1.2.0     DA       New abstract classes Loadable, Saveable
## -- 2021-06-21  1.3.0     SY       Add extensions in classes Loadable,
## --                                Saveable, DataPlotting & DataStoring.
## -- 2021-07-01  1.4.0     SY       Extend save/load functionalities
## -- 2021-08-20  1.5.0     DA       Added property class Plottable
## -- 2021-08-28  1.5.1     DA       Added constant C_VAR0 to class DataStoring
## -- 2021-09-11  1.5.0     MRD      Change Header information to match our new library name
## -- 2021-10-06  1.5.2     DA       Moved class DataStoring to new module mlpro.bf.data.py and
## --                                classes DataPlotting, Plottable to new module mlpro.bf.plot.py
## -- 2021-10-07  1.6.0     DA       Class Log: 
## --                                - colored text depending on log type 
## --                                - new method set_log_level()
## -- 2021-10-25  1.7.0     SY       Add new class ScientificObject
## -- 2021-11-03  1.7.1     DA       Class Log: new type C_LOG_TYPE_SUCCESS for success messages 
## -- 2021-11-15  1.7.2     DA       Class Log: 
## --                                - method set_log_level() removed
## --                                - parameter p_logging is the new log level now
## --                                Class Saveable: new constant C_SUFFIX
## -- 2021-12-07  1.7.3     SY       Add a new attribute in ScientificObject
## -- 2021-12-31  1.7.4     DA       Class Log: udpated docstrings
## -- 2022-07-27  1.7.5     DA       A little refactoring
## -- 2022-08-21  1.7.6     DA       A little refactoring
## -- 2022-10-02  1.8.0     DA       Class Log:
## --                                - new methods Log.get_name(), Log.set_name()
## --                                - method log(): C_NAME in quotation marks
## -- 2022-10-29  1.8.1     DA       Class Log: removed call of switch_logging() from __init__()
## -- 2022-11-04  1.8.2     DA       Class Timer: refactoring
## -- 2022-11-07  1.9.0     DA       Class Log: new method get_log_level()
## -- 2023-01-14  1.9.1     SY       Add class Label
## -- 2023-01-31  1.9.2     SY       Renaming class Label to PersonalisedStamp
## -- 2023-02-22  2.0.0     DA       Class Saveable: new custom method _save()
## -- 2023-03-27  2.1.0     DA       Refactoring persistence:
## --                                - new class Id
## --                                - renamed class LoadSave to Persistent
## --                                - merged classes Load, Save into Persistent
## --                                - logging
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.1.0 (2023-03-27)

This module provides various classes with elementry functionalities for reuse in higher level classes. 
For example: logging, persistence, timer...
"""


from datetime import datetime, timedelta
from time import sleep
import dill as pkl
import os
import sys
import uuid
from mlpro.bf.exceptions import *


# Global dictionary to store paths of pickle files during runtime
g_persistence_file_paths = {}




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Id:
    """
    Property class that inherits a unique id and related get/set-methods to a child class.

    Parameters
    ----------
    p_id
        Optional external id

    Attributes
    _id
        Unique id of the object.

    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_id = None):
        self.set_id(p_id=p_id)


## -------------------------------------------------------------------------------------------------
    def get_id(self):
        return self._id
    

## -------------------------------------------------------------------------------------------------
    def set_id(self, p_id = None):
        """
        Sets/generates a new id.

        Parameters
        ----------
        p_id
            Optional external id. If None, a unique id is generated.
        """

        if p_id is not None:
            self._id = p_id
        else:
            self._id = uuid.uuid4()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Log:
    """
    This class adds elementry log functionality to inherited classes.

    Parameters
    ----------
    p_logging
        Log level (see constants C_LOG_*). Default: Log.C_LOG_ALL

    """

    C_TYPE              = '????'
    C_NAME              = '????'

    # Types of log lines
    C_LOG_TYPE_I        = 'I'  # Information
    C_LOG_TYPE_W        = 'W'  # Warning
    C_LOG_TYPE_E        = 'E'  # Error
    C_LOG_TYPE_S        = 'S'  # Success / Milestone

    C_LOG_TYPES         = [C_LOG_TYPE_I, C_LOG_TYPE_W, C_LOG_TYPE_E, C_LOG_TYPE_S]

    C_COL_WARNING       = '\033[93m'  # Yellow
    C_COL_ERROR         = '\033[91m'  # Red
    C_COL_SUCCESS       = '\033[32m'  # Green
    C_COL_RESET         = '\033[0m'  # Reset color

    # Log levels
    C_LOG_ALL           = True
    C_LOG_NOTHING       = False
    C_LOG_WE            = C_LOG_TYPE_W
    C_LOG_E             = C_LOG_TYPE_E

    C_LOG_LEVELS        = [C_LOG_ALL, C_LOG_NOTHING, C_LOG_WE, C_LOG_E]

    # Internals
    C_INST_MSG          = True

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=C_LOG_ALL):
        if p_logging not in self.C_LOG_LEVELS: 
            raise ParamError('Wrong log level. See class Log for valid log levels')
        self._level = p_logging

        if self.C_INST_MSG:
            self.log(self.C_LOG_TYPE_I, 'Instantiated')
            self.C_INST_MSG = False


## -------------------------------------------------------------------------------------------------
    def get_name(self) -> str:
        return self.C_NAME


## -------------------------------------------------------------------------------------------------
    def set_name(self, p_name:str):
        self.C_NAME = p_name


## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging):
        """
        Sets new log level. 

        Parameters
        ----------
        p_logging     
            Log level (constant C_LOG_LEVELS contains valid values)

        """

        if p_logging not in self.C_LOG_LEVELS: raise ParamError('Wrong log level. See class Log for valid log levels')
        self._level = p_logging


 ## -------------------------------------------------------------------------------------------------
    def get_log_level(self):
        return self._level

 
 ## -------------------------------------------------------------------------------------------------
    def log(self, p_type, *p_args):
        """
        Writes log line to standard output in format:
        yyyy-mm-dd  hh:mm:ss.mmmmmm  [p_type  C_TYPE C_NAME]: [p_args] 

        Parameters:
            p_type      type of log entry
            p_args      log informations

        Returns: 
            Nothing
        """

        if not self._level: return

        if self._level == self.C_LOG_WE:
            if (p_type == self.C_LOG_TYPE_I) or (p_type == self.C_LOG_TYPE_S): return
        elif self._level == self.C_LOG_E:
            if (p_type == self.C_LOG_TYPE_I) or (p_type == self.C_LOG_TYPE_S) or (p_type == self.C_LOG_TYPE_W): return

        now = datetime.now()

        if p_type == self.C_LOG_TYPE_W:
            col = self.C_COL_WARNING
        elif p_type == self.C_LOG_TYPE_E:
            col = self.C_COL_ERROR
        elif p_type == self.C_LOG_TYPE_S:
            col = self.C_COL_SUCCESS
        else:
            col = self.C_COL_RESET

        print(col + '%04d-%02d-%02d  %02d:%02d:%02d.%06d ' % (
        now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond),
              p_type + '  ' + self.C_TYPE + ' "' + self.C_NAME + '":', *p_args, self.C_COL_RESET)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Persistent (Id, Log):
    """
    Property class that inherits persistence to its child classes.

    Parameters
    ----------
    p_id
        Optional external id
    p_logging
        Log level (see constants C_LOG_*). Default: Log.C_LOG_ALL

    Attributes
    ----------
    C_PERSISTENCE_VERSION : str
        Version of the implementation of the persistence. Shall be raised in child classes whenever 
        an incompatible change has been done.
    C_SUFFIX : str = '.pkl'
        Default suffix for pickled result files.
    """

    C_PERSISTENCE_VERSION : str = '1.0.0'
    C_SUFFIX : str              = '.pkl'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_id=None, p_logging=Log.C_LOG_ALL):
        Id.__init__(self, p_id=p_id)
        Log.__init__(self, p_logging=p_logging)

        # Persistent objects need a unique filename
        self.set_filename( p_filename_stub = self.__class__.__name__ + '[' + str(self.get_id()) + ']')
        self._persistence_version = self.C_PERSISTENCE_VERSION


## -------------------------------------------------------------------------------------------------
    def get_filename_stub(self) -> str:
        """
        Returns the unique filename of the object without a suffix.

        Returns
        -------
        filename_stub : str
            Filename stub.
        """

        return self._filename_stub


## -------------------------------------------------------------------------------------------------
    def get_filename(self) -> str:
        """
        Returns the full unique filename of the object including the suffix.

        Returns
        -------
        filename : str
            Full filename.
        """

        return self._filename
    

## -------------------------------------------------------------------------------------------------
    def set_filename(self, p_filename_stub:str, p_suffix:str=None):
        self._filename_stub = p_filename_stub
        
        if p_suffix is not None:
            self._suffix = p_suffix 
        else:
            self._suffix = self.C_SUFFIX

        self._filename = self._filename_stub + self._suffix
    

## -------------------------------------------------------------------------------------------------
    def _get_path(self) -> str:
        """
        Internal helper method to determine the current path for loading/saving external data.
        """

        # 1 Check: is self the path donator?
        try:
            return g_persistence_file_paths[self.get_filename()]
        except:
            pass


        # 2 Locate path donator in call stack
        frame_id = 1
        path = ''
        path_donator_bak = self
        
        while True:
            try:
                frame = sys._getframe(frame_id)
            except:
                break

            try:
                path_donator = frame.f_locals['self']
                if path_donator != path_donator_bak:
                    path_donator_bak = path_donator
                    filename = path_donator.get_filename()
                    path = g_persistence_file_paths[filename]
                    break
            except:
                pass
            
            frame_id += 1

        return path


## -------------------------------------------------------------------------------------------------
    @classmethod
    def load( cls, p_path:str, p_filename:str ):
        """
        Static method to load an object of the current class from a file using pickle/dill. During
        unpickling the given file, standard method __setstate__() is called. This in turn is implemented
        specifically and calls the MLPro custom method _complete_state(). This method allows the
        completion of the unpickled object from further externally stored data.

        Parameters
        ----------
        p_path : str
            Path where file will be saved
        p_filename : str = None      
            File name (if None an internal filename will be used)

        Returns
        -------
        Object 
            Object of the given class that was unpickled from the given file.
        """

        g_persistence_file_paths[p_filename] = p_path

        obj = pkl.load(open(p_path + os.sep + p_filename, 'rb'))

        obj.log(Log.C_LOG_TYPE_I, 'Object loaded from file "' + p_path + os.sep + p_filename + '"')

        return obj
    

## -------------------------------------------------------------------------------------------------
    def __setstate__(self, p_state:dict):
        """
        Python standard method to set the internal object state during unpickling from file.
        The custom method _complete_state() is called to complete the state from further external
        sources.

        Parameters
        ----------
        p_state : dict
            Incoming object state dictionary to be completed.
        """

        # 1 Check for compatible version of persistence
        try:
            ver_file = p_state['_persistence_version']
        except:
            ver_file = None

        if ( ver_file is None ) or ( ver_file != self.C_PERSISTENCE_VERSION ):
            raise ParamError('Pickle file', p_state['_filename'], 'not compatible!')
                     
        # 2 Update object state 
        self.__dict__.update(p_state)

        # 3 Call custom method to complete the object state
        self._os_sep = os.sep
        self._complete_state( p_path = self._get_path(),
                              p_os_sep = os.sep,
                              p_filename_stub = self.get_filename_stub() )


## -------------------------------------------------------------------------------------------------
    def _complete_state(self, p_path:str, p_os_sep:str, p_filename_stub:str):
        """
        Custom method to complete the object state (=self) from external data sources. This method
        is called by standard method __setstate__() during unpickling the object from an external
        file. 

        Parameters
        ----------
        p_path : str
            Path of the object pickle file (and further optional related files)
        p_os_sep : str
            OS-specific path separator.
        p_filename_stub : str
            Filename stub to be used for further optional custom data files
        """

        pass


## -------------------------------------------------------------------------------------------------
    def save(self, p_path:str, p_filename:str=None) -> bool:
        """
        Saves the object to the given path and file name using pickle/dill. If file name is None, a 
        unique inernal file name is used (recommended). During pickling the Python standard method
        __getstate() is called. This in turn is implemented specifically and calls the MLPro custom
        method _reduce_state(). This method allows to reduce unpickleable components from the object
        state before pickling. These components can optionally be stored in separate files of a 
        suitable format.

        Parameters
        ----------
        p_path : str
            Path where file will be saved
        p_filename : str = None      
            File name (if None an internal filename will be used)

        Returns
        -------
        successful : bool
            True, if file content was saved successfully. False otherwise.
        """

        # 1 Create folder if it doesn't exist
        if not os.path.exists(p_path): os.makedirs(p_path)

        if p_filename is not None:
            filename = p_filename
        else:
            filename = self.get_filename()

        g_persistence_file_paths[filename] = p_path

        pkl.dump( obj=self, 
                  file=open(p_path + os.sep + filename, "wb"),
                  protocol=pkl.HIGHEST_PROTOCOL )
        
        self.log(Log.C_LOG_TYPE_I, 'Object saved to file "' + p_path + os.sep + filename + '"')
        return True


## -------------------------------------------------------------------------------------------------
    def __getstate__(self) -> dict:
        """
        Python standard method to get the internal object state during pickling from file.
        The custom method _reduce_state() is called to remove data or object references from the
        state that can not be pickled.

        Returns
        ----------
        state : dict
            (Reduced) object state dictionary to be pickled.
        """

        state = self.__dict__.copy()

        self._reduce_state( p_state = state, 
                            p_path = self._get_path(), 
                            p_os_sep = os.sep,
                            p_filename_stub = self.get_filename_stub() )           

        return state


## -------------------------------------------------------------------------------------------------
    def _reduce_state(self, p_state:dict, p_path:str, p_os_sep:str, p_filename_stub:str):
        """
        Custom method to reduce the given object state by components that can not be pickled. 
        Further data files can be created in the given path and should use the given filename stub.

        Parameters
        ----------
        p_state : dict
            Object state dictionary to be reduced by components that can not be pickled.
        p_path : str
            Path to store further optional custom data files
        p_os_sep : str
            OS-specific path separator.
        p_filename_stub : str
            Filename stub to be used for further optional custom data files
        """

        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Timer:
    """
    Timer class in two time modes (real/virtual) and with simple lap management.

    Parameters
    ----------
    p_mode : int         
        C_MODE_REAL for real time mode or C_MODE_VIRTUAL for virtual time mode
    p_lap_duration : timedelta = None
        Optional duration of a single lap.
    p_lap_limit : int = C_LAP_LIMIT    
        Maximum number of laps until the lap counter restarts with 0  
    """

    C_MODE_REAL         = 0  # Real time
    C_MODE_VIRTUAL      = 1  # Virtual time

    C_LAP_LIMIT         = 999999

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_mode:int, p_lap_duration:timedelta=None, p_lap_limit:int=C_LAP_LIMIT):

        self._mode = p_mode
        self._lap_duration = p_lap_duration

        if p_lap_limit == 0:
            self._lap_limit = self.C_LAP_LIMIT
        else:
            self._lap_limit = p_lap_limit

        self.reset()


## -------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        """
        Resets timer.

        Returns: 
            Nothing
        """

        self.time = timedelta(0, 0, 0)
        self.lap_time = timedelta(0, 0, 0)
        self.lap_id = 0

        if self._mode == self.C_MODE_REAL:
            self.timer_start_real = datetime.now()
            self.lap_start_real = self.timer_start_real
            self.time_real = self.timer_start_real


## -------------------------------------------------------------------------------------------------
    def get_time(self) -> timedelta:
        if self._mode == self.C_MODE_REAL:
            self.time_real = datetime.now()
            self.time = self.time_real - self.timer_start_real

        return self.time


## -------------------------------------------------------------------------------------------------
    def get_lap_time(self) -> timedelta:
        if self._mode == self.C_MODE_REAL:
            self.lap_time = datetime.now() - self.lap_start_real

        return self.lap_time


## -------------------------------------------------------------------------------------------------
    def get_lap_id(self):
        return self.lap_id


## -------------------------------------------------------------------------------------------------
    def add_time(self, p_delta: timedelta):
        if self._mode == self.C_MODE_VIRTUAL:
            self.lap_time = self.lap_time + p_delta
            self.time = self.time + p_delta


## -------------------------------------------------------------------------------------------------
    def finish_lap(self) -> bool:
        """
        Finishes the current lap. In timer mode C_MODE_REAL the remaining time
        until the end of the lap will be paused. 

        Returns: 
            True, if the remaining time to the next lap was positive. False, if 
            the timer timed out.
        """

        timeout = False

        # Compute delay until next lap
        if self._lap_duration is not None:
            delay = self._lap_duration - self.get_lap_time()

            # Check for timeout
            if delay < timedelta(0, 0, 0):
                timeout = True
                delay = timedelta(0, 0, 0)

            # Handle delay depending on timer mode
            if self._mode == self.C_MODE_REAL:
                # Wait until next lap start
                sleep(delay.total_seconds())
            else:
                # Just set next lap start time
                self.time = self.time + delay

        # Update lap data
        self.lap_id = divmod(self.lap_id + 1, self._lap_limit)[1]
        self.lap_time = timedelta(0, 0, 0)
        self.lap_start_real = datetime.now()

        return not timeout





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TStamp:
    """
    This class provides elementry time stamp functionality for inherited classes.
    """


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_tstamp: timedelta = None):
        self.set_tstamp(p_tstamp)


## -------------------------------------------------------------------------------------------------
    def get_tstamp(self) -> timedelta:
        return self.tstamp


## -------------------------------------------------------------------------------------------------
    def set_tstamp(self, p_tstamp: timedelta):
        self.tstamp = p_tstamp





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class  ScientificObject:
    """
    This class provides elementary functionality for storing a scientific reference.
    """

    C_SCIREF_TYPE_NONE = None
    C_SCIREF_TYPE_ARTICLE = "Journal Article"
    C_SCIREF_TYPE_BOOK = "Book"
    C_SCIREF_TYPE_ONLINE = "Online"
    C_SCIREF_TYPE_PROCEEDINGS = "Proceedings"
    C_SCIREF_TYPE_TECHREPORT = "Technical Report"
    C_SCIREF_TYPE_UNPUBLISHED = "Unpublished"

    C_SCIREF_TYPE = C_SCIREF_TYPE_NONE
    C_SCIREF_AUTHOR = None
    C_SCIREF_TITLE = None
    C_SCIREF_JOURNAL = None
    C_SCIREF_ABSTRACT = None
    C_SCIREF_VOLUME = None
    C_SCIREF_NUMBER = None
    C_SCIREF_PAGES = None
    C_SCIREF_YEAR = None
    C_SCIREF_MONTH = None
    C_SCIREF_DAY = None
    C_SCIREF_DOI = None
    C_SCIREF_KEYWORDS = None
    C_SCIREF_ISBN = None
    C_SCIREF_SERIES = None
    C_SCIREF_PUBLISHER = None
    C_SCIREF_CITY = None
    C_SCIREF_COUNTRY = None
    C_SCIREF_URL = None
    C_SCIREF_CHAPTER = None
    C_SCIREF_BOOKTITLE = None
    C_SCIREF_INSTITUTION = None
    C_SCIREF_CONFERENCE = None
    C_SCIREF_NOTES = None
    C_SCIREF_EDITOR = None





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class PersonalisedStamp (Id):
    """
    This class serves as a base class of label to set up a name and id for another class.
    
    Parameters
    ----------
    p_name : str
        name of the created class.
    p_id : int
        unique id of the created class. Default: None.
        
    Attributes
    ----------
    C_NAME : str
        name of the created class. Default: ''.
    """

    C_NAME = ''

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name:str, p_id:int=None):

        self.C_NAME = p_name

        if p_name != '':
            self.set_name(p_name)
        else:
            raise NotImplementedError('Please add a name!')
        
        self.set_id(p_id)


## -------------------------------------------------------------------------------------------------
    def set_name(self, p_name:str):
        """
        This method provides a functionality to set an unique name.

        Parameters
        ----------
        p_name : str
            An unique name.

        """
        self._name = p_name
        self.C_NAME = p_name


## -------------------------------------------------------------------------------------------------
    def get_name(self) -> str:
        """
        This method provides a functionality to get the unique name.

        Returns
        -------
        str
            The unique name of the related component.

        """
        return self._name