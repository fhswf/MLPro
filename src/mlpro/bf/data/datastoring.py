## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.data
## -- Module  : datastoring.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-04-16  0.0.0     DA       Creation
## -- 2021-05-29  1.0.0     DA       Release of first version
## -- 2021-06-16  1.1.0     SY       Adding the first version of data storing,
## --                                data plotting, and data saving classes
## -- 2021-06-21  1.2.0     SY       Add extensions in classes Loadable,
## --                                Saveable, DataPlotting & DataStoring.
## -- 2021-08-28  1.2.1     DA       Added constant C_VAR0 to class DataStoring
## -- 2021-09-18  1.2.1     MRD      Buffer Class Implementation. Add new parameter buffer
## --                                to the Adaptive Class
## -- 2021-09-19  1.2.2     MRD      Improvement on Buffer Class. Implement new base class
## --                                Buffer Element and BufferRnd
## -- 2021-09-22  1.3.0     MRD      New classes BufferElement, Buffer, BufferRnd
## -- 2021-09-25  1.3.1     MRD      Add __len__ functionality for SARBuffer
## -- 2023-02-09  1.3.2     MRD      Beautify
## -- 2023-03-02  1.3.3     SY       Update load_data in DataStoring
## -- 2024-04-28  1.4.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.4.0 (2024-04-28)

This module provides various elementary data management classes.

"""


import math
import os
import csv
import copy



# Export list for public API
__all__ = [ 'DataStoring' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DataStoring: #(Persistent):
    """
    This class provides a functionality to store values of variables during
    training/simulation.
    """

    C_VAR0 = 'Frame ID'


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_variables):
        """
        Parameters:
            p_variable    List of variable names
        """
        
        self.memory_dict = {}
        self.names = p_variables
        self.frame_id = {}
        for name in self.names:
            self.memory_dict[name] = {}
            self.frame_id[name] = []


## -------------------------------------------------------------------------------------------------
    def add_frame(self, p_frame_id):
        """
        To store unique sections in a variable (e.g episodes in RL, etc.)
        """
        
        for name in self.names:
            self.memory_dict[name][p_frame_id] = []
            self.frame_id[name].append(p_frame_id)


## -------------------------------------------------------------------------------------------------
    def memorize(self, p_variable, p_frame_id, p_value):
        """
        To store a particular variable into a memory
        """
        
        self.memory_dict[p_variable][p_frame_id].append(p_value)


## -------------------------------------------------------------------------------------------------
    def get_values(self, p_variable, p_frame_id=None):
        """
        To obtain value from the memory
        """
        
        if p_frame_id == None:
            return self.memory_dict[p_variable]
        else:
            return self.memory_dict[p_variable][p_frame_id]


## -------------------------------------------------------------------------------------------------
    def list_to_chunks(self, p_data, p_chunksize):
        NumChunks = int(math.ceil(len(p_data) / (p_chunksize * 1.0)))
        retval = []
        for chunk in range(NumChunks):
            retval.append(sum(p_data[chunk * p_chunksize: (chunk + 1) * p_chunksize]) / (1.0 * p_chunksize))
        return retval


## -------------------------------------------------------------------------------------------------
    def compress(self, p_chunksize):
        for name in self.names:
            for ep in len(self.memory_dict[name]):
                self.memory_dict[name][ep] = self.list_to_chunks(self.memory_dict[name][ep], p_chunksize)


## -------------------------------------------------------------------------------------------------
    def save_data(self, p_path, p_filename=None, p_delimiter="\t") -> bool:
        """
        To save stored data in memory_dict as a readable file format
        """

        if (p_filename is not None) and (p_filename != ''):
            self.filename = p_filename
        else:
            self.filename = self._generate_filename()

        if self.filename is None:
            return False

        try:
            if not os.path.exists(p_path):
                os.makedirs(p_path)
            path_save = p_path + os.sep + self.filename + ".csv"
            with open(path_save, "w", newline="") as write_file:
                header = copy.deepcopy(self.names)
                header.insert(0, self.C_VAR0)
                writer = csv.writer(write_file, delimiter=p_delimiter, quoting=csv.QUOTE_ALL)
                writer.writerow(header)
                writer = csv.writer(write_file, delimiter=p_delimiter)
                for frame in self.frame_id[self.names[0]]:
                    for idx in range(len(self.memory_dict[self.names[0]][frame])):
                        row = []
                        row.append(frame)
                        for name in self.names:
                            row.append(self.memory_dict[name][frame][idx])
                        writer.writerow(row)
            return True
        except:
            return False


## -------------------------------------------------------------------------------------------------
    def load_data(self,
                  p_path,
                  p_filename,
                  p_delimiter="\t",
                  p_frame=True,
                  p_header=True) -> bool:
        """
        To load data from a readable file format and store them into the DataStoring class format
        """

        try:
            path_load = p_path + os.sep + p_filename
            with open(path_load, "r") as read_file:
                reader       = csv.reader(read_file, delimiter=p_delimiter)
                names        = False
                str_memorize = False
                for row in reader:
                    if names is False:
                        if p_header:
                            if p_frame:
                                del row[0:1]
                            self.__init__(row)
                        else:
                            if p_frame:
                                del row[0:1]
                            row_title = []
                            for i in range(len(row)):
                                row_title.append('Data_%i'%(i+1))
                            self.__init__(row_title)
                            str_memorize = True
                        names = True
                    else:
                        str_memorize = True
                        
                    if str_memorize:
                        if p_frame:
                            column = 1
                            for name in self.names:
                                if row[0] not in self.frame_id[name]:
                                    self.add_frame(row[0])
                                try:
                                    self.memorize(name, row[0], float(row[column]))
                                except:
                                    self.memorize(name, row[0],(row[column]))
                                column += 1
                        else:
                            column = 0
                            for name in self.names:
                                if 'Frame_0' not in self.frame_id[name]:
                                    self.add_frame('Frame_0')
                                try:
                                    self.memorize(name, 'Frame_0', float(row[column]))
                                except:
                                    self.memorize(name, 'Frame_0',(row[column]))
                                column += 1
            return True
        except:
            return False

