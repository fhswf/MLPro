## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro.bf
## -- Module  : data
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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.1 (2021-09-25)

This module provides various elementary data management classes.
"""

from datetime import datetime, timedelta
# from time import sleep
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle as pkl
import os
import csv
import copy
from mlpro.bf.various import LoadSave
import random


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DataStoring(LoadSave):
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
            self.filename = self.generate_filename()

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
    def load_data(self, p_path, p_filename, p_delimiter="\t") -> bool:
        """
        To load data from a readable file format and store them into the DataStoring class format
        """

        try:
            path_load = p_path + os.sep + p_filename
            with open(path_load, "r") as read_file:
                reader = csv.reader(read_file, delimiter=p_delimiter)
                header = True
                for row in reader:
                    if header:
                        del row[0:1]
                        self.__init__(row)
                        header = False
                    else:
                        column = 1
                        for name in self.names:
                            if row[0] not in self.frame_id[name]:
                                self.add_frame(row[0])
                            self.memorize(name, row[0], float(row[column]))
                            column += 1
            return True
        except:
            return False


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BufferElement:
    """
    Base class implementation for buffer element
    """

    ## -------------------------------------------------------------------------------------------------
    def __init__(self, p_element: dict) -> None:
        """
        Parameters:
            p_element (dict): Buffer element in dictionary
        """

        self._element = {}

        self.add_value_element(p_element)

    ## -------------------------------------------------------------------------------------------------
    def add_value_element(self, p_val: dict):
        """
        Adding new value to the element container

        Parameters:
            p_val (dict): Elements in dictionary
        """
        self._element = {**self._element, **p_val}

    ## -------------------------------------------------------------------------------------------------
    def get_data(self):
        """
        Get the buffer element.

        Returns:
            Returns the buffer element.
        """

        return self._element


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Buffer:
    """
    Base class implementation for buffer management.
    """

    ## -------------------------------------------------------------------------------------------------
    def __init__(self, p_size=1):
        """
        Parameters:
            p_size (int, optional): Buffer size. Defaults to 1.
        """
        self._size = p_size
        self._data_buffer = {}

    ## -------------------------------------------------------------------------------------------------
    def add_element(self, p_elem: BufferElement):
        """
        Add element to the buffer.

        Parameters:
            p_elem (BufferElement): Element of Buffer
        """
        self._data_buffer = {**p_elem.get_data(), **self._data_buffer}
        for key, value in self._data_buffer.items():
            if key in p_elem.get_data() and key in self._data_buffer:
                if not isinstance(self._data_buffer[key], list):
                    self._data_buffer[key] = [p_elem.get_data()[key]]
                else:
                    self._data_buffer[key].append(p_elem.get_data()[key])

                if len(self._data_buffer[key]) > self._size:
                    self._data_buffer[key].pop(-len(self._data_buffer[key]))

    ## -------------------------------------------------------------------------------------------------
    def clear(self):
        """
        Resets buffer.
        """

        self._data_buffer.clear()

    ## -------------------------------------------------------------------------------------------------
    def get_latest(self):
        """
        Returns latest buffered element. 
        """

        try:
            return {key: self._data_buffer[key][-1] for key in self._data_buffer}
        except:
            return None

    ## -------------------------------------------------------------------------------------------------
    def get_all(self):
        """
        Return all buffered elements.

        """
        return self._data_buffer

    ## -------------------------------------------------------------------------------------------------
    def get_sample(self, p_num: int):
        """
        Sample some element from the buffer.

        Parameters:
            p_num (int): Number of sample

        Returns:
            Samples in dictionary
        """
        return self._extract_rows(self._gen_sample_ind(p_num))

    ## -------------------------------------------------------------------------------------------------
    def _gen_sample_ind(self, p_num: int) -> list:
        """
        Generate random indices from the buffer.

        Parameters:
            p_num (int): Number of sample

        Returns:
            List of incides
        """
        raise NotImplementedError

    ## -------------------------------------------------------------------------------------------------
    def _extract_rows(self, p_list_idx: list):
        """
        Extract the element in the buffer based on a
        list of indices.

        Parameters:
            p_list_idx (list): List of indices

        Returns:
            Samples in dictionary
        """
        rows = {}
        for key in self._data_buffer:
            rows[key] = [self._data_buffer[key][i] for i in p_list_idx]
        return rows

    ## -------------------------------------------------------------------------------------------------
    def is_full(self) -> bool:
        """
        Check if the buffer is full.

        Returns:
            True, if the buffer is full
        """
        keys = list(self._data_buffer.keys())
        return len(self._data_buffer[keys[0]]) >= self._size

    ## -------------------------------------------------------------------------------------------------
    def __len__(self):
        keys = list(self._data_buffer.keys())
        return len(self._data_buffer[keys[0]])


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BufferRnd(Buffer):
    """
    Buffer implmentation with random sampling
    """

    ## -------------------------------------------------------------------------------------------------
    def _gen_sample_ind(self, p_num: int) -> list:
        """
        Generate random indicies

        Parameters:
            p_num (int): Number of sample

        Returns:
            List of indicies
        """
        keys = list(self._data_buffer.keys())
        return random.sample(list(range(0, len(self._data_buffer[keys[0]]))), p_num)
