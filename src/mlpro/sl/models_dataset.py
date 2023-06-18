## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl
## -- Module  : models_data.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-18  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-06-13)

This module provides dataset classes for supervised learning tasks.
"""


from mlpro.bf.various import Log
from mlpro.bf.math import *
from mlpro.bf.events import *
import random




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Dataset(Log, EventManager):


    """

    Parameters
    ----------
    p_input_space
    p_output_space
    p_output_cls
    p_data_class
    p_feature_dataset
    p_label_dataset
    p_batch_size
    p_drop_short
    p_shuffle
    p_logging
    """
    C_FETCH_SINGLE = 0
    C_FETCH_BATCH = 1

    C_SPLIT_NONE = 0

    C_MODE_TRAIN = 1
    C_MODE_EVAL = 2
    C_MODE_TEST = 3

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_feature_space : ESpace,
                 p_label_space : ESpace,
                 p_output_cls : type,
                 p_data_class : type,
                 p_feature_dataset,
                 p_label_dataset,
                 p_batch_size : int,
                 p_drop_short : bool,
                 p_shuffle : bool,
                 p_eval_split : float,
                 p_test_split : float,
                 p_settings,
                 p_logging = Log.C_LOG_ALL):


        Log.__init__(self, p_logging = p_logging)
        self._feature_space = p_feature_space
        self._label_space = p_label_space
        self._output_cls = p_output_cls
        self._data_class = p_data_class
        self._feature_dataset = p_feature_dataset
        self._label_dataset = p_label_dataset
        self._batch_size = p_batch_size

        if self._batch_size > 1:
            self._mode = self.C_FETCH_BATCH

        self._shuffle = p_shuffle
        self._drop_short = p_drop_short
        self._eval_split = p_eval_split
        self._test_split = p_test_split

        if self._eval_split is None and self._test_split is None:
            self._split = self.C_SPLIT_NONE
        else:
            self.setup_splits()

        self._settings = p_settings

        self._num_instances = self.__len__()
        self._feature_space, self._label_space = self._setup_spaces()
        self._indexes_train = list(range(self._num_instances))
        self.reset(p_shuffle = self._shuffle)


## -------------------------------------------------------------------------------------------------
    def __iter__(self, p_seed = 0):

        self.reset(p_seed)
        return self


## -------------------------------------------------------------------------------------------------
    def __next__(self):

        if self._mode == self.C_FETCH_BATCH:
            return self.get_next_batch()
        else:
            return self.get_next()


## -------------------------------------------------------------------------------------------------
    def __getitem__(self, p_index):

        return self.get_data(p_index)


## -------------------------------------------------------------------------------------------------
    def __len__(self):

        return len(self._feature_dataset)


## -------------------------------------------------------------------------------------------------
    def setup(self, p_settings):

        self._setup(p_settings)


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_settings):

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def _setup_spaces():


        return None, None

## -------------------------------------------------------------------------------------------------
    def _setup_split(self):

        if self.eval_split is not None:
            self._indexes_eval = self._indexes_train[0:int(self.eval_split * len(self._indexes_train))]
            del self._indexes_train[0:int(self.eval_split * len(self._indexes_train))]

        if self.test_split is not None:
            self._indexes_test = self._indexes_train[0:int(self.test_split * len(self._indexes_train))]
            del self._indexes_train[0:int(self.test_split * len(self._indexes_train))]


## -------------------------------------------------------------------------------------------------
    def reset(self, p_shuffle = None, p_seed = None, p_epoch = 0):

        self._indexes = list(range(self._num_instances))

        if p_shuffle:
            random.shuffle(self._indexes)

        self.setup(self._settings)

        return self._reset(p_seed = p_seed, p_shuffle = p_shuffle)


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_shuffle, p_seed):

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_data(self, p_index):

        features = self._feature_dataset[p_index]
        labels = self._label_space[p_index]

        return features, labels


## -------------------------------------------------------------------------------------------------
    def get_next(self):
        # Return an Instance with first 'batch size' features and corresponding labels as a single label
        raise NotImplementedError



## -------------------------------------------------------------------------------------------------
    def get_next_batch(self):

        if self._drop_short:
            if 2*self._batch_size > len(self._indexes):
                self._last_batch = True

        else:
            if self._batch_size > len(self._indexes):
                self._last_batch = True

        indexes = self._indexes[0:self._batch_size]
        del self._indexes[0:self._batch_size]
        batch = [(self._feature_dataset[i], self._label_dataset[i]) for i in indexes]

        return batch