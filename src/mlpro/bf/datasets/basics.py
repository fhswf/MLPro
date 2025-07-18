## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.datasets
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-18  0.0.0     LSB      Creation
## -- 2023-07-24  1.0.0     LSB      Release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-07-24)

This module provides dataset classes for supervised learning tasks.
"""


import os
import random

import numpy as np
import pandas as pd

from mlpro.bf.various import Log
from mlpro.bf.events import *
from mlpro.bf.exceptions import ParamError, Error
from mlpro.bf.math import *
from mlpro.bf.math.normalizers import NormalizerMinMax



# Export list for public API
__all__ = [ 'Dataset',
            'SASDataset' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Dataset(Log):


    """
    This class serves as the base class for the Dataset Models in MLPro, that can include the native datasets,
    third party datasets and custom datasets. This class provides functionality to pre-process, setup and load the
    data from the datasets based on provided configurations.

        Parameters
        ----------
        p_output_cls:
            The output class from the dataset, default is Element.
        p_feature_dataset:
            The feature Dataset as an array.
        p_label_dataset:
            The label Dataset as an array.
        p_batch_size:int
            Batch Size to deliver the data.
        p_drop_short:bool
            Whether the final batch shall be dropped in case of insufficient data.
        p_shuffle:bool
            If the data shall be shuffled before delivery.
        p_eval_split:float
            The amount of data to be separated for evaluation, as a factor of 1.
        p_test_split:float
            The amount of data to be separated for evaluation, as a factor of 1.
        p_normalize:bool
            Whether the data shall be normalized or not.
        p_settings:
            Additional Dataset specific settings.
        p_logging:
            The logging level of the Dataset.

    """
    C_TYPE = 'Dataset'

    C_FETCH_SINGLE = 0
    C_FETCH_BATCH = 1

    C_MODE_TRAIN = 1
    C_MODE_EVAL = 2
    C_MODE_TEST = 3

    C_CLS_ELEM = Element

    # to be implemented
    C_CLS_ARR = np.array
    C_CLS_LS = list


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_output_cls = C_CLS_ELEM,
                 p_feature_dataset = None,
                 p_label_dataset = None,
                 p_features:list[str] = None,
                 p_labels:list[str] = None,
                 p_label_indexes:list = None,
                 p_batch_size : int = 1,
                 p_drop_short : bool = False,
                 p_shuffle : bool = False,
                 p_eval_split : float = None,
                 p_test_split : float = None,
                 p_normalize: bool = True,
                 p_settings = None,
                 p_logging = Log.C_LOG_ALL):


        Log.__init__(self, p_logging = p_logging)
        self._output_cls = p_output_cls
        self._features = p_features
        self._labels = p_labels
        self._label_indexes = p_label_indexes


        if p_feature_dataset is not None and p_label_dataset is not None:
            self._feature_dataset = p_feature_dataset
            self._label_dataset = p_label_dataset
        elif p_label_dataset:
            raise ParamError("Please also provide feature dataset as p_dataset.")


        self._batch_size = p_batch_size
        self._last_batch = False

        # 1. Setup the mode of data delivery
        if self._batch_size > 1:
            self._fetch_mode = self.C_FETCH_BATCH
        else:
            self._fetch_mode = self.C_FETCH_SINGLE

        self._shuffle = p_shuffle
        self._drop_short = p_drop_short
        self._eval_split = p_eval_split
        self._test_split = p_test_split

        # 2. Setup the meta-data (Shall be a different function? In cases when custom meta-data needs to be included)
        self._num_instances = self.__len__()
        self._indexes_train = list(range(self._num_instances))
        self._indexes = self._indexes_train.copy()

        # 3. Split the dataset
        if self._eval_split is None and self._test_split is None:
            self._split = False
        else:
            self._setup_split()
            self._split = True
            self._mode = self.C_MODE_TRAIN

        self._settings = p_settings


        # Calculating the number of batches
        if self._drop_short:
            self.num_batches = self._num_instances // self._batch_size
        else:
            self.num_batches = self._num_instances // self._batch_size + 1


        # 5. Setting up the feature and label spaces
        self._feature_space, self._label_space = self.setup_spaces()


        # 6. Setting up the normalizers
        # Must be shifted to a different module later specific for data preprocessing
        if p_normalize:
            self._normalize = True
            feature_min_boundaries = np.rint(np.min(self._feature_dataset, axis=0))
            feature_max_boundaries = np.rint(np.max(self._feature_dataset, axis=0))
            feature_boundaries = np.stack((feature_min_boundaries, feature_max_boundaries), axis=1)
            self._normalizer_feature_data = NormalizerMinMax()
            self._normalizer_feature_data.update_parameters(p_boundaries=feature_boundaries)
            label_min_boundaries = np.rint(np.min(self._label_dataset, axis=0))
            label_max_boundaries = np.rint(np.max(self._label_dataset, axis=0))
            label_boundaries = np.stack((label_min_boundaries, label_max_boundaries), axis=1)
            self._normalizer_label_data = NormalizerMinMax()
            self._normalizer_label_data.update_parameters(p_boundaries=label_boundaries)
        else:
            self._normalize = False

        # 7. Reset the dataset
        self.reset(p_shuffle = self._shuffle)


## -------------------------------------------------------------------------------------------------
    def __iter__(self, p_seed = 0):

        """
        This returns the dataset object as an iterator.

        Parameters
        ----------
        p_seed:int
            Seed to reset the data.

        Returns
        -------
        Dataset (Iterator)
            Dataset as an iterator
        """
        self.reset(self._shuffle, p_seed)
        return self


## -------------------------------------------------------------------------------------------------
    def __next__(self):
        """
        This method enables to iterate over the dataset as a Python Iterator Object.

        Returns
        -------
            Returns next data instance from the dataset.
        """

        return self.get_next()


## -------------------------------------------------------------------------------------------------
    def __getitem__(self, p_index):
        """
        Gets the data instance from the dataset at a given instance.

        Parameters
        ----------
        p_index: int
            Index of the data.

        Returns
        -------
        Returns the instance at the given index.
        """

        return self.get_data(p_index)


## -------------------------------------------------------------------------------------------------
    def __len__(self):
        """
        Returns the length of the Dataset. By default returns the length of the Feature Dataset.

        Returns
        -------
        int
            Length of the dataset.

        """
        return len(self._feature_dataset)

## -------------------------------------------------------------------------------------------------
    def set_mode(self, p_mode):
        """
        Sets the mode of the dataset.

        Parameters
        ----------
        p_mode:
            Mode to be set. Valid values are:
            C_MODE_TRAIN: Training Mode
            C_MODE_EVAL: Evaluation Mode
            C_MODE_TEST: Testing Mode

        """

        if p_mode == self.C_MODE_TRAIN:
            self.log(Log.C_LOG_TYPE_I, "Dataset mode set to Training.")
            self._last_batch = False
            self._indexes = self._indexes_train.copy()
        if p_mode == self.C_MODE_EVAL:
            self.log(Log.C_LOG_TYPE_I, "Dataset mode set to Evaluation.")
            self._indexes = self._indexes_eval.copy()
            self._last_batch = False
        if p_mode == self.C_MODE_TEST:
            self.log(Log.C_LOG_TYPE_I, "Dataset mode set to Testing.")
            self._indexes = self._indexes_test.copy()
            self._last_batch = False


## -------------------------------------------------------------------------------------------------
    def setup_spaces(self):
        """
        This is a static custom method to return the feature and label space of the dataset.

        Returns
        -------
        Feature Space, Label Space
            A tuple of feature and label space, respectively.
        """
        feature_space = ESpace()
        label_space = ESpace()
        if self._features is not None:
            for i,feature in enumerate(self._features):
                feature_space.add_dim(Dimension(p_name_long=feature, p_name_short=f'F{i}_{feature[0:5]}'))
        else:
            for i in range(len(self.get_data(0)[0].get_values())):
                feature_space.add_dim(Dimension(p_name_long=f"Feature {i}", p_name_short=f'F_{i}'))

        if self._labels is not None:
            for i,label in enumerate(self._labels):
               label_space.add_dim(Dimension(p_name_long=label, p_name_short=f'L{i}_{label[0:5]}'))
        elif self._label_dataset is not None:
            for i in range(len(self.get_data(0)[1].get_values())):
                label_space.add_dim(Dimension(p_name_long=f"Label {i}", p_name_short=f'L_{i}'))
        elif self._label_indexes:
            for i in range(len(self._label_indexes)):
                label_space.add_dim(Dimension(p_name_long=f"Label {i}", p_name_short=f'L_{i}'))

        return feature_space, label_space

## -------------------------------------------------------------------------------------------------
    def _setup_split(self):

        """
        The method to split the dataset. This method is called during the instantiation of the class. By defualt the
        dataset is split by dividing the dataset as per the factors given. For custom applications, please rewrite
        the method.

        """

        if self._eval_split is not None:
            self._indexes_eval = self._indexes_train[0:int(self._eval_split * len(self._indexes_train))]
            del self._indexes_train[0:int(self._eval_split * len(self._indexes_train))]

        if self._test_split is not None:
            self._indexes_test = self._indexes_train[0:int(self._test_split * len(self._indexes_train))]
            del self._indexes_train[0:int(self._test_split * len(self._indexes_train))]

        self.log(Log.C_LOG_TYPE_I, "Dataset Split.")

## -------------------------------------------------------------------------------------------------
    def reset(self, p_seed = None, p_shuffle = False, **p_kwargs):
        """
        Reset the dataset. Shuffles the indexes of the dataset in case they are to be shuffled.
        Additionally it calls the custom reset method in case of custom implementations.

        Parameters
        ----------
        p_shuffle: bool
            Whether the dataset shall be shuffled.
        p_seed: int
            Seed for the purpose of reproducibility.
        p_kwargs: dict
            Additional key worded arguments for custom reset.

        """
        # Shuffle the entire indices
        if not self._split:
            self._indexes.clear()
            self._indexes.extend(self._indexes_train.copy())
            if p_shuffle or self._shuffle:
                random.shuffle(self._indexes)

        # Shuffle each split
        elif self._split and (p_shuffle or self._shuffle):
            random.shuffle(self._indexes_train)
            if self._eval_split:
                random.shuffle(self._indexes_eval)
            if self._test_split:
                random.shuffle(self._indexes_test)

        self._last_batch = False
        self._reset(p_seed=p_seed, p_shuffle=p_shuffle, **p_kwargs)
        self.log(Log.C_LOG_TYPE_I, "Dataset is reset.")


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed, p_shuffle = False, **p_kwargs):
        """
        Custom reset method for the dataset.

        Parameters
        ----------
        p_seed: int
            Seed for the purpose of reproducibility.
        p_shuffle: bool
            Whether the dataset shall be reshuffled.
        p_kwargs: dict
            Additional key worded arguments for custom reset.


        """
        pass


## -------------------------------------------------------------------------------------------------
    def get_data(self, p_index):
        """
        Get a data instance from the dataset at the given instance.

        Parameters
        ----------
        p_index: int
            Index from which the data shall be fetched.

        Returns
        -------
            The data instance at the given instance.
        """
        # 1. Fetch feature and label data and normalize if needed
        features = self._feature_dataset[p_index]
        if self._normalize:
            features = self._normalizer_feature_data.normalize(p_data=features)
        if self._label_indexes:
            labels = []
            for id in self._label_indexes:
                labels.append(features.pop(id))
                label_obj = self._output_cls(self._label_space)
                label_obj.set_values(labels)
        elif self._label_dataset is not None:
            labels = self._label_dataset[p_index]
            if self._normalize:
                labels = self._normalizer_label_data.normalize(p_data=labels)
            label_obj = self._output_cls(self._label_space)
            label_obj.set_values(labels)
        feature_obj = self._output_cls(self._feature_space)
        feature_obj.set_values(features)


        try:
            return feature_obj, label_obj
        except:
            return feature_obj


## -------------------------------------------------------------------------------------------------
    def get_next(self):

        """
        Gets the next data instance from the list of indexes prepared for data delivery to a scnenario.

        Returns
        -------
        Next data instance.
        """
        if self._fetch_mode == self.C_FETCH_BATCH:
            return self.get_next_batch()
        else:
            return self.get_next_instance()

## -------------------------------------------------------------------------------------------------
    def get_next_instance(self):

        """
        Returns a next single instance as a tuple of feature and label object.

        Returns
        -------
        [(feature_element, label_element)]
            Returns a tuple of feature and label object. This is wrapped in a list for compatibility with iterating
            behaviour, and to keep it conformative with a batch delivery.
        """

        # Return an Instance with first 'batch size' features and corresponding labels as a single label
        self.log(Log.C_LOG_TYPE_I, "Getting next Instance.")

        # 1. Check if last batch
        if len(self._indexes) == 1:
            self._last_batch = True
            self.log(Log.C_LOG_TYPE_I, "Last Instance to be delivered.")

        # 2. Check if no indexes remaining
        elif len(self._indexes) == 0:
            raise Error("End of Data. Please watch the _last_batch attribute.")

        # 3. Assign next instance
        index = self._indexes[0]
        del self._indexes[0]

        # 4. Get data from that instance, and deliver as BatchElements
        val = self.get_data(index)

        feature_element = BatchElement(self._feature_space)
        feature_element.set_values([val[0].get_values()])
        if self._label_dataset is not None:
            label_element = BatchElement(self._label_space)
            label_element.set_values([val[1].get_values()])

            return [(feature_element, label_element)]

        return feature_element


## -------------------------------------------------------------------------------------------------
    def get_next_batch(self):
        """
        Gets the next batch of data from the dataset for a scenario.

        Returns
        -------
        [(feature_objects, label_objects)]
            A tuple of feature objects and label objects, equal to the batch size, wrapped into a list.
        """
        self.log(Log.C_LOG_TYPE_I, "Getting next batch of data.")

        # 1. Check if last instance
        if len(self._indexes) == 0:
            raise Error("End of Data. Please watch the _last_batch attribute.")

        # 2. Check if last batch
        if self._drop_short:
            if 2*self._batch_size > len(self._indexes):
                self._last_batch = True
                self.log("Last batch being delivered.")
                indexes = self._indexes[0:self._batch_size]
                del self._indexes[0:self._batch_size]


        elif self._batch_size >= len(self._indexes):
            self._last_batch = True
            self.log(Log.C_LOG_TYPE_I, "Last batch being delivered.")
            indexes = self._indexes[:]
            self._indexes.clear()


        else:
            indexes = self._indexes[0:self._batch_size]
            del self._indexes[0:self._batch_size]


        feature_values = []
        label_values = []

        # 3. Get values and deliver them as BatchElement
        if self._output_cls == Element:
            for index in indexes:
                vals = self.get_data(index)
                feature_values.append(vals[0].get_values())
                if self._label_dataset is not None or self._label_indexes:
                    label_values.append(vals[1].get_values())
            feature_batch = BatchElement(self._feature_space)
            feature_batch.set_values(feature_values)
            if self._label_dataset is not None or self._label_indexes:
                label_batch = BatchElement(self._label_space)
                label_batch.set_values(label_values)
                return [(feature_batch, label_batch)]

            return [(feature_batch)]

        else:
            raise ParamError("This output class is not yet supported for this dataset")









## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SASDataset(Dataset):
    """
    A custom dataset model class, that creates a dataset based on the historical state and action
    data generated by a training run of an MLPro Environment.

    Parameters
    ----------
    p_state_fpath: str
        Path+name of the csv file with record of states of the system.
    p_action_fpath: str
        Path+name of the csv file with record of actions performed on the system.
    p_state_space:ESpace
        State space of the system.
    p_action_space:ESpace
        Action space of the system.
    p_episode_col:str
        The name of the column that contains the episode information, so that the data before and after the end
        of episode can be handled.
    p_delimiter: str
        The delimiter for the CSV file. Default is '\t'.
    p_drop_columns: list['str']
        List of names of columns to be discarded from the dataset.
    p_batch_size: int
        Batch size of the data to be delivered in a scenario.
    p_drop_short: bool
        Whether the last batch shall be dropped in case of insufficient data.
    p_shuffle: bool
        Whether the data shall be shuffled before delivery.
    p_eval_split: float
        The amount of data to split for the purpose of evaluation, as a factor of 1.
    p_test_split: float
        The amount of data to be split for the purpose of testing, as a factor of 1.
    p_settings:
        Additional Settings for custom applications.
    p_logging:
        Log level for the dataset.
    """


    ## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_path,
                 p_state_fname,
                 p_action_fname,
                 p_state_space : ESpace,
                 p_action_space : ESpace,
                 p_op_state_indexes:list[int] = None,
                 p_episode_col = 'Episode ID',
                 p_delimiter = '\t',
                 p_drop_columns = ['Episode ID', 'Cycle', 'Day', 'Second', 'Microsecond'],
                 p_batch_size : int = 1,
                 p_drop_short : bool = False,
                 p_shuffle : bool = False,
                 p_normalize:bool = False,
                 p_eval_split : float = 0,
                 p_test_split : float = 0,
                 p_settings = None,
                 p_logging = Log.C_LOG_ALL
                 ):

        # 1. Setup feature and label dataset from the csv files
        feature_dataset, label_dataset = self._setup_dataset(p_path = p_path,
                                                            p_state_fname=p_state_fname,
                                                            p_action_fname=p_action_fname,
                                                            p_drop_columns=p_drop_columns,
                                                            p_episode_col=p_episode_col,
                                                            p_delimiter=p_delimiter)


        self._state_space = p_state_space.copy(True)
        self._action_space = p_action_space.copy(True)
        self._op_state_indexes = p_op_state_indexes
        # feature_space, label_space = self.setup_spaces()


        Dataset.__init__(self,
            p_feature_dataset = feature_dataset,
            p_label_dataset = label_dataset,
            p_batch_size=p_batch_size,
            p_eval_split=p_eval_split,
            p_test_split=p_test_split,
            p_shuffle=p_shuffle,
            p_normalize=p_normalize,
            p_drop_short=p_drop_short,
            p_settings=p_settings,
            p_logging=p_logging)




## -------------------------------------------------------------------------------------------------
    def setup_spaces(self):
        """
        Set's up the feature and label spaces of the dataset. Not a static method in this case.

        Returns
        -------
        feature_space, label_space: (ESpace, ESpace)
            A tuple of feature and label space of the dataset.
        """

        if self._op_state_indexes:
            label_space = self._state_space.spawn(p_id_list=[self._state_space.get_dim_ids()[i] for i in self._op_state_indexes])
        else:
            label_space = self._state_space.copy()

        feature_space = self._state_space.copy(p_new_dim_ids=True)

        for dim in self._action_space.get_dims():
            feature_space.add_dim(dim)


        return feature_space, label_space


## -------------------------------------------------------------------------------------------------
    def _setup_dataset(self,
                       p_path:str,
                       p_state_fname: str,
                       p_action_fname: str,
                       p_drop_columns: list,
                       p_episode_col: str,
                       p_delimiter='\t'):
        """
        Sets up the dataset. The CSV files are read and processed to generate the feature and label dataset.

        Parameters
        ----------
        state_fpath:str
            Path + Name for the CSV file with states data.
        action_fpath:str
            Path + Name for the CSV file with action data.
        drop_columns: List[str]
            List of columns to be discarded from the dataset.
        episode_col: str
            The name of the column that contains the episode information.
        delimiter: str
            Delimiter for the CSV files. Default is '\t'

        Returns
        -------

        """

        # Fetching states without dropping columns
        self._states = pd.read_csv(filepath_or_buffer=p_path+os.sep+p_state_fname,
                                    delimiter=p_delimiter)

        # Finding the episode change ids
        episodes_entry_change = self._states[p_episode_col].diff()
        episode_idx = episodes_entry_change[episodes_entry_change.ne(0)].index-1

        # Drop the columns
        self._states = self._states.drop(columns=p_drop_columns, axis=0)

        # Fetch actions and drop the columns
        self._actions = pd.read_csv(filepath_or_buffer=p_path+os.sep+p_action_fname,
                                    delimiter=p_delimiter).drop(columns=p_drop_columns, axis=0)

        # Create input df, with state and action
        input = pd.concat([self._states, self._actions], axis=1, copy=True).iloc[:-1].reset_index(drop=True)
        # Drop rows at episode change
        input = input.drop(labels=episode_idx[1:]).values
        # Create output df
        output = self._states.iloc[1:].reset_index(drop=True)
        # Drop rows at episode change
        output = output.drop(labels=episode_idx[1:]).values

        return input, output


## -------------------------------------------------------------------------------------------------
    def get_data(self, p_index):

        if self._op_state_indexes:
            features = self._feature_dataset[p_index]
            labels = [self._label_dataset[p_index][i] for i in self._op_state_indexes]

            feature_obj = BatchElement(self._feature_space)
            feature_obj.set_values(features)
            label_obj = BatchElement(self._label_space)
            label_obj.set_values(labels)

            return feature_obj, label_obj

        else:
            Dataset.get_data(self, p_index= p_index)