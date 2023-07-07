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


from mlpro.bf.math import *
from mlpro.bf.math.normalizers import NormalizerMinMax
from mlpro.bf.events import *
import random
import pandas as pd




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Dataset(Log):


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
        self._feature_dataset = p_feature_dataset
        self._label_dataset = p_label_dataset
        self._batch_size = p_batch_size
        self._last_batch = False

        if self._batch_size > 1:
            self._fetch_mode = self.C_FETCH_BATCH
        else:
            self._fetch_mode = self.C_FETCH_SINGLE

        self._shuffle = p_shuffle
        self._drop_short = p_drop_short
        self._eval_split = p_eval_split
        self._test_split = p_test_split

        self._num_instances = self.__len__()

        self._indexes_train = list(range(self._num_instances))
        self._indexes = self._indexes_train.copy()

        if self._eval_split is None and self._test_split is None:
            self._split = False
        else:
            self._setup_split()
            self._split = True
            self._mode = self.C_MODE_TRAIN

        self._settings = p_settings



        if self._drop_short:
            self.num_batches = self._num_instances // self._batch_size
        else:
            self.num_batches = self._num_instances // self._batch_size + 1


        self._feature_space, self._label_space = self.setup_spaces()

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

        self.reset(p_shuffle = self._shuffle)


## -------------------------------------------------------------------------------------------------
    def __iter__(self, p_seed = 0):

        """

        Parameters
        ----------
        p_seed

        Returns
        -------

        """
        self.reset(p_seed)
        return self


## -------------------------------------------------------------------------------------------------
    def __next__(self):
        """

        Returns
        -------

        """
        return self.get_next()


## -------------------------------------------------------------------------------------------------
    def __getitem__(self, p_index):
        """

        Parameters
        ----------
        p_index

        Returns
        -------

        """
        return self.get_data(p_index)


## -------------------------------------------------------------------------------------------------
    def __len__(self):
        """

        Returns
        -------

        """
        return len(self._feature_dataset)


## -------------------------------------------------------------------------------------------------
    def set_mode(self, p_mode):
        """

        Parameters
        ----------
        p_mode

        Returns
        -------

        """
        if p_mode == self.C_MODE_TRAIN:
            self._indexes = self._indexes_train.copy()
        if p_mode == self.C_MODE_EVAL:
            self._indexes = self._indexes_eval.copy()
        if p_mode == self.C_MODE_TEST:
            self._indexes = self._indexes_test.copy()


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        """

        Returns
        -------

        """

        return None, None

## -------------------------------------------------------------------------------------------------
    def _setup_split(self):
        """

        Returns
        -------

        """
        if self._eval_split is not None:
            self._indexes_eval = self._indexes_train[0:int(self._eval_split * len(self._indexes_train))]
            del self._indexes_train[0:int(self._eval_split * len(self._indexes_train))]

        if self._test_split is not None:
            self._indexes_test = self._indexes_train[0:int(self._test_split * len(self._indexes_train))]
            del self._indexes_train[0:int(self._test_split * len(self._indexes_train))]


## -------------------------------------------------------------------------------------------------
    def reset(self, p_shuffle = None, p_seed = None, p_epoch = 0):
        """

        Parameters
        ----------
        p_shuffle
        p_seed
        p_epoch

        Returns
        -------

        """
        if not self._split:
            self._indexes.clear()
            self._indexes.extend(self._indexes_train.copy())
            if p_shuffle:
                random.shuffle(self._indexes)

        elif self._split and p_shuffle:
            random.shuffle(self._indexes_train)
            random.shuffle(self._indexes_eval)
            random.shuffle(self._indexes_eval)

        self._last_batch = False
        return self._reset(p_seed=p_seed, p_shuffle=p_shuffle, p_epoch=p_epoch)


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed, p_shuffle = False, p_epoch = 0):
        """

        Parameters
        ----------
        p_seed
        p_shuffle
        p_epoch

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def get_data(self, p_index):
        """

        Parameters
        ----------
        p_index

        Returns
        -------

        """
        if self._normalize:
            features = self._normalizer_feature_data.normalize(p_data=self._feature_dataset[p_index])
            labels = self._normalizer_label_data.normalize(p_data=self._label_dataset[p_index])
        else:
            features = self._feature_dataset[p_index]
            labels = self._label_dataset[p_index]

        if self._output_cls == Element:
            feature_obj = self._output_cls(self._feature_space)
            feature_obj.set_values(features)
            label_obj = self._output_cls(self._label_space)
            label_obj.set_values(labels)

        else:
            raise ParamError("The output class is not yet supported for this item")

        return feature_obj, label_obj

## -------------------------------------------------------------------------------------------------
    def get_next(self):

        """

        Returns
        -------

        """
        if self._fetch_mode == self.C_FETCH_BATCH:
            return self.get_next_batch()
        else:
            return self.get_next_instance()

## -------------------------------------------------------------------------------------------------
    def get_next_instance(self):

        """

        Returns
        -------

        """
        # Return an Instance with first 'batch size' features and corresponding labels as a single label
        if len(self._indexes) == 1:
            self._last_batch = True

        elif len(self._indexes) == 0:
            raise Error("End of Data. Please watch the _last_batch attribute.")

        index = self._indexes[0]
        del self._indexes[0]

        val = self.get_data(index)

        feature_element = Element(self._feature_space)
        feature_element.set_values(val[0].get_values())
        label_element = Element(self._label_space)
        label_element.set_values(val[1].get_values())

        return [(feature_element, label_element)]


## -------------------------------------------------------------------------------------------------
    def get_next_batch(self):
        """

        Returns
        -------

        """
        if self._last_batch:
            indexes = self._indexes
            self._indexes.clear()

        elif len(self._indexes) == 0:
            raise Error("End of Data. Please watch the _last_batch attribute.")

        else:
            if self._drop_short:
                if 2*self._batch_size > len(self._indexes):
                    self._last_batch = True

            else:
                if self._batch_size >= len(self._indexes):
                    self._last_batch = True

            indexes = self._indexes[0:self._batch_size]
            del self._indexes[0:self._batch_size]

        feature_values = []
        label_values = []


        if self._output_cls == Element:
            for index in indexes:
                vals = self.get_data(index)
                feature_values.append(vals[0].get_values())
                label_values.append(vals[1].get_values())
            feature_batch = BatchElement(self._feature_space)
            feature_batch.set_values(feature_values)
            label_batch = BatchElement(self._label_space)
            label_batch.set_values(label_values)

        else:
            raise ParamError("This output class is not yet supported for this dataset")

        return [(feature_batch, label_batch)]






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SASDataset(Dataset):
    """

    Parameters
    ----------
    p_state_fpath
    p_action_fpath
    p_state_space
    p_action_space
    p_episode_col
    p_delimiter
    p_drop_columns
    p_batch_size
    p_drop_short
    p_shuffle
    p_eval_split
    p_test_split
    p_settings
    p_logging
    """


    ## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_state_fpath,
                 p_action_fpath,
                 p_state_space : ESpace,
                 p_action_space : ESpace,
                 p_episode_col = 'Episode ID',
                 p_delimiter = '\t',
                 p_drop_columns = ['Episode ID', 'Cycle', 'Day', 'Second', 'Microsecond'],
                 p_batch_size : int = 1,
                 p_drop_short : bool = False,
                 p_shuffle : bool = False,
                 p_eval_split : float = 0,
                 p_test_split : float = 0,
                 p_settings = None,
                 p_logging = Log.C_LOG_ALL
                 ):

        feature_dataset, label_dataset = self._setup_dataset(state_fpath=p_state_fpath,
            action_fpath=p_action_fpath,
            drop_columns=p_drop_columns,
            episode_col=p_episode_col,
            delimiter=p_delimiter)


        self._state_space = p_state_space.copy(True)
        self._action_space = p_action_space.copy(True)

        # feature_space, label_space = self.setup_spaces()


        Dataset.__init__(self,
            p_feature_dataset = feature_dataset,
            p_label_dataset = label_dataset,
            p_batch_size=p_batch_size,
            p_eval_split=p_eval_split,
            p_test_split=p_test_split,
            p_shuffle=p_shuffle,
            p_drop_short=p_drop_short,
            p_settings=p_settings,
            p_logging=p_logging)




## -------------------------------------------------------------------------------------------------
    def setup_spaces(self):
        """

        Parameters
        ----------
        p_state_space
        p_action_space

        Returns
        -------

        """

        label_space = self._state_space.copy()

        for dim in self._action_space.get_dims():
            self._state_space.add_dim(dim)

        feature_space = self._state_space.copy(p_new_dim_ids=True)

        return feature_space, label_space


## -------------------------------------------------------------------------------------------------
    def _setup_dataset(self,
                       state_fpath: str,
                       action_fpath: str,
                       drop_columns: list,
                       episode_col: str,
                       delimiter='\t'):
        """

        Parameters
        ----------
        state_fpath
        action_fpath
        drop_columns
        episode_col
        delimiter

        Returns
        -------

        """

        # Fetching states without dropping columns
        self._states = pd.read_csv(filepath_or_buffer=state_fpath, delimiter=delimiter)

        # Finding the episode change ids
        episodes_entry_change = self._states[episode_col].diff()
        episode_idx = episodes_entry_change[episodes_entry_change.ne(0)].index-1

        # Drop the columns
        self._states = self._states.drop(columns=drop_columns, axis=0)

        # Fetch actions and drop the columns
        self._actions = pd.read_csv(filepath_or_buffer=action_fpath, delimiter=delimiter).drop(columns=drop_columns,
            axis=0)

        # Create input df, with state and action
        input = pd.concat([self._states, self._actions], axis=1, copy=True).iloc[:-1].reset_index(drop=True)
        # Drop rows at episode change
        input = input.drop(labels=episode_idx[1:]).values
        # Create output df
        output = self._states.iloc[1:].reset_index(drop=True)
        # Drop rows at episode change
        output = output.drop(labels=episode_idx[1:]).values

        return input, output


# ## -------------------------------------------------------------------------------------------------
#     def get_data(self, p_index):
#         """
#
#         Parameters
#         ----------
#         p_index
#
#         Returns
#         -------
#
#         """
#         features = self._feature_dataset.iloc[p_index].values
#         feature_obj = self._output_cls(self._feature_space).set_values(features)
#
#         labels = self._states.iloc[p_index+1].values
#         label_obj = self._output_cls(self._label_space).set_values(labels)
#
#         return feature_obj, label_obj