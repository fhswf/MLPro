## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.math
## -- Module  : normalizers.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-09-16  0.0.0     LSB      Creation
## -- 2022-09-20  1.0.0     LSB      Release of first version
## -- 2022-09-23  1.0.1     LSB      Refactoring
## -- 2022-09-26  1.0.2     LSB      Refatoring and reduced custom normalize and denormalize methods
## -- 2022-10-01  1.0.3     LSB      Refactoring and redefining the update parameter method
## -- 2022-10-16  1.0.4     LSB      Updating z-transform parameters based on a new data/element(np.ndarray)
## -- 2022-10-18  1.0.5     LSB      Refactoring following the review
## -- 2022-11-03  1.0.6     LSB      Refactoring new update methods
## -- 2022-11-03  1.0.7     LSB      Refactoring
## -- 2022-12-09  1.0.8     LSB      Handling zero division error
## -- 2022-12-09  1.0.9     LSB      Returning same object after normalization and denormalization
## -- 2022-12-29  1.0.10    LSB      Bug Fix
## -- 2022-12-30  1.0.11    LSB      Bug Fix ZTransform
## -- 2023-01-07  1.0.12    LSB      Bug Fix
## -- 2023-01-12  1.0.13    LSB      Bug Fix
## -- 2023-02-13  1.0.14    LSB      BugFix: Changed the direct reference to p_param to a copy object
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.14 (2023-02-13)
This module provides base class for Normalizers and normalizer objects including MinMax normalization and
normalization by Z transformation.

"""

from mlpro.bf.math import *
import numpy as np
from typing import Union


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Normalizer:
    """
    Base template class for normalizer objects.
    """

    ## -------------------------------------------------------------------------------------------------
    def __init__(self):

        self._param = None
        self._param_old = None
        self._param_new = None


    ## -------------------------------------------------------------------------------------------------
    def _set_parameters(self, p_param):
        """
        custom method to set the normalization parameters

        Parameters
        ----------
        p_set:Set
            Set related to the elements to be normalized

        Returns
        -------
        boolean:True
            Returns true after setting the parameters
        """
        self._param = p_param.copy()


    ## -------------------------------------------------------------------------------------------------
    def normalize(self, p_data: Union[Element, np.ndarray]):
        """
        Method to normalize a data (Element/ndarray) element based on MinMax or Z-transformation

        Parameters
        ----------
        p_data:Element or a numpy array
            Data element to be normalized

        Returns
        -------
        element:Element or numpy array
            Normalized Data
        """

        if self._param is None:
            raise ImplementationError('Normalization parameters not set')
        if isinstance(p_data, Element):
            p_data.set_values(np.multiply(p_data.get_values(), self._param[0]) - self._param[1])
        elif isinstance(p_data, np.ndarray):
            p_data = np.multiply(p_data, self._param[0]) - self._param[1]
        else:
            raise ParamError('Wrong data type provided for normalization')
        return p_data


    ## -------------------------------------------------------------------------------------------------
    def denormalize(self, p_data: Union[Element, np.ndarray]):
        """
        Method to denormalize a data (Element/ndarray) element based on MinMax or Z-transformation

        Parameters
        ----------
        p_data:Element or a numpy array
            Data element to be denormalized

        Returns
        -------
        element:Element or numpy array
            Denormalized Data
        """

        if self._param is None:
            raise ImplementationError('Normalization parameters not set')

        if isinstance(p_data, Element):

            p_data.set_values(np.multiply(p_data.get_values(), 1 / self._param[0]) + (
                    self._param[1] / self._param[0]))

        elif isinstance(p_data, np.ndarray):
            p_data = np.multiply(p_data, 1 / self._param[0]) + \
                     (self._param[1] / self._param[0])
            p_data = np.nan_to_num(p_data)
        else:
            raise ParamError('Wrong datatype provided for denormalization')

        return p_data


    ## -------------------------------------------------------------------------------------------------
    def renormalize(self, p_data: Union[Element, np.ndarray]):
        """
        Method to denormalize and renormalize an element based on old and current normalization parameters.

        Parameters
        ----------
        p_data:Element or numpy array
            Element to be renormalized.

        Returns
        -------
        renormalized_element:Element or numpy array
            Renormalized Data

        """

        self._set_parameters(self._param_old)
        denormalized_element = self.denormalize(p_data)
        self._set_parameters(self._param_new)
        renormalized_element = self.normalize(denormalized_element)
        return renormalized_element


    ## -------------------------------------------------------------------------------------------------
    def update_parameters(self, p_data: Union[Set, Element, np.ndarray]):
        """
        Custom method to update normalization parameters.

        Parameters
        ----------
        p_data
            arguments specific to normalization parameters. Check the normalizer objects for specific parameters

        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NormalizerMinMax(Normalizer):
    """
    Class to normalize elements based on MinMax normalization.
    """

    ## -------------------------------------------------------------------------------------------------
    def update_parameters(self, p_set: Set = None, p_boundaries: Union[list, np.ndarray] = None):
        """
        Method to update the normalization parameters of MinMax normalizer.

        Parameters
        ----------
        p_set:Set
            Set related to the elements to be normalized
        p_boundaries:ndarray
            array consisting of boundaries related to the dimension of the array

        """
        if self._param_new is not None: self._param_old = self._param_new.copy()

        try:
            if self._param_new is None: self._param_new = np.zeros([2, len(p_set.get_dim_ids())], dtype=np.float64)
            boundaries = [p_set.get_dim(i).get_boundaries() for i in p_set.get_dim_ids()]

        except:
            try:
                if self._param_new is None: self._param_new = np.zeros([(len(p_boundaries)), (len(p_boundaries))])
                boundaries = p_boundaries.reshape(-1, 2)

            except:
                raise ParamError("Wrong parameters provided for update. Please provide a set as p_set or boundaries as "
                                 "p_boundaries")

        for i, boundary in enumerate(boundaries):
            if (boundary[1] - boundary[0]) == 0:
                self._param_new[0][i] = 0
            else:
                self._param_new[0][i] = (2 / (boundary[1] - boundary[0]))
            if (boundary[1] - boundary[0]) == 0:
                self._param_new[0][i] = 0
            else:
                self._param_new[1][i] = (2 * boundary[0] / (boundary[1] - boundary[0]) + 1)

        if self._param is not None:
            self._param_old = self._param.copy()
        else:
            self._param_old = self._param_new.copy()
        self._param = self._param_new.copy()
        pass


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NormalizerZTrans(Normalizer):
    """
    Class for Normalization based on Z transformation.
    """

    ## -------------------------------------------------------------------------------------------------
    def update_parameters(self,
                          p_dataset: np.ndarray = None,
                          p_data_new: Union[Element, np.ndarray] = None,
                          p_data_del: Union[Element, np.ndarray] = None):
        """
        Method to update the normalization parameters for Z transformer

        Parameters
        ----------
        p_dataset:numpy array
            Dataset related to the elements to be normalized. Using this parameter will reset the normalization
            parameters based on the dataset provided.
        p_data_new:Element or numpy array
            New element to update the normalization parameters. Using this parameter will set/update the
            normalization parameters based on the data provided.
        p_data_del:Element or Numpy array
            Old element that is replaced with the new element.

        """
        try:
            data_new = p_data_new.get_values()
        except:
            data_new = p_data_new

        try:
            data_del = p_data_del.get_values()
        except:
            data_del = p_data_del

        if self._param_new is not None: self._param_old = self._param_new.copy()

        if data_new is None and data_del is None and isinstance(p_dataset, np.ndarray):
            self._std = np.std(p_dataset, axis=0, dtype=np.float64)
            self._mean = np.mean(p_dataset, axis=0, dtype=np.float64)
            self._n = len(p_dataset)
            if self._param_new is None: self._param_new = np.zeros([2, self._std.shape[-1]])

        elif isinstance(data_new, np.ndarray) and data_del is None and p_dataset is None:
            # this try/except block checks if the parameters are previously set with a dataset, otherwise sets the
            # parameters based on a single element
            try:
                old_mean = self._mean.copy()
                self._n += 1
                self._mean = (old_mean * (self._n - 1) + data_new) / (self._n)
                self._std = np.sqrt((np.square(self._std) * (self._n - 1)
                                     + (data_new - self._mean) * (data_new - old_mean)) / (self._n))

            except:
                self._n = 1
                self._mean = data_new.copy()
                self._std = np.zeros(shape=data_new.shape)

            if self._param_new is None: self._param_new = np.zeros([2, data_new.shape[-1]])

        elif isinstance(data_new, np.ndarray) and isinstance(data_del, np.ndarray) and not p_dataset:
            try:
                old_mean = self._mean.copy()
                self._mean = old_mean + ((data_new - data_del) / (self._n))
                self._std = np.sqrt(np.square(self._std) + (
                    ((np.square(data_new) - np.square(data_del)) - self._n * (np.square(
                        self._mean) - np.square(old_mean)))) / self._n)

            except:
                raise ParamError("Normalization parameters are not initialised prior to updating with replacing a "
                                 "data element")

        else:
            raise ParamError("Wrong parameters for update_parameters(). Please either provide a dataset as p_dataset "
                             "or a new data element as p_data ")

        self._param_new[0] = np.divide(1, self._std, out = np.zeros_like(self._std), where = self._std!=0)
        self._param_new[1] = np.divide(self._mean, self._std, out = np.zeros_like(self._std), where = self._std!=0)
        # self._param_new[1 == np.inf] = 0
        if self._param is not None:
            self._param_old = self._param.copy()
        else:
            self._param_old = self._param_new.copy()
        self._param = self._param_new.copy()
