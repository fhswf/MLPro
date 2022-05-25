## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : math.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-05-23  0.0.0     DA       Creation 
## -- 2021-05-30  1.0.0     DA       Release of first version
## -- 2021-08-26  1.1.0     DA       Class Dimension extended by base set and description
## -- 2021-09-11  1.1.0     MRD      Change Header information to match our new library name
## -- 2021-09-23  1.2.0     DA       Changes to deal with big data objects:
## --                                - new class DataObject
## --                                - new base set type 'D' in class Dimension
## --                                - changes in class Element: list instead of np.array
## -- 2021-10-25  1.3.0     DA       New class Function
## -- 2021-12-03  1.3.1     DA       New methods Dimension.copy(), Set.copy(), Set.append()
## -- 2021-12-03  1.3.2     MRD      Fix Set.append() due to the usage of max() on empty list
## -- 2022-01-21  1.4.0     DA       New class TrendAnalyzer
## -- 2022-02-25  1.4.1     SY       Class Dimension extended by auto generated ID
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.4.1 (2022-02-25)

This module provides basic mathematical classes .
"""

import numpy as np
from itertools import repeat
import uuid





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Dimension:
    """
    Objects of this type specify properties of a dimension of a set.

    Parameters:
    -----------
    p_name_short : str
        Short name of dimension
    p_base_set 
        Base set of dimension. See constants C_BASE_SET_*. Default = C_BASE_SET_R.
    p_name_long :str
        Long name of dimension (optional)
    p_name_latex : str
        LaTeX name of dimension (optional)
    p_unit : str
        Unit (optional)
    p_unit_latex : str
        LaTeX code of unit (optional)
    p_boundaries : List
        List with minimum and maximum value (optional)
    p_description : str
        Description of dimension (optional)

    """

    C_BASE_SET_R = 'R'  # real numbers
    C_BASE_SET_N = 'N'  # natural numbers
    C_BASE_SET_Z = 'Z'  # integer numbers
    C_BASE_SET_DO = 'DO'  # (big) data objects (like images, point clouds, ...)

    ## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name_short, p_base_set=C_BASE_SET_R, p_name_long='', p_name_latex='', p_unit='',
                 p_unit_latex='', p_boundaries=[], p_description='') -> None:
        self._id = str(uuid.uuid4())
        self._name_short = p_name_short
        self._base_set = p_base_set
        self._name_long = p_name_long
        self._name_latex = p_name_latex
        self._unit = p_unit
        self._unit_latex = p_unit_latex
        self._boundaries = p_boundaries
        self._description = p_description

    ## -------------------------------------------------------------------------------------------------
    def get_id(self):
        return self._id

    ## -------------------------------------------------------------------------------------------------
    def get_name_short(self):
        return self._name_short

    ## -------------------------------------------------------------------------------------------------
    def get_base_set(self):
        return self._base_set

    ## -------------------------------------------------------------------------------------------------
    def get_name_long(self):
        return self._name_long

    ## -------------------------------------------------------------------------------------------------
    def get_name_latex(self):
        return self._name_latex

    ## -------------------------------------------------------------------------------------------------
    def get_unit(self):
        return self._unit

    ## -------------------------------------------------------------------------------------------------
    def get_unit_latex(self):
        return self._unit_latex

    ## -------------------------------------------------------------------------------------------------
    def get_boundaries(self):
        return self._boundaries

    ## -------------------------------------------------------------------------------------------------
    def get_description(self):
        return self._description

    ## -------------------------------------------------------------------------------------------------
    def copy(self):
        return self.__class__(p_name_short=self._name_short,
                              p_base_set=self._base_set,
                              p_name_long=self._name_long,
                              p_name_latex=self._name_latex,
                              p_unit=self._unit,
                              p_unit_latex=self._unit_latex,
                              p_boundaries=self._boundaries,
                              p_description=self._description)


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Set:
    """
    Objects of this type describe a (multivariate) set in a mathematical sense.
    """

    ## -------------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        self._dim_list = []
        self._dim_ids = []

    ## -------------------------------------------------------------------------------------------------
    def add_dim(self, p_dim: Dimension):
        """
        Raises the dimensionality of the set by adding a new dimension.

        Parameters:
            p_dim       Dimension to be added
        """

        self._dim_ids.append(p_dim.get_id())
        self._dim_list.append(p_dim)

    ## -------------------------------------------------------------------------------------------------
    def get_dim(self, p_id) -> Dimension:
        """
        Returns the dimension specified by it's unique id.
        """

        return self._dim_list[self._dim_ids.index(p_id)]

    ## -------------------------------------------------------------------------------------------------
    def get_num_dim(self):
        """
        Returns the dimensionality of the set (=number of dimensions of the set).
        """

        return len(self._dim_list)

    ## -------------------------------------------------------------------------------------------------
    def get_dim_ids(self):
        """
        Returns the unique ids of the related dimensions.
        """

        return self._dim_ids

    ## -------------------------------------------------------------------------------------------------
    def spawn(self, p_id_list: list):
        """
        Spawns a new class with same type and a subset of dimensions specified
        by an index list.

        Parameters:
            p_id_list       List of indices of dimensions to be adopted

        Returns:
            New object with subset of dimensions
        """

        new_set = self.__class__()
        for i in p_id_list:
            new_set.add_dim(self._dim_list[self._dim_ids.index(i)])

        return new_set

    ## -------------------------------------------------------------------------------------------------
    def copy(self, p_new_dim_ids=True):
        new_set = self.__class__()

        if p_new_dim_ids:
            for dim_id in self.get_dim_ids():
                new_set.add_dim(self.get_dim(dim_id).copy())
        else:
            for dim_id in self.get_dim_ids():
                new_set.add_dim(self.get_dim(dim_id))

        return new_set

    ## -------------------------------------------------------------------------------------------------
    def append(self, p_set, p_new_dim_ids=True):
        if p_new_dim_ids:
            for dim_id in p_set.get_dim_ids():
                self.add_dim(p_set.get_dim(dim_id).copy())
        else:
            for dim_id in p_set.get_dim_ids():
                self.add_dim(p_set.get_dim(dim_id))


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DataObject:
    """
    Container class for (big) data objects of any type with optional additional meta data.
    """

    ## -------------------------------------------------------------------------------------------------
    def __init__(self, p_data, *p_meta_data) -> None:
        self._data = p_data
        self._meta_data = p_meta_data

    ## -------------------------------------------------------------------------------------------------
    def get_data(self):
        return self._data

    ## -------------------------------------------------------------------------------------------------
    def get_meta_data(self) -> tuple:
        return self._meta_data


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Element:
    """
    Element of a (multivariate) set.
    """

    ## -------------------------------------------------------------------------------------------------
    def __init__(self, p_set: Set) -> None:
        self._set = p_set
        self._values = list(repeat(0, self._set.get_num_dim()))

    ## -------------------------------------------------------------------------------------------------
    def get_related_set(self) -> Set:
        return self._set

    ## -------------------------------------------------------------------------------------------------
    def get_dim_ids(self) -> list:
        return self._set.get_dim_ids()

    ## -------------------------------------------------------------------------------------------------
    def get_values(self):
        return self._values

    ## -------------------------------------------------------------------------------------------------
    def set_values(self, p_values):
        """
        Overwrites the values of all components of the element.

        Parameters:
            p_values        Something iterable with same length as number of element dimensions.
        """

        self._values = p_values

    ## -------------------------------------------------------------------------------------------------
    def get_value(self, p_dim_id):
        return self._values[self._set.get_dim_ids().index(p_dim_id)]

    ## -------------------------------------------------------------------------------------------------
    def set_value(self, p_dim_id, p_value):
        self._values[self._set.get_dim_ids().index(p_dim_id)] = p_value


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ElementList:
    """
    List of Element objects.
    """

    ## -------------------------------------------------------------------------------------------------
    def __init__(self, ):
        self._elem_list = []
        self._elem_ids = []

    ## -------------------------------------------------------------------------------------------------
    def add_elem(self, p_id, p_elem: Element):
        """
        Adds an element object under it's id in the internal element list.

        Parameters:
            p_id        Unique id of the element
            p_elem      Element object to be added
        """

        self._elem_ids.append(p_id)
        self._elem_list.append(p_elem)

    ## -------------------------------------------------------------------------------------------------
    def get_elem_ids(self) -> list:
        return self._elem_ids

    ## -------------------------------------------------------------------------------------------------
    def get_elem(self, p_id) -> Element:
        return self._elem_list[self._elem_ids.index(p_id)]


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MSpace(Set):
    """
    Objects of this type represent a metric space. The method distance implements the metric of the 
    space.
    """

    ## -------------------------------------------------------------------------------------------------
    def distance(self, p_e1: Element, p_e2: Element):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ESpace(MSpace):
    """
    Objects of this type represent an Euclidian space. The distance method
    implements the Euclidian norm.
    """

    ## -------------------------------------------------------------------------------------------------
    def distance(self, p_e1: Element, p_e2: Element):
        return np.sum((np.array(p_e1.get_values()) - np.array(p_e2.get_values())) ** 2) ** 0.5


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Function:
    """
    Model class for an elementary bi-multivariate mathematical function that maps elements of a
    multivariate input space to elements of a multivariate output space.
    """

    ## -------------------------------------------------------------------------------------------------
    def __init__(self, p_input_space: MSpace, p_output_space: MSpace, p_output_elem_cls=Element):
        """
        Parameters:
            p_input_space       Input space
            p_output_space      Output space
            p_output_elem_cls   Output element class (compatible to class Element)
        """

        self._input_space = p_input_space
        self._output_space = p_output_space
        self._output_elem_cls = p_output_elem_cls

    ## -------------------------------------------------------------------------------------------------
    def map(self, p_input: Element) -> Element:
        """
        Maps a multivariate abscissa/input element to a multivariate ordinate/output element. 
        """

        output = self._output_elem_cls(self._output_space)
        self._map(p_input, output)
        return output

    ## -------------------------------------------------------------------------------------------------
    def _map(self, p_input: Element, p_output: Element):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TrendAnalyzer:
    """
    Template class for trend analysis. Values can be added and the implemented algorithm decides
    about the trend.

    Parameters
    ----------
    p_horizon : int
        This number of last values is buffered as base for the trend computation.

    """

    C_TREND_POSITIVE = 1
    C_TREND_NEGATIVE = -1
    C_TREND_NEUTRAL = 0

    ## -------------------------------------------------------------------------------------------------
    def __init__(self, p_horizon: int):
        self._horizon = p_horizon
        self._trend = self.C_TREND_NEUTRAL
        self._buffer = []

    ## -------------------------------------------------------------------------------------------------
    def add_value(self, p_value: float) -> int:
        """
        Adds a new value to the buffer and computes the trend of the buffered values by calling the
        custom methodd _compute_trend().

        Parameters
        ----------
        p_value : float
            Value to be added to the internal buffer.

        Returns
        -------
        trend : int
            Trend of the buffered values. Possible values are C_TREND_POSITIVE, C_TREND_NEGATIVE, C_TREND_NEUTRAL.

        """

        self._buffer.append(p_value)
        if len(self._buffer) > self._horizon: self._buffer.pop()

        trend = self._compute_trend(np.asarray(self._buffer))
        if trend > 0:
            self._trend = self.C_TREND_POSITIVE
        elif trend < 0:
            self._trend = self.C_TREND_NEGATIVE
        else:
            self._trend = self.C_TREND_NEUTRAL

        return self._trend

    ## -------------------------------------------------------------------------------------------------
    def _compute_trend(self, p_values: np.ndarray) -> int:
        """
        Custom method for trend algorithm.

        Parameters
        ----------
        p_values : np.ndarray
            Numpy array with values to be analyzed.

        Returns
        -------
        trend : float
            Where a value >0 means a positive trend, 0 a neutral trend and <0 a negative trend.

        """

        raise NotImplementedError

    ## -------------------------------------------------------------------------------------------------
    def get_trend(self) -> int:
        """
        Returns the trend of the currently buffered values.

        Returns
        -------
        trend : int
            Valid values are: -1 for negative trend, 0 for a neutral trend and 1 for a positive trend.

        """

        return self._trend
