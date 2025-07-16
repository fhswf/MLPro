## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.math.normalizers
## -- Module  : basics.py
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
## -- 2024-04-30  1.1.0     DA       Refactoring and new class Renormalizable
## -- 2024-05-23  1.2.0     DA       Method Normalizer._set_parameters(): little optimization
## -- 2024-07-12  1.2.1     LSB      Renormalization error
## -- 2025-06-24  1.3.0     DA       Refactoring and extension
## -- 2025-06-25  1.4.0     DA       Method Normalizer.renormalize(): tuning of dim-wise renormalization
## -- 2025-06-30  2.0.0     DA       Class Normalizer: new parent class Scaler
## -- 2025-07-07  2.0.1     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.0.1 (2025-07-07)

This module provides base class for Normalizers and normalizer objects including MinMax normalization and
normalization by Z transformation.
"""


import numpy as np

from mlpro.bf.exceptions import ParamError
from mlpro.bf.math import Data, Element, Scaler



# Export list for public API
__all__ = [ 'Normalizer',
            'Renormalizable' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Normalizer (Scaler):
    """
    Base class for normalizers. A normalizer in MLPro is a linear scaler with a fixed parameter 
    structure providing the following basic operations:

    Normalization   (= scaling):   scaled_data   = unscaled_data * self._param[0] + self._param[1]
    Denormalization (= unscaling): unscaled_data = ( scaled_data - self._param[1] ) / self._param[0] 
    Renormalization (= unscaling with old params and scaling with new params)

    self._param_old and self._param_new are Numpy arrays of shape (2, n_dims) where n_dims is the number of dimensions
    of the data to be normalized. The first row contains the scaling factors and the second row
    contains the offsets for each dimension.

    Parameters
    ----------
    See class Scaler for details.
    """

## -------------------------------------------------------------------------------------------------
    def normalize( self, 
                   p_data : Data, 
                   p_dim : int = None,
                   p_param = None ) -> Data:
        """
        Normalizes the specified data.

        Parameters
        ----------
        p_data : Data
            Data to be normalized.
        p_dim : int = None
            Optional index of the dimension to be normalized.
        p_param = None
            Optional parameter set to be applied to the normalization. If None the set stored in
            self._param_new is used.

        Returns
        -------
        Data
            The normalized data.
        """

        return self.scale( p_data = p_data,
                           p_dim = p_dim,
                           p_param = p_param )


## -------------------------------------------------------------------------------------------------
    def denormalize( self, 
                     p_data : Data, 
                     p_dim : int = None,
                     p_param = None ) -> Data:
        """
        Denormalizes the specified data.

        Parameters
        ----------
        p_data : Data
            Data to be denormalized.
        p_dim : int = None
            Optional index of the dimension to be denormalized.
        p_param = None
            Optional parameter set to be applied to the denormalization. If None the set stored in
            self._param_old is used.

        Returns
        -------
        Data
            The normalized data.
        """

        return self.unscale( p_data = p_data,
                             p_dim = p_dim,
                             p_param = p_param )
    

## -------------------------------------------------------------------------------------------------
    def renormalize( self, 
                     p_data : Data, 
                     p_dim : int = None,
                     p_param_old = None,
                     p_param_new = None ) -> Data:
        """
        Renormalizes the specified data by denormalizing them with previous parameters stored in 
        _param_old and normalize them with the current parameters in _param_new.

        Parameters
        ----------
        p_data : Data
            Data to be renormalized.
        p_dim : int = None
            Optional index of the dimension to be renormalized.
        p_param_old = None
            Optional parameter set to be applied to the denormalization. If None the set stored in
            self._param_old is used.
        p_param_new = None
            Optional parameter set to be applied to the normalizaion. If None the set stored in
            self._param_new is used.

        Returns
        -------
        Data
            The renormalized data.
        """

        if ( self._param_old is None ) and ( p_param_old is None ): return p_data
        
        if ( p_dim is not None ) and np.array_equal(self._param_new[:,p_dim], self._param_old[:,p_dim] ):
            return p_data
        
        return self.rescale( p_data = p_data, 
                             p_dim = p_dim,
                             p_param_old = p_param_old,
                             p_param_new = p_param_new )
    

## -------------------------------------------------------------------------------------------------
    def _map( self, 
              p_input : Element, 
              p_output : Element = None, 
              p_dim : int = None ) -> Element:

        scale, offset = self._param
        output = p_input if p_output is None else p_output
        values = p_input.get_values()

        if p_dim is None:
            values = values * scale + offset
        else:
            values[p_dim] = values[p_dim] * scale[p_dim] + offset[p_dim]

        output.set_values( p_values = values )
        return output
    

## -------------------------------------------------------------------------------------------------
    def _map_ndarray( self, 
                      p_input : np.ndarray, 
                      p_output : np.ndarray = None, 
                      p_dim = None ) -> np.ndarray:
        
        scale, offset = self._param[0], self._param[1]
        output = p_input if p_output is None else p_output
    
        if p_dim is None:
            np.multiply(p_input, scale, out=output)
            np.add(output, offset, out=output)
        else:
            np.multiply(p_input[:,p_dim], scale, out=output[:,p_dim])
            np.add(output[:,p_dim], offset, out=output[:,p_dim])

        return output
        

## -------------------------------------------------------------------------------------------------
    def _map_list( self, 
                   p_input : list, 
                   p_output : list = None, 
                   p_dim = None ) -> list:
        
        if p_dim is None:
            raise ParamError('Method Normalizer._map_list() needs a valid p_dim')
        
        scale, offset = self._param
        output = p_input if p_output is None else p_output

        input = np.array(p_input) 
        np.multiply(input, scale[p_dim], out=input)
        np.add(input, offset[p_dim], out=input)
        output[:] = input.tolist()

        return output
    

## -------------------------------------------------------------------------------------------------
    def _map_inverse( self, 
                      p_input : Element, 
                      p_output : Element = None, 
                      p_dim : int = None ) -> Element:

        scale, offset = self._param
        output = p_input if p_output is None else p_output
        values = p_input.get_values()

        if p_dim is None:
            values = ( values - offset ) / scale
        else:
            values[p_dim] = ( values[p_dim] - offset[p_dim] ) / scale[p_dim]

        output.set_values( p_values = values)
        return output
    

## -------------------------------------------------------------------------------------------------
    def _map_inverse_ndarray( self, 
                              p_input : np.ndarray, 
                              p_output : np.ndarray = None, 
                              p_dim = None ) -> np.ndarray:

        scale, offset = self._param[0], self._param[1]
        output = p_input if p_output is None else p_output

        if p_dim is None:
            np.subtract(p_input, offset, out=output)
            np.divide(output, scale, out=output)
        else:
            np.subtract(p_input[:,p_dim], offset, out=output[:,p_dim])
            np.divide(output[:,p_dim], scale, out=output[:,p_dim])

        return output       
    

## -------------------------------------------------------------------------------------------------
    def _map_inverse_list( self, 
                           p_input : list, 
                           p_output : list = None, 
                           p_dim = None ) -> list:
        
        if p_dim is None:
            raise ParamError('Method Normalizer._map_inverse_list() needs a valid p_dim')

        scale, offset = self._param
        output = p_input if p_output is None else p_output

        input = np.array(p_input) 
        np.subtract(input, offset[p_dim], out=input)
        np.divide(input, scale[p_dim], out=input)
        output[:] = input.tolist()

        return output





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Renormalizable:
    """
    Property class to add custom renomalization of internally stored data.
    """

## -------------------------------------------------------------------------------------------------
    def renormalize( self, p_normalizer : Normalizer ):
        """
        Custom method to renormalize internally stored data. 

        Parameters
        ----------
        p_normalizer : Normalizier
            Suitable normalizer object to be used for renormalization.
        """

        pass