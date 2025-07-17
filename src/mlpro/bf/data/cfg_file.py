## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.data
## -- Module  : cfg_file.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-12-11  0.1.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2024-12-11)

This module provides classes to deal with persistent configuration data. 

"""

from pathlib import Path
import json



# Export list for public API
__all__ = [ 'ConfigFile' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ConfigFile:
    """
    Stores configuration data in a local JSON file.

    Parameters
    ----------
    p_fname : str
        Name of the local JSON file.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_fname: str):
        self._fname  = p_fname


## -------------------------------------------------------------------------------------------------
    def get(self, p_key):
        """
        Returns the values stored for the specified key. If no values were found an exception is
        raised.

        Parameters
        ----------
        p_key
            Key.

        Returns
        -------
        values
            Values stored for the specified key.
        """

        with open(self._fname, "r") as file:
            config = json.load(file)

        return config[p_key]
 

## -------------------------------------------------------------------------------------------------
    def set(self, p_key, p_values) -> bool:
        """
        Stores the values of the specified key.

        Parameters
        ----------
        p_key
            Key.
        p_values
            Values to be stored.

        Returns
        -------
        bool
            True, if storing was successfull. False otherwise.
        """

        try:
            file_path = Path(self._fname)
            file_path.parent.mkdir(parents=True, exist_ok=True)   

            try:
                with file_path.open("r+") as file:
                    try:
                        config = json.load(file)
                    except:
                        config = {}

                    config[p_key] = p_values
                    file.seek(0)
                    json.dump(config, file, indent=4)
                    file.truncate()
            except:
                with file_path.open("w") as file:
                    config = { p_key : p_values }
                    json.dump(config, file, indent=4)

            return True
        
        except:
            return False