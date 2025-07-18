"""
### MLPro-BF – Basic Functions (`mlpro.bf`)

This sub-framework represents the functional substructure for all higher-level machine learning frameworks of MLPro. 
It is organized into a total of five layers and contains the following sub-modules:

**Layer 0 - Elementary Functions**

- :mod:`mlpro.bf.various` - Logging, time measurement, ...
- :mod:`mlpro.bf.exceptions` - Exceptions
- :mod:`mlpro.bf.data` - Elementary data management
- :mod:`mlpro.bf.plot` - Plot functionalities


**Layer 1 - Computation**

- :mod:`mlpro.bf.events` - Event management
- :mod:`mlpro.bf.ops` - Operational classes
- :mod:`mlpro.bf.mt` - Multitasking (i.e. multithreading and multiprocessing)


**Layer 2 - Mathematics**

- :mod:`mlpro.bf.math` - Mathematics


**Layer 3 - Application Support**

- :mod:`mlpro.bf.datasets` - Datasets
- :mod:`mlpro.bf.physics` - Physics
- :mod:`mlpro.bf.streams` - Data stream processing
- :mod:`mlpro.bf.systems` - State-based systems
- :mod:`mlpro.bf.control` - Closed-loop control


**Layer 4 - Machine Learning**  

- :mod:`mlpro.bf.ml` - Basics of machine learning

---

The most commonly used classes can be imported from **mlpro.bf** directly:

- :class:`mlpro.bf.various.Log` - Central logging system
- :class:`mlpro.bf.various.TStampType` - Time stamp type
- :class:`mlpro.bf.ops.Mode` - Operating mode definition
- :class:`mlpro.bf.plot.PlotSettings` - Plot configuration class
- :class:`mlpro.bf.exceptions.ParamError` - Parameter error
- :class:`mlpro.bf.exceptions.Error` - General error
- :class:`mlpro.bf.exceptions.ImplementationError` - Implementation error

---

**Learn more:** `MLPro-BF documentation <https://mlpro.readthedocs.io/en/latest/content/02_basic_functions/mlpro_bf/main.html>`_
"""

from .various import Log, TStampType
from .ops import Mode
from .exceptions import ParamError, Error, ImplementationError
from .plot import PlotSettings


# Export list for public API
__all__ = [ 'Log',
            'TStampType',
            'Mode',
            'PlotSettings',
            'ParamError',
            'Error',
            'ImplementationError' ]
