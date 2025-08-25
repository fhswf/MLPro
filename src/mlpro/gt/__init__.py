"""
### MLPro-GT – Game Theory (`mlpro.gt`)

This sub-framework provides model classes and tools for Game Theory, focusing on both dynamic and native game theory tasks.  
It enables the modeling, analysis, and solution of game-theoretic problems within the MLPro ecosystem.

MLPro-GT is organized into the following sub-modules:

- **Dynamic Games** (`mlpro.gt.dynamicgames`):
  - :mod:`mlpro.gt.dynamicgames.basics` – Model classes for tasks related to Game Theory in dynamic games
  - :mod:`mlpro.gt.dynamicgames.potential` – Model classes for Potential Games in dynamic programming
  - :mod:`mlpro.gt.dynamicgames.stackelberg` – Model classes for Stackelberg Games in dynamic programming

- **Native Game Theory** (`mlpro.gt.native`):
  - :mod:`mlpro.gt.native` – Model classes for tasks related to native game theory

---

**Learn more:** `MLPro-GT documentation <https://mlpro.readthedocs.io/en/latest/content/03_machine_learning/mlpro_gt/main.html>`_

"""

from .dynamicgames import *