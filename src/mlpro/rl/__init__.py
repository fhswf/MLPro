"""
### MLPro-RL – Reinforcement Learning (`mlpro.rl`)

This sub-framework provides classes and tools for Reinforcement Learning (RL), supporting the implementation, training, and evaluation of RL scenarios and agents.
It builds upon `mlpro.bf` and integrates seamlessly with other MLPro components.

MLPro-RL is organized around model classes and includes the following sub-modules:

- :mod:`mlpro.rl.models_env` – Model classes for environments
- :mod:`mlpro.rl.models_env_ada` – Model classes for adaptive environment models
- :mod:`mlpro.rl.models_agents` – Model classes for policies, model-free and model-based agents, and multi-agent systems
- :mod:`mlpro.rl.models_train` – Model classes for defining, executing, and managing RL training scenarios

---

The most commonly used classes can be imported from **mlpro.rl** directly:

- :class:`mlpro.rl.models_env.Environment` – Base class for RL environments
- :class:`mlpro.rl.models_env.Reward` – Base class for reward functions
- :class:`mlpro.rl.models_env_ada.EnvModel` – Base class for model-based agent
- :class:`mlpro.rl.models_agents.Agent` – Base class for agents
- :class:`mlpro.rl.models_agents.Policy` – Base class for policies
- :class:`mlpro.rl.models_train.RLScenario` – RL training scenario framework
- :class:`mlpro.rl.models_train.RLTraining` – Training loop and agent management

**Learn more:** `MLPro-RL documentation <https://mlpro.readthedocs.io/en/latest/content/03_machine_learning/mlpro_rl/main.html>`_

"""

from .models import *
