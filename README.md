[![py37](https://github.com/fhswf/MLPro/actions/workflows/python37.yml/badge.svg)](https://github.com/fhswf/MLPro/actions/workflows/python37.yml)
[![py38](https://github.com/fhswf/MLPro/actions/workflows/python38.yml/badge.svg)](https://github.com/fhswf/MLPro/actions/workflows/python38.yml)
[![py39](https://github.com/fhswf/MLPro/actions/workflows/python39.yml/badge.svg)](https://github.com/fhswf/MLPro/actions/workflows/python39.yml)
[![Documentation Status](https://readthedocs.org/projects/mlpro/badge/?version=latest)](https://mlpro.readthedocs.io/en/latest/?badge=latest)

<img src="doc/\logo/original/logo.png" align="right" width="40%"/>
# MLPro - Machine Learning Professional
Machine Learning Professional - A Synoptic Framework for Standardized Machine Learning Tasks in Python

MLPro was developed in 2021 by [Automation Technology and Learning Systems team at Fachhochschule Südwestfalen](https://www.fh-swf.de/de/forschung___transfer_4/labore_3/labs/labor_fuer_automatisierungstechnik__soest_1/standardseite_57.php)

MLPro provides complete, standardized, and reusable functionalities to support your scientific research, educational task, or industrial project in machine learning.

In the first version of MLPro, we provide a standardized Python package for reinforcement learning (RL) and game theoretical (GT) approaches, including environments, algorithms, multi-agent RL (MARL), model-based RL (MBRL) and many more. Additionally, we incorporate the available third party packages by developing wrapper classes to enable our users to reuse the third party packages in MLPro.

# Main Features

-   Test-driven development (CI/CD concept)
-   Clean code and constructed through Object-Oriented Programming
-   Ready-to-use functionalities
-   Usability in scientific, industrial and educational contexts
-   Extensible, maintainable, understandable
-   Attractive UI support (available soon)
-   Reuse of available state-of-the-art implementations
-   Clear documentations

# Documentation

The Documentation is available on : [MLPRORTD](https://www.google.com)

# Installation

## Prerequisites
MLPro requires Python 3.7+

```bash
pip install mlpro
```

# Example
This examples shows how to train CartPole-V1 with Stable-Baselines3 wrapper

```python
import gym

from stable_baselines3 import PPO

from mlpro.rl.models import *
from mlpro.wrappers.openai_gym import WrEnvGYM2MLPro
from mlpro.wrappers.sb3 import WrPolicySB32MLPro

class MyScenario(RLScenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):
        gym_env     = gym.make('CartPole-v1')
        self._env   = WrEnvGYM2MLPro(gym_env, p_logging=p_logging) 

        policy_sb3 = PPO(
                    policy="MlpPolicy",
                    n_steps=5, 
                    env=None,
                    _init_setup_model=False)

        policy_wrapped = WrPolicySB32MLPro(
                p_sb3_policy=policy_sb3, 
                p_observation_space=self._env.get_state_space(),
                p_action_space=self._env.get_action_space(),
                p_ada=p_ada,
                p_logging=p_logging)
        
        return Agent(
            p_policy=policy_wrapped,   
            p_envmodel=None,
            p_name='Smith',
            p_ada=p_ada,
            p_logging=p_logging
        )

training = RLTraining(
        p_scenario_cls=MyScenario,
        p_cycle_limit=1000,
        p_max_adaptations=0,
        p_max_stagnations=0,
        p_visualize=True,
        p_logging=Log.C_LOG_ALL )

training.run()
```

# Implemented Wrappers

| **Features**                | **Status** |
| --------------------------- | ----------------------|
| OpenAI-Gym | :heavy_check_mark: |
| Stable-Baselines3               | :heavy_check_mark: |
| PettingZoo         | :heavy_check_mark: |
| Hyperopt             | :heavy_check_mark: |

# Mainteners
MLPro is currently maintained by [Detlef Arend](https://github.com/detlefarend), [M Rizky Diprasetya](https://github.com/rizkydiprasetya), [Steve Yuwono](https://github.com/steveyuwono), [William Budiatmadjaja](https://github.com/budiatmadjajaWill)

# How to contribute
If you want to contribute, please read [CONTRIBUTING.md](https://github.com/fhswf/MLPro/blob/master/CONTRIBUTING.md)