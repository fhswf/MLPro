.. _target_bf_ml:

Layer 4 - Machine learning
==========================

One of the fundamental concepts in MLPro is to anchor universal standards for machine learning already in the basic 
functions. This shall facilitate the creation of higher ML functionalities and ensure their recombinability. The 
challenge here is to capture the nature of machine learning on an abstract and general level while 
establishing concrete templates and processes and solving elementary subtasks. The highest layer 4 
of the basic functions of MLPro is dedicated to this topic.

.. image:: layer4_machine_learning/images/MLPro-BF-ML_Overview.drawio.png
    :scale: 45 %

The focus of the consideration is the :ref:`adaptive model <target_bf_ml_model>` with its elementary properties

- Adaptivity
- Executability
- Parameterizability
- Persistence

Of course, this model does not exist just on its own. Rather, it interacts with a simulated or real object.
For example, in the case of offline supervised learning, this can be a data set, in the case of online unsupervised 
learning, a data stream, or in the case of reinforcement learning, a state-based system. Topics like this are 
covered in MLPro in higher-level ML frameworks. On an abstract level, however, we introduce the  
:ref:`ML scenario <target_bf_ml_scenario>` for this because although we do not yet know anything about the 
concrete ML application, we know that an adaptive model is involved. Furthermore, we propagate that the application 
is executable in "simulation" or "real operation" mode.

In MLPro, a model's :ref:`training and the tuning <target_bf_ml_train_and_tune>`  of its hyperparameters are based 
on such an ML scenario. For tuning, powerful packages from third parties are already integrated at this low level 
using wrapper technology.

In more complex applications, it can be helpful to group multiple models and allow communication between them. 
It would be desirable here to optimally utilize the system resources through parallel processing. For this purpose, 
MLPro provides :ref:`adaptive workflows <target_bf_ml_workflows>`.

In the field of systems engineering, the creation of suitable simulations or digital twins is an essential aspect.
However, suppose a system cannot be described mathematically precisely enough due to its complexity or unknown 
influencing factors. In that case, machine learning methods can also be used to imitate the system behavior based 
on historical data and/or online monitoring. For this purpose, MLPro provides standards for 
:ref:`adaptive systems <target_bf_ml_asystems>`.


**Learn more**

.. toctree::
   :maxdepth: 1
   :glob:

   layer4_machine_learning/*


**Cross reference**

- :ref:`Related howtos <target_howtos_bf_ml>`
- :ref:`API reference BF-ML - Machine learning <target_api_bf_ml>`
