.. _target_getstarted_SL:
Getting Started
---------------

As mentioned in the introductory section, MLPro-SL's functionalities are still limited and not ready to be labelled as the first version.
However, we are working on it to enhance MLPro-SL and bring you full supervised learning functionalities soon.

At the moment, we provide a basic template class for :ref:`supervised learning adaptive function <target_bf_sl_afct>`, which has been extended to the feedforward neural network.
We introduce MLP as a sample of the MLPro-SL model and provide a ready-to-use :ref:`PyTorch-based <_target_sl_afct_pool_pytorch>` multilayer perceptron network.

No experience with MLPro? To learn more about MLPro, please refer to the :ref:`Getting Started page of MLPro <target_mlpro_getstarted>`.

After following the below step-by-step guideline, we expect the user understands the MLPro-SL in practice and starts using MLPro-SL.

**1. What is Supervised Learning?**
   If you have not worked with supervised learning before, we recommend starting with an understanding of its basic concepts.
   There are numerous references, articles, papers, books, and videos available online that explain supervised learning.
   In brief, supervised learning is a type of machine learning where a model is trained on a labeled dataset, meaning each training example is paired with the correct output.
   The model learns to map inputs to the correct outputs by minimizing the difference between its predictions and the actual labels.
   Once trained, it can then make predictions on new, unseen data based on the patterns it has learned.

**2. What is MLPro-SL?**
   We assume you have a fundamental understanding of MLPro and supervised learning.
   Therefore, you need to familiarize yourself with the overview of MLPro-SL by following these steps:

   (a) :ref:`MLPro-SL introduction page <target_overview_SL>`

**3. Understanding Adapative Function in MLPro-SL**
   Firstly, it is essential to understand the adaptive function in MLPro-SL, which can be found on  :ref:`this page <target_bf_sl_afct>`.

   Next, you can refer to our how-to files related to the adaptive function in MLPro-SL, which is used for model-based RL, as outlined below:

   (a) `Howto RL-MB-001: Train and Reload Model Based Agent (Gymnasium) <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/04_howtos_mb/howto_rl_mb_001_train_and_reload_model_based_agent_gym%20copy.html>`_

   (b) :ref:`Howto RL-MB-002: MBRL with MPC on Grid World Environment <Howto MB RL 001>`

   For more advanced supervised learning techniques in model-based RL, such as using a native model-based RL network, you can refer to the following example:

   (c) `Howto RL-MB-002: MBRL on RobotHTM Environment <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/04_howtos_mb/howto_rl_mb_002_robothtm_environment.html>`_


**4. Additional Guidance**
   After completing the previous steps, we encourage you to practice with MLPro-SL and begin using this subpackage for your supervised learning activities. 
   For more advanced features, we recommend reviewing the following files:

   (a) :ref:`API Reference: MLPro-SL <target_api_sl>`
   
   (b) :ref:`API Reference: MLPro-SL Pool of Objects <target_api_pool_sl>`