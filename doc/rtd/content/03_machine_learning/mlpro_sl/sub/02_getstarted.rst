.. _target_getstarted_SL:
Getting Started
---------------

As mentioned in the introductory section, MLPro-SL's functionalities are still limited and not ready to be labelled as the first version.
However, we are working on it to enhance MLPro-SL and bring you full supervised learning functionalities soon.

At the moment, we provide a basic template class for :ref:`supervised learning adaptive function <target_bf_sl_afct>`, which has been extended to the feedforward neural network.
We introduce MLP as a sample of the MLPro-SL model and provide a ready-to-use :ref:`PyTorch-based <_target_sl_afct_pool_pytorch>` multilayer perceptron network.

After following the below step-by-step guideline, we expect the user understands the MLPro-SL in practice and starts using MLPro-SL.

**1. What is MLPro?**
   If you are a first-time user of MLPro, you might wonder what is MLPro.
   Therefore, we recommend initially start with understanding MLPro by checking out the following steps:

   (a) :ref:`MLPro: An Introduction <target_mlpro_introduction>`

   (b) `introduction video of MLPro <https://ars.els-cdn.com/content/image/1-s2.0-S2665963822001051-mmc1.mp4>`_

   (c) :ref:`installing and getting started with MLPro <target_mlpro_getstarted>`

   (d) `MLPro paper in Software Impact journal <https://doi.org/10.1016/j.simpa.2022.100421>`_

**2. What is Supervised Learning?**
   If you have not dealt with supervised learning, we recommend starting to understand at least the basic concept of supervised learning.
   There are plenty of references, articles, papers, books, or videos on the internet that explains supervised learning.
   As an overview, supervised learning is a type of machine learning in which a model is trained on a labelled dataset to predict the output for new/unseen inputs.
   Supervised learning can be used to build a predictive model that can make predictions based on available data.

**3. What is MLPro-SL?**
   We expect that you have a basic knowledge of MLPro and supervised learning.
   Therefore, you need to understand the overview of MLPro-SL by following the steps below:

   (a) :ref:`MLPro-SL introduction page <target_overview_SL>`

**4. Understanding Adapative Function in MLPro-SL**
   First of all, it is important to understand the adaptive function in MLPro-SL, which can be found on :ref:`this page <target_bf_sl_afct>`.

   Then, you can start following some of our howto files related to the adaptive function in MLPro-SL, which is used for model-based RL, as follows:

   (a) :ref:`Howto RL-MB-001: MBRL on RobotHTM Environment <Howto MB RL 001>`

   (b) :ref:`Howto RL-MB-002: MBRL with MPC on Grid World Environment <Howto MB RL 002>`
   
   (c) :ref:`Howto RL-MB-003: Train and Reload Model Based Agent (Gym) <Howto MB RL 003>`


**5. Additional Guidance**
   After following the previous steps, we hope that you could practice MLPro-SL and start using this subpackage for your SL-related activities.
   For more advanced features, we highly recommend you to check out the following files:

   (a) :ref:`API Reference: MLPro-SL <target_api_sl>`
   
   (b) :ref:`API Reference: MLPro-SL Pool of Objects <target_api_pool_sl>`