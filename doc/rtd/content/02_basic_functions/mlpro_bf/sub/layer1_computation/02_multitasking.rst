.. _target_bf_mt:
Multitasking
============

In MLPro, the two essential techniques **Multithreading** and **Multiprocessing** for the asynchronous execution of 
program parts are summarized under multitasking. Both techniques allow the use of several or even all cores of the 
CPU(s), which can significantly increase the execution speed of the respective program. The basic prerequisite for 
this is, of course, that significant parts of the program can actually be executed in parallel.

While with multithreading program parts are executed in parallel within the same process, with multiprocessing they 
are started in separate processes. The latter uses computing resources more effectively but requires more 
administrative effort on the part of the operating system. Therefore, it is hard to say in advance which of the two 
techniques is the better choice for a specific implementation. What can be said, however, is that multithreading is 
the more straightforward technique to use with good acceleration of parallel programs. Multiprocessing requires a 
little more detailed knowledge of the internal mechanisms of the runtime environment. It unfolds its strengths, in 
particular with long-running parallel program parts, since the administrative overhead is not so significant here. 
The pros and cons of multithreading and multiprocessing can be excellently discussed. For this purpose, reference is 
made to relevant sources on the Internet.

The details of the multitasking implemented in MLPro are explained in more detail below:

**Range**

In the context of multitasking, the range describes the degree of parallelism of a program function. There are three 
different degrees:

- Serial
- Parallel with multithreading
- Parallel with multiprocessing


**Asynchronous execution of methods**

In order to enable the asynchronous execution of methods of a class, MLPro provides the property class **Async**. 
In particular, this includes the method **_start_async()**, which allows the execution of another method in a separate 
thread or process. 


**Tasks und Workflows**

A fundamental and consistently used concept in MLPro is that of tasks and workflows. A task is a class that can be 
executed in one of the three possible ranges mentioned above. Tasks can in turn be grouped into workflows. 
Any **directed graphs** of tasks can be set up and processed massively parallel via corresponding predecessor 
relationships. Successor tasks are informed about the termination of their predecessors via event technology. The 
**Task** and **Workflow** classes of the same name have once again been implemented as property classes. They are 
consistently reused in higher functions through inheritance. Some examples are :ref:`Stream Processing <target_bf_streams_processing>` 
and the :ref:`Adaptive Workflows <target_bf_ml_workflows>`.


**Gap unter MacOS**

At the time of the creation of MLPro, a technical problem related to multiprocessing occurred on Apple computers 
under MacOS. This is documented at `Race condition when using multiprocessing BaseManager and Pool in Python3 <https://github.com/python/cpython/issues/88321>`_. It is recommended to 
check MLPro functionalities in multiprocessing mode on MacOS-based computers very carefully and, if in doubt, to use 
multithreading.


**Cross Reference**

- :ref:`Howto BF-MT-001: Multitasking - Parallel Algorithms <Howto BF MT 001>`
- :ref:`Howto BF-MT-002: Multitasking - Tasks and Workflows <Howto BF MT 002>`
- :ref:`API Reference BF-MT - Multitasking <target_api_bf_mt>`


