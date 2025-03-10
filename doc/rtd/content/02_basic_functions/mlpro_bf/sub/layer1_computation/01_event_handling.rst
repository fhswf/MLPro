.. _target_bf_event:
Event handling
==============

Event handling is a widely used standard technique in software development. And that's how it found its way into MLPro. 
The mechanism is inherited by higher classes in the form of the property class **EventHandler**. This class allows 
event handler methods to be registered and events to be triggered, which in turn call registered handlers. An object of 
type **Event** is passed to each handler. This contains further information about the context of the event.

The event handling functionality is used extensively in MLPro. A large number of higher classes use this mechanism. 
Based on this, even an event-oriented adaptation mechanism is cultivated in 
:ref:`Layer 4 - Machine Learning <target_bf_ml>`.


**Cross reference**

- :ref:`Howto BF-EH-001: Event handling <Howto BF EH 001>`
- :ref:`API reference BF-EVENTS - Event handling <target_api_bf_event>`

