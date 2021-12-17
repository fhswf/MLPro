# industrial_robot_status_controller

[![Build Status: Travis CI](https://travis-ci.com/gavanderhoorn/industrial_robot_status_controller.svg?branch=master)](https://travis-ci.com/gavanderhoorn/industrial_robot_status_controller)
[![ROS build Status - Kinetic](http://build.ros.org/job/Kdev__industrial_robot_status_controller__ubuntu_xenial_amd64/badge/icon)](http://build.ros.org/view/Kdev/job/Kdev__industrial_robot_status_controller__ubuntu_xenial_amd64/)
[![ROS build Status - Melodic](http://build.ros.org/job/Mdev__industrial_robot_status_controller__ubuntu_bionic_amd64/badge/icon)](http://build.ros.org/view/Mdev/job/Mdev__industrial_robot_status_controller__ubuntu_bionic_amd64/)
[![Github Issues](https://img.shields.io/github/issues/gavanderhoorn/industrial_robot_status_controller.svg)](http://github.com/gavanderhoorn/industrial_robot_status_controller/issues)

[![license - apache 2.0](https://img.shields.io/:license-Apache%202.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)


## Overview

These packages provide a `ros_control` compatible controller that publishes "robot status" of (industrial) robots in the form of [RobotStatus][] messages.

`hardware_interface`s need to expose an `IndustrialRobotStatusHandle` resource, which they update with information from a/the robot controller. The `IndustrialRobotStatusController` will then take this and populate the fields of the `RobotStatus` message with the information from the `IndustrialRobotStatusHandle`.

Note: the controller does not implement any logic to *derive* the values of the fields in `RobotStatus`, it merely transforms `IndustrialRobotStatusHandle` into a `RobotStatus` message. The `hardware_interface` is responsible for implementing the correct logic to populate the fields based on information about the status of the robot controller that it interfaces with.


## Installation

### Binaries

These packages have been released through the ROS buildfarm. Only ROS Kinetic and Melodic are supported on Ubuntu Xenial and Bionic (`i386`, `amd64`, `armhf` and `arm64`).

The following command may be used to install the controller on a supported platform:

```bash
sudo apt-get install ros-kinetic-industrial-robot-status-controller
```

Replace `kinetic` with `melodic` when installing on ROS Melodic.

Note: `industrial_robot_status_interface` is only needed when integrating the interface into a `hardware_interface`.

You can now continue with configuring your `ros_control` controller stack. See the [Example - Controller](#controller-1) section below for an example controller configuration.

### From source

This should only be needed in case:

 1. there is no release available for a particular platform
 1. the current release does not include a certain feature or fix available in the source repository
 1. changes to the packages are needed

In other cases a binary installation is preferred.

To build the packages, the following commands may be used in a Catkin workspace (example uses [catkin_tools](https://github.com/catkin/catkin_tools), but `catkin_make` should also work):

```bash
# this assumes there is a Catkin workspace at '/path/to/catkin_ws' and it contains a 'src' sub directory
$ git -C /path/to/catkin_ws/src clone https://github.com/gavanderhoorn/industrial_robot_status_controller.git
$ rosdep update
$ rosdep install --from-paths /path/to/catkin_ws/src -i
$ cd /path/to/catkin_ws
$ catkin build
$ source devel/setup.bash
```

At this point the controller can be used with any `ros_control`-based stack. See the [Example - Controller](#controller-1) section below for an example controller configuration.


## Usage/Integration

### Interface

`#include` the `industrial_robot_status_interface/industrial_robot_status_interface.h` header and store an instance of `industrial_robot_status_interface::IndustrialRobotStatusInterface` somewhere in the `hardware_interface`. Store an instance of `IndustrialRobotStatusHandle` as well.

Register the `IndustrialRobotStatusHandle` with the `IndustrialRobotStatusInterface` and finally register the `IndustrialRobotStatusInterface` with `ros_control` via `InterfaceManager::registerInterface(..)`.

Periodically update the `IndustrialRobotStatusHandle` (typically in `RobotHW::read(..)`) and set the fields (to `TriState::UNKNOWN`, `TriState::FALSE` or `TriState::TRUE`), based upon information from the robot controller available to the `hardware_interface`.

### Controller

As the controller does not implement any logic itself, but merely transforms a `IndustrialRobotStatusHandle` into a `RobotStatus` message, it's fully reusable and requires no source code changes (custom robot status derivation logic should be implemented in the `hardware_interface`).

The controller supports two configurable settings:

 - `publish_rate`: rate at which `RobotStatus` messages should be published (default: 10 Hz)
 - `handle_name` : name of the `IndustrialRobotStatusHandle` resource exposed by the `hardware_interface` (default: `"industrial_robot_status_handle"`)

See the *Example* section below for a full configuration example of the controller.


## Example

This section shows how to integrate the interface into a `hardware_interface` and how to add a stanza to a `ros_control` configuration file to load the controller.

### Interface

Initialising and registering the `IndustrialRobotStatusHandle` and `IndustrialRobotStatusInterface`:

```c++
...

// these could be class members of course
industrial_robot_status_interface::RobotStatus robot_status_resource_{};
industrial_robot_status_interface::IndustrialRobotStatusInterface robot_status_interface_{};

...

// somewhere in the initialisation of the hardware_interface
robot_status_interface_.registerHandle(
  industrial_robot_status_interface::IndustrialRobotStatusHandle(
    "industrial_robot_status_handle", robot_status_resource_));
registerInterface(&robot_status_interface_);

...

```

Note: the first argument to the `IndustrialRobotStatusHandle` ctor can be anything, as long as the controller is configured to look for the same resource handle name. The default is `"industrial_robot_status_handle"`.

Finally: update the members of `robot_status_resource_` with data from the controller:

```c++
...

// somewhere in RobotHW::read(..)
using industrial_robot_status_interface::TriState;
using industrial_robot_status_interface::RobotMode;

// set defaults
robot_status_resource_.in_motion       = TriState::UNKNOWN;
// skip other fields for brevity
robot_status_resource_.mode            = RobotMode::UNKNOWN;
robot_status_resource_.error_code      = 0;

// use controller info to set real values
// note: 'latest_robot_data_' is some variable that contains
//       the latest state received from whatever this hw interface
//       is communicating with.
if (latest_robot_data_.state == MyRobot::IS_MOVING)
  robot_status_resource_.in_motion = TriState::TRUE;

...

```

### Controller

Add the controller as usual to a `ros_control` configuration `.yaml`:

```yaml
robot_status_controller:
  type: industrial_robot_status_controller/IndustrialRobotStatusController
  handle_name: my_custom_status_handle
  publish_rate: 10

```

Here the `IndustrialRobotStatusHandle` was registered under a different resource name, so `handle_name` is set to `"my_custom_status_handle"`.


## Additional information

Refer to the [RobotStatus][] message documentation for more information on how the values for the fields should be derived.


[RobotStatus]: http://docs.ros.org/latest/api/industrial_msgs/html/msg/RobotStatus.html
