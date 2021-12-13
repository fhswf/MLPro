/**
 * Copyright (c) 2019, G.A. vd. Hoorn (TU Delft Robotics Institute)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @author G.A. vd. Hoorn (TU Delft Robotics Institute)
 */

#include <industrial_robot_status_controller/industrial_robot_status_controller.h>

#include <pluginlib/class_list_macros.hpp>

#include <memory>


namespace industrial_robot_status_controller
{


bool IndustrialRobotStatusController::init(
  InterfaceType* hw,
  ros::NodeHandle& root_node_handle,
  ros::NodeHandle& controller_node_handle)
{
  if ((robot_status_interface_ = hw) == nullptr)
  {
    ROS_ERROR("IndustrialRobotStatusController: Could not get Industrial "
              "Robot Status interface from hardware");
    return false;
  }

  publish_rate_ = 10.0;
  if (!controller_node_handle.getParam("publish_rate", publish_rate_))
  {
    ROS_INFO_STREAM("IndustrialRobotStatusController: no 'publish_rate' "
                    "parameter. Using default " << publish_rate_ << " [Hz].");
  }

  std::string handle_name = "industrial_robot_status_handle";
  if (!controller_node_handle.getParam("handle_name", handle_name))
  {
    ROS_INFO_STREAM("IndustrialRobotStatusController: no 'handle_name' "
                    "parameter. Using default '" << handle_name << "'.");
  }

  try
  {
    robot_status_handle_ = std::make_unique<HandleType>(
        robot_status_interface_->getHandle(handle_name));
  }
  catch (const hardware_interface::HardwareInterfaceException& ex)
  {
    ROS_ERROR_STREAM("IndustrialRobotStatusController: exception getting "
                     "robot status handle: " << ex.what());
    return false;
  }

  // init publisher for robot status
  robot_status_pub_.init(root_node_handle, "robot_status", 1);

  return true;
}


void IndustrialRobotStatusController::starting(const ros::Time& time)
{
  last_publish_time_ = time;
}


static industrial_msgs::TriState::_val_type convert(const industrial_robot_status_interface::TriState& tristate)
{
  if (tristate == industrial_robot_status_interface::TriState::TRUE)
    return industrial_msgs::TriState::TRUE;
  if (tristate == industrial_robot_status_interface::TriState::FALSE)
    return industrial_msgs::TriState::FALSE;
  if (tristate == industrial_robot_status_interface::TriState::UNKNOWN)
    return industrial_msgs::TriState::UNKNOWN;
}


static industrial_msgs::RobotMode::_val_type convert(const industrial_robot_status_interface::RobotMode& robot_mode)
{
  if (robot_mode == industrial_robot_status_interface::RobotMode::MANUAL)
    return industrial_msgs::RobotMode::MANUAL;
  if (robot_mode == industrial_robot_status_interface::RobotMode::AUTO)
    return industrial_msgs::RobotMode::AUTO;
  if (robot_mode == industrial_robot_status_interface::RobotMode::UNKNOWN)
    return industrial_msgs::RobotMode::UNKNOWN;
}


void IndustrialRobotStatusController::update(const ros::Time& time, const ros::Duration& /*period*/)
{
  // obey configured publication rate
  if (!(publish_rate_ > 0.0 && last_publish_time_ + ros::Duration(1.0/publish_rate_) < time))
    return;

  if(robot_status_pub_.trylock())
  {
    // we're actually publishing, so increment time
    last_publish_time_ = last_publish_time_ + ros::Duration(1.0/publish_rate_);

    // TODO: should we use stamp from hw RobotStatus instead?
    robot_status_pub_.msg_.header.stamp = time;

    // publish msg based on last state from hw interface
    auto robot_status_hw = robot_status_handle_->getRobotStatus();

    // simple conversion operation
    robot_status_pub_.msg_.in_motion.val       = convert(robot_status_hw.in_motion);
    robot_status_pub_.msg_.motion_possible.val = convert(robot_status_hw.motion_possible);
    robot_status_pub_.msg_.drives_powered.val  = convert(robot_status_hw.drives_powered);
    robot_status_pub_.msg_.e_stopped.val       = convert(robot_status_hw.e_stopped);
    robot_status_pub_.msg_.in_error.val        = convert(robot_status_hw.in_error);
    robot_status_pub_.msg_.mode.val            = convert(robot_status_hw.mode);
    robot_status_pub_.msg_.error_code          = robot_status_hw.error_code;

    robot_status_pub_.unlockAndPublish();
  }
}


} // namespace industrial_robot_status_controller

PLUGINLIB_EXPORT_CLASS(industrial_robot_status_controller::IndustrialRobotStatusController, controller_interface::ControllerBase);
