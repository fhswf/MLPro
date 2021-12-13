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

#ifndef INDUSTRIAL_ROBOT_STATUS_CONTROLLER_INDUSTRIAL_ROBOT_STATUS_CONTROLLER_H_
#define INDUSTRIAL_ROBOT_STATUS_CONTROLLER_INDUSTRIAL_ROBOT_STATUS_CONTROLLER_H_

#include <industrial_robot_status_interface/industrial_robot_status_interface.h>

#include <controller_interface/controller.h>
#include <industrial_msgs/RobotStatus.h>
#include <realtime_tools/realtime_publisher.h>

#include <memory>


namespace industrial_robot_status_controller
{


using InterfaceType = industrial_robot_status_interface::IndustrialRobotStatusInterface;
using HandleType = industrial_robot_status_interface::IndustrialRobotStatusHandle;


class IndustrialRobotStatusController
    : public controller_interface::Controller<InterfaceType>
{
 public:
  IndustrialRobotStatusController() = default;

  virtual ~IndustrialRobotStatusController() override {}

  bool init(
    InterfaceType* hw,
    ros::NodeHandle& root_node_handle,
    ros::NodeHandle& controller_node_handle) override;
  void starting(const ros::Time& time) override;
  void update(const ros::Time& time, const ros::Duration& period) override;

 private:
  InterfaceType* robot_status_interface_{};
  std::unique_ptr<HandleType> robot_status_handle_{};

  realtime_tools::RealtimePublisher<industrial_msgs::RobotStatus> robot_status_pub_;
  ros::Time last_publish_time_;
  double publish_rate_;
};


} // namespace industrial_robot_status_controller

#endif
