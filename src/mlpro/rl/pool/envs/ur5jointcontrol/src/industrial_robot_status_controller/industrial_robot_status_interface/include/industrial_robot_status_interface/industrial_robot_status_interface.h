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

#ifndef INDUSTRIAL_ROBOT_STATUS_INTERFACE_INDUSTRIAL_ROBOT_STATUS_INTERFACE_H_
#define INDUSTRIAL_ROBOT_STATUS_INTERFACE_INDUSTRIAL_ROBOT_STATUS_INTERFACE_H_

#include <hardware_interface/internal/hardware_resource_manager.h>

#include <string>


namespace industrial_robot_status_interface
{

// following enums mirror ROS-Industrial's simple_message enums
// TODO: should we just re-use those?

enum class TriState : int8_t
{
  UNKNOWN = -1,
  FALSE   =  0,
  TRUE    =  1,
};

enum class RobotMode : int8_t
{
  UNKNOWN = -1,
  MANUAL  =  1,
  AUTO    =  2,
};

typedef std::int32_t ErrorCode;

struct RobotStatus
{
  RobotMode mode;
  TriState  e_stopped;
  TriState  drives_powered;
  TriState  motion_possible;
  TriState  in_motion;
  TriState  in_error;
  ErrorCode error_code;
};


class IndustrialRobotStatusHandle
{
 public:
  IndustrialRobotStatusHandle() = delete;

  IndustrialRobotStatusHandle(const std::string& name, RobotStatus& robot_status)
      : name_(name), robot_status_(&robot_status) {}

  const std::string& getName() const noexcept { return name_; }

  const RobotStatus& getRobotStatus() const noexcept { return *robot_status_; }

 private:
  std::string name_;
  const RobotStatus* robot_status_;
};


class IndustrialRobotStatusInterface : public hardware_interface::HardwareResourceManager<IndustrialRobotStatusHandle> {};


}  // namespace industrial_robot_status_interface


#endif // INDUSTRIAL_ROBOT_STATUS_INTERFACE_INDUSTRIAL_ROBOT_STATUS_INTERFACE_H_
