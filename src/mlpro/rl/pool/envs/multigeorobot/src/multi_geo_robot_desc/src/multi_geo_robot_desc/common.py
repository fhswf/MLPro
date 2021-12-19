#!/usr/bin/env python
# Templating for Arm Robot
import em
import rospkg
import os
import math
import yaml
import copy


def init_joint():
    joint = {
            'name' : '',
            'type' : '',
            'parent' : '',
            'child' : '',
            'limit' : 0,
            'ox' : 0,
            'oy' : 0,
            'oz' : 0,
            'rr' : 0,
            'rp' : 0,
            'ry' : 0,
            'ax' : 0,
            'ay' : 0,
            'az' : 0
        }
    return joint

def armrobot_creator(robotType, manyArm, arm_lengths, arm_masses, arm_radiuses, adapter_masses, armJoints, tooltipJoints, gripperOn, dummy):

    rospack = rospkg.RosPack()
    world_dir = os.path.join(rospack.get_path('multi_geo_robot_desc'), 'models')
    urdf_dir = os.path.join(rospack.get_path('multi_geo_robot_desc'), 'urdf')

    template_file = os.path.join(world_dir, 'robot_creator.xacro.template')

    links = []
    jointsR = []

    # Template Data
    # Base Model
    base_mass = 1000
    base_inertia_tensor_ixx = 0.000012/0.002494
    base_inertia_tensor_iyy = 0.000012/0.002494
    base_inertia_tensor_izz = 0.000024/0.002494
    base_model = {
        'name' : "base_model",
        'mass' : base_mass,
        'cgx' : 0.000002,
        'cgy' : -0,
        'cgz' : 0.019926,
        'ixx' : base_inertia_tensor_ixx*base_mass,
        'iyy' : base_inertia_tensor_iyy*base_mass,
        'izz' : base_inertia_tensor_izz*base_mass
    }

    links.append(base_model['name'])

    # Arm
    arms = []
    for i in range(manyArm):
        adapters = []
        adapter_num = 2
        if robotType == "2D":
            if i == 0:
                adapter_num = 1
        for j in range(adapter_num):
            adapter_name = "adapter_" + str(i+1) + "_" + str(j+1)
            adapter_mass = adapter_masses
            adapter_inertia_tensor_ixx = (0.000002/0.001157)
            adapter_inertia_tensor_iyy = (0.000002/0.001157)
            adapter_inertia_tensor_izz = (0.000002/0.001157)
            adapter = {
                'name' : adapter_name,
                'mass' : adapter_mass,
                'cgx' : 0,
                'cgy' : -0.003725,
                'cgz' : 0.003727,
                'ixx' : adapter_inertia_tensor_ixx*adapter_mass,
                'iyy' : adapter_inertia_tensor_iyy*adapter_mass,
                'izz' : adapter_inertia_tensor_izz*adapter_mass
            }
            adapters.append(adapter)
            links.append(adapter['name'])

        arm_name = "arm_" + str(i+1)
        arm_mass = arm_masses[i]
        arm_length = arm_lengths[i]
        arm_radius = arm_radiuses
        arm_inertia_tensor_ixx = ((1.0/12.0) * arm_mass * ((3.0 * arm_radius * arm_radius) + (arm_length * arm_length)))
        arm_inertia_tensor_iyy = ((1.0/12.0) * arm_mass * ((3.0 * arm_radius * arm_radius) + (arm_length * arm_length)))
        arm_inertia_tensor_izz = ((1.0/12.0) * arm_mass * (arm_radius * arm_radius))
        arm = {
            'name' : arm_name,
            'adapters' : adapters,
            'mass' : arm_mass,
            'length' : arm_length,
            'radius' : arm_radius,
            'cgx' : 0,
            'cgy' : 0,
            'cgz' : 0,
            'ixx' : arm_inertia_tensor_ixx,
            'iyy' : arm_inertia_tensor_iyy,
            'izz' : arm_inertia_tensor_izz
            }

        arms.append(arm)
        links.append(arm['name'])

    tooltip_name = "tooltip"
    tooltip_mass = 10
    tooltip_length = 0.01
    tooltip_radius = arm_radiuses
    tooltip_inertia_tensor_ixx = ((1.0/12.0) * tooltip_mass * ((3.0 * tooltip_radius * tooltip_radius) + (tooltip_length * tooltip_length)))
    tooltip_inertia_tensor_iyy = ((1.0/12.0) * tooltip_mass * ((3.0 * tooltip_radius * tooltip_radius) + (tooltip_length * tooltip_length)))
    tooltip_inertia_tensor_izz = ((1.0/12.0) * tooltip_mass * (tooltip_radius * tooltip_radius))

    tooltip = {
        'name' : tooltip_name,
        'mass' : tooltip_mass,
        'length' : tooltip_length,
        'radius' : tooltip_radius,
        'cgx' : 0,
        'cgy' : 0,
        'cgz' : 0,
        'ixx' : tooltip_inertia_tensor_ixx,
        'iyy' : tooltip_inertia_tensor_iyy,
        'izz' : tooltip_inertia_tensor_izz
    }
    links.append(tooltip['name'])

    # Joints
    joints = []
    parent = base_model['name']

    # Joint Arm
    for i in range(manyArm):
        adapter_num = 2
        if robotType == "2D":
            if i == 0:
                adapter_num = 1
        # Adapter
        for j in range(adapter_num):
            joint = init_joint()
            joint['parent'] = parent
            joint['child'] = arms[i]['adapters'][j]['name']
            joint['name'] = joint['parent'] + "_" + joint['child']
            joint['type'] = 'fixed'
            if armJoints[i][j]:
                joint['type'] = 'revolute'
                jointsR.append(joint['name'])
            else:
                joint['type'] = 'fixed'
                joint['limit'] = 1

            if "base_model" in parent:
                joint['az'] = 1
                joint['oz'] = 0.12
                joint['rp'] = math.pi
                if robotType == "2D":
                    joint['ry'] = math.pi/2
            elif "adapter" in parent:
                joint['rr'] = math.pi
                if robotType == "2D":
                    if i == 0:
                        joint['rp'] = math.pi/2
                if (i+1)%2 == 0:
                    joint['az'] = 1
                    joint['oz'] = 0.12
                else:
                    joint['ay'] = 1
                    joint['oy'] = -0.12
            elif "arm" in parent:
                joint['oz'] = (0.06+(arms[i-1]['length']/2))
                joint['rr'] = math.pi
                if robotType == "2D":
                    if i == 1:
                        joint['ry'] = math.pi/2
                if (i)%2 == 0:
                    joint['rr'] = math.pi
                    joint['az'] = 1
                else:
                    joint['ay'] = 1
                    joint['rr'] = math.pi/2
                        

            parent = joint['child']
            joints.append(joint)

        joint = init_joint()
        joint['parent'] = parent
        joint['child'] = arms[i]['name']
        joint['name'] = joint['parent'] + "_" + joint['child']
        joint['type'] = 'fixed'
        if armJoints[i][-1]:
            joint['type'] = 'revolute'
            jointsR.append(joint['name'])
        else:
            joint['type'] = 'fixed'
            joint['limit'] = 1
        
        joint['az'] = 1

        if (i+1)%2 == 0:
            joint['rr'] = math.pi/2
            joint['oy'] = -(0.06+(arms[i]['length']/2))
        else:
            joint['oz'] = (0.06+(arms[i]['length']/2))
            if robotType == "2D":
                if i == 0:
                    joint['rr'] = math.pi/2
                    joint['rp'] = math.pi/2
                    joint['oy'] = -(0.06+(arms[i]['length']/2))
                    joint['oz'] = 0

        parent = joint['child']
        joints.append(joint)


    # Joint Tooltip
    joint = init_joint()
    joint['parent'] = arms[-1]['name']
    joint['child'] = tooltip['name']
    joint['name'] = joint['parent'] + "_" + joint['child']
    joint['type'] = 'fixed'
    if tooltipJoints:
        joint['type'] = 'revolute'
        jointsR.append(joint['name'])

    joint['az'] = 1
    joint['oz'] = (0.005+(arms[-1]['length']/2))
    joints.append(joint)

    dummies = {}
    dummies['active'] = False
    # EEF Dummy
    dummies['name'] = "eef_dummy"
    dummies['active'] = dummy
    links.append(dummies['name'])

    gripper = {}
    gripper["active"] = False
    if gripperOn:
        gripper["active"] = True

    template_data = {
        'base_model' : base_model,
        'arms' : arms,
        'dummy' : dummies,
        'tooltip' : tooltip,
        'joints' : joints,
        'gripper' : gripper
    }

    files = {}

    # Templating
    data = None
    with open(template_file, 'r') as f:
        data = f.read()
    files['robot.urdf.xacro'] = em.expand(data, template_data)

    # Saving the tamplate
    for name, content in files.items():
        if name.endswith('.template'):
            name = name[:-len('.template')]

        name = os.path.basename(name)
        file_path = os.path.join(world_dir, name)
        with open(file_path, 'w+') as f:
            f.write(content)


    # Convert Xacro to URDF
    xacrofile = os.path.join(world_dir, 'robot_environment.xacro')
    urdffile = os.path.join(urdf_dir, 'model.urdf')
    os.system('rosrun xacro xacro --inorder {} > {}'.format(xacrofile,urdffile))
    
    print("Robot Configuration Generated")
    return links, jointsR

def armrobot_controller_creator(joints,gripperOn):

    rospack = rospkg.RosPack()
    config_dir = os.path.join(rospack.get_path('multi_geo_robot_desc'), 'config')
    config_file = os.path.join(config_dir, 'robotConfig.yaml')
    moveit_file = os.path.join(config_dir, 'controllers.yaml')
    template_srdf_file = os.path.join(config_dir, 'multi_geo_robot.srdf.template')

    # Arm Controller
    main = {}    

    constraints_arm = {}
    constraints_arm["goal_time"] = 0.6
    constraints_arm["stopped_velocity_tolerance"] = 0.05

    for joint_name in joints:
     constraints_arm[joint_name] = {"trajectory" : 0.1, "goal" : 0.1}
    
    arm_controller = {}
    arm_controller["type"] = "position_controllers/JointTrajectoryController"
    arm_controller["joints"] = copy.deepcopy(joints)
    arm_controller["constraints"] = constraints_arm
    arm_controller["state_publish_rate"] = 125
    arm_controller["action_monitor_rate"] = 10
    arm_controller["stop_trajectory_duration"] = 0.5

    joint_group = {}
    joint_group["type"] = "position_controllers/JointGroupPositionController"
    joint_group["joints"] = copy.deepcopy(joints)

    main["arm_controller"] = arm_controller
    main["joint_group_position_controller"] = joint_group

    # Gripper Controller
    if gripperOn:
        gripper_controller = {}
        gripper_controller["type"] = "position_controllers/GripperActionController"
        gripper_controller["joint"] = "finger_joint"
        gripper_controller["action_monitor_rate"] = 20
        gripper_controller["goal_tolerance"] = 0.01
        gripper_controller["max_effort"] = 5
        gripper_controller["stall_velocity_threshold"] = 0.01
        gripper_controller["stall_timeout"] = 0.5

        main["gripper_controller"] = gripper_controller

    with open(config_file, 'w') as file:
        documents = yaml.dump(main, file)

    # Moveit Controller
    moveit_controller = {}
    controller_list = []

    controller_item = {}
    controller_item["name"] = "arm_controller"
    controller_item["action_ns"] = "follow_joint_trajectory"
    controller_item["type"] = "FollowJointTrajectory"
    controller_item["joints"] = copy.deepcopy(joints)

    controller_list.append(controller_item)

    moveit_controller["controller_list"] = controller_list

    with open(moveit_file, 'w') as file:
        documents = yaml.dump(moveit_controller, file)

    # SRDF Configuration

    template_data = {}

    # Templating
    files = {}
    data = None
    with open(template_srdf_file, 'r') as f:
        data = f.read()
    files['multi_geo_robot.srdf'] = em.expand(data, template_data)

    # Saving the tamplate
    for name, content in files.items():
        if name.endswith('.template'):
            name = name[:-len('.template')]

        name = os.path.basename(name)
        file_path = os.path.join(config_dir, name)
        with open(file_path, 'w+') as f:
            f.write(content)
    