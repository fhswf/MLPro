#!/usr/bin/env python3
# import geometry_msgs.msg
import rospy
import actionlib
from robotiq_2f_gripper_msgs.msg import CommandRobotiqGripperFeedback, CommandRobotiqGripperResult, CommandRobotiqGripperAction, CommandRobotiqGripperGoal
from robotiq_2f_gripper_control.robotiq_2f_gripper_driver import Robotiq2FingerGripperDriver as Robotiq
import control_msgs.msg
from std_msgs.msg import String
from gazebo_msgs.msg import ContactsState
from gazebo_ros_link_attacher.srv import Attach, AttachRequest
import math

rospy.init_node('gripper_command')
sim = rospy.get_param('~sim','false')
if sim:
    print("""
            
            THIS IS A SIMULATION 
            
            """)
else:
    print("""
            
            Using Real Hardware 
            
            """)
if sim:
    client = actionlib.SimpleActionClient('/gripper_controller/gripper_cmd', control_msgs.msg.GripperCommandAction) 
else:
    client = actionlib.SimpleActionClient('/gripper_controller/gripper_cmd', CommandRobotiqGripperAction)

client.wait_for_server()

attach_srv = rospy.ServiceProxy('/link_attacher_node/attach',Attach)
attach_srv.wait_for_service(timeout=5)
rospy.loginfo("Created ServiceProxy to /link_attacher_node/attach")
detach_srv = rospy.ServiceProxy('/link_attacher_node/detach',Attach)
detach_srv.wait_for_service(timeout=5)
rospy.loginfo("Created ServiceProxy to /link_attacher_node/detach")
rospy.loginfo("Gripper Contact Handler Ready")

class Comparator():
    def __init__(self):
        self.process=False
        self.r_contact_col_name=None
        self.r_contact_norm=None
        self.l_contact_col_name=None
        self.l_contact_norm=None
    def r_call(self,contact_array):
        if self.process==False:
            return
        for state in contact_array.states:
            self.r_contact_col_name=(state.collision1_name if state.collision1_name !=
                                     "robot::right_inner_finger::right_inner_finger_pad_collision_1"
                                     else state.collision2_name)
            for c_n in state.contact_normals:
                if (math.isclose(0,c_n.x,abs_tol=0.01) and 
                    math.isclose(1,c_n.y,rel_tol=0.1) and 
                    math.isclose(0,c_n.z,abs_tol=0.01)):
                    self.r_contact_norm=True
                    break
                else:
                    self.r_contact_norm=False
                    continue
            self.cancel_goal()
        return
                
    def l_call(self,contact_array):
        if self.process==False:
            return
        for state in contact_array.states:
            self.l_contact_col_name=(state.collision1_name if state.collision1_name !=
                                     "robot::left_inner_finger::left_inner_finger_pad_collision_1"
                                     else state.collision2_name)
            for c_n in state.contact_normals:
                if (math.isclose(0,c_n.x,abs_tol=0.01) and 
                    math.isclose(-1,c_n.y,rel_tol=0.1) and 
                    math.isclose(0,c_n.z,abs_tol=0.01)):
                    self.l_contact_norm=True
                    break
                else:
                    self.l_contact_norm=False
                    continue
            self.cancel_goal()
        return
    def cancel_goal(self):
        if self.r_contact_col_name==self.l_contact_col_name: #and self.r_contact_norm and self.l_contact_norm :
            self.process=False
            
            client.cancel_goal()
            #Robotiq.stop(client, block=False)
            
            #ATTACH LINK
            req = AttachRequest()
            req.model_name_1 = "robot"
            req.link_name_1 = "left_inner_finger"
            req.model_name_2 = self.l_contact_col_name.split("::")[0]
            req.link_name_2 = self.l_contact_col_name.split("::")[1]
            rospy.loginfo("Attaching {} and {}".format(req.model_name_2+"::"+req.link_name_2,"robot::left_inner_finger"))
            attach_srv.call(req)
        return
        	
def gripper_client(given):
    try:
        if given.data=="close":
            comparator.process=True
            rospy.loginfo('Closing Gripper')
            
            if sim:
                goal = control_msgs.msg.GripperCommandGoal()
                goal.command.position = 0.695  
                goal.command.max_effort = 5
                client.send_goal(goal)
            else:
                Robotiq.close(client, speed=0.05, force=5, block = False)
            
            if(client.wait_for_result(timeout=rospy.Duration(1))):
                return client.get_result()
            else:
                return ("stalled: True")
                
        if given.data=="open":
            comparator.process=False
            rospy.loginfo("Opening Gripper")
            if sim:
                #DETACH LINK
                try:
                    req = AttachRequest()
                    req.model_name_1 = "robot"
                    req.link_name_1 = "left_inner_finger"
                    req.model_name_2 = comparator.l_contact_col_name.split("::")[0]
                    req.link_name_2 = comparator.l_contact_col_name.split("::")[1]
                    rospy.loginfo("Detaching {} and {}".format(
                        req.model_name_2+"::"+req.link_name_2,
                        "robot::left_inner_finger"))
                    detach_srv.call(req)
                except:
                    rospy.loginfo("Link Not Detached")
                    
                comparator.r_contact_col_name=None
                comparator.r_contact_norm=None
                comparator.l_contact_col_name=None
                comparator.l_contact_norm=None
                
                
                goal = control_msgs.msg.GripperCommandGoal()
                goal.command.position = 0  
                goal.command.max_effort = 5
                client.send_goal(goal)
            else:
                Robotiq.open(client, speed=0.1, force=5, block = False)
                
            if(client.wait_for_result(timeout=rospy.Duration(1))):
                return client.get_result()
            else:
                return ("stalled: True")
    except rospy.ROSInterruptException:
        print ("Program interrupted before completion")
        
comparator= Comparator()
if sim:
    right_sub = rospy.Subscriber('/right_inner_finger_pad_bumper/', ContactsState, comparator.r_call)
    left_sub = rospy.Subscriber('/left_inner_finger_pad_bumper/', ContactsState, comparator.l_call)

gripper_sub = rospy.Subscriber('/gripper_controller/command', String, gripper_client)

rospy.loginfo('gripper action server initialized')
rospy.spin()
