#!/usr/bin/env python
import sys
sys.path.append('/home/siml/catkin_ws/src/side_walking/src')
import poly_traj_generation
#from poly_traj_generation import PolyTrajGeneration
    #from poly_traj_generation_y import PolyTrajGenerationY
    #from poly_traj_generation_2intents import PolyTrajGenerationTwoIntents
from nodes import Nodes
from check import Check
from pomdp_follower import PomdpFollower
from full_knowledge_collaborative_planner import FullKnowledgeCollaborativePlanner

import math as m
import numpy as np
import numpy.linalg as LA
import time
import matplotlib.pyplot as plt

import rospy
import std_msgs.msg as std_msg
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
import geometry_msgs.msg as geom_msg
from geometry_msgs.msg import PoseStamped, Twist
import visualization_msgs.msg as vis_msg
from gazebo_msgs.srv import GetModelState, GetModelStateRequest


from leg_tracker.msg import Person
from leg_tracker.msg import PersonArray
#import gazebo_msgs.msg  as gazebo_msg
#from utils import linear_interp_traj
#from holonomic_controller import HolonomicController

#argv: python ped_crossing.py arg1 arg2 arg3 arg4 arg5 arg6
#arg1: helmet id for ped, arg2: chair id for ped_goal arg3: robot goal chair id
#arg4: timing_sm, arg5: safety_sm, arg6: max_accelx (0.4), arg7: max_accely (0.4)
class SideWalking :
    def __init__(self,sim_mode=True,planner='Full',prior=[1,0]):
        self.sim = sim_mode
        self.planner = planner # Full, Pomdp
        # sim_mode parameter initialization
        self.mapID = 0
        self.subgoal_ind = 1 # to be set, input by arg

    	# general parameters
        self.x_ref = []
        self.v = 1.0 #desired group velocity
        self.sm = 1.0 #safety margin
        self.dt = 0.1 
        self.weight = [1,1]
        self.H = 3
        #    self.ped_helmet_id = int(sys.argv[1])
    	#    self.ped_goal_chair_id = int(sys.argv[2])
    	#    self.rob_goal_chair_id = int(sys.argv[3])

    	#    # ped/robot goal information
    	#    self.ped_goal = np.zeros((2,1))
        #self.rob_goal = np.zeros((2,1)
        self.vh = self.v #defult, to be estimate
      	self.vr = self.v #nominal speed
      	self.T = 17.0
    	self.human_replan_rate = 1
    	self.robot_replan_rate = 1

    	self.robot_inside = 1
    	self.ww = 0.0
    	self.wc = 1.0
    	self.wd = [0,0]
    	self.wd[0] = 1.0
    	self.wd[1] = 1.0
    	self.theta = prior
        self.view_thr = -1
        self.prob_thr = 0.1
        #self.subgoals = np.full((2,2),np.Inf)

        #subgoals
        self.p1_goal = [np.Inf,np.Inf]
        self.p1_goal.append([np.Inf,np.Inf])
        self.p2_goal = [np.Inf,np.Inf]
        self.p2_goal.append([np.Inf,np.Inf])
        self.costmap = []
        self.costmap0s = [0,0]
        self.costmap_res = 0.025

        if self.sim :
            inters_timing = 8.5
            subg_inters = inters_timing*self.ped_vel + self.ped_pos
            self.subgoals = np.zeros((2,2))
            self.subgoals[0][0] = -1.0*self.v*(self.T-inters_timing)
            self.subgoals[1][1] = 1.0*self.v*(self.T-inters_timing)
            self.subgoals[0][:] = self.subgoals[0][:] + subg_inters[0]
            self.subgoals[1][:] = self.subgoals[1][:] + subg_inters[1]

            if self.mapID==1:
                self.subgoals[0][1] = -self.subgoals[0][0]
                self.subgoals[1][1] = self.subgoals[1][0]


      	#self.ped_pos = np.zeros((2,1))
      	self.ped_pos = [np.Inf for i in range(2)]
      	self.rob_pos = [np.Inf for i in range(2)]
        self.ped_rel_pos = [0,0]
        self.ped_id = np.Inf
        self.ped_vel = [0.0 for i in range(2)]
        self.rob_vel = [0.0 for i in range(2)]
        self.rob_yaw = 0.0
        self.R = np.zeros((2,2))
        self.R[0][0] = 1.0
        self.R[1][1] = 1.0
        #self.ped_pos_old = np.zeros((2,1))
    	#self.rob_pos_old = np.zeros((2,1))

        #self.rob_intent = 0 # 0: pass first
        #self.ped_intent = 1 # 1: yield

        self.rob_reaction_time = 4.5


    	#self.max_accelx = 0.0
        #self.max_accely = 0.1
        # self.max_accelx = sys.argv[5]
        # self.max_accely = sys.argv[6]
      	self.recovery_gain = 4.0
      	self.local_traj_duration = 4.0

        self.Xtraj = []
        self.Ytraj = []
        self.Xvel = []
        self.Yvel = []
        self.x_pos_ref = 0.0
        self.y_pos_ref = 0.0
        self.x_vel_ref = 0.0
        self.y_vel_ref = 0.0

      	self.angular_vel_z = 0.0

      	self.calibrated = False
      	self.start = False
        #helmets and chairs:
    	#if self.sim:
        #        rospy.wait_for_service('/gazebo/get_model_state')

      	self.num_obs = 3
      	#obs_time
      	self.obs_time = 0.0
      	self.obs_time_sec = 0
      	self.obs_time_nsec = 0
        self.obs_pose = np.full((self.num_obs, 2), np.inf)
        self.obs_pre_pose = np.full((self.num_obs, 2), np.inf)
        self.obs_vel = np.full((self.num_obs, 2), np.inf)
        self.obs_pose_update_time = [np.inf, np.inf, np.inf, np.inf, np.inf]
        self.corner_pose = np.full((4,2), np.inf)

        self.velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        #self.cmd_state_pub = rospy.Publisher('/cmd_state', Twist, queue_size=1)
        #self.marker_pub = rospy.Publisher('ped_pos', vis_msg.Marker, latch=False, queue_size=1)
        #self.line_seg_pub = rospy.Publisher('path', vis_msg.Marker, latch=False, queue_size=1)
        #self.robot_marker_pub = rospy.Publisher('robot', vis_msg.Marker, latch=False, queue_size=1)
        #self.goal_marker_pub = rospy.Publisher('goal', vis_msg.Marker, latch=False, queue_size=1)

        self.vel_msg = Twist()
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)
        self.ped_sub = rospy.Subscriber('/people_tracked',PersonArray,self.ped_callback,queue_size=1)
        self.goal_pose_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_pose_callback, queue_size=1)
        #self.costmap_sub = rospy.Subscriber('/move_base/local_costmap/costmap',OccupancyGrid, self.costmap_callback, queue_size=1)
        self.costmap_sub = rospy.Subscriber('/local_map',OccupancyGrid, self.costmap_callback, queue_size=1)
        self.run()


        # Callbacks
    def costmap_callback(self,map_msg) :
        if not np.isnan(map_msg.data[0]) :
            self.costmap = np.copy(map_msg.data)
            self.costmap0s[0] = map_msg.info.origin.position.x-map_msg.info.width/2.0*map_msg.info.resolution
            self.costmap0s[1] = map_msg.info.origin.position.y-map_msg.info.height/2.0*map_msg.info.resolution
            self.costmap_res = map_msg.info.resolution


    def odom_callback(self,odom_msg) :
        if not np.isnan(odom_msg.pose.pose.position.x):
            self.rob_pos[0] = odom_msg.pose.pose.position.x
            self.rob_pos[1] = odom_msg.pose.pose.position.y
            if odom_msg.twist.twist.linear.x < 1.2 and odom_msg.twist.twist.linear.y < 1.2 :
                self.rob_vel[0] = 0.3*self.rob_vel[0] + 0.7*odom_msg.twist.twist.linear.x
                self.rob_vel[1] = 0.3*self.rob_vel[1] + 0.7*odom_msg.twist.twist.linear.y
            q = odom_msg.pose.pose.orientation
            self.rob_yaw = m.atan2(2.0*(q.x*q.y + q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)



    def ped_callback(self,person_msg) :
        people = person_msg.people
        #print 'ped callback reached id = {}'.format(len(people))
        ped_found = False 
        supp_ped_found = False
        supp_ped_pos = [0,0]
        supp_ped_id = np.Inf

        for i in range(len(people)):
            if len(people)==0 :
                ped_found = True
                continue

            person = people[i]
            pos = []
            pos.append(person.pose.position.x)
            dist = np.sqrt(pos[0]**2+person.pose.position.y**2)
            if not np.isnan(person.pose.position.x) and len(pos)>0 and not ped_found and dist <2.5:
                if self.ped_id == np.Inf:
                    self.ped_id = person.id
                    self.ped_rel_pos[0] = person.pose.position.x
                    self.ped_rel_pos[1] = person.pose.position.y
                    self.ped_vel = [0,0]
                    ped_found = True
                    print 'pedestrian initialized'
                    break
                
                ped_pos_new = [person.pose.position.x, person.pose.position.y]    
                if self.ped_id == person.id :
                    self.ped_rel_pos[0] = ped_pos_new[0]
                    self.ped_rel_pos[1] = ped_pos_new[1]
                    #print 'pedestrian pose={}'.format(ped_pos_new)
                    ped_found = True
                    break
                elif not supp_ped_found :
                    supp_ped_pos[0] = ped_pos_new[0]
                    supp_ped_pos[1] = ped_pos_new[1]
                    supp_ped_id = person.id
                    supp_ped_index = i
                    supp_ped_found = True

        if not ped_found and supp_ped_found:
            print 'pedestrian target updated'
            self.ped_id = supp_ped_id
            self.ped_rel_pos = supp_ped_pos
            self.ped_vel = [0,0]


    def goal_pose_callback(self, goal_pose_msg):
        self.goal_x = goal_pose_msg.pose.position.x
        self.goal_y = goal_pose_msg.pose.position.y
        q = goal_pose_msg.pose.orientation
            #self.goal_yaw = math.atan2(dd2.0*(q.x*q.y + q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)


        # Main Function
    def run(self):
        collision_thr = 50
        human_path_found = 0
        robot_path_found = 0
        rob_traj_duration = 1.0/self.robot_replan_rate
        ped_traj_duration = 1.0/self.human_replan_rate
        ped_goal = np.full((2,1),np.Inf)
        rob_goal = np.full((2,1),np.Inf)
        #initialization: poses, goals
        #set p1_goal, robot state, ped state TODO
        self.rob_vel[0] = 0
        self.rob_vel[1] = 0
        #self.ped_pos[1] = np.copy(self.obs_pose[self.ped_helmet_id,1])
        #self.ped_pos[0] = np.copy(self.obs_pose[self.ped_helmet_id,0])

        start_yaw = 0.0
        #initialization: system level
        cmd_pub_rate = 100
        traj_pub_rate = 1/self.dt
        rate = rospy.Rate(traj_pub_rate)
        vel_msg_ = Twist()

        plot_count = 0
        vel_count = 0
        vel_n = 5
        vel_est = np.zeros((2,vel_n))
        goal_thr = 2.5
        initial_time = time.time()
        program_start_time = time.time()
        robot_time_del_plan = rob_traj_duration

        #initialization: pedestrian velcity estimate, robot fixed command
        vR_ = np.zeros(2)
        vR_baseline = np.zeros(2)
        vR_accel = np.zeros(2)
        vh_filtered = 0.0
        vh_filtered_old = 0.0
        vh_fixed = 0
        vh_fixed_count = 0
        vh_fixed_value = 1.0
        tbc_ped = np.inf
        vR_fixed = np.zeros(2)
        convergence_count = 0
        vel_thr = 0.2
        alpha = 0.2


        


        while (not rospy.is_shutdown()):
            current_time = time.time()
            robot_time_del_plan = current_time - initial_time
            self.ped_pos[0] = self.ped_rel_pos[0]+self.rob_pos[0]
            self.ped_pos[1] = self.ped_rel_pos[1]+self.rob_pos[1]
            if not self.start :# check ped pose, robot pose, goals
            	criteria0 = (self.rob_pos[1]!=np.Inf and self.rob_pos[0]!=np.Inf)
            	criteria1 = (self.ped_rel_pos[1]!=0 and self.ped_rel_pos[0]!=0)
                #if criteria1 :
                    #print 'criteria1 fixed'
                    #print 'ped_pos = {}'.format(self.ped_pos)

                #if criteria0 :
                    #print 'criteria0 fixed'
                    #print 'rob_pos = {}'.format(self.rob_pos)

            	if criteria0 and criteria1 :
                    #initialization
                    print 'criteria1 and 2 fixed'
                    initial_time = current_time
                    #robot_time_del_plan = rob_traj_duration
                    self.x_vel_ref = 0
                    self.y_vel_ref = 0
                    self.vel_msg.linear.x = 0.0
                    self.vel_msg.linear.y = 0.0
                    initial_rob_pos = self.rob_pos
                    initial_ped_pos = [0,0]
                    initial_ped_pos[0] = self.ped_rel_pos[0]+initial_rob_pos[0]
                    initial_ped_pos[1] = self.ped_rel_pos[1]+initial_rob_pos[1]
                    x1 = initial_ped_pos
                    x2 = initial_rob_pos
                    start_yaw = self.rob_yaw
                    print 'initial_rob x = {}'.format(initial_rob_pos[0])
                    print 'initial_ped x = {}'.format(initial_ped_pos[0])
                    print 'initial_rob y = {}'.format(initial_rob_pos[1])
                    print 'initial_ped y = {}'.format(initial_ped_pos[1])
                    

                    self.p1_goal[0] = [initial_ped_pos[0]+10,initial_ped_pos[1]+2]
                    self.p1_goal[1] = [initial_ped_pos[0]+5, initial_ped_pos[1]-5]
                    self.p2_goal[0] = [initial_ped_pos[0]+10,initial_ped_pos[1]+2]
                    self.p2_goal[1] = [initial_ped_pos[0]+5, initial_ped_pos[1]-7]


                    p1_goal = self.p1_goal[1]
                    p2_goal = self.p2_goal[1]
                    raw_input('Environment Ready, Press Any Key to Start')
                    self.start = True
                else :
                    continue
            else:


                #ped_goal = self.subgoal[:][subgoal_ind]
                #initialization: sanity check on pedestrian and robot readings to current_xx_po
                #if self.sim :
                ## gazebo
                #    model_state_result = self.get_model_srv(self.request)

                if  robot_time_del_plan > (rob_traj_duration - self.dt):
                    #replan
                    new_ped_vel = [(self.ped_pos[0]-x1[0])/robot_time_del_plan,(self.ped_pos[1]-x1[1])/robot_time_del_plan]
                    self.ped_vel[0] = (1-alpha)*self.ped_vel[0]+alpha*new_ped_vel[0]
                    self.ped_vel[1] = (1-alpha)*self.ped_vel[1]+alpha*new_ped_vel[1]
                    self.v = np.sqrt(self.ped_vel[0]**2+self.ped_vel[1]**2)
                    self.v = min(self.v,1.0)
                    rel_dist = np.sqrt(self.ped_rel_pos[0]**2+self.ped_rel_pos[1]**2)
                    if rel_dist < 2.5:
                        new_sm = rel_dist
                        self.sm = (1-alpha)*self.sm+alpha*new_sm
                    x1_ = self.ped_pos+self.ped_vel
                    x2_ = self.rob_pos+self.rob_vel
                            

                    if np.sqrt(self.ped_vel[0]**2+self.ped_vel[1]**2) > vel_thr:
                        print 'dist = {}'.format(rel_dist)
                        print 'sm = {}'.format(self.sm)
                        #print 'rob_pos = {}'.format(self.rob_pos)
                        #print 'ped_pos = {}'.format(self.ped_pos)
                        #print 'node initial state ped, rob= {},{}'.format(x1_,x2_)
                        if self.planner=='Full' :
                            agentID=1
                            robot_path_found, xtraj, ytraj, pathx_, pathy_, rob_s_angz, path_fScore, current = FullKnowledgeCollaborativePlanner(x1_,x2_,self.dt,rob_traj_duration,self.weight,agentID,self.H,p1_goal,p2_goal,self.sm,self.v,self.ww,self.costmap,self.costmap0s,collision_thr,self.costmap_res)
                            self.Xtraj = np.copy(pathx_)
                            self.Ytraj = np.copy(pathy_)
                            human_trajx = xtraj
                            human_trajy = ytraj
                            
                        elif self.planner== 'Pomdp' :
                            agentID=2
                            print 'pomdp called'
                            stime = time.time()
                            robot_path_found, xtraj, ytraj, pathx_, pathy_, rob_s_angz, path_fScore, current = PomdpFollower(x1_,x2_,self.dt,rob_traj_duration,self.weight,agentID,self.H,self.theta,self.view_thr,self.prob_thr,self.p1_goal,self.p2_goal,self.sm,self.v,self.ww,self.wc,self.wd,self.costmap,self.costmap0s,collision_thr,self.costmap_res)
                            time_del = time.time()-stime
                            print 'pomdp computation time = {}'.format(time_del)
                            self.Xtraj = xtraj
                            self.Ytraj = ytraj
                            human_trajx = pathx_
                            human_trajy = pathy_
                            #if robot_path_found:
                        
                            #else :
                            #    print 'pomdp plan not found'

                        if robot_path_found :
                            x1 = x1_
                            x2 = x2_
                            robot_time_del_plan = 0
                            initial_time = current_time
                            terminate_index = int(np.floor(rob_traj_duration/self.dt))
                            #print 'traj generated = {}'.format(pathx_)

                traj_index = int(np.ceil(robot_time_del_plan/self.dt))
                if len(self.Xtraj)>traj_index:
                    self.x_pos_ref = np.copy(self.Xtraj[traj_index])
                    self.y_pos_ref = np.copy(self.Ytraj[traj_index])
                    self.x_vel_ref = (self.x_pos_ref - self.Xtraj[traj_index-1])/self.dt
                    self.y_vel_ref = (self.y_pos_ref - self.Xtraj[traj_index-1])/self.dt
                    
                else:
                    print 'INDEX exceeds trajectory output, x_vel_ref = {}'.format(self.x_vel_ref)
                    self.x_pos_ref = self.rob_pos[0]+self.x_vel_ref*self.dt
                    self.y_pos_ref = self.rob_pos[1]+self.y_vel_ref*self.dt


                
        
                ss_error = 1.2
                self.vel_msg.linear.x = ss_error*(self.x_vel_ref*m.cos(self.rob_yaw) + self.y_vel_ref*m.sin(self.rob_yaw))
                self.vel_msg.linear.y = ss_error*(-self.x_vel_ref*m.sin(self.rob_yaw) + self.y_vel_ref*m.cos(self.rob_yaw))
                omega = 1.0
                damping = 1.0
                v_ref_ = np.sqrt(self.x_vel_ref**2+self.y_vel_ref**2)
                if v_ref_ > 0.01 :
                    v_ref = [self.x_vel_ref/v_ref_,self.y_vel_ref/v_ref_]
                    yaw_ref = np.sign(np.cross(v_ref,[1,0]))*np.arccos(np.dot([1,0],v_ref))
                    self.vel_msg.angular.z = omega*(yaw_ref-self.rob_yaw) + 2*damping*omega*(0-self.angular_vel_z)
                
                collision = Check([self.x_pos_ref],[self.y_pos_ref],self.costmap,self.costmap0s,collision_thr,self.costmap_res) #TODO, check collision from map
                if collision:
                    self.vel_msg.linear.x = 0.0
                    self.vel_msg.linear.y = 0.0

                self.velocity_pub.publish(self.vel_msg)


    	    # Plotting starts here (actual code ends here)
            plt.cla()
            plt.axis('equal')
            x_min = initial_rob_pos[0]-10
            x_max = initial_rob_pos[0]+15
            y_min = initial_rob_pos[1]-15
            y_max = initial_rob_pos[1]+10
            plt.xlim((x_min, x_max))
            plt.ylim((y_min, y_max))
            plt.grid(True)
            plt.autoscale(False)
            plt.plot(self.ped_pos[0],self.ped_pos[1],'or')
            plt.plot(self.rob_pos[0],self.rob_pos[1],'og')
            plt.plot(self.p1_goal[0][0],self.p1_goal[0][1],'ok')
            plt.plot(self.p1_goal[1][0],self.p1_goal[1][1],'ok')

            if len(self.costmap)>0 :
                obstacle = [l for l in range(0,len(self.costmap),10) if self.costmap[l]>collision_thr]
                #print 'costmap = {}'.format(self.costmap[0])
                width = int(np.sqrt(len(self.costmap)))
                my = [int(np.floor(l/width))*self.costmap_res+self.costmap0s[1] for l in obstacle]
                mx = [np.mod(obstacle[k],width)*self.costmap_res+self.costmap0s[0] for k in range(len(obstacle))]
                #print 'resolution = {}'.format(self.costmap_res)
                #print 'obstacle, mx, my = {},{},{}'.format(obstacle[0],mx[0],my[0])
            if len(self.Xtraj)>0 :
                plt.plot(self.Xtraj,self.Ytraj,'*g')
               # print 'length of path = {}'.format(len(self.Xtraj))
                plt.plot(human_trajx,human_trajy,'r')
                if len(self.costmap)>0 :
                    plt.plot(mx,my,'ok')

            plt.pause(0.1)            
            #plt.plot(cn_node.human_trajy,cn_node.human_trajy,'*g')
    	    #plt.plot(self.p1_goal[0],self.p1_goal[1],'ok')
    	    #plt.plot(self.p2_goal[0],self.p2_goal[1],'om')
            #print 'self.Xtraj = {}'.format(self.Xtraj)       

                            

              

            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('side_walking_node')
    #arg1: helmet id for ped, arg2: chair id for ped_goal
    #arg3: timing_sm, arg4: safety_sm, arg5: max_accelx (0.4), arg6: max_accely (0.4)
    side_walking = SideWalking(sim_mode=False, planner='Full',prior=[0,1])
    rospy.spin()
