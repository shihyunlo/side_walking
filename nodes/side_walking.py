#!/usr/bin/env python
import sys
sys.path.append('/home/siml/catkin_ws/src/side_walking/src')
import poly_traj_generation
#from poly_traj_generation import PolyTrajGeneration
    #from poly_traj_generation_y import PolyTrajGenerationY
    #from poly_traj_generation_2intents import PolyTrajGenerationTwoIntents
from nodes import Nodes
from check import Check
from check import Map2Odom
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
from geometry_msgs.msg import PoseStamped, Twist, PoseWithCovarianceStamped
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
        self.planner = planner # Full, Pomdp, IPomdp
        if self.planner == 'IPomdp' :
            self.goal_ind = 1 #goal known by the robot
        # sim_mode parameter initialization
        self.mapID = 0
        self.subgoal_ind = 1 # to be set, input by arg

    	# general parameters
        self.x_ref = []
        self.v = 1.0 #desired group velocity
        self.sm = 1.0 #safety margin
        self.dt = 0.1 
        self.weight = [1,1]
        self.H = 2
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
        self.wh = 8.0
    	self.ww = 0.0
    	self.wc = 0.1
    	self.wd = [0,0]
    	self.wd[0] = 2.0
    	self.wd[1] = 2.0
    	self.theta = prior
        self.view_thr = -1
        self.prob_thr = 0.15
        self.collision_thr = 75
        #self.subgoals = np.full((2,2),np.Inf)

        #subgoals
        self.p1_goal = [[np.Inf,np.Inf]]
        self.p1_goal.append([np.Inf,np.Inf])
        self.p2_goal = [[np.Inf,np.Inf]]
        self.p2_goal.append([np.Inf,np.Inf])
        self.intersection = [[np.Inf, np.Inf]]
        self.goal0 = []
        self.odom_costmap = []
        self.odom_costmap0s = [0,0]
        self.odom_costmap_res = 0.025
        self.odom_costmap_rr = 0
        self.odom_costmap_h = 0
        self.map_costmap = []
        self.map_costmap0s = [0,0]
        self.map_costmap_res = 0.025
        self.map_costmap_rr = 0


        self.ds_size = 16
        self.ds_map_costmap = []
        self.ds_map_costmap0s = [0,0]
        self.ds_map_costmap_res = 0.025*self.ds_size
        self.ds_map_costmap_rr = 0

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
        self.rob_pos_odom = [np.Inf for i in range(2)]
        self.ped_rel_pos = [0,0]
        self.ped_id = np.Inf
        self.ped_vel = [0.0 for i in range(2)]
        self.rob_vel = [0.0 for i in range(2)]
        self.rob_vel_odom = [0.0 for i in range(2)]
        self.rob_yaw = np.Inf
        self.rob_yaw_odom = np.Inf
        self.rob_yaw_tracking = np.Inf
        self.rob_angz = 0.0
        self.r_m2o = np.Inf
        self.t_m2o = [0.0,0.0]
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
        self.iXtraj = [[],[]]
        self.iYtraj = [[],[]]
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

        self.velocity_pub = rospy.Publisher('/lp_cmd_vel', Twist, queue_size=1)
        #self.cmd_state_pub = rospy.Publisher('/cmd_state', Twist, queue_size=1)
        #self.marker_pub = rospy.Publisher('ped_pos', vis_msg.Marker, latch=False, queue_size=1)
        #self.line_seg_pub = rospy.Publisher('path', vis_msg.Marker, latch=False, queue_size=1)
        #self.robot_marker_pub = rospy.Publisher('robot', vis_msg.Marker, latch=False, queue_size=1)
        #self.goal_marker_pub = rospy.Publisher('goal', vis_msg.Marker, latch=False, queue_size=1)

        self.vel_msg = Twist()
        self.amcl_sub = rospy.Subscriber('/amcl_pose',PoseWithCovarianceStamped,self.amcl_callback,queue_size=1)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)
        self.ped_sub = rospy.Subscriber('/people_tracked',PersonArray,self.ped_callback,queue_size=1)
        self.goal_pose_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_pose_callback, queue_size=1)
        self.odom_costmap_sub = rospy.Subscriber('/move_base/local_costmap/costmap',OccupancyGrid, self.odom_costmap_callback, queue_size=1)
        self.map_costmap_sub = rospy.Subscriber('/map',OccupancyGrid, self.map_costmap_callback, queue_size=1)
        
        #self.costmap_sub = rospy.Subscriber('/local_map',OccupancyGrid, self.costmap_callback, queue_size=1)
        self.run()


        # Callbacks
    def odom_costmap_callback(self,map_msg) :
        if not np.isnan(map_msg.data[0]) :
            self.odom_costmap = np.copy(map_msg.data)
            self.odom_costmap0s[0] = map_msg.info.origin.position.x-(map_msg.info.width/2.0)*map_msg.info.resolution
            #self.rob_pos[1]
            #print 'map_msg.info.width={}'.format(map_msg.info.width)
            #print 'map_resolution={}'.format(map_msg.info.resolution)

            self.odom_costmap0s[1] = map_msg.info.origin.position.y-(map_msg.info.height/2.0)*map_msg.info.resolution
            self.odom_costmap_res = map_msg.info.resolution
            self.odom_costmap_rr = map_msg.info.width
            self.odom_costmap_h = map_msg.info.height

    def map_costmap_callback(self,map_msg) :
        if not np.isnan(map_msg.data[0]) :
            self.map_costmap = np.copy(map_msg.data)
            self.map_costmap0s[0] = map_msg.info.origin.position.x-(map_msg.info.width/2.0)*map_msg.info.resolution
            self.map_costmap0s[0] = map_msg.info.origin.position.x
            #self.rob_pos[1]
            #print 'map_msg.info.width={}'.format(map_msg.info.width)
            #print 'map_resolution={}'.format(map_msg.info.resolution)

            self.map_costmap0s[1] = map_msg.info.origin.position.y-(map_msg.info.height/2.0)*map_msg.info.resolution
            self.map_costmap0s[1] = map_msg.info.origin.position.y
            
            self.map_costmap_res = map_msg.info.resolution
            self.map_costmap_rr = map_msg.info.width
            self.map_costmap_h = map_msg.info.height
 


    def odom_callback(self,odom_msg) :
        if not np.isnan(odom_msg.pose.pose.position.x):
            self.rob_pos_odom[0] = odom_msg.pose.pose.position.x
            self.rob_pos_odom[1] = odom_msg.pose.pose.position.y
            if odom_msg.twist.twist.linear.x < 1.2 and odom_msg.twist.twist.linear.y < 1.2 :
                self.rob_vel_odom[0] = 0.3*self.rob_vel_odom[0] + 0.7*odom_msg.twist.twist.linear.x
                self.rob_vel_odom[1] = 0.3*self.rob_vel_odom[1] + 0.7*odom_msg.twist.twist.linear.y
            q = odom_msg.pose.pose.orientation
            self.rob_yaw_odom = m.atan2(2.0*(q.x*q.y + q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)
            self.rob_yaw_odom = 2.0*m.acos(q.w)*np.sign(q.z)
            #TODO: Check this:2.0? used to:1.0
            if abs(self.rob_yaw_odom)>m.pi :
                self.rob_yaw_odom = self.rob_yaw_odom - 2.0*np.sign(self.rob_yaw_odom)*m.pi
            
            self.rob_angz = odom_msg.twist.twist.angular.z
            #print 'rob_yaw_odom = {}'.format(self.rob_yaw_odom)


    def amcl_callback(self,amcl_msg) :
        if not np.isnan(amcl_msg.pose.pose.position.x) :
            self.rob_pos[0] = amcl_msg.pose.pose.position.x
            self.rob_pos[1] = amcl_msg.pose.pose.position.y
            #self.rob_pos[0] = 24.88
            #self.rob_pos[1] = 15
            #self.rob_pos[0] = 23.03
            #self.rob_pos[1] = 14.8
            #self.rob_pos[0] = 17.5
            #self.rob_pos[1] = 14.14
            #self.rob_pos[0] = 19.08
            #self.rob_pos[1] = 14.37
            self.rob_pos[0] = 34.606
            self.rob_pos[1] = -50.853
            #if amcl_msg.twist.twist.linear.x < 1.2 and amcl_msg.twist.twist.linear.y < 1.2 :
            #    self.rob_vel[0] = 0.3*self.rob_vel[0] + 0.7*amcl_msg.twist.twist.linear.x
            #    self.rob_vel[1] = 0.3*self.rob_vel[1] + 0.7*amcl_msg.twist.twist.linear.y
            q = amcl_msg.pose.pose.orientation
            self.rob_yaw = m.atan2(2.0*(q.x*q.y + q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)
            self.rob_yaw = 2.0*m.acos(q.w)*np.sign(q.z)
            #TODO:Check this: 2.0 used to 1.0
            if abs(self.rob_yaw)>m.pi :
                self.rob_yaw = self.rob_yaw - 2.0*np.sign(self.rob_yaw)*m.pi
            #print 'rob_yaw = {}'.format(self.rob_yaw)


    def ped_callback(self,person_msg) :
        people = person_msg.people
        #print 'ped callback reached id = {}'.format(len(people))
        ped_found = False 
        supp_ped_found = False
        supp_ped_list = []
        supp_ped_id_list = []
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
            if not np.isnan(person.pose.position.x) and len(pos)>0 and not ped_found and dist <1.5 and person.pose.position.x>0.25:
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
        collision_thr = self.collision_thr
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

        rob_intersection_reached = 0
        ped_intersection_reached = 0
        rob_goal0_reached = 0

        i_trajx = [[],[]]
        i_trajy = [[],[]]
        i_trajx_ = [[],[]]
        i_trajy_ = [[],[]]
        i_path_found = [0,0]

        rob_yaw_ = np.Inf
        preloaded_traj = False
        preloaded_traj_loaded = False

        turning_timing = np.Inf
        if self.planner == 'IPomdp':
            turning_timing = 1.8
        elif self.planner == 'IPomdpBaseline':
            turning_timing = 3.8
        


        while (not rospy.is_shutdown()):
            rob_yaw_update = False
            if self.rob_yaw!=np.Inf and self.rob_yaw_odom!=np.Inf :

                #TODO: check if this affects the accuracy
                if self.r_m2o ==np.Inf: 
                    self.r_m2o = self.rob_yaw_odom - self.rob_yaw
                self.t_m2o[0] = self.odom_costmap0s[0]+(self.odom_costmap_res*self.odom_costmap_rr/2.0) - (np.cos(self.r_m2o)*self.rob_pos[0]-np.sin(self.r_m2o)*self.rob_pos[1])
                self.t_m2o[1] = self.odom_costmap0s[1]+(self.odom_costmap_res*self.odom_costmap_h/2.0) - (np.sin(self.r_m2o)*self.rob_pos[0]+np.cos(self.r_m2o)*self.rob_pos[1])
                self.rob_yaw_tracking = self.rob_yaw_odom - self.r_m2o
                #print 'rob_yaw_tracking = {}'.format(self.rob_yaw_tracking)
                #print 'rob_ yaw_odom = {}'.format(self.r_m2o)
                
                if (self.rob_yaw_tracking-self.rob_yaw)>0.1 :
            
                    print 'rob_yaw tracking -                            amcl = {}'.format(self.rob_yaw_tracking-self.rob_yaw)

            #print 'amcl, odom yaw = {},{}'.format(self.rob_yaw,self.rob_yaw_odom)   
            #plt.cla()
            #plt.axis('equal')
            #x_min = -15
            #x_max = -5
            #_min = 0
            #y_max = 15
            #lt.xlim((x_min, x_max))
            #plt.ylim((y_min, y_max))
            #plt.grid(True)
            #plt.autoscale(False)
            #m2o_x,m2o_y = Map2Odom([self.rob_pos[0]+1],[self.rob_pos[1]],self.r_m2o,self.t_m2o)
            #plt.plot(m2o_x,m2o_y,'og')

            #plt.plot(self.costmap0s[0]+2,self.costmap0s[1]+2,'or')
            #obstacle = [l for l in range(0,len(self.costmap),4) if self.costmap[l]>collision_thr]
            ##print 'costmap = {}'.format(self.costmap[0])
            #width = int(np.sqrt(len(self.costmap)))
            #my = [int(np.floor(l/width))*self.costmap_res+self.costmap0s[1] for l in obstacle]
            #mx = [np.mod(obstacle[k],width)*self.costmap_res+self.costmap0s[0] for k in range(len(obstacle))]
            #plt.plot(mx,my,'ok')

            #print 'r_m2o = {}'.format(self.r_m2o)
            #print 't_m2o = {}'.format(self.t_m2o)
            #plt.pause(0.5)
            #self.vel_msg.linear.x = 0.3
            #self.vel_msg.linear.y = 0
            #omega = 0.4
            #damping = 1.0
            #self.x_pos_ref = 8.4
            #self.y_pos_ref = 12.4
            #self.x_vel_ref = self.x_pos_ref-self.rob_pos[0]
            #self.y_vel_ref = self.y_pos_ref-self.rob_pos[1]
            #v_ref_ = np.sqrt(self.x_vel_ref**2+self.y_vel_ref**2)
            ##print 'v_ref_ = {}'.format(v_ref_)
            #if v_ref_ > 0.01 :
            #    v_ref = [self.x_vel_ref/v_ref_,self.y_vel_ref/v_ref_]
            #    #print 'v_ref = {}'.format(v_ref)
            #    yaw_ref = m.asin(np.cross([1,0],v_ref))
            #    while abs(yaw_ref>m.pi) :
            #        yaw_ref = yaw_ref - np.sign(yaw_ref)*m.pi 
            #    self.vel_msg.angular.z = omega*(yaw_ref-self.rob_yaw) + 2*damping*omega*(0-self.vel_msg.angular.z)
            #    #print 'z = {}'.format(self.vel_msg.angular.z)
            #collision = Check([self.x_pos_ref],[self.y_pos_ref],self.costmap,self.costmap0s,collision_thr,self.costmap_res,self.r_m2o,self.t_m2o) 
            #if collision:
            #    self.vel_msg.linear.x = 0.0
            #    self.vel_msg.linear.y = 0.0
            #    print 'Realtime Collision!!!'

            #self.velocity_pub.publish(self.vel_msg)
            #print 'rob_yaw = {}'.format(self.rob_yaw)



            current_time = time.time()
            robot_time_del_plan = current_time - initial_time
            self.ped_pos[0] = np.cos(self.rob_yaw)*self.ped_rel_pos[0]-np.sin(self.rob_yaw)*self.ped_rel_pos[1]+self.rob_pos[0]
            self.ped_pos[1] = np.sin(self.rob_yaw)*self.ped_rel_pos[0]+np.cos(self.rob_yaw)*self.ped_rel_pos[1]+self.rob_pos[1]
            #print 'rob_yaw = {}'.format(self.rob_yaw)
            if not self.start :# check ped pose, robot pose, goals
            	criteria0 = (self.rob_pos[1]!=np.Inf and self.rob_pos[0]!=np.Inf)
            	criteria1 = (self.ped_rel_pos[1]!=0 and self.ped_rel_pos[0]!=0)
                criteria2 = (self.rob_yaw!=np.Inf) and (self.rob_yaw_odom!=np.Inf)
                #if criteria1 :
                #    print 'criteria1 fixed'
                #    print 'ped_pos = {}'.format(self.ped_pos)

                #if criteria0 :
                #    print 'criteria0 fixed'
                #    print 'rob_pos = {}'.format(self.rob_pos)
                if len(self.map_costmap)>0 and len(self.ds_map_costmap)==0:
                    self.ds_size = 10
                    self.ds_map_costmap0s[0] = self.map_costmap0s[0]+self.map_costmap_res*self.ds_size/2.0
                    self.ds_map_costmap0s[1] = self.map_costmap0s[1]+self.map_costmap_res*self.ds_size/2.0
                    self.ds_map_costmap_res = self.map_costmap_res*self.ds_size
                    patch = []
                    for i in range(0,self.ds_size) :
                        patch = patch + range(i*self.map_costmap_rr,i*self.map_costmap_rr+self.ds_size)
                    
                    #for i in range(0,len(self.map_costmap)) :
                    #    if map_costmap[i]>self.collision_thr:


                    
                    for j in range(int(np.floor(self.map_costmap_h/self.ds_size))) :
                        for i in range(int(np.floor(self.map_costmap_rr/self.ds_size))) :
                            ds_collision = len([1 for k in patch if self.map_costmap[j*self.ds_size*self.map_costmap_rr+i*self.ds_size+k]>self.collision_thr])
                            ds_collision = ds_collision*(self.collision_thr+1)
                            self.ds_map_costmap.append(ds_collision)
                    
                    self.ds_map_costmap_rr = int(np.floor(self.map_costmap_rr/self.ds_size))



            	if criteria0 and criteria1 and criteria2:
                    #initialization
                    print 'criteria0, 1 and 2 fixed'
                    initial_time = current_time
                    #robot_time_del_plan = rob_traj_duration
                    self.x_vel_ref = 0
                    self.y_vel_ref = 0
                    self.vel_msg.linear.x = 0.0
                    self.vel_msg.linear.y = 0.0
                    initial_rob_pos = self.rob_pos
                    initial_ped_pos = [0,0]
                    print 'rob yaw = {}'.format(self.rob_yaw)
                    initial_ped_pos[0] = np.cos(-self.rob_yaw)*self.ped_rel_pos[0]-np.sin(-self.rob_yaw)*self.ped_rel_pos[1]+initial_rob_pos[0]
                    initial_ped_pos[1] = np.sin(-self.rob_yaw)*self.ped_rel_pos[0]+np.cos(-self.rob_yaw)*self.ped_rel_pos[1]+initial_rob_pos[1]
                    

                    x1 = [0,0,0,0]
                    x1[0] = initial_ped_pos[0]
                    x1[1] = initial_ped_pos[1]
                    x2 = [0,0,0,0]
                    x2[0] = initial_rob_pos[0]
                    x2[1] = initial_rob_pos[1]
                    start_yaw = self.rob_yaw
                    print 'initial_rob x = {}'.format(initial_rob_pos[0])
                    print 'initial_ped x = {}'.format(initial_ped_pos[0])
                    print 'initial_rob y = {}'.format(initial_rob_pos[1])
                    print 'initial_ped y = {}'.format(initial_ped_pos[1])
                    

                    # self.p1_goal[0] = [initial_ped_pos[0]+10,initial_ped_pos[1]+2]
                    # self.p1_goal[1] = [initial_ped_pos[0]+5, initial_ped_pos[1]-5]
                    # self.p2_goal[0] = [initial_ped_pos[0]+10,initial_ped_pos[1]+2]
                    # self.p2_goal[1] = [initial_ped_pos[0]+5, initial_ped_pos[1]-7]


                    # #for pilot test
                    # self.intersection[0] = [27.05,14.95]

                    # if initial_rob_pos[0]>self.intersection[0][0]:
                    #     self.p1_goal[0] = [21.95, 14.49]
                    #     self.p2_goal[0] = [21.95, 14.49]
                    # else :
                    #     self.p1_goal[0] = [31.76,16.06]
                    #     self.p2_goal[0] = [31.76,16.06]


                    self.p2_goal[0] = [35.782,-55.718]
                    self.p2_goal[1] = [36.959,-60.777]
                    goal_dir = [0.2307,-0.973]
                    # self.p1_goal[1] = [28.35,10.9]
                    # self.p2_goal[1] = [28.35,10.9]

                    


                    #self.p1_goal[1] = [28.16,5]
                    #self.p2_goal[1] = [28.16,5]


                    # p1_goal = self.p1_goal[1]
                    # p2_goal = self.p2_goal[1]
                    p1_goal = self.p2_goal[1]
                    p2_goal = self.p2_goal[1]
                    plt.cla()
                    plt.axis('equal')
                    x_min = initial_rob_pos[0]-10
                    x_max = initial_rob_pos[0]+15
                    y_min = initial_rob_pos[1]-5
                    y_max = initial_rob_pos[1]

                    plt.xlim((x_min, x_max))
                    plt.ylim((y_min, y_max))
                    plt.grid(True)
                    plt.autoscale(False)          

                    if len(self.map_costmap)>0 :
                        obstacle = [l for l in range(0,len(self.map_costmap),10) if self.map_costmap[l]>collision_thr]
                        width = self.map_costmap_rr
                        my = [int(np.floor(l/width))*self.map_costmap_res+self.map_costmap0s[1] for l in obstacle]
                        mx = [np.mod(obstacle[k],width)*self.map_costmap_res+self.map_costmap0s[0] for k in range(len(obstacle))]
                        plt.plot(mx,my,'ok')

                    plt.plot(self.ped_pos[0],self.ped_pos[1],'or')
                    plt.plot(self.rob_pos[0],self.rob_pos[1],'og')
                    plt.plot(self.p1_goal[0][0],self.p1_goal[0][1],'*r')
                    plt.plot(self.p1_goal[1][0],self.p1_goal[1][1],'*g')
                    raw_input('Environment Ready, Press Any Key to Start')
                    plt.pause(1) 

                    self.start = True
                else :
                    continue
            else:


                #ped_goal = self.subgoal[:][subgoal_ind]
                #initialization: sanity check on pedestrian and robot readings to current_xx_po
                #if self.sim :
                ## gazebo
                #    model_state_result = self.get_model_srv(self.request)
                if abs(self.rob_yaw-rob_yaw_)>0.01:
                    rob_yaw_update = True

                rob_yaw_ = self.rob_yaw
                if self.rob_yaw!=np.Inf and self.rob_yaw_odom!=np.Inf and rob_yaw_update: #t_m2o only update only once r_m2o updated
                    self.t_m2o[0] = self.odom_costmap0s[0]+(self.odom_costmap_res*self.odom_costmap_rr/2.0) - (np.cos(self.r_m2o)*self.rob_pos[0]-np.sin(self.r_m2o)*self.rob_pos[1])
                    self.t_m2o[1] = self.odom_costmap0s[1]+(self.odom_costmap_res*self.odom_costmap_h/2.0) - (np.sin(self.r_m2o)*self.rob_pos[0]+np.cos(self.r_m2o)*self.rob_pos[1])
                    
                if  robot_time_del_plan > (rob_traj_duration - self.dt):
                    #replan
                    new_ped_vel = [(self.ped_pos[0]-x1[0])/robot_time_del_plan,(self.ped_pos[1]-x1[1])/robot_time_del_plan]
                    self.ped_vel[0] = (1-alpha)*self.ped_vel[0]+alpha*new_ped_vel[0]
                    self.ped_vel[1] = (1-alpha)*self.ped_vel[1]+alpha*new_ped_vel[1]
                    self.v = np.sqrt(self.ped_vel[0]**2+self.ped_vel[1]**2)
                    self.v = min(self.v,1.4)
                    self.v = max(self.v,0.5)
                    #self.v = 0.5
                    #print 'self.v ={}'.format(self.v)
                    rel_dist = np.sqrt(self.ped_rel_pos[0]**2+self.ped_rel_pos[1]**2)
                    if rel_dist < 2.5:
                        new_sm = rel_dist
                        self.sm = (1-alpha)*self.sm+alpha*new_sm
                        self.sm = min(self.sm,0.4)

                    x1_ = [0,0,0,0]
                    x1_[0] = self.ped_pos[0]+self.ped_vel[0]
                    x1_[1] = self.ped_pos[1]+self.ped_vel[1]
                    x1_[2] = np.cos(self.rob_yaw)*max(0.1,self.v)
                    x1_[3] = np.sin(self.rob_yaw)*max(0.1,self.v)
                    x2_ = [0,0,0,0]
                    x2_[0] = self.rob_pos[0]+self.rob_vel[0]
                    x2_[1] = self.rob_pos[1]+self.rob_vel[1]
                    x2_[2] = np.cos(self.rob_yaw)*max(0.1,self.v)
                    x2_[3] = np.sin(self.rob_yaw)*max(0.1,self.v)
                    
                    #print 'velocity = {}{}'.format(x1_[2]/self.v,x1_[3]/self.v)         

                    #if np.sqrt(self.ped_vel[0]**2+self.ped_vel[1]**2) > vel_thr:
                     
                    if self.v > vel_thr:
                        #print 'dist = {}'.format(rel_dist)
                        #print 'sm = {}'.format(self.sm)
                        
                        #print 'rob_pos = {}'.format(self.rob_pos)
                        #print 'ped_pos = {}'.format(self.ped_pos)
                        #print 'node initial state ped, rob= {},{}'.format(x1_,x2_)

                        #anticipated intersection
                        inters_now = self.intersection[0]
                        goal0now = self.p2_goal[1]
                        #print 'goal0now={}'.format(goal0now)
                        #print 'rob_pos = {}'.format(self.rob_pos)

                        #TODO: assuming more than one intersection to reach
                        rob_goal0_dist = np.sqrt((self.rob_pos[0]-goal0now[0])**2+(self.rob_pos[1]-goal0now[1])**2)
                        # rob_intersect_dist = np.sqrt((self.rob_pos[0]-inters_now[0])**2+(self.rob_pos[1]-inters_now[1])**2)
                        # ped_intersect_dist = np.sqrt((self.ped_pos[0]-inters_now[0])**2+(self.ped_pos[1]-inters_now[1])**2)


                        if not rob_goal0_reached and rob_goal0_dist<(self.v*turning_timing+0.1) :
                            rob_goal0_reached = 1



                        if not rob_goal0_reached or self.planner=='Test':
                            #to walk together with a constant speed.. TODODO: goal direction?
                            agentID=1
                            print 'default planner called'
                            goal_ind = 1
                            p1_goal = self.p1_goal[goal_ind]
                            p2_goal = self.p2_goal[goal_ind]
                            stime = time.time()
                            robot_path_found, pathx_, pathy_, xtraj, ytraj, rob_s_angz, path_fScore, current = FullKnowledgeCollaborativePlanner(x1_,x2_,self.dt,rob_traj_duration,self.weight,agentID,self.H,p1_goal,p2_goal,self.sm,self.v,self.ww,self.ds_map_costmap,self.ds_map_costmap0s,collision_thr,self.ds_map_costmap_res,self.ds_map_costmap_rr,self.r_m2o,self.t_m2o)
                            time_del = time.time()-stime

                            agentID=2
                            #print 'pomdp called'
                            #stime = time.time()
                            robot_path_found, xtraj, ytraj, pathx_, pathy_, rob_s_angz, path_fScore, current = PomdpFollower(x1_,x2_,self.dt,rob_traj_duration,self.weight,agentID,self.H,[0,1.0],self.view_thr,self.prob_thr,self.p1_goal,self.p2_goal,self.sm,self.v,self.ww,self.wc,self.wd,self.map_costmap,self.map_costmap0s,collision_thr,self.map_costmap_res,self.map_costmap_rr,self.r_m2o,self.t_m2o,self.wh)
                            #time_del = time.time()-stime
                            #print 'pomdp computation time = {}'.format(time_del)
                            #self.Xtraj = xtraj
                            #self.Ytraj = ytraj
                            #human_trajx = pathx_
                            #human_trajy = pathy_
                            #print 'pomdp traj length = {}'.format(len(xtraj))
                            #print 'pomdp traj length = {}'.format(len(pathx_))
                        else :
                            if self.planner == 'IPomdpBaseline' :
                                #TODODO: goal direction?
                                if not preloaded_traj_loaded:
                                    robot_path_found,xtraj,ytraj = IPomdpBaslineTraj(x2_,self.v,goal_dir)
                                    robot_path_found = 1
                                    preloaded_traj_loaded = 1
                                else:
                                    robot_path_found = False

                                preloaded_traj = True


                            # if self.planner=='Full' :
                            #     agentID=1
                            #     print 'full planner called'
                            #     ##p1_goal = self.ped_pos
                            #     ##p2_goal = self.ped_pos
                            #     goal_ind = self.theta.index(max(self.theta))
                            #     p1_goal = self.p1_goal[goal_ind]
                            #     p2_goal = self.p2_goal[goal_ind]
                            #     stime = time.time()
                            #     robot_path_found, pathx_, pathy_, xtraj, ytraj, rob_s_angz, path_fScore, current = FullKnowledgeCollaborativePlanner(x1_,x2_,self.dt,rob_traj_duration,self.weight,agentID,self.H,p1_goal,p2_goal,self.sm,self.v,self.ww,self.ds_map_costmap,self.ds_map_costmap0s,collision_thr,self.ds_map_costmap_res,self.ds_map_costmap_rr,self.r_m2o,self.t_m2o)
                            #     time_del = time.time()-stime
                            #     #print 'full- planner computation time = {}'.format(time_del)
                                
                            # elif self.planner== 'Pomdp' :
                            #     agentID=2
                            #     #print 'pomdp called'
                            #     stime = time.time()
                            #     robot_path_found, xtraj, ytraj, pathx_, pathy_, rob_s_angz, path_fScore, current = PomdpFollower(x1_,x2_,self.dt,rob_traj_duration,self.weight,agentID,self.H,self.theta,self.view_thr,self.prob_thr,self.p1_goal,self.p2_goal,self.sm,self.v,self.ww,self.wc,self.wd,self.ds_map_costmap,self.ds_map_costmap0s,collision_thr,self.ds_map_costmap_res,self.ds_map_costmap_rr,self.r_m2o,self.t_m2o,self.wh)
                            #     time_del = time.time()-stime
                            #     print 'pomdp computation time = {}'.format(time_del)
                            elif self.planner== 'IPomdp' :
                                #TODODO: goal direction?
                                if not preloaded_traj_loaded :
                                    robot_path_found, xtraj, ytraj = IPomdpTraj(x2_,self.v,goal_dir)
                                    robot_path_found = 1
                                    preloaded_traj_loaded = 1
                                else :
                                    robot_path_found = False

                                preloaded_traj = True
                                #robot_path_found, xtraj, ytraj, pathx_, pathy_, rob_s_angz, path_fScore, current = PomdpLeader(x1_,x2_,self.dt,rob_traj_duration,self.weight,agentID,self.H,self.theta,self.view_thr,self.prob_thr,self.p1_goal,self.p2_goal,self.sm,self.v,self.ww,self.wc,self.wd,self.ds_map_costmap,self.ds_map_costmap0s,collision_thr,self.ds_map_costmap_res,self.ds_map_costmap_rr,self.r_m2o,self.t_m2o,self.wh,self.goal_ind)
                                
                                #if robot_path_found:
                            
                                #else :
                                #    print 'pomdp plan not found'





                        if preloaded_traj_loaded and (robot_time_del_plan > (self.rob_traj_duration*(turning_timing+0.2))):
                            #go back to call the default planner#TODODO: goal location
                            agentID=1
                            print 'default planner called'
                            goal_ind = 1
                            p1_goal = self.p1_goal[goal_ind]
                            p2_goal = self.p2_goal[goal_ind]
                            stime = time.time()
                            robot_path_found, pathx_, pathy_, xtraj, ytraj, rob_s_angz, path_fScore, current = FullKnowledgeCollaborativePlanner(x1_,x2_,self.dt,rob_traj_duration,self.weight,agentID,self.H,p1_goal,p2_goal,self.sm,self.v,self.ww,self.ds_map_costmap,self.ds_map_costmap0s,collision_thr,self.ds_map_costmap_res,self.ds_map_costmap_rr,self.r_m2o,self.t_m2o)
                            time_del = time.time()-stime
                            agentID=2
                            #print 'pomdp called'
                            #stime = time.time()
                            robot_path_found, xtraj, ytraj, pathx_, pathy_, rob_s_angz, path_fScore, current = PomdpFollower(x1_,x2_,self.dt,rob_traj_duration,self.weight,agentID,self.H,[0,1.0],self.view_thr,self.prob_thr,self.p1_goal,self.p2_goal,self.sm,self.v,self.ww,self.wc,self.wd,self.map_costmap,self.map_costmap0s,collision_thr,self.map_costmap_res,self.map_costmap_rr,self.r_m2o,self.t_m2o,self.wh)
                            



                        if robot_path_found :
                            self.Xtraj = xtraj
                            self.Ytraj = ytraj
                            human_trajx = pathx_
                            human_trajy = pathy_
                            x1 = x1_
                            x2 = x2_
                            robot_time_del_plan = 0
                            initial_time = current_time
                            terminate_index = int(np.floor(rob_traj_duration/self.dt))
                            #print 'traj generated = {}'.format(pathx_)
                        else :
                            robot_time_del_plan = time.time()-initial_time



                            

                traj_index = int(np.ceil(robot_time_del_plan/self.dt))
                if len(self.Xtraj)>traj_index:
                    
                    #print 'x_pos_ref now and previous = {} {}'.format(self.x_pos_ref,self.Xtraj[traj_index-1])
                    #print 'y_pos_ref now and previous = {} {}'.format(self.y_pos_ref,self.Ytraj[traj_index-1])
                    if traj_index>0 :
                        self.x_vel_ref = (self.Xtraj[traj_index] - self.Xtraj[traj_index-1])/self.dt
                        self.y_vel_ref = (self.Ytraj[traj_index] - self.Ytraj[traj_index-1])/self.dt
                    

                    self.x_pos_ref = np.copy(self.Xtraj[traj_index])
                    self.y_pos_ref = np.copy(self.Ytraj[traj_index])

                    if abs(self.x_vel_ref)>2 or abs(self.y_vel_ref)>2:
                        print 'x_y vel_ref  = {} {}'.format(self.x_vel_ref,self.y_vel_ref)
                        print 'Xtraj = {}'.format(self.Xtraj)
                        print 'Ytraj = {}'.format(self.Ytraj)
                        print 'bad index = {}'.format(traj_index)

                else:
                    print 'INDEX exceeds trajectory output, x_vel_ref = {}'.format(self.x_vel_ref)
                    self.x_pos_ref = self.rob_pos[0]+self.x_vel_ref*self.dt
                    self.y_pos_ref = self.rob_pos[1]+self.y_vel_ref*self.dt


                
        
                ss_error = 1.2
                max_vel = 0.4
                vel_x = ss_error*(self.x_vel_ref*m.cos(self.rob_yaw) + self.y_vel_ref*m.sin(self.rob_yaw))
                vel_y = ss_error*(-self.x_vel_ref*m.sin(self.rob_yaw) + self.y_vel_ref*m.cos(self.rob_yaw))
                vel_xy = min(max_vel,(np.sqrt(vel_x**2+vel_y**2)))
                self.vel_msg.linear.x = vel_xy
                #self.vel_msg.linear.x = 0
                #self.vel_msg.linear.y = 0
                omega = 0.5
                damping = 1.0
                v_ref_ = np.sqrt(self.x_vel_ref**2+self.y_vel_ref**2)
                
                if v_ref_ > 0.01 :
                    v_ref = [self.x_vel_ref/v_ref_,self.y_vel_ref/v_ref_]
                    #print 'v_ref = {}'.format(v_ref)
                    yaw_ref = m.acos(v_ref[0])
                    yaw_ref = yaw_ref*np.sign(v_ref[1])
                    #yaw_ref = m.asin(np.cross([1,0],v_ref))
                    #while abs(yaw_ref>m.pi) :
                    #    yaw_ref = yaw_ref - np.sign(yaw_ref)*2.0*m.pi 
                    self.vel_msg.angular.z = omega*(yaw_ref-self.rob_yaw) + 2*damping*omega*(0-self.rob_angz)
                    #print 'yaw_ref, self.yaw = {}, {}'.format(yaw_ref,self.rob_yaw)
                    #print 'rob ang z = {}'.format(self.rob_angz)
                    #print 'ang vel = {}'.format(self.vel_msg.angular.z)        

                #collision check using online updated costmap 
                #collision = Check([self.x_pos_ref],[self.y_pos_ref],self.odom_costmap,self.odom_costmap0s,collision_thr,self.odom_costmap_res,self.odom_costmap_rr,self.r_m2o,self.t_m2o) 
                collision = Check([self.x_pos_ref],[self.y_pos_ref],self.ds_map_costmap,self.ds_map_costmap0s,collision_thr,self.ds_map_costmap_res,self.ds_map_costmap_rr) 
                                


                if collision:
                    self.vel_msg.linear.x = 0.0                   
                    print 'Realtime Collision!!!'
                    print 'robot position = {}'.format(self.rob_pos)
                    print 'robot position reference = {}'.format([self.x_pos_ref,self.y_pos_ref])
                    self.velocity_pub.publish(self.vel_msg)
                    break
                #print 'vel_msg = {},{}'.format(self.vel_msg.angular.z,self.vel_msg.linear.x)
                self.velocity_pub.publish(self.vel_msg)

                        # Plotting starts here (actual code ends here)

            plt.cla()
            plt.axis('equal')
            x_min = initial_rob_pos[0]-10
            x_max = initial_rob_pos[0]+15
            y_min = initial_rob_pos[1]-5
            y_max = initial_rob_pos[1]

            plt.xlim((x_min, x_max))
            plt.ylim((y_min, y_max))
            plt.grid(True)
            plt.autoscale(False)          

            if len(self.map_costmap)>0 :
                obstacle = [l for l in range(0,len(self.map_costmap),10) if self.map_costmap[l]>collision_thr]
                width = self.map_costmap_rr
                my = [int(np.floor(l/width))*self.map_costmap_res+self.map_costmap0s[1] for l in obstacle]
                mx = [np.mod(obstacle[k],width)*self.map_costmap_res+self.map_costmap0s[0] for k in range(len(obstacle))]
                plt.plot(mx,my,'ok')

            plt.plot(self.ped_pos[0],self.ped_pos[1],'or')
            plt.plot(self.rob_pos[0],self.rob_pos[1],'og')
            plt.plot(self.p1_goal[0][0],self.p1_goal[0][1],'*r')
            plt.plot(self.p1_goal[1][0],self.p1_goal[1][1],'*g')
            #if len(self.ds_map_costmap)>0 :
            #    obstacle = [l for l in range(0,len(self.ds_map_costmap),1) if self.ds_map_costmap[l]>collision_thr]
            #    width = self.ds_map_costmap_rr
            #    my = [int(np.floor(l/width))*self.ds_map_costmap_res+self.ds_map_costmap0s[1] for l in obstacle]
            #    mx = [np.mod(obstacle[k],width)*self.ds_map_costmap_res+self.ds_map_costmap0s[0] for k in range(len(obstacle))]
            #    plt.plot(mx,my,'ok')
            

            if len(self.Xtraj)>0 :
                plt.plot(self.Xtraj,self.Ytraj,'*g')
                plt.plot(human_trajx,human_trajy,'r')



            if len(self.iXtraj[0])>0:
                plt.plot(self.iXtraj[0],self.iYtraj[0],'m')
                plt.plot(self.iXtraj[1],self.iYtraj[1],'m')
                   
            plt.pause(0.02) 




   
             

            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('side_walking_node')
    #side_walking = SideWalking(sim_mode=False, planner='Full',prior=[1,0])
    side_walking = SideWalking(sim_mode=False, planner='IPomdp',prior=[0.5,0.5])
    rospy.spin()
