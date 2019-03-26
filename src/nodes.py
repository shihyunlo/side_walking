
import numpy as np
class Nodes :
    def __init__(self,parent=0,start_time=0,end_time=0,has=[],ras=[],x_ped=[],x_rob=[],theta=np.Inf):
        self.parent = parent
        self.start_time = start_time
        self.end_time = end_time
        self.theta = theta
        self.human_trajx = []
        self.human_trajy = []
        self.human_velx = []
        self.human_vely = []
        self.robot_trajx = []
        self.robot_trajy = []
        self.robot_velx = []
        self.robot_vely = []
        self.robot_angz = []
        #self.robot_action_sequence = ras
        self.x_ped = x_ped
        self.x_rob = x_rob
        self.gScore_ = []
        self.fScore_ = np.Inf #unused..?
        self.intent = 2
        self.ped_intent_history = []
        self.theta_new = theta
        self.view = 0
        self.trajt = []
        #if len(has)>0:
        #    self.human_action_sequence = has
        #if len(ras)>0:
        #    self.robot_action_sequence = ras
        #if len(x_ped)>0:
        #    self.x_ped = x_ped
        #if len(x_rob)>0:
        #    self.x_rob = x_rob

