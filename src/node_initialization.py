import numpy as np
import sys
from nodes import Nodes
def NodeInitialization (x_rob,x_ped,weight,theta,p1_goal,p2_goal,v):

    gScore_ = [0,0]
    gScore = np.dot(gScore_,weight)

    initial_node = Nodes()
    initial_node.parent = 0
    initial_node.
