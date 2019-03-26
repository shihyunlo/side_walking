import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append('/home/siml/catkin_ws/src/side_walking/src')
from nodes import Nodes
from check import Check
from side_by_side_action_sampling import SideBySideActionSampling
from reconstruct_path_from_node import ReconstructPathFromNode

def FullKnowledgeCollaborativePlanner(x1,x2,dt,traj_duration,weight,agentID,H,p1_goal,p2_goal,sm,v,ww,costmap,costmap0s,cthr,cres):
    path_robot = []
    path_fScore = []
    path_intent = []
    path_found = False
    initial_node = Nodes()
    initial_node.x_rob = x2
    initial_node.x_ped = x1
    initial_node.gScore_ = [0,0]
    #initial_node.theta = theta_hat
    #initial_node.theta_new = theta_hat
    gScore = []
    gScore.append(0)
    gScore_ = []
    fScore = []
    fScore.append(np.Inf)
    totalNodes = []
    totalNodes.append(initial_node)
    closeSet = []
    openSet = []
    openSet.append(0)
    cameFrom = []
    cameFrom.append(-1)
    rob_hScore = np.Inf

    min_dist = sm-0.1
    min_vel = 0.3

    x_rob = initial_node.x_rob
    x_ped = initial_node.x_ped
    pathx = []
    pathx_ = []
    pathy = []
    pathy_ = []
    rob_s_angz = []
    current = []

    #plt.cla()
    


    while len(openSet)>0 :
        current_fScores = [fScore[j] for j in openSet]
        min_fScore = min(current_fScores)
        current_node_id = fScore.index(min_fScore)
        if current_node_id>len(totalNodes):
            print 'cureent node id greater than total node size, error'
            break

        current = totalNodes[current_node_id]
        #print 'cn_node_id = {}'.format(current_node_id)
        #print 'cn node x_rob = {}'.format(current.x_rob)
        #print 'cn_node x_ped = {}'.format(current.x_ped)
        #print 'cn_node trajx = {}'.format(current.robot_trajx)
        #print 'cn_node trajy = {}'.format(current.robot_trajy)
        



        if (current.start_time > (traj_duration*H+0.1)) or (rob_hScore<1.0):
            pathx, pathy, pathx_, pathy_, rob_s_angz, path_intent = ReconstructPathFromNode(cameFrom,current_node_id,totalNodes,agentID)
            path_found = True
            path_fScore = min_fScore
            #print 'path found'
            #print 'pathx_ = {}'.format(pathx_)
            #print 'pathy_ = {}'.format(pathy_)


            break
        #print 'openSet = {}'.format(openSet)
        #print 'fScore = {}'.format(fScore)
        openSet.pop(openSet.index(current_node_id))
        closeSet.append(current_node_id)

        ang_num = 2
        sp_num = 1
        neighbors = SideBySideActionSampling(current,dt,traj_duration,agentID,v,sm,p1_goal,p2_goal,ang_num,sp_num)

        ped_intent_real = 2

        ped_travel_cost = traj_duration
        rob_travel_cost = traj_duration
        #print 'len neighbors = {}'.format(len(neighbors))
        for i in range(len(neighbors)):
            cn_node = neighbors[i]
            #collision1 = CollisionCheck([cn_node.human_trajx,cn_node.human_trajy],[cn_node.robot_trajx,cn_node.robot_trajy],min_dist-0.02,min_vel)
            #collision2 = MapCollisionCheck(cn_node)
            # TODO
            collision1 = Check(cn_node.robot_trajx,cn_node.robot_trajy,costmap,costmap0s,cthr,cres)
            collision2 = 0
            if collision1 or collision2 :
                continue

            cn_node_id = len(totalNodes)
            #cn_node.ped_intent_history.append(current.ped_intent_history)
            #cn_node.ped_intent_history.append(ped_intent_real)
            
            #print 'current x_ped = {}'.format(current.x_ped)
            #print 'len cn node x ped, x_rob = {},{}'.format(cn_node.x_ped,cn_node.x_rob)
            #print 'x ped = {}'.format(x_ped)
            human_velx = cn_node.x_ped[2]
            human_vely = cn_node.x_ped[3]
            human_vel = np.sqrt(human_velx**2+human_vely**2)

            gScore_new = np.add(current.gScore_,[rob_travel_cost,ped_travel_cost])
            gScore_new = np.add(gScore_new,[0,ww*((human_vel-v)**2)])
            cn_node.gScore_ = gScore_new
            gScore.append(np.dot(weight,gScore_new)+np.random.normal()*0.01) #add noise to avoid floating error
            cameFrom.append(current_node_id)


            x_ped_ = cn_node.x_ped
            pHx = x_ped_[0]
            pHy = x_ped_[1]
            x_rob_ = cn_node.x_rob
            pRx = x_rob_[0]
            pRy = x_rob_[1]

            ped_del = np.add([pHx,pHy],[-p1_goal[0],-p1_goal[1]])
            rob_del = np.add([pRx,pRy],[-p2_goal[0],-p2_goal[1]])
            ped_hScore = np.sqrt(np.dot(ped_del,ped_del))/v
            rob_hScore = np.sqrt(np.dot(rob_del,rob_del))/v
            hScore = np.dot(weight,[rob_hScore,ped_hScore])
            fScore.append(gScore[-1]+hScore)
            openSet.append(cn_node_id)
            totalNodes.append(cn_node)

            #print 'robot_trajx = {},{}'.format(cn_node_id,cn_node.robot_trajx)
            #print 'node end time = {}'.format(cn_node.end_time)
            #print 'robot_trajx = {}{}'.format(ccn_node.robot_trajx[0]mcn_node.robot_trajx[-1])
            
            #plt.plot(cn_node.robot_trajx,cn_node.robot_trajx,'*r')
            #plt.plot(cn_node.human_trajy,cn_node.human_trajy,'*g')
            

   
    #plt.pause(0.1)
    if not path_found:
        print 'fk path not found after openSet cleared'
        print 'total node length = {}'.format(len(totalNodes))
    #else :

        #print 'gScore = {}'.format(gScore_new)

    return path_found, pathx, pathy, pathx_, pathy_, rob_s_angz, path_fScore, current
