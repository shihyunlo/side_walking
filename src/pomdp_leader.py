import numpy as np
import numpy.linalg as linalg
import math as m
import sys
sys.path.append('/home/siml/catkin_ws/src/side_walking/src')
from full_knowledge_collaborative_planner import FullKnowledgeCollaborativePlanner
from navigation_action_sampling import NavigationActionSampling
from reconstruct_path_from_node import ReconstructPathFromNode

from nodes import Nodes
from check import Check
from obs_likelihood import ObsLikelihood
def IPomdpTraj(x,v,goal_dir,dt):
    pathx = []
    pathy = []
    pos = [x[0],x[1]]
    t = np.dot(range(0,10),dt)
    ang_del = 37.5/360*np.pi

    #rotate, for segment 1
    vTx = goal_dir[0]
    vTy = goal_dir[1]
    velx = np.dot(vTx,np.cos(np.dot(t,ang_accel))) + np.dot(vTy,np.sin(np.dot(t,ang_accel)))
    vely = -np.dot(vTx,np.sin(np.dot(t,ang_accel))) + np.dot(vTy,np.cos(np.dot(t,ang_accel)))
    for j in range(0,len(t)):
        pos[0] = pos[0] + velx[j]*dt + (v-1.0)*goal_dir[0]*j*dt
        pos[1] = pos[1] + vely[j]*dt + (v-1.0)*goal_dir[1]*j*dt
        pathx.append(pos[0])
        pathy.append(pos[1])

    #rotate, for segment 2
    for j in range(0,len(t)):
        pos[0] = pos[0] + velx[-1-j]*dt + (v-1.0)*goal_dir[0]*dt
        pos[1] = pos[1] + vely[-1-j]*dt + (v-1.0)*goal_dir[1]*dt
        pathx.append(pos[0])
        pathy.append(pos[1])

    for j in range(0,len(t)):
        pos[0] = pos[0] + goal_dir[0]*v*dt
        pos[1] = pos[1] + goal_dir[1]*v*dt
        pathx.append(pos[0])
        pathy.append(pos[1])


    path_found = True
    return path_found, pathx, pathy

def IPomdpBaselineTraj(x,v,goal_dir):
    pathx = []
    pathy = []
    pos = [x[0],x[1]]
    t = np.dot(range(0,10),dt)
    ang_del = 37.5/360*np.pi

    #rotate, for segment 1
    vTx = goal_dir[0]
    vTy = goal_dir[1]
    velx = np.dot(vTx,np.cos(np.dot(t,ang_accel))) + np.dot(vTy,np.sin(np.dot(t,ang_accel)))
    vely = -np.dot(vTx,np.sin(np.dot(t,ang_accel))) + np.dot(vTy,np.cos(np.dot(t,ang_accel)))
    for j in range(0,len(t)):
        pos[0] = pos[0] + velx[j]*dt + (v-1.0)*goal_dir[0]*j*dt
        pos[1] = pos[1] + vely[j]*dt + (v-1.0)*goal_dir[1]*j*dt
        pathx.append(pos[0])
        pathy.append(pos[1])

    # rotate, segment 2
    vTx = velx[-1]
    vTy = vely[-1]
    velx = np.dot(vTx,np.cos(np.dot(t,ang_accel))) + np.dot(vTy,np.sin(np.dot(t,ang_accel)))
    vely = -np.dot(vTx,np.sin(np.dot(t,ang_accel))) + np.dot(vTy,np.cos(np.dot(t,ang_accel)))
    for j in range(0,len(t)):
        pos[0] = pos[0] + velx[j]*dt + (v-1.0)*goal_dir[0]*j*dt
        pos[1] = pos[1] + vely[j]*dt + (v-1.0)*goal_dir[1]*j*dt
        pathx.append(pos[0])
        pathy.append(pos[1])

    #rotate, segment 3
    vTx = velx[-1]
    vTy = vely[-1]
    ang_accel = -ang_accel
    velx = np.dot(vTx,np.cos(np.dot(t,ang_accel))) + np.dot(vTy,np.sin(np.dot(t,ang_accel)))
    vely = -np.dot(vTx,np.sin(np.dot(t,ang_accel))) + np.dot(vTy,np.cos(np.dot(t,ang_accel)))
    for j in range(0,len(t)):
        pos[0] = pos[0] + velx[j]*dt + (v-1.0)*goal_dir[0]*j*dt
        pos[1] = pos[1] + vely[j]*dt + (v-1.0)*goal_dir[1]*j*dt
        pathx.append(pos[0])
        pathy.append(pos[1])

    #rotate, segment 4
    vTx = velx[-1]
    vTy = vely[-1]
    velx = np.dot(vTx,np.cos(np.dot(t,ang_accel))) + np.dot(vTy,np.sin(np.dot(t,ang_accel)))
    vely = -np.dot(vTx,np.sin(np.dot(t,ang_accel))) + np.dot(vTy,np.cos(np.dot(t,ang_accel)))
    for j in range(0,len(t)):
        pos[0] = pos[0] + velx[j]*dt + (v-1.0)*goal_dir[0]*j*dt
        pos[1] = pos[1] + vely[j]*dt + (v-1.0)*goal_dir[1]*j*dt
        pathx.append(pos[0])
        pathy.append(pos[1])

    #rotate, segment 5
    vTx = velx[-1]
    vTy = vely[-1]
    velx = np.dot(vTx,np.cos(np.dot(t,ang_accel))) + np.dot(vTy,np.sin(np.dot(t,ang_accel)))
    vely = -np.dot(vTx,np.sin(np.dot(t,ang_accel))) + np.dot(vTy,np.cos(np.dot(t,ang_accel)))
    for j in range(0,len(t)):
        pos[0] = pos[0] + velx[j]*dt + (v-1.0)*goal_dir[0]*j*dt
        pos[1] = pos[1] + vely[j]*dt + (v-1.0)*goal_dir[1]*j*dt
        pathx.append(pos[0])
        pathy.append(pos[1])


    path_found = True
    return path_found, pathx, pathy

def PartnerCollisionCheck(traj1,traj2,thr1,thr2) :
#traj1 may contain several segments
  traj1x = traj1[0] #human
  traj1y = traj1[1]
  traj2x = traj2[0] #robot
  traj2y = traj2[1]
  velx = traj2[2]
  vely = traj2[3]
  #print 'traj1x len = {}'.format(len(traj1x))
  #print 'traj1x len = {}'.format(len(traj1x[0]))
  
  for i in range(len(traj1x)) :
    collision = len([j for j in range(len(traj2x)) if np.sqrt((traj1x[i][j]-traj2x[j])**2+(traj1y[i][j]-traj2y[j])**2)<thr1 and np.sqrt(velx[j]**2+vely[j]**2)>thr2])>0

  return collision






def PomdpLeader(x1,x2,dt,traj_duration,weight,agentID,H,theta_hat,view_thr,thr,p1_goal,p2_goal,sm,v,ww,wc,wd,costmap,costmap0s,cthr,cres,rr,r_m2o,t_m2o,wh,goal_ind):
    path_robot = []
    path_fScore = []
    path_intent = []
    pathx = []
    pathy = []
    pathx_ = []
    pathy_ = []
    robot_s_angz = []
    path_fScore = np.Inf
    current = []
    path_found = False
    initial_node = Nodes()
    initial_node.x_rob = x2
    initial_node.x_ped = x1
    initial_node.theta = theta_hat
    initial_node.theta_new = theta_hat
    initial_node.gScore_ = [0,0]
    gScore = []
    gScore.append(0)
    gScore_ = []
    gScore_.append([0,0])
    fScore = []
    fScore.append(np.Inf)
    fScore_ = []
    fScore_.append([np.Inf, np.Inf])
    totalNodes = []
    totalNodes.append(initial_node)
    closeSet = []
    openSet = []
    openSet.append(0)
    cameFrom = []
    cameFrom.append(-1)
    rob_hScore = np.Inf
    cn_node = []
    cn_node.append(initial_node)
    current = initial_node

    min_dist = 0.3
    min_vel = 0.3

    x_rob = initial_node.x_rob
    x_ped = initial_node.x_ped
    theta = initial_node.theta

    if agentID==1 :
        agentID_=2
    elif agentID==2 :
        agentID_=1

    belief_num = len(theta)
    r_path_found = []
    r_x = []
    r_y = []
    r_x_ = []
    r_y_ = []
    rob_s_angz = []

    rollout_path_found = []
    rollout_x = []
    rollout_x_ = []
    rollout_y = []
    rollout_y_ = []
    robot_s_angz = []
    
    pomdp_path_found = 1

    #for i in range(belief_num):
    #    if theta[i]>thr:
    #        r_path_found, r_x, r_y, r_x_, r_y_, rob_s_angz, path_fScore, current_ = FullKnowledgeCollaborativePlanner(x_ped,x_rob,dt,traj_duration,weight,agentID_,H,p1_goal[i],p2_goal[i],sm,v,ww,costmap,costmap0s,cthr,cres,rr,r_m2o,t_m2o)
    #        if not r_path_found :
    #            fk_path_found = 0

    #    rollout_path_found.append(r_path_found)
    #    rollout_x.append(r_x)
    #    rollout_x_.append(r_x_)
    #    rollout_y.append(r_y)
    #    rollout_y_.append(r_y_)
    #    robot_s_angz.append(rob_s_angz)

    while len(openSet)>0 :
        current_fScores = [fScore[j] for j in openSet]
        min_fScore = min(current_fScores)
        current_node_id = fScore.index(min_fScore)
        if current_node_id>len(totalNodes):
           print 'cureent node id greater than total node size, error'
           break

        current = totalNodes[current_node_id]

        if (current.end_time > traj_duration*H) or (rob_hScore<1.0):
           pathx, pathy, pathx_, pathy_, rob_s_angz, path_intent = ReconstructPathFromNode(cameFrom,current_node_id,totalNodes,agentID)
           path_found = True
           path_fScore = min_fScore
           pathx_ = r_x
           pathy_ = r_y
           #print 'rob_s_angz = {}'.format(rob_s_angz)
           #print 'rob_hScore = {}'.format(rob_hScore)
           #print 'pomdp path vel = {}'.format(np.sqrt((pathx[0]-pathx[1])**2+(pathy[0]-pathy[1])**2)/dt)
           break

        if current_node_id in openSet :
            openSet.pop(openSet.index(current_node_id))
        else :
            print 'current node id {} is not in list'.format(current_node_id)
            print 'totalNodes length = {}'.format(len(totalNodes))
            print 'closedSet = {}'.format(closeSet)
            print 'openSet = {}'.format(openSet)
            break


        closeSet.append(current_node_id)

        theta = current.theta_new
        num_ti = [i for i in range(len(theta)) if theta[i]>thr]

        ang_num = 1
        sp_num = 1
        H_ = max(1,H-2)

        if current.end_time< (traj_duration*H_):
            sp_num=1


        # Pomdp transition rollout
        pomdp_id = 2
        r_path_found, r_x, r_y, r_x_, r_y_, rob_s_angz, path_fScore, current_ = PomdpFollower(x_ped,x_rob,dt,traj_duration,weight,pomdp_id,H,theta,view_thr,thr,p1_goal,p2_goal,sm,v,ww,wc,wd,costmap,costmap0s,cthr,cres,rr,r_m2o,t_m2o,wh)
        if not r_path_found :
            pomdp_path_found = 0
            print 'pomdp rollout not found'
            continue

        start_ind = int(np.floor(current.end_time/dt))
        num_traj = int(np.floor(traj_duration/dt))
        #olloutX = []
        #RolloutY = []
        #RolloutX_ = []
        #RolloutY_ = []
        Rollout = []

        #for j in num_ti :
        #    #print 'len rollout[j] = {}'.format(len(rollout_x[j]))
        #    #print 'start index = {}, end index = {}'.format(start_ind,start_ind+num_traj)
        #    RolloutX.append([rollout_x[j][k] for k in range(start_ind,start_ind+num_traj,1)])
        #    RolloutX_.append([rollout_x_[j][k] for k in range(start_ind,start_ind+num_traj,1)])
        #    RolloutY.append([rollout_y[j][k] for k in range(start_ind,start_ind+num_traj,1)])
        #    RolloutY_.append([rollout_y_[j][k] for k in range(start_ind,start_ind+num_traj,1)])
        RolloutX = r_x
        RolloutY = r_y
        RolloutX_ = r_x_
        RolloutY_ = r_y_

        Rollout.append(RolloutX)
        Rollout.append(RolloutY)


        #Action space sampling
#        neighbors = NavigationActionSampling(current,dt,traj_duration,agentID,v,sm,p1_goal,p2_goal,ang_num,sp_num,len(num_ti))
        neighbors = SideBySideActionSampling(current,dt,traj_duration,agentID,v,sm,p1_goal,p2_goal,ang_num,sp_num)

        #print 'neighbor size = {}x{}'.format(len(neighbors),len(neighbors[0]))
        #print 'current node start time and end time = {},{}'.format(current.start_time,current.end_time)
        #print 'child node start time and end time = {},{}'.format(neighbors[0][0].start_time,neighbors[0][0].end_time)
        ped_travel_cost = traj_duration
        rob_travel_cost = traj_duration
        
      
      
        #print 'start ind, s+num = {},{}'.format(start_ind,start_ind+num_traj)
        #print 'neighbor size = {},{}'.format(len(neighbors),len(neighbors[0]))
        #print 'rollout length = {},{}'.format(len(rollout_x),len(rollout_x[0]))


        ####TODO: Start here!
        for i in range(len(neighbors[0])):
            #nodes with the same sampled action
            cn_node = [neighbors[j][i] for j in range(len(neighbors))]
            traj = [] 
            vel = [] #agent velocity
            x = [] # agent position
            #print 'robot velocity = {}'.format(cn_node[0].robot_velx[-1]**2+cn_node[0].robot_vely[-1]**2)
            if agentID==1 :
                traj.append([cn_node[0].human_trajx[j] for j in range(num_traj)])
                traj.append([cn_node[0].human_trajy[j] for j in range(num_traj)])
                traj.append([cn_node[0].human_velx[j] for j in range(num_traj)])
                traj.append([cn_node[0].human_vely[j] for j in range(num_traj)])  
                vel.append(cn_node[0].x_ped[2])
                vel.append(cn_node[0].x_ped[3])
                x = [cn_node[0].x_ped[0],cn_node[0].x_ped[1]]
          
          
            else :
                traj.append([cn_node[0].robot_trajx[j] for j in range(num_traj)])
                traj.append([cn_node[0].robot_trajy[j] for j in range(num_traj)])
                traj.append([cn_node[0].robot_velx[j] for j in range(num_traj)])
                traj.append([cn_node[0].robot_vely[j] for j in range(num_traj)])
                #print 'cn_node vel = {}'.format(cn_node[0].x_rob)
                vel.append(cn_node[0].x_rob[2])
                vel.append(cn_node[0].x_rob[3])
                x = [cn_node[0].x_rob[0],cn_node[0].x_rob[1]]

            #collision1 = Check(cn_node[0].robot_trajx,cn_node[0].robot_trajy,costmap,costmap0s,cthr,cres,rr,r_m2o,t_m2o)
            collision1 = Check(cn_node[0].robot_trajx,cn_node[0].robot_trajy,costmap,costmap0s,cthr,cres,rr)
            
            collision2 = PartnerCollisionCheck(Rollout,traj,min_dist-0.02,min_vel)
            if collision1 or collision2 :
                #print 'collision 1,2 = {}{}'.format(collision1,collision2)
                #print 'Rollout = {}'.format(Rollout)
                #rint 'traj = {}'.format(traj)

                continue
            #print 'RolloutX_ = {}'.format(RolloutX_)
            #print 'vel = {}'.format(vel)
            wdist = 0.0
            x_ = [] #partner state
            x__ = [] #agent state appended
            for j in range(len(RolloutX_)) :
                for k in range(len(RolloutX_[j])):
                    if np.isnan(RolloutX_[j][k]) or np.isnan(RolloutY_[j][k]):
                        print 'rollout is nan'
                    if np.isnan(traj[0][k]) or np.isnan(traj[1][k]):
                        print 'traj is nan'
                    wdist = wdist + theta[num_ti[j]]*np.sqrt((RolloutX_[j][k]-traj[0][j])**2+(RolloutY_[j][k]-traj[1][j])**2)
                x_.append([RolloutX[j][-1],RolloutY[j][-1]])
                x__.append(x)

            if agentID == 1 :
                x_ped_ = x__
                x_rob_ = x_
            else :
                x_rob_ = x__
                x_ped_ = x_


            cdist = [wdist, wdist]
            cdist[agentID_] = 0.0
            cdist = [0,0] #TODO
            #chvel = [0,ww*((v-np.sqrt(cn_node[0].x_ped[2]**2+cn_node[0].x_ped[3]**2))**2)]
            chvel = [0,0]
            gScore_new = [0,0]
            #print 'gScorw = {}'.format(current.gScore_)
            gScore_new[0] = current.gScore_[0] + rob_travel_cost + chvel[0] + cdist[0]
            gScore_new[1] = current.gScore_[1] + ped_travel_cost + chvel[1] + cdist[1]
            gScoreNew = np.dot(gScore_new,weight)

            for j in num_ti :
                #bayesian belief update based on obs:traj_
                if len(num_ti)<belief_num :
                    # assuming only two states
                    theta = [0 for k in range(len(theta))]
                    theta[j] = 1.0
                else :
                    obs = x_[j]
                    distx = [np.sqrt((x_[k][0]-obs[0])**2+(x_[k][1]-obs[1])**2) for k in range(len(x_))]
                    distx[j] = np.Inf
                    obs_x = [obs[0]-x[0],obs[1]-x[1]]
                    view = np.dot(vel,obs_x)/np.sqrt(np.dot(obs_x,obs_x))/np.sqrt(np.dot(vel,vel))
                    cn_node[j].view = view
                    if view>view_thr :
                        #belief updates
                        mindistx = min(distx)
                        minind = distx.index(mindistx)
                        #update: other
                        obs_s1 = ObsLikelihood(wd[j],mindistx)
                        obs_s = [1.0-obs_s1 for k in range(belief_num)]
                        #update: self
                        obs_s[j] = obs_s1
                        s = theta #??
                        o = np.dot(obs_s,s)
                        s_new = np.multiply(obs_s,s)
                        theta = [k/o for k in s_new]

                cn_ind = num_ti.index(j)
                cn_node[cn_ind].theta_new = theta
                cn_node[cn_ind].gScore_ = gScore_new
                #cost_to_go using the updated theta
                #print 'p1_goal and ped_pos = {},{}'.format(p1_goal,x_ped_)
                #print 'p2_goal and rob_pos = {},{}'.format(p2_goal,x_rob_)

                ped_dtgo = [np.sqrt((p1_goal[num_ti[k]][0]-x_ped_[k][0])**2+(p1_goal[num_ti[k]][1]-x_ped_[k][1])**2)*theta[num_ti[k]]/v for k in range(len(x_ped_)) ]
                ped_hScore = np.sum(ped_dtgo)
                rob_dtgo = [np.sqrt((p2_goal[num_ti[k]][0]-x_rob_[k][0])**2+(p2_goal[num_ti[k]][1]-x_rob_[k][1])**2)*theta[num_ti[k]]/v for k in range(len(x_rob_)) ]
                rob_hScore = np.sum(rob_dtgo)
                hScore = np.dot(weight,[rob_hScore,ped_hScore])
                gScore.append(gScoreNew)
                gScore_.append(gScore_new)
                fScore_.append(np.add(gScore_new,[wh*weight[0]*rob_hScore,wh*weight[1]*ped_hScore])) 
                fScore.append(gScoreNew+wh*hScore+np.random.normal()*0.01)

                #cost-to-go
                cameFrom.append(current_node_id)
                openSet.append(len(totalNodes))
                totalNodes.append(cn_node[cn_ind])

            if (len(totalNodes)-len(openSet))>400 :
                print 'gScore_ = {}'.format(gScore_[-1])
                print 'gScoreNew = {}'.format(gScoreNew)
                print 'fScore = {}'.format(fScore[-1])
                #print 'too many nodes'
                break

    if not path_found:
        print 'pomdp path not found after openSet cleared'
        print 'total node length = {}'.format(len(totalNodes)) 
        print 'cn_node end time = {}'.format(cn_node[0].end_time)
        print 'current node time = {}'.format(current.end_time)
    #else:
    #    print 'total node length = {}'.format(len(totalNodes))       



    return path_found, pathx, pathy, pathx_,pathy_, robot_s_angz, path_fScore, current

