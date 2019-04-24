import numpy as np
import math as m
import sys
sys.path.append('/home/siml/catkin_ws/src/side_walking/src')
from nodes import Nodes
import matplotlib.pyplot as plt

def NavigationActionSampling(parent,dt,traj_duration,agentID,v,sm,p1_goal,p2_goal,ang_num,sp_num,num_obs):
    t = np.dot(range(0,int(traj_duration/dt)),dt)
    ang_del = 37.5/360*np.pi
    sp_del = 0.4

    count = 0
    act_num = (2*ang_num+1)*(2*sp_num+1)
    Actions = []

    dv_ = np.dot(range(-sp_num,sp_num+1),sp_del)
    ang_accel_ = np.dot(range(-ang_num,ang_num+1),ang_del)
    dv_mesh, ang_mesh = np.meshgrid(dv_,ang_accel_)
    dv_sampled = dv_mesh.reshape(1,len(dv_)*len(ang_accel_))
    ang_sampled = ang_mesh.reshape(1,len(dv_)*len(ang_accel_))


    #plt.cla()
    #plt.axis('equal')

    #plt.grid(True)
    #plt.autoscale(False)



    #print 'len(dv_), len(ang_accel_) = {},{}'.format(len(dv_),len(ang_accel_))
    for i in range(len(dv_)*len(ang_accel_)):
        action = Nodes(parent,parent.end_time,parent.end_time+traj_duration,[],[],[],parent.theta_new)
        dv = dv_sampled[0][i]
        #print 'dv = {}'.format(dv)
        ang_accel = ang_sampled[0][i]
        count = count + 1
        #initialization
        if (agentID==1):
            init_x_ped = parent.x_ped[0]
            init_y_ped = parent.x_ped[1]
            pos = [init_x_ped,init_y_ped,0,0]
            vx_ind = 2
            vy_ind = 3
            #vTravellerx = pos[2]
            #vTravellery = pos[3]
            vTx = p1_goal[0][0]-pos[0]
            vTy = p1_goal[0][1]-pos[1]
            vTx = parent.x_ped[2]
            vTy = parent.x_ped[3]
        else :
            init_x_rob = parent.x_rob[0]
            init_y_rob = parent.x_rob[1]
            pos = [init_x_rob,init_y_rob,0,0]
            vx_ind = 5
            vy_ind = 6
            #vTravellerx = pos[5]
            #vTravellery = pos[6]
            vTx = p2_goal[0][0]-pos[0]
            vTy = p2_goal[0][1]-pos[1]
            vTx = parent.x_rob[2]
            vTy = parent.x_rob[3]

        #print 'x_rob = {},{}'.format(parent.x_rob[2],parent.x_rob[3])
        vTx_ = vTx/np.sqrt(vTx**2+vTy**2)*(v+dv)
        vTy_ = vTy/np.sqrt(vTx**2+vTy**2)*(v+dv)
        trajx = []
        #trajx.append(pos[0])
        trajy = []
        #trajy.append(pos[1])
        velx = np.dot(vTx_,np.cos(np.dot(t,ang_accel))) + np.dot(vTy_,np.sin(np.dot(t,ang_accel)))
        vely = -np.dot(vTx_,np.sin(np.dot(t,ang_accel))) + np.dot(vTy_,np.cos(np.dot(t,ang_accel)))



        for j in range(0,len(t)):
            pos[0] = pos[0] + velx[j]*dt
            pos[1] = pos[1] + vely[j]*dt
            trajx.append(pos[0])
            trajy.append(pos[1])

            #plt.plot(pos[0],pos[1],'ok')

            

        time = np.add(t,parent.end_time)
        action.trajt = time

        if dv<0 :
            action.intent = 1
        else :
            action.intent = 0


        if agentID==1 :
            action.human_trajx = trajx
            action.human_trajy = trajy
            action.human_velx = velx
            action.human_vely = vely
            x_ped_ = []
            x_ped_.append(pos[0])
            x_ped_.append(pos[1])
            x_ped_.append(velx[-1])
            x_ped_.append(vely[-1])
            action.x_ped = x_ped_
        else :
            action.robot_trajx = trajx
            action.robot_trajy = trajy
            action.robot_velx = velx
            action.robot_vely = vely
            action.robot_angz = [ang_accel for k in range(len(t))]
            x_rob_ = []
            x_rob_.append(pos[0])
            x_rob_.append(pos[1])
            x_rob_.append(velx[-1])
            x_rob_.append(vely[-1])
            action.x_rob = x_rob_

        Actions.append(action)


    Actions_ = []

    for i in range(max(num_obs,1)):
        Actions_.append(Actions)
    #print 'Actions_[0][0].rob_pos = {}'

    plt.pause(0.01)
    return Actions_


