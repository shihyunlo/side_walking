import numpy as np
import math as m
import sys
sys.path.append('/home/siml/catkin_ws/src/side_walking/src')
from nodes import Nodes

def SideBySideActionSampling(parent,dt,traj_duration,agentID,v,sm,p1_goal,p2_goal,ang_num,sp_num):
    t = np.dot(range(0,int(np.ceil((traj_duration/dt)))),dt)
    ang_del = 37.5/360*np.pi
    sp_del = 0.4

    count = 0
    if agentID==1: #human
        sp_num=0
    else :
        sp_num=1

    act_num = (2*ang_num+1)*(2*sp_num+1)
    Actions = []
    #for i in range(act_num):
    #    Actions.append(Nodes(parent,parent.end_time,parent.end_time+traj_duration,[],[],[],[],parent.theta_new))

    dv_ = np.dot(range(-sp_num,sp_num+1),sp_del)
    ang_accel_ = np.dot(range(-ang_num,ang_num+1),ang_del)

    dv_mesh, ang_mesh = np.meshgrid(dv_,ang_accel_)

    dv_sampled = dv_mesh.reshape(1,len(dv_)*len(ang_accel_))
    ang_sampled = ang_mesh.reshape(1,len(dv_)*len(ang_accel_))


    init_x_ped = parent.x_ped[0]
    init_y_ped = parent.x_ped[1]
    init_x_rob = parent.x_rob[0]
    init_y_rob = parent.x_rob[1]
    
    for i in range(act_num):
        action = Nodes(parent,parent.end_time,parent.end_time+traj_duration,[],[],[],[],parent.theta_new)
        dv = dv_sampled[0][i]
        ang_accel = ang_sampled[0][i]
        count = count + 1
        #initialization
        if (agentID==1):
            pos = [init_x_ped,init_y_ped]
            pos_ = [init_x_rob,init_y_rob]
            vx_ind = 2
            vy_ind = 3
            #vTravellerx = pos[2]
            #vTravellery = pos[3]
            #vTx = p1_goal[0]-pos[0]
            #vTy = p1_goal[1]-pos[1]
            vTx = parent.x_ped[2]
            vTy = parent.x_ped[3]

        else :
            pos = [init_x_rob,init_y_rob]
            pos_ = [init_x_ped,init_y_ped]
            vx_ind = 5
            vy_ind = 6
            #vTravellerx = pos[2]
            #vTravellery = pos[3]
            #vTx = p2_goal[0]-pos[0]
            #vTy = p2_goal[1]-pos[1]
            vTx = parent.x_rob[2]
            vTy = parent.x_rob[3]

        #print 'x_rob full = {},{}'.format(parent.x_rob[2],parent.x_rob[3])
        #v = np.sqrt(parent.x_ped[2]**2+parent.x_ped[3]**2)
        vTx_ = vTx/np.sqrt(vTx**2+vTy**2)*(v+dv)
        vTy_ = vTy/np.sqrt(vTx**2+vTy**2)*(v+dv)
        trajx = []
        #trajx.append(pos[0])
        trajy = []
        #trajy.append(pos[1])
        trajx_ = []
        #trajx_.append(pos_[0])
        trajy_ = []
        #trajy_.append(pos_[1])
        #print 'vTx_ and t and ang_accel = {}{}{}'.format(vTx_,t,ang_accel)
        velx = np.dot(vTx_,np.cos(np.dot(t,ang_accel))) + np.dot(vTy_,np.sin(np.dot(t,ang_accel)))
        vely = -np.dot(vTx_,np.sin(np.dot(t,ang_accel))) + np.dot(vTy_,np.cos(np.dot(t,ang_accel)))

        for j in range(0,len(t)):
            pos[0] = pos[0] + velx[j]*dt
            pos[1] = pos[1] + vely[j]*dt
            s_ = np.cross([velx[j],vely[j]],np.add([pos_[0],pos_[1]],[-pos[0],-pos[1]]))
            the_ = np.sign(s_)*np.pi/2
            pos_[0] = pos[0]+np.dot([np.cos(the_),-np.sin(the_)],[velx[j],vely[j]])*sm
            pos_[1] = pos[1]+np.dot([np.sin(the_),np.cos(the_)],[velx[j],vely[j]])*sm
            trajx.append(pos[0])
            trajy.append(pos[1])
            trajx_.append(pos_[0])
            trajy_.append(pos_[1])

            #plt.cla()
            #plt.axis('equal')
            x_min = -10
            x_max = 10
            y_min = 5
            y_max = 20
            #plt.xlim((x_min, x_max))
            #plt.ylim((y_min, y_max))
            #plt.grid(True)
            #plt.autoscale(False)
            #plt.plot(self.ped_pos[0],self.ped_pos[1],'ok')
            #plt.plot(self.rob_pos[0],self.rob_pos[1],'om')
            #if len(self.Xtraj)>0 :
            #    plt.plot(self.Xtraj,self.Ytraj,'r')
            #    plt.plot(xtraj,ytraj,'g')
            #    plt.pause(0.1)
        time = np.add(t,parent.end_time+dt)
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
            action.robot_trajx = trajx_
            action.robot_trajy = trajy_
            action.robot_velx = velx
            action.robot_vely = vely
            x_ped_ = []
            x_ped_.append(pos[0])
            x_ped_.append(pos[1])
            x_ped_.append(velx[-1])
            x_ped_.append(vely[-1])
            x_rob_ = []
            x_rob_.append(pos_[0])
            x_rob_.append(pos_[1])
            x_rob_.append(velx[-1])
            x_rob_.append(vely[-1])
            action.x_rob = x_rob_
            action.x_ped = x_ped_  
            action.robot_angz = [ang_accel for k in range(len(t))]            


        else :
            action.robot_trajx = trajx
            action.robot_trajy = trajy
            action.robot_velx = velx
            action.robot_vely = vely
            action.human_trajx = trajx_
            action.human_trajy = trajy_
            action.human_velx = velx
            action.human_vely = vely
            x_ped_ = []
            x_ped_.append(pos_[0])
            x_ped_.append(pos_[1])
            x_ped_.append(velx[-1])
            x_ped_.append(vely[-1])
            x_rob_ = []
            x_rob_.append(pos[0])
            x_rob_.append(pos[1])
            x_rob_.append(velx[-1])
            x_rob_.append(vely[-1])
            action.x_rob = x_rob_
            action.x_ped = x_ped_           
            action.robot_angz = [ang_accel for k in range(len(t))] 

           
        Actions.append(action)
    return Actions


