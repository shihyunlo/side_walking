import numpy as np
import numpy.linalg as linalg
import sys
def Map2Odom(trajx_,trajy_,r_m2o,t_m2o) :
    trajx = [t_m2o[0]+np.cos(r_m2o)*trajx_[i] - np.sin(r_m2o)*trajy_[i] for i in range(len(trajx_))]
    trajy = [t_m2o[1]+np.sin(r_m2o)*trajx_[i] + np.cos(r_m2o)*trajy_[i] for i in range(len(trajy_))]

    return trajx, trajy

def Check(trajx,trajy,costmap,costmap0s,cthr,cres,rr,r_m2o= [],t_m2o=[]):
    if not len(t_m2o)==0 : #local map, needs odom to map transform
        trajx, trajy = Map2Odom(trajx,trajy,r_m2o,t_m2o)
    
    
    collision = 0
    x0 = costmap0s[0]
    y0 = costmap0s[1]
    #rr = int(np.sqrt(len(costmap)))
    rad = 0.2
    patch_size = int(rad/cres)
    patch = []
    for i in range(-patch_size,patch_size+1) :
        patch = patch + range(i*rr-patch_size,i*rr+patch_size+1)
 #   for i in range(len(trajx)) :
 #       mx = int((trajx[i]-x0)/cres)
 #       my = int((trajy[i]-y0)/cres)
 #       index = mx+my*rr
 #       count = 0
 #       not_collision = 1
 #       collision_count = 0
 #       for i in patch :
 #           if (index+max(patch))<len(costmap) and (index+min(patch))>0 :
 #               count = count+1
 #               not_collision = costmap[index+i]<cthr
 #               if not_collision :
 #                   continue
 #               else:
 #                   collision_count = collision_count + 1
#        if collision_count==count :
#            collision = 1
#            break


    #patch = 0    
    for i in range(len(trajx)) :
        mx = int((trajx[i]-x0)/cres)
        my = int((trajy[i]-y0)/cres)
        index = mx+my*rr
        if (index+max(patch))< len(costmap) and (index+min(patch))>0 :
            collision = (costmap[index]>cthr)
            #collision = len([(costmap[index+i] >cthr) for i in patch])>0
            if collision :
                break        
        
    return collision
