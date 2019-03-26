import numpy as np

def ReconstructPathFromNode(cameFrom,current_node_id,totalNodes,agentID):
    key = current_node_id
    pathx = []
    pathy = []
    pathx_ = []
    pathy_ = []
    rob_s_angz = []
    path_intent = []

    while not key==0 :
        current_node = totalNodes[key]
        seqxR = current_node.robot_trajx
        seqyR = current_node.robot_trajy
        angzR = current_node.robot_angz
        seqxH = current_node.human_trajx
        seqyH = current_node.human_trajy

        t = [j for j in range(len(current_node.trajt)) if current_node.trajt[j]>(current_node.start_time-0.1) and current_node.trajt[j]<(current_node.end_time+0.1)]
        if len(seqxR)>0 :
            #current_s_xR = [seqxR[tt] for tt in t]
            #current_s_yR = [seqyR[tt] for tt in t]
            #current_a_zR = [angzR[tt] for tt in t]
            current_s_xR = seqxR
            current_s_yR = seqyR
            
        if len(seqxH)>0 :
            #current_s_xH = [seqxH[tt] for tt in t]
            #current_s_yH = [seqyH[tt] for tt in t]
            current_s_xH = seqxH
            current_s_yH = seqyH

        #print 'len(current_s_xR) = {}'.format(len(current_s_xR))
        #current_a_zR = [angzR[tt] for tt in t]
        current_a_zR = angzR
        current_a_zR = current_a_zR+rob_s_angz
        rob_s_angz = current_a_zR

        if agentID== 1 :
            current_s_xH = current_s_xH+pathx
            pathx = current_s_xH
            current_s_yH = current_s_yH+pathy
            pathy = current_s_yH
            current_s_xR = current_s_xR+pathx_
            pathx_ = current_s_xR
            current_s_yR = current_s_yR+pathy_
            pathy_ = current_s_yR
        else :

            current_s_xR = current_s_xR+pathx
            pathx = current_s_xR
            current_s_yR = current_s_yR+pathy
            pathy = current_s_yR
            #current_s_xH = current_s_xH+pathx_
            #pathx_ = current_s_xH
            #current_s_yH = current_s_yH+pathy_
            #pathy_ = current_s_yH


        current_intent = current_node.intent
        path_intent = [current_intent]+path_intent
        key = cameFrom[key]
        #print 'len(pathx_) = {}'.format(len(pathx_))
        #print 'len(pathx) = {}'.format(len(pathx))
        #print 'key = {}'.format(key)

    return pathx, pathy, pathx_, pathy_, rob_s_angz, path_intent
