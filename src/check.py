import numpy as np
import numpy.linalg as linalg
import sys


def Check(trajx,trajy,costmap,costmap0s,cthr,cres):
    collision = 0
    x0 = costmap0s[0]
    y0 = costmap0s[1]

    for i in range(len(trajx)) :
        mx = int((trajx[i]-x0)/cres)
        my = int((trajy[i]-y0)/cres)
        index = mx+my*int(np.sqrt(len(costmap)))
        if index<len(costmap):
            collision = (costmap[index]>cthr)
        if collision :
            break

    return collision

