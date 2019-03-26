import numpy as np

def ObsLikelihood(w,x):
    return 1.0/(1.0+np.exp(-np.dot(w,x)))
