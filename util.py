from longRandomWalk import *
from TilingNumberLine import *
from TilingWithInertia import *
from copy import deepcopy
import numpy as np

def MCeducate(nlt=None, episodes = 10000, decaying=True):
    if type(nlt) == type(None):
        nlt = NumLineTiling() 

    learningTrace = [deepcopy(nlt)]

    for e in range(episodes):
        if decaying:
            nlt.alpha = 1.0/(1 + e)
        lrw = LRW()
        trace, reward = lrw.episode()
        for state in trace:
           nlt.moveVal(state, reward)
        learningTrace.append(deepcopy(nlt))
    
    return nlt, learningTrace


def TDeducate(nlt=None, episodes = 10000, decaying=True):
    if type(nlt) == type(None):
        nlt = NumLineTiling() 
    learningTrace = [deepcopy(nlt)]

    for e in range(episodes):
        if decaying:
            nlt.alpha = 1.0/(1 + e)
        else:
            nlt.alpha = 1e-4
        lrw = LRW()
        trace, reward = lrw.episode()
        nlt.moveVal(trace[-1], reward)
        for i in range(1, len(trace)-1):
           nlt.moveVal(trace[-1-i], nlt.getVal(trace[-i]))
        learningTrace.append(deepcopy(nlt))
    
    return nlt, learningTrace



if __name__ == "__main__":
    print("Running procedure to evaluate the number line.")
    nlt, learningTrace = MCeducate()

