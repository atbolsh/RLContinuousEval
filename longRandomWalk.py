import numpy as np
from copy import deepcopy


# Note that allowed states include 0 but not self.size

class LRW:
    """Numberline of size 'size' (default 1000), every turn, 
steps up to stepsize right or left at random. Reward of 1 by terminating 
on the right, or of -1 for terminating on the left."""

    def __init__(self, size=1000, stepsize=100):
        self.size = size
        self.stepsize = stepsize
        self.initial = int(size/2)
        self.state = self.initial
        self.complete = False
    
    def currentState(self):
        return self.state, self.complete
    
    def move(self):
        if self.complete:
            print "Game over! Restart."
            return self.state, 0, self.complete

        step = np.random.randint(0 - self.stepsize, 
                                 self.stepsize + 1)
        self.state += step

        if self.state < 0:
            reward = -1
            self.state = 0
            self.complete = True
        elif self.state >= self.size:
            reward = 1
            self.state = self.size -1
            self.complete = True
        else:
            reward = 0
        
        return self.state, reward, self.complete
     
    def episode(self):
        state, _ = self.currentState()
        trace = [state]
        while not (self.complete):
            state, reward, _ = self.move()
            trace.append(state)
        return trace, reward #Last reward will be correct.



