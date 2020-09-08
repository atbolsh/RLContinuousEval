import numpy as np
from copy import deepcopy


# Note that allowed states include 0 but not self.size

class NumLineTiling:

    def __init__(self, size=1000, blockSize=100, interval=10, alpha = 0.01):
        self.size = size
        self.blockSize = blockSize
        self.interval = interval
        self.alpha = alpha
        
        self.numBlocks = 1 + max(int((size - blockSize)/interval), 0)
        self.blockWeights = np.zeros(self.numBlocks, dtype='float64')
        ## Possible randomization commes here
        self.initBlockWeights()


    def initBlockWeights(self):
        return None # No change for now; might add randomization later.
    
    
    def getBlocks(self, state):
        if (state >= self.size) or (state < 0):
            return -1, -1
        
        highest = int(state / self.interval) #Highest index of a block that contains state

        # At interval = 10 and blockSize = 100, I don't want state 100 to be in block 0;
        # each block gets exactly 100 states
        lowest  = int((self.interval + state - self.blockSize) / self.interval) 
        
        lowest = max(lowest, 0)
        highest = min(highest, self.numBlocks - 1)
        
        return lowest, highest + 1 #Plus one is there for slicing purposes.


    def getVal(self, state):
        l, h = self.getBlocks(state)
        
        if l == -1: # Out of bounds.
            return 0
        
        return sum(self.blockWeights[l:h])/(h - l) # Take average to avoid weird edge effects. 
    

    def moveVal(self, state, target, alpha=-1): #Moves prediction towards target; generalizes to neighbors, of course.
        if alpha <= 0: #None specified
            alpha = self.alpha
        l, h = self.getBlocks(state)

        if l == -1: # Out of bounds.
            return None

        v = self.getVal(state)
        delta = target - v
        gradStep = alpha*(delta / (h - l))
        
        for i in range(l, h):
            self.blockWeights[i] += gradStep
        
        return None


