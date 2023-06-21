import numpy as np
import random

class BatchMaker:

    def __init__(self, batch_size, data):

        self.batch_size = batch_size
        self.data = data
        assert len(self.data.shape) == 2
        self.n = data.shape[0]
        self.r = self.n % self.batch_size
        
        if self.r != 0:
            print(f"{self.r} samples remaining")
    
    def make_indices(self,):

        indices = np.arange(start=0, stop=self.n, step=1, dtype=np.int32)
        random.shuffle(indices)

        indices = np.array(np.split(indices, self.batch_size))

        return indices



    
