import numpy as np


class MinHash:
    def __init__(self, num_hashes):
        self.num_hashes = num_hashes
        self.salts = np.arange(self.num_hashes)

    def signature(self, s):
        sig = [np.inf] * self.num_hashes
        for i in range(self.num_hashes):
            sig[i] = min([hash(' '.join(x) + str(self.salts[i])) for x in s])
        return np.array(sig)
