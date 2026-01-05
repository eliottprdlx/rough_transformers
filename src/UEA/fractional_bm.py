import numpy as np
import fbm
import random

# Experiment taken from : https://arxiv.org/abs/1905.08494.
class FractionalBrownianMotion():
    def __init__(self, n_paths, n_samples, hursts):
        self.n_paths = n_paths
        self.n_samples = n_samples
        self.hursts = hursts
    
    def generate_fbm(self):
        X, y = [], []
        # X should be of shape (n_paths, n_features, n_samples)
        for i in range(self.n_paths):
            H = random.choice(self.hursts)
            fbm_sample = fbm.FBM(n=self.n_samples-1, hurst=H, length=1.0, method='daviesharte')
            X.append(fbm_sample.fbm().reshape(1, -1))
            y.append(self.hursts.index(H))
        X = np.stack(X)
        y = np.array(y)
        return X, y