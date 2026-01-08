from fractional_bm import FractionalBrownianMotion
import os

local = os.path.join("src", "UEA", "main.py")
target_dir = os.path.join("results")

FBM = FractionalBrownianMotion(n_paths=5, n_samples=1000, hursts=[0.2, 0.4, 0.6, 0.8])
FBM.plot_sample([0.4, 0.45, 0.5, 0.55], path=os.path.join(target_dir, "fbm_sample_paths.png"))