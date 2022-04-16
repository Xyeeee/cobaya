import os
import matplotlib.pyplot as plt
import numpy as np
from anesthetic import MCMCSamples, NestedSamples

plt.switch_backend("TkAgg")
# root = 'icelake/camb_default'
# planck_samples = MCMCSamples(root=root)
poly_samples = NestedSamples(root="beta0.12/default_polychord_raw/default")
# poly_samples.posterior_points(beta=1)
# poly_samples = poly_samples.filter(items=["logA", "beta"])
# fig, ax = poly_samples.plot_2d(poly_samples.columns[:6])
poly_samples.gui()
# old_samples = NestedSamples(root="chains/test")
# old_samples = old_samples.filter(items=["logA"])
# old_samples.gui()
plt.show()
