import os
import matplotlib.pyplot as plt
import numpy as np
from anesthetic import MCMCSamples, NestedSamples


plt.switch_backend("TkAgg")
root = 'icelake/camb_default'

poly_samples = NestedSamples(root="chains/test")
# poly_samples.tex = planck_samples.tex

f = open(test.stats)
[ list(i)[4:]  for i in f.readlines[30:27+30] ]

paramnames = poly_samples.columns[:6].tolist()

# paramnames = np.append(paramnames, 'beta')
# Plotting
fig, ax = poly_samples.plot_2d(paramnames)
poly_samples.plot_2d(ax)

poly_samples.gui()
plt.show()
