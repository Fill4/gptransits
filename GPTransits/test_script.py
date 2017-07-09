#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

gpResults = np.genfromtxt('celerite_nojitter.results')
diamondsResults = np.genfromtxt('diamonds.results')

plt.rc('text', usetex=False)

fig = plt.figure("Amplitudes_1 (S0)", figsize=(10, 7))
plt.scatter(gpResults[:,1]*gpResults[:,2], diamondsResults[:,1], s=30)
plt.title('Amplitude 1 S_0 (ppm)')
plt.xlabel('GP')
plt.ylabel('DIAMONDS')
plt.tight_layout()

fig2 = plt.figure("Timescales_1 (w0)", figsize=(10, 7))
plt.scatter(gpResults[:,2], diamondsResults[:,2], s=30)
plt.title("Timescale w0 (muHz)")
plt.xlabel('GP')
plt.ylabel('DIAMONDS')
plt.tight_layout()

'''
fig3 = plt.figure("Amplitudes_2 (S0)", figsize=(10, 7))
plt.scatter(gpResults[:,1]*gpResults[:,2], diamondsResults[:,3], s=30)
plt.title("Amplitudes_2 (S0)")
plt.xlabel('GP')
plt.ylabel('DIAMONDS')
plt.tight_layout()

fig4 = plt.figure("Timescales_2 (w0)", figsize=(10, 7))
plt.scatter(gpResults[:,2], diamondsResults[:,4], s=30)
plt.title("Timescales_2 (w0)")
plt.xlabel('GP')
plt.ylabel('DIAMONDS')
plt.tight_layout()
'''

plt.show()