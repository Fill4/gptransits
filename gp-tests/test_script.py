import matplotlib.pyplot as plt
import numpy as np

timeResults = np.genfromtxt('results.dat')
freqResults = np.genfromtxt('diamonds_results.dat')

fig = plt.figure("Amplitudes_1 (S0)", figsize=(10, 7))
plt.scatter(timeResults[:,1], freqResults[:,1], s=30)
plt.title("Amplitudes_1 (S0)")
plt.xlabel('Amplitude (ppm) Timeseries')
plt.xlabel('Amplitude (ppm) Diamonds')
plt.tight_layout()

fig2 = plt.figure("Timescales_1 (w0)", figsize=(10, 7))
plt.scatter(timeResults[:,2], freqResults[:,2], s=30)
plt.title("Timescales_1 (w0)")
plt.xlabel('Timescale (muHz) Timeseries')
plt.xlabel('Timescale (muHz) Diamonds')
plt.tight_layout()

fig3 = plt.figure("Amplitudes_2 (S0)", figsize=(10, 7))
plt.scatter(timeResults[:,1], freqResults[:,3], s=30)
plt.title("Amplitudes_2 (S0)")
plt.xlabel('Amplitude (ppm) Timeseries')
plt.xlabel('Amplitude (ppm) Diamonds')
plt.tight_layout()

fig4 = plt.figure("Timescales_2 (w0)", figsize=(10, 7))
plt.scatter(timeResults[:,2], freqResults[:,4], s=30)
plt.title("Timescales_2 (w0)")
plt.xlabel('Timescale (muHz) Timeseries')
plt.xlabel('Timescale (muHz) Diamonds')
plt.tight_layout()

plt.show()