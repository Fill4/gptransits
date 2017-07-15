#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

celerite_1kernel = np.genfromtxt('celerite_mast_1kernel.results')
celerite_2kernel_small = np.genfromtxt('celerite_mast_2kernel_small.results')
celerite_2kernel_large =  np.genfromtxt('celerite_mast_2kernel_large.results')
#george_nojitter = np.genfromtxt('george_nojitter.results')
#george_expsq = np.genfromtxt('george_expsq.results')
diamonds = np.genfromtxt('diamonds.results')
diamonds2 = np.genfromtxt('diamonds2.results')

factor = np.pi*2
#factor = 1

# Normalization
celerite_1kernel[:,1:4] = np.sqrt(celerite_1kernel[:,1:4]*celerite_1kernel[:,1:4]/(4*np.pi**(1/4)))
celerite_1kernel[:,4:7] = celerite_1kernel[:,4:7]*1e6/(24*3600*factor)

celerite_2kernel_small[:,1:4] = np.sqrt(celerite_2kernel_small[:,1:4]*celerite_2kernel_small[:,1:4]/(4*np.pi**(1/4)))
celerite_2kernel_small[:,4:7] = celerite_2kernel_small[:,4:7]*1e6/(24*3600*factor)

celerite_2kernel_large[:,1:4] = np.sqrt(celerite_2kernel_large[:,1:4]*celerite_2kernel_large[:,1:4]/(4*np.pi**(1/4)))
celerite_2kernel_large[:,4:7] = celerite_2kernel_large[:,4:7]*1e6/(24*3600*factor)

############################################################################
############################################################################


fig = plt.figure('Amplitude_1', figsize=(15, 7))
fig.suptitle(r'Amplitude $S_0$ - One Kernel', fontsize=26)

#------------------------------------

ax1 = fig.add_subplot(121)
ax1.errorbar(celerite_1kernel[:,1], diamonds[:,1], xerr=[celerite_1kernel[:,2], celerite_1kernel[:,3]], yerr=[diamonds[:,2], diamonds[:,3]], fmt='ok', ecolor='b', capthick=2)
#ax1.scatter(celerite_nojitter[:,1], diamonds[:,1], s=30)
x = np.linspace(min(celerite_1kernel[:,1]*0.8), max(celerite_1kernel[:,1]*1.2), num=500)

linreg = linregress(celerite_1kernel[:,1], diamonds[:,1])
rvalue = linreg.rvalue
ax1.plot(x, x*linreg.slope + linreg.intercept, 'r')
ax1.plot(x, x, 'r--')
print(linreg.slope)

ax1.set_title('Diamonds 1 --> R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax1.set_xlabel('GP (ppm)',fontsize=14)
ax1.set_ylabel('DIAMONDS (ppm)',fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------

ax2 = fig.add_subplot(122)
ax2.errorbar(celerite_1kernel[:,1], diamonds2[:,1], xerr=[celerite_1kernel[:,2], celerite_1kernel[:,3]], yerr=[diamonds2[:,2], diamonds2[:,3]], fmt='ok', ecolor='b', capthick=2)
#ax2.scatter(celerite_kallinger[:,1], diamonds[:,1], s=30)
x = np.linspace(min(celerite_1kernel[:,1]*0.8), max(celerite_1kernel[:,1]*1.2), num=500)

linreg = linregress(celerite_1kernel[:,1], diamonds2[:,1])
rvalue = linreg.rvalue
ax2.plot(x, x*linreg.slope + linreg.intercept, 'r')
ax1.plot(x, x, 'r--')
print(linreg.slope)

ax2.set_title('Diamonds 2 --> - R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax2.set_xlabel('GP (ppm)',fontsize=14)
ax2.set_ylabel('DIAMONDS (ppm)',fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)

fig.tight_layout()
fig.subplots_adjust(top=0.85)
plt.savefig('comparison_amp_1.png', dpi = 200)

#------------------------------------

fig2 = plt.figure('Amplitude_2', figsize=(15, 7))
fig2.suptitle(r'Amplitude $S_0$ - Two Kernels', fontsize=26)


ax3 = fig2.add_subplot(121)
ax3.errorbar(celerite_2kernel_small[:,1], diamonds[:,1], xerr=[celerite_2kernel_small[:,2], celerite_2kernel_small[:,3]], yerr=[diamonds[:,2], diamonds[:,3]], fmt='ok', ecolor='b', capthick=2)
#ax2.scatter(celerite_kallinger[:,1], diamonds[:,1], s=30)
x = np.linspace(min(celerite_2kernel_small[:,1]*0.8), max(celerite_2kernel_small[:,1]*1.2), num=500)

linreg = linregress(celerite_2kernel_small[:,1], diamonds[:,1])
rvalue = linreg.rvalue
ax3.plot(x, x*linreg.slope + linreg.intercept, 'r')
ax1.plot(x, x, 'r--')
print(linreg.slope)

ax3.set_title('Small Kernel - Diamonds 1 --> R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax3.set_xlabel('GP (ppm)',fontsize=14)
ax3.set_ylabel('DIAMONDS (ppm)',fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------

ax4 = fig2.add_subplot(122)
ax4.errorbar(celerite_2kernel_large[:,1], diamonds2[:,1], xerr=[celerite_2kernel_large[:,2], celerite_2kernel_large[:,3]], yerr=[diamonds2[:,2], diamonds2[:,3]], fmt='ok', ecolor='b', capthick=2)
#ax2.scatter(celerite_kallinger[:,1], diamonds[:,1], s=30)
x = np.linspace(min(celerite_2kernel_large[:,1]*0.8), max(celerite_2kernel_large[:,1]*1.2), num=500)

linreg = linregress(celerite_2kernel_large[:,1], diamonds2[:,1])
rvalue = linreg.rvalue
ax4.plot(x, x*linreg.slope + linreg.intercept, 'r')
ax1.plot(x, x, 'r--')
print(linreg.slope)

ax4.set_title('Large Kernel - Diamonds 2 --> R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax4.set_xlabel('GP (ppm)',fontsize=14)
ax4.set_ylabel('DIAMONDS (ppm)',fontsize=14)
ax4.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------

fig2.tight_layout()
fig2.subplots_adjust(top=0.85)
plt.savefig('comparison_amp_2.png', dpi = 200)

#############################################################################
#############################################################################
########################################

fig3 = plt.figure('Frequency_1', figsize=(15, 7))
fig3.suptitle(r'Characteristic Frequency $\omega_0$ - One Kernel', fontsize=26)

#------------------------------------

ax1 = fig3.add_subplot(121)
ax1.errorbar(celerite_1kernel[:,4], diamonds[:,4], xerr=[celerite_1kernel[:,5], celerite_1kernel[:,6]], yerr=[diamonds[:,5], diamonds[:,6]], fmt='ok', ecolor='b', capthick=2)
x = np.linspace(min(celerite_1kernel[:,4]*0.8), max(celerite_1kernel[:,4]*1.2), num=500)

linreg = linregress(celerite_1kernel[:,4], diamonds[:,4])
rvalue = linreg.rvalue
ax1.plot(x, x*linreg.slope + linreg.intercept, 'r')
#ax1.plot(x, x, 'r--')
print(linreg.slope)

ax1.set_title(r'Low $\omega_0$ Diamonds --> R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax1.set_xlabel(r'GP ($\mu$Hz)',fontsize=14)
ax1.set_ylabel(r'DIAMONDS ($\mu$Hz)',fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------

ax2 = fig3.add_subplot(122)
ax2.errorbar(celerite_1kernel[:,4], diamonds2[:,4], xerr=[celerite_1kernel[:,5], celerite_1kernel[:,6]], yerr=[diamonds2[:,5], diamonds2[:,6]], fmt='ok', ecolor='b', capthick=2)
x = np.linspace(min(celerite_1kernel[:,4]*0.8), max(celerite_1kernel[:,4]*1.2), num=500)

linreg = linregress(celerite_1kernel[:,4], diamonds2[:,4])
rvalue = linreg.rvalue
ax2.plot(x, x*linreg.slope + linreg.intercept, 'r')
#ax2.plot(x, x, 'r--')
print(linreg.slope)
              
ax2.set_title(r'High $\omega_0$ Diamonds --> R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax2.set_xlabel(r'GP ($\mu$Hz)',fontsize=14)
ax2.set_ylabel(r'DIAMONDS ($\mu$Hz)',fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)

fig3.tight_layout()
fig3.subplots_adjust(top=0.85)
plt.savefig('comparison_freq_1.png', dpi = 200)

#------------------------------------

fig4 = plt.figure('Frequency_2', figsize=(15, 7))
fig4.suptitle(r'Characteristic Frequency $\omega_0$ - Two Kernels', fontsize=26)

ax3 = fig4.add_subplot(121)
ax3.errorbar(celerite_2kernel_small[:,4], diamonds[:,4], xerr=[celerite_2kernel_small[:,5], celerite_2kernel_small[:,6]], yerr=[diamonds[:,5], diamonds[:,6]], fmt='ok', ecolor='b', capthick=2)
x = np.linspace(min(celerite_2kernel_small[:,4]*0.8), max(celerite_2kernel_small[:,4]*1.2), num=500)

linreg = linregress(celerite_2kernel_small[:,4], diamonds[:,4])
rvalue = linreg.rvalue
ax3.plot(x, x*linreg.slope + linreg.intercept, 'r')
ax3.plot(x, x, 'r--')
print(linreg.slope)

ax3.set_title(r'Low $\omega_0$ Kernel - Low $\omega_0$ Diamonds --> R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax3.set_xlabel(r'GP ($\mu$Hz)',fontsize=14)
ax3.set_ylabel(r'DIAMONDS ($\mu$Hz)',fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------
              
ax4 = fig4.add_subplot(122)
ax4.errorbar(celerite_2kernel_large[:,4], diamonds2[:,4], xerr=[celerite_2kernel_large[:,5], celerite_2kernel_large[:,6]], yerr=[diamonds2[:,5], diamonds2[:,6]], fmt='ok', ecolor='b', capthick=2)
x = np.linspace(min(celerite_2kernel_large[:,4]*0.8), max(celerite_2kernel_large[:,4]*1.2), num=500)

linreg = linregress(celerite_2kernel_large[:,4], diamonds2[:,4])
rvalue = linreg.rvalue
ax4.plot(x, x*linreg.slope + linreg.intercept, 'r')
ax4.plot(x, x, 'r--')
print(linreg.slope)
              
ax4.set_title(r'High $\omega_0$ Kernel - High $\omega_0$ Diamonds --> R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax4.set_xlabel(r'GP ($\mu$Hz)',fontsize=14)
ax4.set_ylabel(r'DIAMONDS ($\mu$Hz)',fontsize=14)
ax4.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------
              
fig4.tight_layout()
fig4.subplots_adjust(top=0.85)
plt.savefig('comparison_freq_2.png', dpi = 200)
