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

#factor = np.pi*2
factor = 1

# Normalization
celerite_1kernel[:,1:4] = np.sqrt(celerite_1kernel[:,1:4]*celerite_1kernel[:,1:4])
celerite_1kernel[:,4:7] = celerite_1kernel[:,4:7]*1e6/(24*3600*factor)

celerite_2kernel_small[:,1:4] = np.sqrt(celerite_2kernel_small[:,1:4]*celerite_2kernel_small[:,1:4])
celerite_2kernel_small[:,4:7] = celerite_2kernel_small[:,4:7]*1e6/(24*3600*factor)

celerite_2kernel_large[:,1:4] = np.sqrt(celerite_2kernel_large[:,1:4]*celerite_2kernel_large[:,1:4])
celerite_2kernel_large[:,4:7] = celerite_2kernel_large[:,4:7]*1e6/(24*3600*factor)

############################################################################
############################################################################


fig = plt.figure('Amplitude', figsize=(14, 10))
fig.suptitle(r'Amplitude $S_0$', fontsize=26)

#------------------------------------

ax1 = fig.add_subplot(221)
ax1.errorbar(celerite_1kernel[:,1], diamonds[:,1], xerr=[celerite_1kernel[:,2], celerite_1kernel[:,3]], yerr=[diamonds[:,2], diamonds[:,3]], fmt='ok', ecolor='b', capthick=2)
#ax1.scatter(celerite_nojitter[:,1], diamonds[:,1], s=30)
x = np.linspace(min(celerite_1kernel[:,1]*0.8), max(celerite_1kernel[:,1]*1.2), num=500)

linreg = linregress(celerite_1kernel[:,1], diamonds[:,1])
rvalue = linreg.rvalue
ax1.plot(x, x*linreg.slope + linreg.intercept, 'r')

ax1.set_title('One Kernel - Diamonds 1 --> R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax1.set_xlabel('GP (ppm)',fontsize=14)
ax1.set_ylabel('DIAMONDS (ppm)',fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------

ax2 = fig.add_subplot(222)
ax2.errorbar(celerite_1kernel[:,1], diamonds2[:,1], xerr=[celerite_1kernel[:,2], celerite_1kernel[:,3]], yerr=[diamonds2[:,2], diamonds2[:,3]], fmt='ok', ecolor='b', capthick=2)
#ax2.scatter(celerite_kallinger[:,1], diamonds[:,1], s=30)
x = np.linspace(min(celerite_1kernel[:,1]*0.8), max(celerite_1kernel[:,1]*1.2), num=500)

linreg = linregress(celerite_1kernel[:,1], diamonds2[:,1])
rvalue = linreg.rvalue
ax2.plot(x, x*linreg.slope + linreg.intercept, 'r')

ax2.set_title('One Kernel - Diamonds 2 --> - R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax2.set_xlabel('GP (ppm)',fontsize=14)
ax2.set_ylabel('DIAMONDS (ppm)',fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------

ax3 = fig.add_subplot(223)
ax3.errorbar(celerite_2kernel_small[:,1], diamonds[:,1], xerr=[celerite_2kernel_small[:,2], celerite_2kernel_small[:,3]], yerr=[diamonds[:,2], diamonds[:,3]], fmt='ok', ecolor='b', capthick=2)
#ax2.scatter(celerite_kallinger[:,1], diamonds[:,1], s=30)
x = np.linspace(min(celerite_2kernel_small[:,1]*0.8), max(celerite_2kernel_small[:,1]*1.2), num=500)

linreg = linregress(celerite_2kernel_small[:,1], diamonds[:,1])
rvalue = linreg.rvalue
ax3.plot(x, x*linreg.slope + linreg.intercept, 'r')

ax3.set_title('Two Kernels Small - Diamonds 1 --> R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax3.set_xlabel('GP (ppm)',fontsize=14)
ax3.set_ylabel('DIAMONDS (ppm)',fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------

ax4 = fig.add_subplot(224)
ax4.errorbar(celerite_2kernel_large[:,1], diamonds2[:,1], xerr=[celerite_2kernel_large[:,2], celerite_2kernel_large[:,3]], yerr=[diamonds2[:,2], diamonds2[:,3]], fmt='ok', ecolor='b', capthick=2)
#ax2.scatter(celerite_kallinger[:,1], diamonds[:,1], s=30)
x = np.linspace(min(celerite_2kernel_large[:,1]*0.8), max(celerite_2kernel_large[:,1]*1.2), num=500)

linreg = linregress(celerite_2kernel_large[:,1], diamonds2[:,1])
rvalue = linreg.rvalue
ax4.plot(x, x*linreg.slope + linreg.intercept, 'r')

ax4.set_title('Two Kernels Large - Diamonds 2 --> R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax4.set_xlabel('GP (ppm)',fontsize=14)
ax4.set_ylabel('DIAMONDS (ppm)',fontsize=14)
ax4.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------

fig.tight_layout()
fig.subplots_adjust(top=0.92)
#plt.show()


#############################################################################
#############################################################################
########################################


fig2 = plt.figure('Frequency', figsize=(14, 10))
fig2.suptitle(r'Characteristic Frequency', fontsize=26)

#------------------------------------

ax1 = fig2.add_subplot(221)
ax1.errorbar(celerite_1kernel[:,4], diamonds[:,4], xerr=[celerite_1kernel[:,5], celerite_1kernel[:,6]], yerr=[diamonds[:,5], diamonds[:,6]], fmt='ok', ecolor='b', capthick=2)
x = np.linspace(min(celerite_1kernel[:,4]*0.8), max(celerite_1kernel[:,4]*1.2), num=500)

linreg = linregress(celerite_1kernel[:,4], diamonds[:,4])
rvalue = linreg.rvalue
ax1.plot(x, x*linreg.slope + linreg.intercept, 'r')

ax1.set_title('One Kernel - Diamonds 1 --> R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax1.set_xlabel(r'GP ($\mu$Hz)',fontsize=14)
ax1.set_ylabel(r'DIAMONDS ($\mu$Hz)',fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------

ax2 = fig2.add_subplot(222)
ax2.errorbar(celerite_1kernel[:,4], diamonds2[:,4], xerr=[celerite_1kernel[:,5], celerite_1kernel[:,6]], yerr=[diamonds2[:,5], diamonds2[:,6]], fmt='ok', ecolor='b', capthick=2)
x = np.linspace(min(celerite_1kernel[:,4]*0.8), max(celerite_1kernel[:,4]*1.2), num=500)

linreg = linregress(celerite_1kernel[:,4], diamonds2[:,4])
rvalue = linreg.rvalue
ax2.plot(x, x*linreg.slope + linreg.intercept, 'r')
              
ax2.set_title('One Kernel - Diamonds 2 --> R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax2.set_xlabel(r'GP ($\mu$Hz)',fontsize=14)
ax2.set_ylabel(r'DIAMONDS ($\mu$Hz)',fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------

ax3 = fig2.add_subplot(223)
ax3.errorbar(celerite_2kernel_small[:,4], diamonds[:,4], xerr=[celerite_2kernel_small[:,5], celerite_2kernel_small[:,6]], yerr=[diamonds[:,5], diamonds[:,6]], fmt='ok', ecolor='b', capthick=2)
x = np.linspace(min(celerite_2kernel_small[:,4]*0.8), max(celerite_2kernel_small[:,4]*1.2), num=500)

linreg = linregress(celerite_2kernel_small[:,4], diamonds[:,4])
rvalue = linreg.rvalue
ax3.plot(x, x*linreg.slope + linreg.intercept, 'r')

ax3.set_title('Two Kernels Small - Diamonds 1 --> R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax3.set_xlabel(r'GP ($\mu$Hz)',fontsize=14)
ax3.set_ylabel(r'DIAMONDS ($\mu$Hz)',fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------
              
ax4 = fig2.add_subplot(224)
ax4.errorbar(celerite_2kernel_large[:,4], diamonds2[:,4], xerr=[celerite_2kernel_large[:,5], celerite_2kernel_large[:,6]], yerr=[diamonds2[:,5], diamonds2[:,6]], fmt='ok', ecolor='b', capthick=2)
x = np.linspace(min(celerite_2kernel_large[:,4]*0.8), max(celerite_2kernel_large[:,4]*1.2), num=500)

linreg = linregress(celerite_2kernel_large[:,4], diamonds2[:,4])
rvalue = linreg.rvalue
ax4.plot(x, x*linreg.slope + linreg.intercept, 'r')
              
ax4.set_title('Two Kernels Large - Diamonds 2 --> R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax4.set_xlabel(r'GP ($\mu$Hz)',fontsize=14)
ax4.set_ylabel(r'DIAMONDS ($\mu$Hz)',fontsize=14)
ax4.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------
              
fig2.tight_layout()
fig2.subplots_adjust(top=0.92)
plt.show()