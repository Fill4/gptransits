#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

celerite_nojitter = np.genfromtxt('celerite_nojitter.results')
celerite_kallinger = np.genfromtxt('celerite_kallinger.results')
george_nojitter = np.genfromtxt('george_nojitter.results')
george_expsq = np.genfromtxt('george_expsq.results')
diamonds = np.genfromtxt('diamonds.results')

############################################################################
############################################################################

fig = plt.figure('Amplitude', figsize=(16, 12))
fig.suptitle(r'Amplitude $S_0$', fontsize=26)

#------------------------------------

ax1 = fig.add_subplot(221)
ax1.scatter(celerite_nojitter[:,1]*celerite_nojitter[:,2], diamonds[:,1], s=30)
#ax1.scatter(celerite_nojitter[:,1], diamonds[:,1], s=30)

rvalue = linregress(celerite_nojitter[:,1]*celerite_nojitter[:,2], diamonds[:,1]).rvalue

ax1.set_title('Celerite Kallinger - R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax1.set_xlabel('GP (ppm)',fontsize=14)
ax1.set_ylabel('DIAMONDS (ppm)',fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------

ax2 = fig.add_subplot(222)
ax2.scatter(celerite_kallinger[:,1]*celerite_kallinger[:,2], diamonds[:,1], s=30)
#ax2.scatter(celerite_kallinger[:,1], diamonds[:,1], s=30)

rvalue = linregress(celerite_kallinger[:,1]*celerite_kallinger[:,2], diamonds[:,1]).rvalue

ax2.set_title('Celerite Kallinger + White Noise - R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax2.set_xlabel('GP (ppm)',fontsize=14)
ax2.set_ylabel('DIAMONDS (ppm)',fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------

ax3 = fig.add_subplot(223)
ax3.scatter(george_nojitter[:,1], diamonds[:,1], s=30)

rvalue = linregress(george_nojitter[:,1], diamonds[:,1]).rvalue

ax3.set_title('George Exp Sq - R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax3.set_xlabel('GP (ppm)',fontsize=14)
ax3.set_ylabel('DIAMONDS (ppm)',fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------

ax4 = fig.add_subplot(224)
ax4.scatter(george_expsq[:,1], diamonds[:,1], s=30)

rvalue = linregress(george_expsq[:,1], diamonds[:,1]).rvalue

ax4.set_title('George Exp Sq + White Noise - R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax4.set_xlabel('GP (ppm)',fontsize=14)
ax4.set_ylabel('DIAMONDS (ppm)',fontsize=14)
ax4.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------

fig.tight_layout()
fig.subplots_adjust(top=0.92)
#plt.show()

#############################################################################
#############################################################################

fig2 = plt.figure('Timescale', figsize=(16, 12))
fig2.suptitle(r'Timescale', fontsize=26)

#------------------------------------

ax1 = fig2.add_subplot(221)
ax1.scatter(celerite_nojitter[:,2], diamonds[:,2], s=30)
x = np.linspace(min(celerite_nojitter[:,2])*0.8, max(celerite_nojitter[:,2])*1.2, num=500)

linreg = linregress(celerite_nojitter[:,2], diamonds[:,2])
rvalue = linreg.rvalue
ax1.plot(x, x*linreg.slope + linreg.intercept)

ax1.set_title('Celerite Kallinger - R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax1.set_xlabel(r'GP [$w_0$ ($\mu$Hz)]',fontsize=14)
ax1.set_ylabel(r'DIAMONDS [$w_0$ ($\mu$Hz)]',fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------

ax2 = fig2.add_subplot(222)
ax2.scatter(celerite_kallinger[:,2], diamonds[:,2], s=30)
x = np.linspace(min(celerite_kallinger[:,2])*0.8, max(celerite_kallinger[:,2])*1.2, num=500)
              
linreg = linregress(celerite_kallinger[:,2], diamonds[:,2])
rvalue = linreg.rvalue
ax2.plot(x, x*linreg.slope + linreg.intercept)
              
ax2.set_title('Celerite Kallinger + White Noise - R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax2.set_xlabel(r'GP [$w_0$ ($\mu$Hz)]',fontsize=14)
ax2.set_ylabel(r'DIAMONDS [$w_0$ ($\mu$Hz)]',fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------

ax3 = fig2.add_subplot(223)
ax3.scatter(george_nojitter[:,2], diamonds[:,2], s=30)
x = np.linspace(min(george_nojitter[:,2])*0.9, max(george_nojitter[:,2])*1.1, num=500)

linreg = linregress(george_nojitter[:,2], diamonds[:,2])
rvalue = linreg.rvalue
ax3.plot(x, x*linreg.slope + linreg.intercept)

ax3.set_title('George Exp Sq - R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax3.set_xlabel(r'GP [$\tau$ (days)]',fontsize=14)
ax3.set_ylabel(r'DIAMONDS [$w_0$ ($\mu$Hz)]',fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------
              
ax4 = fig2.add_subplot(224)
ax4.scatter(george_expsq[:,2], diamonds[:,2], s=30)
x = np.linspace(min(george_expsq[:,2])*0.9, max(george_expsq[:,2])*1.1, num=500)
              
linreg = linregress(george_expsq[:,2], diamonds[:,2])
rvalue = linreg.rvalue
ax4.plot(x, x*linreg.slope + linreg.intercept)
              
ax4.set_title('George Exp Sq + White Noise - R Value = {:6f}'.format(abs(rvalue)), fontsize=16)
ax4.set_xlabel(r'GP [$\tau$ (days)]',fontsize=14)
ax4.set_ylabel(r'DIAMONDS [$w_0$ ($\mu$Hz)]',fontsize=14)
ax4.tick_params(axis='both', which='major', labelsize=14)

#------------------------------------
              
fig2.tight_layout()
fig2.subplots_adjust(top=0.92)
plt.show()