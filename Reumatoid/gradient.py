#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:40:52 2022

@author: wakamototamaki
"""

#Notch1-DLL1-NICD1_Notchc1_DLLc1_Notch2-DLL2-NICD2_Notchc2_DLLc2_hes1 model
#2-d
#graph
#NSCLC
# %%
import matplotlib.pyplot as plt
import random
import numpy as np
from matplotlib import animation
from tqdm import trange

# %%
N = 20000 #時間刻み数
s = 0.01 #時間の刻み幅
T =  N*s#最終時刻

Ly = 30 #細胞の数（たて）
Lx = 60 #細胞の数（よこ)
ITVL = N/100

# %%
#initial condition
e = 0.1
Nm = e * np.random.rand(Ly, Lx+1) #Notch1 in membrane
Dm = e * np.random.rand(Ly, Lx+1)+5.1 #DLL1 in membrane
Nc = e * np.random.rand(Ly, Lx+1) #Notch1 in cytosol
Dc = e * np.random.rand(Ly, Lx+1)+5.1 #DLL1 in cytosol
I  = e * np.random.rand(Ly, Lx+1)+0.5 #NICD1
#Nm = e * np.zeros((Ly, Lx)) #Notch1 in membrane
#Dm = e * np.zeros((Ly, Lx))+5.1 #DLL1 in membrane
#Nc = e * np.zeros((Ly, Lx)) #Notch1 in cytosol
#Dc = e * np.zeros((Ly, Lx))+5.1 #DLL1 in cytosol
#I  = e * np.zeros((Ly, Lx))+0.5 #NICD1

# %%
#parameter
#binding rate
alpha1 = 1.0

#move to membrane
gammaN = 2.0
gammaD = 1.0

#basal decay
mu1 = 1.0
mu2 = 1.0
mu3 = 0.01
mu4 = 1.0
mu5 = 0.5

#parameter of function
beta1 = 1.1
beta2 = 20.0
beta3 = 100.0
beta4 = 1.0

sigma_v = 1.0
sig = np.arange(0, Lx+1, 1.0)
sigma_s = np.zeros((Ly, Lx+1))
for i in range(Ly):
    sigma_s[i, :] = 1.0*np.exp(-5*(sig/Lx)*(sig/Lx))

# %%
#ODEs
def Notchm(nm, d_ave, nc):
    return -alpha1*sigma_s*d_ave*nm - mu1*nm + gammaN*nc
def Deltam(dm, n_ave, dc):
    return -alpha1*sigma_s*n_ave*dm - mu2*dm + gammaD*dc
def NICD(ac, d_ave, nm):
    return (alpha1*sigma_s/sigma_v)*d_ave*nm - mu3*ac
def Notchc(nc, i):
    return beta2*(i**2)/(beta1 + i**2) - (mu4 + gammaN*sigma_v)*nc
def Deltac(dc, i):
    return beta4/(1 + beta3*(i**2)) - (mu5 + gammaD*sigma_v)*dc


# %%
fig, ax = plt.subplots() #fig,axオブジェクトを作成
#ax.set_xticks([])
#ax.set_yticks([])
ims = []
im1 = plt.imshow(Nm[:, 1:], interpolation='nearest', animated=True, vmin=0, vmax=7, cmap='inferno')
title = plt.text(0.5, 1.1, f't= {0}', transform=plt.gca().transAxes, ha='center', va='center', fontsize="large")
im2 = fig.colorbar(im1, ax=ax)
ims.append([im1, title])

n_ave = np.zeros((Ly, Lx+1))
d_ave = np.zeros((Ly, Lx+1))

for l in trange(N):
    #corner
    n_ave[0, Lx]    = (Nm[0, -1]    + Nm[1, Lx])/2
    n_ave[Ly-1, Lx] = (Nm[-1, Lx] + Nm[Ly-1, -1])/2
    d_ave[0, Lx]    = (Dm[0, -1]    + Dm[1, Lx])/2
    d_ave[Ly-1, Lx] = (Dm[-1, Lx] + Dm[Ly-1, -1])/2
    #edge
    n_ave[0, 1:-1]    = (Nm[1, 1:-1]  + Nm[0, :-2]    + Nm[0, 2:])/3
    n_ave[Ly-1, 1:-1] = (Nm[-1, 1:-1] + Nm[Ly-1, :-2] + Nm[Ly-1, 2:])/3
    n_ave[1:-1, Lx] = (Nm[1:-1, -1] + Nm[:-2, Lx] + Nm[2:, Lx])/3
    d_ave[0, 1:-1]    = (Dm[1, 1:-1]  + Dm[0, :-2]    + Dm[0, 2:])/3
    d_ave[Ly-1, 1:-1] = (Dm[-1, 1:-1] + Dm[Ly-1, :-2] + Dm[Ly-1, 2:])/3
    d_ave[1:-1, Lx] = (Dm[1:-1, -1] + Dm[:-2, Lx] + Dm[2:, Lx])/3
    #other
    n_ave[1:-1, 1:-1] = (Nm[:-2, 1:-1] + Nm[2:,1:-1] + Nm[1:-1, :-2] + Nm[1:-1, 2:])/4
    d_ave[1:-1, 1:-1] = (Dm[:-2, 1:-1] + Dm[2:,1:-1] + Dm[1:-1, :-2] + Dm[1:-1, 2:])/4
    #Notch in membrane
    r11 = Notchm(Nm, d_ave, Nc)
    r21 = Deltam(Dm, n_ave, Dc)
    r31 = NICD(I, d_ave, Nm)
    r41 = Notchc(Nc, I)
    r51 = Deltac(Dc, I)

    r12 = Notchm(Nm+(s/2)*r11, d_ave, Nc+(s/2)*r41)
    r22 = Deltam(Dm+(s/2)*r21, n_ave, Dc+(s/2)*r51)
    r32 = NICD(I+(s/2)*r31, d_ave, Nm+(s/2)*r11)
    r42 = Notchc(Nc+(s/2)*r41, I+(s/2)*r31)
    r52 = Deltac(Dc+(s/2)*r51, I+(s/2)*r31)

    r13 = Notchm(Nm+(s/2)*r12, d_ave, Nc+(s/2)*r42)
    r23 = Deltam(Dm+(s/2)*r22, n_ave, Dc+(s/2)*r52)
    r33 = NICD(I+(s/2)*r32, d_ave, Nm+(s/2)*r12)
    r43 = Notchc(Nc+(s/2)*r42, I+(s/2)*r32)
    r53 = Deltac(Dc+(s/2)*r52, I+(s/2)*r32)

    r14 = Notchm(Nm+s*r13, d_ave, Nc+s*r43)
    r24 = Deltam(Dm+s*r23, n_ave, Dc+s*r53)
    r34 = NICD(I+s*r33, d_ave, Nm+s*r13)
    r44 = Notchc(Nc+s*r43, I+s*r33)
    r54 = Deltac(Dc+s*r53, I+s*r33)
    Nm_new = Nm + (s/6)*(r11 + 2*r12 + 2*r13 + r14)
    #Delta in membrane
    Dm_new = Dm + (s/6)*(r21 + 2*r22 + 2*r23 + r24)
    #NICD
    I_new = I + (s/6)*(r31 + 2*r32 + 2*r33 + r34)
    #Notch in cytosol 
    Nc_new = Nc + (s/6)*(r41 + 2*r42 + 2*r43 + r44)
    Dc_new = Dc + (s/6)*(r51 + 2*r52 + 2*r53 + r54)
    #updata
    Nm[:, 1:] = Nm_new[:, 1:]
    Dm[:, 1:] = Dm_new[:, 1:]
    I[:, 1:]  = I_new[:, 1:]
    Nc[:, 1:] = Nc_new[:, 1:]
    Dc[:, 1:] = Dc_new[:, 1:]
    Nm[Nm<0] = 0
    Dm[Dm<0] = 0
    Nc[Nc<0] = 0
    Dc[Dc<0] = 0
    I[I<0]   = 0

    if (l+1) % ITVL == 0:
        im1 = plt.imshow(Nm[:,1:], interpolation='nearest', animated=True, vmin=0, vmax=7, cmap='inferno')
        title = plt.text(0.5, 1.1, f't= {(l+1)*s}', transform=plt.gca().transAxes, ha='center', va='center', fontsize="large")
        ims.append([im1, title])
# %%
anim = animation.ArtistAnimation(fig, ims, interval=150)
anim.save('gradient_python.gif', writer="pillow")
plt.show()
# %%
