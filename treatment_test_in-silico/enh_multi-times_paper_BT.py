#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 09:57:20 2023

@author: wakamototamaki
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import copy
# %%
N = 30000 #時間刻み数
T = 500 #最終時刻
s = T/N #時間の刻み幅
ver = 100 #細胞の数（たて）
wid = 100 #細胞の数（よこ）
ITVL = int(N/50)

t = np.linspace(0, T, N+1)
T1 = 500
N1 = 150000
t1 = np.linspace(T, T+T1, N1+1)
t2 = t1+T1

# %%
canc = np.zeros((4, 6))
# %%
#function
#activation and inhibition of Hes-1 by NICD
def hes(x1, x2):
    return nu0 + (nu2*(x1**2)/(nu1 + x1**2))*(1 - ((x2**2)/(nu3 + x2**2)))

#ODEs
def Notchm1(x, d_ave1, d_ave2, nc1):
    return (1+p_a)*(-alpha1*d_ave1*x - alpha2*d_ave2*x) - mu1*x + (1+p_g)*gamma1*nc1
def Notchm2(x, d_ave1, d_ave2, nc2):
    return (1+p_a)*(-alpha5*d_ave1*x - alpha6*d_ave2*x) - mu1*x + (1+p_g)*gamma1*nc2
    
def Deltam1(x, n_ave1, n_ave2, dc1):
    return (1+p_a)*(-alpha1*n_ave1*x -alpha5*n_ave2*x) - mu2*x + gamma2*dc1
def Deltam2(x, n_ave1, n_ave2, dc2):
    return (1+p_a)*(-alpha2*n_ave1*x - alpha6*n_ave2*x) - mu2*x + gamma2*dc2
    
def NICD1(x, d_ave1, d_ave2, nm1):
    return (1+p_a)*(alpha1*d_ave1*nm1 + alpha2*d_ave2*nm1) - mu4*x    
def NICD2(x, d_ave1, d_ave2, nm2):
    return (1+p_a)*(alpha5*d_ave1*nm2 + alpha6*d_ave2*nm2)  - mu4*x
    
def Notchc1(x, i1):
    return (1+p_b)*beta2_1*(i1**2)/(beta1 + i1**2) - (mu5 + (1+p_g)*gamma1)*x    
def Notchc2(x, i2):
    return (1+p_b)*beta2_2*(i2**2)/(beta1 + i2**2) - (mu5 + (1+p_g)*gamma1)*x
    
def Deltac1(x, hh):
    return beta4_1/(beta3 + hh**2) - (mu6 + gamma2)*x    
def Deltac2(x, hh):
    return beta4_2/(beta3 + hh**2) - (mu6 + gamma2)*x

def LCHes1(x, i1, i2):
    return hes(i1, i2)-mu0*x
def BTHes1(x, i1, i2):
    return hes(i2, i1)-mu0*x

# %%
#runge-kutta
def ud_m(func, m, u1, u2, c): #function of Notch and Delta in membrane
    r11 = func(m,  u1, u2, c)
    r12 = func(m+(s/2)*r11, u1, u2, c)
    r13 = func(m+(s/2)*r12, u1, u2, c)
    r14 = func(m+s*r13, u1, u2, c)
    return m + (s/6)*(r11 + 2*r12 + 2*r13 + r14)
def ud_i(func, i, d1, d2, nm): #function of NICD
    r11 = func(i,  d1, d2, nm)
    r12 = func(i+(s/2)*r11, d1, d2, nm)
    r13 = func(i+(s/2)*r12, d1, d2, nm)
    r14 = func(i+s*r13, d1, d2, nm)
    return i + (s/6)*(r11 + 2*r12 + 2*r13 + r14)
def ud_c(func, c, ih): #function of Notch and Delta in cytosol
    r11 = func(c,  ih)
    r12 = func(c+(s/2)*r11, ih)
    r13 = func(c+(s/2)*r12, ih)
    r14 = func(c+s*r13, ih)
    return c + (s/6)*(r11 + 2*r12 + 2*r13 + r14)
def ud_h(func, h, i1, i2): #function of HES-1
    r11 = func(h, i1, i2)
    r12 = func(h+(s/2)*r11, i1, i2)
    r13 = func(h+(s/2)*r12, i1, i2)
    r14 = func(h+s*r13, i1, i2)
    return h + (s/6)*(r11 + 2*r12 + 2*r13 + r14)

# %%
#sequence
Nm1 = np.zeros((ver+2, wid+2)) #Notch1 in membrane
Dm1 = np.zeros((ver+2, wid+2)) #DLL1 in membrane
Nc1 = np.zeros((ver+2, wid+2)) #Notch1 in cytosol
Dc1 = np.zeros((ver+2, wid+2)) #DLL1 in cytosol
I1 = np.zeros((ver+2, wid+2)) #NICD1
Nm2 = np.zeros((ver+2, wid+2)) #Notch2 in membrane
Dm2 = np.zeros((ver+2, wid+2)) #DLL2 in membrane
Nc2 = np.zeros((ver+2, wid+2)) #Notch2 in cytosol
Dc2 = np.zeros((ver+2, wid+2)) #DLL2 in cytosol
I2 = np.zeros((ver+2, wid+2)) #NICD2
H1 = np.zeros((ver+2, wid+2)) #Hes1

# %%
#initial condition
e = 0.01
Nm1[1:-1, 1:-1] = e*np.random.rand(ver, wid)
Nm2[1:-1, 1:-1] = e*np.random.rand(ver, wid)
Dm1[1:-1, 1:-1] = e*np.random.rand(ver, wid)
Dm2[1:-1, 1:-1] = e*np.random.rand(ver, wid)
Nc1[1:-1, 1:-1] = e*np.random.rand(ver, wid)
Nc2[1:-1, 1:-1] = e*np.random.rand(ver, wid)
Dc1[1:-1, 1:-1] = e*np.random.rand(ver, wid)
Dc2[1:-1, 1:-1] = e*np.random.rand(ver, wid)
I1[1:-1, 1:-1] = e*np.random.rand(ver, wid)
I2[1:-1, 1:-1] = e*np.random.rand(ver, wid)
H1[1:-1, 1:-1] = e*np.random.rand(ver, wid)
H_0 = copy.copy(H1[1:-1, 1:-1])

# %%
#parameter
#binding rate
alpha1 = 6.0
alpha2 = 10.0
alpha5 = 8.0
alpha6 = 6.0
#basal decay
mu0 = 0.5 ###
mu1 = 1.0
mu2 = 1.0
mu4 = 0.1
mu5 = 1.0
mu6 = 0.5
#move to membrane
gamma1 = 2.0
gamma2 = 1.0
#parameter of function
beta1 = 0.1
beta2_1 = 5.0
beta2_2 = 8.0
beta3 = 2.0
beta4_1 = 8.0
beta4_2 = 8.0
#parameter of Hes-1
nu0 = 0.5
nu1 = 5.0
nu2 = 25.0
nu3 = 5.0

# %%
p_a = 0.0
p_b = 0.0
p_g = 0.0

# %%
for l in trange(N-1):
    n1_ave = (Nm1[0:-2, 1:-1] + Nm1[2:,1:-1] + Nm1[1:-1, 0:-2] + Nm1[1:-1, 2:])/4
    n2_ave = (Nm2[0:-2, 1:-1] + Nm2[2:,1:-1] + Nm2[1:-1, 0:-2] + Nm2[1:-1, 2:])/4
    d1_ave = (Dm1[0:-2, 1:-1] + Dm1[2:,1:-1] + Dm1[1:-1, 0:-2] + Dm1[1:-1, 2:])/4
    d2_ave = (Dm2[0:-2, 1:-1] + Dm2[2:,1:-1] + Dm2[1:-1, 0:-2] + Dm2[1:-1, 2:])/4
#Delta in membrane
    Dm1[1:-1, 1:-1] = ud_m(Deltam1, Dm1[1:-1, 1:-1], n1_ave, n2_ave, Dc1[1:-1, 1:-1])
    Dm2[1:-1, 1:-1] = ud_m(Deltam2, Dm2[1:-1, 1:-1], n1_ave, n2_ave, Dc2[1:-1, 1:-1])
#Delta in cytosol
    Dc1[1:-1, 1:-1] = ud_c(Deltac1, Dc1[1:-1, 1:-1], H1[1:-1, 1:-1])
    Dc2[1:-1, 1:-1] = ud_c(Deltac2, Dc2[1:-1, 1:-1], H1[1:-1, 1:-1])
#Hes1
    H1[1:-1, 1:-1] = ud_h(BTHes1, H1[1:-1, 1:-1], I1[1:-1, 1:-1], I2[1:-1, 1:-1])
#Notch in membrane
    Nm1_new = ud_m(Notchm1, Nm1[1:-1, 1:-1], d1_ave, d2_ave, Nc1[1:-1, 1:-1])
    Nm2_new = ud_m(Notchm2, Nm2[1:-1, 1:-1], d1_ave, d2_ave, Nc2[1:-1, 1:-1])
#Notch in cytosol 
    Nc1[1:-1, 1:-1] = ud_c(Notchc1, Nc1[1:-1, 1:-1], I1[1:-1, 1:-1])
    Nc2[1:-1, 1:-1] = ud_c(Notchc2, Nc2[1:-1, 1:-1], I2[1:-1, 1:-1])
#NICD
    I1[1:-1, 1:-1] = ud_i(NICD1, I1[1:-1, 1:-1], d1_ave, d2_ave, Nm1[1:-1, 1:-1])
    I2[1:-1, 1:-1] = ud_i(NICD2, I2[1:-1, 1:-1], d1_ave, d2_ave, Nm2[1:-1, 1:-1])
    Nm1[1:-1, 1:-1] = Nm1_new
    Nm2[1:-1, 1:-1] = Nm2_new
    
h = H1[1:-1, 1:-1]-H_0
hmax = np.max(h)
canc[0, :] = np.count_nonzero(h/hmax > 0.75)

# %%
#p_a = 0.0
p_a = np.random.rand(ver, wid)
#p_b = 0.0
#p_b = np.random.rand(ver, wid)
#p_g = 0.0
#p_g = np.random.rand(ver, wid)

Nm1[1:-1, 1:-1] = e*np.random.rand(ver, wid)
Nm2[1:-1, 1:-1] = e*np.random.rand(ver, wid)
Dm1[1:-1, 1:-1] = e*np.random.rand(ver, wid)
Dm2[1:-1, 1:-1] = e*np.random.rand(ver, wid)
Nc1[1:-1, 1:-1] = e*np.random.rand(ver, wid)
Nc2[1:-1, 1:-1] = e*np.random.rand(ver, wid)
Dc1[1:-1, 1:-1] = e*np.random.rand(ver, wid)
Dc2[1:-1, 1:-1] = e*np.random.rand(ver, wid)
I1[1:-1, 1:-1] = e*np.random.rand(ver, wid)
I2[1:-1, 1:-1] = e*np.random.rand(ver, wid)
H1[1:-1, 1:-1] = e*np.random.rand(ver, wid)
H_0 = copy.copy(H1[1:-1, 1:-1])

# %%
for l in trange(N-1):
    n1_ave = (Nm1[0:-2, 1:-1] + Nm1[2:,1:-1] + Nm1[1:-1, 0:-2] + Nm1[1:-1, 2:])/4
    n2_ave = (Nm2[0:-2, 1:-1] + Nm2[2:,1:-1] + Nm2[1:-1, 0:-2] + Nm2[1:-1, 2:])/4
    d1_ave = (Dm1[0:-2, 1:-1] + Dm1[2:,1:-1] + Dm1[1:-1, 0:-2] + Dm1[1:-1, 2:])/4
    d2_ave = (Dm2[0:-2, 1:-1] + Dm2[2:,1:-1] + Dm2[1:-1, 0:-2] + Dm2[1:-1, 2:])/4
#Delta in membrane
    Dm1[1:-1, 1:-1] = ud_m(Deltam1, Dm1[1:-1, 1:-1], n1_ave, n2_ave, Dc1[1:-1, 1:-1])
    Dm2[1:-1, 1:-1] = ud_m(Deltam2, Dm2[1:-1, 1:-1], n1_ave, n2_ave, Dc2[1:-1, 1:-1])
#Delta in cytosol
    Dc1[1:-1, 1:-1] = ud_c(Deltac1, Dc1[1:-1, 1:-1], H1[1:-1, 1:-1])
    Dc2[1:-1, 1:-1] = ud_c(Deltac2, Dc2[1:-1, 1:-1], H1[1:-1, 1:-1])
#Hes1
    H1[1:-1, 1:-1] = ud_h(BTHes1, H1[1:-1, 1:-1], I1[1:-1, 1:-1], I2[1:-1, 1:-1])
#Notch in membrane
    Nm1_new = ud_m(Notchm1, Nm1[1:-1, 1:-1], d1_ave, d2_ave, Nc1[1:-1, 1:-1])
    Nm2_new = ud_m(Notchm2, Nm2[1:-1, 1:-1], d1_ave, d2_ave, Nc2[1:-1, 1:-1])
#Notch in cytosol 
    Nc1[1:-1, 1:-1] = ud_c(Notchc1, Nc1[1:-1, 1:-1], I1[1:-1, 1:-1])
    Nc2[1:-1, 1:-1] = ud_c(Notchc2, Nc2[1:-1, 1:-1], I2[1:-1, 1:-1])
#NICD
    I1[1:-1, 1:-1] = ud_i(NICD1, I1[1:-1, 1:-1], d1_ave, d2_ave, Nm1[1:-1, 1:-1])
    I2[1:-1, 1:-1] = ud_i(NICD2, I2[1:-1, 1:-1], d1_ave, d2_ave, Nm2[1:-1, 1:-1])
    Nm1[1:-1, 1:-1] = Nm1_new
    Nm2[1:-1, 1:-1] = Nm2_new

h = H1[1:-1, 1:-1]-H_0
hmax = np.max(h)
canc[1, 0:2] = np.count_nonzero(h/hmax > 0.75)

# %%
nm1_c = copy.copy(Nm1[1:-1, 1:-1])
nm2_c = copy.copy(Nm2[1:-1, 1:-1])
dm1_c = copy.copy(Dm1[1:-1, 1:-1])
dm2_c = copy.copy(Dm2[1:-1, 1:-1])
i1_c = copy.copy(I1[1:-1, 1:-1])
i2_c = copy.copy(I2[1:-1, 1:-1])
nc1_c = copy.copy(Nc1[1:-1, 1:-1])
nc2_c = copy.copy(Nc2[1:-1, 1:-1])
dc1_c = copy.copy(Dc1[1:-1, 1:-1])
dc2_c = copy.copy(Dc2[1:-1, 1:-1])
h_c = copy.copy(H1[1:-1, 1:-1])

# %%
Nm1[1:-1, 1:-1] = nm1_c
Nm2[1:-1, 1:-1] = nm2_c
Dm1[1:-1, 1:-1] = dm1_c
Dm2[1:-1, 1:-1] = dm2_c
Nc1[1:-1, 1:-1] = nc1_c
Nc2[1:-1, 1:-1] = nc2_c
Dc1[1:-1, 1:-1] = dc1_c
Dc2[1:-1, 1:-1] = dc2_c
I1[1:-1, 1:-1] = i1_c
I2[1:-1, 1:-1] = i2_c
H1[1:-1, 1:-1] = h_c

# %%
#p_a = 0.0
#p_a = np.random.rand(ver, wid)
#p_b = 0.0
p_b = np.random.rand(ver, wid)
p_g = 0.0
#p_g = np.random.rand(ver, wid)

# %%
for l in trange(N1-1):
    n1_ave = (Nm1[0:-2, 1:-1] + Nm1[2:,1:-1] + Nm1[1:-1, 0:-2] + Nm1[1:-1, 2:])/4
    n2_ave = (Nm2[0:-2, 1:-1] + Nm2[2:,1:-1] + Nm2[1:-1, 0:-2] + Nm2[1:-1, 2:])/4
    d1_ave = (Dm1[0:-2, 1:-1] + Dm1[2:,1:-1] + Dm1[1:-1, 0:-2] + Dm1[1:-1, 2:])/4
    d2_ave = (Dm2[0:-2, 1:-1] + Dm2[2:,1:-1] + Dm2[1:-1, 0:-2] + Dm2[1:-1, 2:])/4
#Delta in membrane
    Dm1[1:-1, 1:-1] = ud_m(Deltam1, Dm1[1:-1, 1:-1], n1_ave, n2_ave, Dc1[1:-1, 1:-1])
    Dm2[1:-1, 1:-1] = ud_m(Deltam2, Dm2[1:-1, 1:-1], n1_ave, n2_ave, Dc2[1:-1, 1:-1])
#Delta in cytosol
    Dc1[1:-1, 1:-1] = ud_c(Deltac1, Dc1[1:-1, 1:-1], H1[1:-1, 1:-1])
    Dc2[1:-1, 1:-1] = ud_c(Deltac2, Dc2[1:-1, 1:-1], H1[1:-1, 1:-1])
#Hes1
    H1[1:-1, 1:-1] = ud_h(BTHes1, H1[1:-1, 1:-1], I1[1:-1, 1:-1], I2[1:-1, 1:-1])
#Notch in membrane
    Nm1_new = ud_m(Notchm1, Nm1[1:-1, 1:-1], d1_ave, d2_ave, Nc1[1:-1, 1:-1])
    Nm2_new = ud_m(Notchm2, Nm2[1:-1, 1:-1], d1_ave, d2_ave, Nc2[1:-1, 1:-1])
#Notch in cytosol 
    Nc1[1:-1, 1:-1] = ud_c(Notchc1, Nc1[1:-1, 1:-1], I1[1:-1, 1:-1])
    Nc2[1:-1, 1:-1] = ud_c(Notchc2, Nc2[1:-1, 1:-1], I2[1:-1, 1:-1])
#NICD
    I1[1:-1, 1:-1] = ud_i(NICD1, I1[1:-1, 1:-1], d1_ave, d2_ave, Nm1[1:-1, 1:-1])
    I2[1:-1, 1:-1] = ud_i(NICD2, I2[1:-1, 1:-1], d1_ave, d2_ave, Nm2[1:-1, 1:-1])
    Nm1[1:-1, 1:-1] = Nm1_new
    Nm2[1:-1, 1:-1] = Nm2_new
  
h = H1[1:-1, 1:-1]-H_0
hmax = np.max(h)
canc[2, 0] = np.count_nonzero(h/hmax > 0.75)

# %%
#p_a = 0.0
#p_a = np.random.rand(ver, wid)
#p_b = 0.0
#p_b = np.random.rand(ver, wid)
#p_g = 0.0
p_g = np.random.rand(ver, wid)

# %%
for l in trange(N1-1):
    n1_ave = (Nm1[0:-2, 1:-1] + Nm1[2:,1:-1] + Nm1[1:-1, 0:-2] + Nm1[1:-1, 2:])/4
    n2_ave = (Nm2[0:-2, 1:-1] + Nm2[2:,1:-1] + Nm2[1:-1, 0:-2] + Nm2[1:-1, 2:])/4
    d1_ave = (Dm1[0:-2, 1:-1] + Dm1[2:,1:-1] + Dm1[1:-1, 0:-2] + Dm1[1:-1, 2:])/4
    d2_ave = (Dm2[0:-2, 1:-1] + Dm2[2:,1:-1] + Dm2[1:-1, 0:-2] + Dm2[1:-1, 2:])/4
#Delta in membrane
    Dm1[1:-1, 1:-1] = ud_m(Deltam1, Dm1[1:-1, 1:-1], n1_ave, n2_ave, Dc1[1:-1, 1:-1])
    Dm2[1:-1, 1:-1] = ud_m(Deltam2, Dm2[1:-1, 1:-1], n1_ave, n2_ave, Dc2[1:-1, 1:-1])
#Delta in cytosol
    Dc1[1:-1, 1:-1] = ud_c(Deltac1, Dc1[1:-1, 1:-1], H1[1:-1, 1:-1])
    Dc2[1:-1, 1:-1] = ud_c(Deltac2, Dc2[1:-1, 1:-1], H1[1:-1, 1:-1])
#Hes1
    H1[1:-1, 1:-1] = ud_h(BTHes1, H1[1:-1, 1:-1], I1[1:-1, 1:-1], I2[1:-1, 1:-1])
#Notch in membrane
    Nm1_new = ud_m(Notchm1, Nm1[1:-1, 1:-1], d1_ave, d2_ave, Nc1[1:-1, 1:-1])
    Nm2_new = ud_m(Notchm2, Nm2[1:-1, 1:-1], d1_ave, d2_ave, Nc2[1:-1, 1:-1])
#Notch in cytosol 
    Nc1[1:-1, 1:-1] = ud_c(Notchc1, Nc1[1:-1, 1:-1], I1[1:-1, 1:-1])
    Nc2[1:-1, 1:-1] = ud_c(Notchc2, Nc2[1:-1, 1:-1], I2[1:-1, 1:-1])
#NICD
    I1[1:-1, 1:-1] = ud_i(NICD1, I1[1:-1, 1:-1], d1_ave, d2_ave, Nm1[1:-1, 1:-1])
    I2[1:-1, 1:-1] = ud_i(NICD2, I2[1:-1, 1:-1], d1_ave, d2_ave, Nm2[1:-1, 1:-1])
    Nm1[1:-1, 1:-1] = Nm1_new
    Nm2[1:-1, 1:-1] = Nm2_new

h = H1[1:-1, 1:-1]-H_0
hmax = np.max(h)
canc[3, 0] = np.count_nonzero(h/hmax > 0.75)

# %%
canc = canc/canc[0, 0]

# %%
fig4 = plt.figure(figsize=(10,5))
ax4 = fig4.add_subplot(111)
label=["untreated", "first", "second", "third"]
ax4.set_ylim([0.5, 1.3])
ax4.plot(label, canc[:, 0], marker='o', markersize=10, label="Tp4-Tp5-Tp6")
ax4.plot(label, canc[:, 1], marker='o', markersize=10, label="Tp4-Tp6-Tp5")
ax4.plot(label, canc[:, 2], marker='o', markersize=10, label="Tp5-Tp4-Tp6")
ax4.plot(label, canc[:, 3], marker='o', markersize=10, label="Tp5-Tp6-Tp4")
ax4.plot(label, canc[:, 4], marker='o', markersize=10, label="Tp6-Tp4-Tp5")
ax4.plot(label, canc[:, 5], marker='o', markersize=10, label="Tp6-Tp5-Tp4")
ax4.legend(loc="best", fontsize=10)
fig4.savefig("enh_sametime_BT.png")
plt.show()
