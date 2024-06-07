#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 17:05:17 2022

@author: wakamototamaki
"""

#Notch1-DLL1-NICD1_Notchc1_DLLc1_Notch2-DLL2-NICD2_Notchc2_DLLc2_hes1 model
#2cells
#graph
# %%
import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np

# %%
N = 50000 #時間刻み数
T = 500 #最終時刻
s = T/N #時間の刻み幅
ver = 1 #細胞の数（たて）
wid = 2 #細胞の数（よこ）
ITVL = int(N/10)

# %%
#sequence
t = np.linspace(0, T, N+1)
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
#parameter
#binding rate
alpha1 = 6.0
alpha2s = np.linspace(0.0, 15.0, 16)
alpha5 = 8.0
alpha6 = 6.0
#basal decay
mu0 = 0.5
mu1 = 1.0
mu2 = 1.0
mu4 = 0.1
mu5 = 1.0
mu6 = 0.5
#move to membrane
gammaN = 2.0
gammaD = 1.0
#parameter of function
beta1 = 0.1
beta2_1 = 8.0
beta2_2 = 5.0
beta3 = 2.0
beta4_1 = 8.0
beta4_2 = 8.0
#parameter of Hes-1
nu0 = 0.5
nu1 = 5.0
nu2 = 25.0
nu3 = 5.0

change = alpha2s
'''
alpha1s = np.linspace(0.0, 15.0, 11)
mu0s = np.linspace(0.1, 1.1, 11)[4]
mu125s = np.linspace(0.2, 2.2, 11)[4]
mu4s = np.linspace(0.02, 0.22, 11)[4]
mu6s = np.linspace(0.1, 1.1, 11)[4]
gammaNs = np.linspace(0.0, 4.0, 11)
gammaDs = np.linspace(0.0, 2.0, 11)
beta1s = np.linspace(0.0, 2.0, 11)
beta2_1s = np.linspace(0.0, 10.0, 11)
beta3s = np.linspace(0.4, 4.4, 11)[4]
beta4_1s = np.linspace(0.0, 16.0, 11)
nu0s = np.linspace(0.0, 1.0, 11)
nu1s = np.linspace(0.0, 10.0, 11)
nu2s = np.linspace(0.0, 50.0, 11)
'''

# %%
#initial condition
e = 0.01
y_init = e*np.random.rand(11, ver, wid)
m = change.size

lc1 = np.zeros(m) #Model-LC_cell1
lc2 = np.zeros(m) #Model-LC_cell2
bt1 = np.zeros(m) #Model-BT_cell1
bt2 = np.zeros(m) #Model-BT_cell2

# %%
#activation and inhibition of Hes-1 by NICD
def hes(x1, x2): #x1:activation x2:inhibition
    return nu0 + (nu2*(x1**2)/(nu1 + x1**2))*(1 - ((x2**2)/(nu3 + x2**2)))

#ODEs
def Notchm1(x, d_ave1, d_ave2, nc1):
    return -alpha1*d_ave1*x - alpha2*d_ave2*x - mu1*x + gammaN*nc1
def Notchm2(x, d_ave1, d_ave2, nc2):
    return -alpha5*d_ave1*x - alpha6*d_ave2*x - mu1*x + gammaN*nc2
    
def Deltam1(x, n_ave1, n_ave2, dc1):
    return -alpha1*n_ave1*x -alpha5*n_ave2*x - mu2*x + gammaD*dc1
def Deltam2(x, n_ave1, n_ave2, dc2):
    return -alpha2*n_ave1*x - alpha6*n_ave2*x - mu2*x + gammaD*dc2
    
def NICD1(x, d_ave1, d_ave2, nm1):
    return alpha1*d_ave1*nm1 + alpha2*d_ave2*nm1 - mu4*x
def NICD2(x, d_ave1, d_ave2, nm2):
    return alpha5*d_ave1*nm2 + alpha6*d_ave2*nm2  - mu4*x
    
def Notchc1(x, i1):
    return beta2_1*(i1**2)/(beta1 + i1**2) - (mu5 + gammaN)*x
def Notchc2(x, i2):
    return beta2_2*(i2**2)/(beta1 + i2**2) - (mu5 + gammaN)*x
    
def Deltac1(x, hh):
    return beta4_1/(beta3 + hh**2) - (mu6 + gammaD)*x
def Deltac2(x, hh):
    return beta4_2/(beta3 + hh**2) - (mu6 + gammaD)*x

def LCHes1(x, i1, i2):
    return hes(i1, i2)-mu0*x
def BTHes1(x, i1, i2):
    return hes(i2, i1)-mu0*x

# %%
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
#Model-LC
for q in trange(m):
    alpha2 = change[q]
    Nm1[1:-1, 1:-1] = y_init[0, :, :]
    Nm2[1:-1, 1:-1] = y_init[1, :, :]
    Dm1[1:-1, 1:-1] = y_init[2, :, :]
    Dm2[1:-1, 1:-1] = y_init[3, :, :]
    Nc1[1:-1, 1:-1] = y_init[4, :, :]
    Nc2[1:-1, 1:-1] = y_init[5, :, :]
    Dc1[1:-1, 1:-1] = y_init[6, :, :]
    Dc2[1:-1, 1:-1] = y_init[7, :, :]
    I1[1:-1, 1:-1]  = y_init[8, :, :]
    I2[1:-1, 1:-1]  = y_init[9, :, :]
    H1[1:-1, 1:-1]  = y_init[10, :, :]
    for l in range(N-1):
        n1_ave = Nm1[1, 0:2] + Nm1[1, 2:]
        n2_ave = Nm2[1, 0:2] + Nm2[1, 2:]
        d1_ave = Dm1[1, 0:2] + Dm1[1, 2:]
        d2_ave = Dm2[1, 0:2] + Dm2[1, 2:]
    #Delta in membrane
        Dm1[1:-1, 1:-1] = ud_m(Deltam1, Dm1[1:-1, 1:-1], n1_ave, n2_ave, Dc1[1:-1, 1:-1])
        Dm2[1:-1, 1:-1] = ud_m(Deltam2, Dm2[1:-1, 1:-1], n1_ave, n2_ave, Dc2[1:-1, 1:-1])
    #Delta in cytosol
        Dc1[1:-1, 1:-1] = ud_c(Deltac1, Dc1[1:-1, 1:-1], H1[1:-1, 1:-1])
        Dc2[1:-1, 1:-1] = ud_c(Deltac2, Dc2[1:-1, 1:-1], H1[1:-1, 1:-1])
    #Hes1
        H1[1:-1, 1:-1] = ud_h(LCHes1, H1[1:-1, 1:-1], I1[1:-1, 1:-1], I2[1:-1, 1:-1])
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
    lc1[q] = H1[1, 1]
    lc2[q] = H1[1, 2]

# %%
#Set the result of the calculation with the parameters of the standard to 1
lc1 = lc1/lc1[10]
lc2 = lc2/lc2[10]

# %%
#Model-BT
beta2_1 = 5.0
beta2_2 = 8.0
for q in trange(m):
    alpha2 = change[q]
    Nm1[1:-1, 1:-1] = y_init[0, :, :]
    Nm2[1:-1, 1:-1] = y_init[1, :, :]
    Dm1[1:-1, 1:-1] = y_init[2, :, :]
    Dm2[1:-1, 1:-1] = y_init[3, :, :]
    Nc1[1:-1, 1:-1] = y_init[4, :, :]
    Nc2[1:-1, 1:-1] = y_init[5, :, :]
    Dc1[1:-1, 1:-1] = y_init[6, :, :]
    Dc2[1:-1, 1:-1] = y_init[7, :, :]
    I1[1:-1, 1:-1]  = y_init[8, :, :]
    I2[1:-1, 1:-1]  = y_init[9, :, :]
    H1[1:-1, 1:-1]  = y_init[10, :, :]
    for l in range(N-1):
        n1_ave = Nm1[1, 0:2] + Nm1[1, 2:]
        n2_ave = Nm2[1, 0:2] + Nm2[1, 2:]
        d1_ave = Dm1[1, 0:2] + Dm1[1, 2:]
        d2_ave = Dm2[1, 0:2] + Dm2[1, 2:]
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
    bt1[q] = H1[1, 1]
    bt2[q] = H1[1, 2]

# %%
#Set the result of the calculation with the parameters of the standard to 1
bt1 = bt1/bt1[10]
bt2 = bt2/bt2[10]

# %%
#making figure    
fig1 = plt.figure(figsize=(8,5))
ax1 = fig1.add_subplot(111)
ax1.plot(change, lc1, marker='o', markersize=10, color="g", label="Model-LC cell1")
ax1.plot(change, lc2, marker='o', mfc="white", markersize=10, color="g", label="Model-LC cell2")
ax1.plot(change, bt1, marker='o', color="r", markersize=10, label="Model-BT cell1")
ax1.plot(change, bt2, marker='o', mfc="white", color="r", markersize=10, label="Model-BT cell2")
ax1.set_xlabel(r"$\alpha_{2}$", fontsize=15)
ax1.set_ylabel(r"HES1$(t^*)$", fontsize=15)
ax1.legend(loc="best", fontsize=10)
ax1.set_xlim([change[0], change[m-1]])
ax1.set_ylim([0, 4.0])
fig1.savefig("2cell_hes_a2.png")
plt.show()
# %%
