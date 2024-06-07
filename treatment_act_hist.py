#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:11:49 2023

@author: tamaki
"""

#Notch1-DLL1-NICD1_Notchc1_DLLc1_Notch2-DLL2-NICD2_Notchc2_DLLc2_hes1 model
#graph
import matplotlib.pyplot as plt
import numpy as np

N = 40000 #時間刻み数
T = 500 #最終時刻
s = T/N #時間の刻み幅
ver = 100 #細胞の数（たて）
wid = 100 #細胞の数（よこ）
ITVL = int(N/10)
t = np.linspace(0, T, N+1)

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
beta2_2 = 5.0
beta3 = 2.0
beta4_1 = 8.0
beta4_2 = 8.0
#parameter of Hes-1
nu0 = 0.5
nu1 = 5.0
nu2 = 25.0
nu3 = 5.0

p_g = 0.0
#p_g = np.random.rand(ver+2, wid+2)

p_a = 0.0
#p_a = np.random.rand(ver+2, wid+2)

p_b = 0.0
#p_b = np.random.rand(ver+2, wid+2)

#アニメーションの準備
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

from tqdm import trange
#culculate of untreated
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
    
counter = np.zeros(11)
for i in range(11):
    counter[i] = np.count_nonzero((5.0*i < H1[1:-1, 1:-1]) & (H1[1:-1, 1:-1] < 5.0*(i+1)))/(ver*wid)
fig_a = plt.figure(figsize=(10,5))
ax_a = fig_a.add_subplot(111)
label = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
ax_a.set_xlim(0, 55)
ax_a.bar(label+2.5, counter, width=4.8)
fig_a.savefig("count-cell_ut_nocancer_LC.png")

'''
fig1 = plt.figure(figsize=(10, 5))
ax1 = fig1.add_subplot(121)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.imshow(H1[1:-1, 1:-1], interpolation='nearest', vmin=0, vmax=55, cmap='jet')
ax2 = fig1.add_subplot(122)
ax2.hist(h_lv, bins=11, range=(0, 55), rwidth=0.9)
ax2.set_ylim([0, 4800])
#fig1.savefig("im_act_papbpgLC.png")

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
Nm1[1:-1, 1:-1] = Nm1[1:-1, 1:-1] + Nm1_ini
Nm2[1:-1, 1:-1] = Nm2[1:-1, 1:-1] + Nm2_ini
Dm1[1:-1, 1:-1] = Dm1[1:-1, 1:-1] + Dm1_ini
Dm2[1:-1, 1:-1] = Dm2[1:-1, 1:-1] + Dm2_ini
Nc1[1:-1, 1:-1] = Nc1[1:-1, 1:-1] + Nc1_ini
Nc2[1:-1, 1:-1] = Nc2[1:-1, 1:-1] + Nc2_ini
Dc1[1:-1, 1:-1] = Dc1[1:-1, 1:-1] + Dc1_ini
Dc2[1:-1, 1:-1] = Dc2[1:-1, 1:-1] + Dc2_ini
I1[1:-1, 1:-1] = I1[1:-1, 1:-1] + I1_ini
I2[1:-1, 1:-1] = I2[1:-1, 1:-1] + I2_ini
H1[1:-1, 1:-1] = H1[1:-1, 1:-1] + H1_ini

for l in range(N-1):
    n1_ave[1:-1, 1:-1] = (Nm1[0:-2, 1:-1] + Nm1[2:,1:-1] + Nm1[1:-1, 0:-2] + Nm1[1:-1, 2:])/4
    n2_ave[1:-1, 1:-1] = (Nm2[0:-2, 1:-1] + Nm2[2:,1:-1] + Nm2[1:-1, 0:-2] + Nm2[1:-1, 2:])/4
    d1_ave[1:-1, 1:-1] = (Dm1[0:-2, 1:-1] + Dm1[2:,1:-1] + Dm1[1:-1, 0:-2] + Dm1[1:-1, 2:])/4
    d2_ave[1:-1, 1:-1] = (Dm2[0:-2, 1:-1] + Dm2[2:,1:-1] + Dm2[1:-1, 0:-2] + Dm2[1:-1, 2:])/4
#Notch in membrane      
    r11 = Notchm1(Nm1,  d1_ave, d2_ave, Nc1)
    r12 = Notchm1(Nm1+(s/2)*r11, d1_ave, d2_ave, Nc1)
    r13 = Notchm1(Nm1+(s/2)*r12, d1_ave, d2_ave, Nc1)
    r14 = Notchm1(Nm1+s*r13, d1_ave, d2_ave, Nc1)
    Nm1_new = Nm1 + (s/6)*(r11 + 2*r12 + 2*r13 + r14)
    r61 = Notchm2(Nm2, d1_ave, d2_ave, Nc2)
    r62 = Notchm2(Nm2+(s/2)*r61, d1_ave, d2_ave, Nc2)
    r63 = Notchm2(Nm2+(s/2)*r62, d1_ave, d2_ave, Nc2)
    r64 = Notchm2(Nm2+s*r63, d1_ave, d2_ave, Nc2)
    Nm2_new = Nm2 + (s/6)*(r61 + 2*r62 + 2*r63 + r64)
#Delta in membrane
    r21 = Deltam1(Dm1, n1_ave, n2_ave, Dc1)
    r22 = Deltam1(Dm1+(s/2)*r21, n1_ave, n2_ave, Dc1)
    r23 = Deltam1(Dm1+(s/2)*r22, n1_ave, n2_ave, Dc1)
    r24 = Deltam1(Dm1+s*r23, n1_ave, n2_ave, Dc1)
    Dm1_new = Dm1 + (s/6)*(r21 + 2*r22 + 2*r23 + r24)
    r71 = Deltam2(Dm2, n1_ave, n2_ave, Dc2)
    r72 = Deltam2(Dm2+(s/2)*r71, n1_ave, n2_ave, Dc2)
    r73 = Deltam2(Dm2+(s/2)*r72, n1_ave, n2_ave, Dc2)
    r74 = Deltam2(Dm2+s*r73, n1_ave, n2_ave, Dc2)
    Dm2_new = Dm2 + (s/6)*(r71 + 2*r72 + 2*r73 + r74)
#NICD    
    r31 = NICD1(I1, d1_ave, d2_ave, Nm1)
    r32 = NICD1(I1+(s/2)*r31, d1_ave, d2_ave, Nm1)
    r33 = NICD1(I1+(s/2)*r32, d1_ave, d2_ave, Nm1)
    r34 = NICD1(I1+s*r33, d1_ave, d2_ave, Nm1)
    I1_new = I1 + (s/6)*(r31 + 2*r32 + 2*r33 + r34)
    r81 = NICD2(I2, d1_ave, d2_ave, Nm2)
    r82 = NICD2(I2+(s/2)*r81, d1_ave, d2_ave, Nm2)
    r83 = NICD2(I2+(s/2)*r82, d1_ave, d2_ave, Nm2)
    r84 = NICD2(I2+s*r83, d1_ave, d2_ave, Nm2)
    I2_new = I2 + (s/6)*(r81 + 2*r82 + 2*r83 + r84)
#Notch in cytosol   
    r41 = Notchc1(Nc1, I1)
    r42 = Notchc1(Nc1+(s/2)*r41, I1)
    r43 = Notchc1(Nc1+(s/2)*r42, I1)
    r44 = Notchc1(Nc1+s*r43, I1)
    Nc1_new = Nc1 + (s/6)*(r41 + 2*r42 + 2*r43 + r44)
    r91 = Notchc2(Nc2, I2)
    r92 = Notchc2(Nc2+(s/2)*r91, I2)
    r93 = Notchc2(Nc2+(s/2)*r92, I2)
    r94 = Notchc2(Nc2+s*r93, I2)
    Nc2_new = Nc2 + (s/6)*(r91 + 2*r92 + 2*r93 + r94)
#Delta in cytosol
    r51 = Deltac1(Dc1, H1)
    r52 = Deltac1(Dc1+(s/2)*r51, H1)
    r53 = Deltac1(Dc1+(s/2)*r52, H1)
    r54 = Deltac1(Dc1+s*r53, H1)
    Dc1_new = Dc1 + (s/6)*(r51 + 2*r52 + 2*r53 + r54)
    r101 = Deltac2(Dc2, H1)
    r102 = Deltac2(Dc2+(s/2)*r101, H1)
    r103 = Deltac2(Dc2+(s/2)*r102, H1)
    r104 = Deltac2(Dc2+s*r103, H1)
    Dc2_new = Dc2 + (s/6)*(r101 + 2*r102 + 2*r103 + r104)
#Hes1
    r111 = BTHes1(H1, I1, I2)
    r112 = BTHes1(H1+(s/2)*r111, I1, I2)
    r113 = BTHes1(H1+(s/2)*r112, I1, I2)
    r114 = BTHes1(H1+s*r113, I1, I2)
    H1_new = H1 + (s/6)*(r111 + 2*r112 + 2*r113 + r114)
    Nm1[1:-1, 1:-1] = Nm1_new[1:-1, 1:-1]
    Nm2[1:-1, 1:-1] = Nm2_new [1:-1, 1:-1]
    Dm1[1:-1, 1:-1] = Dm1_new[1:-1, 1:-1]
    Dm2[1:-1, 1:-1] = Dm2_new[1:-1, 1:-1]
    Nc1[1:-1, 1:-1] = Nc1_new[1:-1, 1:-1]
    Nc2[1:-1, 1:-1] = Nc2_new[1:-1, 1:-1]
    Dc1[1:-1, 1:-1] = Dc1_new[1:-1, 1:-1]
    Dc2[1:-1, 1:-1] = Dc2_new[1:-1, 1:-1]
    I1[1:-1, 1:-1] = I1_new[1:-1, 1:-1]
    I2[1:-1, 1:-1] = I2_new[1:-1, 1:-1] 
    H1[1:-1, 1:-1] = H1_new[1:-1, 1:-1]
    if (l+1)%ITVL == 0:
        print(100*(l+1)/N, "%")
h_lv = np.ravel(H1[1:-1, 1:-1])

fig2 = plt.figure(figsize=(10, 5))
ax3 = fig2.add_subplot(121)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.imshow(H1[1:-1, 1:-1], interpolation='nearest', vmin=0, vmax=55, cmap='jet')
ax4 = fig2.add_subplot(122)
ax4.hist(h_lv, bins=11, range=(0, 55), rwidth=0.9)
ax4.set_ylim([0, 4800])
#fig2.savefig("im_act_papbpgBT.png")
'''
plt.show()
