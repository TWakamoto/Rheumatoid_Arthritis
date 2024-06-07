#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:26:08 2023

@author: wakamototamaki
"""

#Notch1-DLL1-NICD1_Notchc1_DLLc1_Notch2-DLL2-NICD2_Notchc2_DLLc2_hes1 model
#graph
import matplotlib.pyplot as plt
import numpy as np

N = 80000 #時間刻み数
T = 1000 #最終時刻
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
Nm1_ini = e*np.random.rand(ver, wid)
Nm2_ini = e*np.random.rand(ver, wid)
Dm1_ini = e*np.random.rand(ver, wid)
Dm2_ini = e*np.random.rand(ver, wid)
Nc1_ini = e*np.random.rand(ver, wid)
Nc2_ini = e*np.random.rand(ver, wid)
Dc1_ini = e*np.random.rand(ver, wid)
Dc2_ini = e*np.random.rand(ver, wid)
I1_ini = e*np.random.rand(ver, wid)
I2_ini = e*np.random.rand(ver, wid)
H1_ini = e*np.random.rand(ver, wid)
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

n1_ave = np.zeros((ver+2, wid+2))
n2_ave = np.zeros((ver+2, wid+2))
d1_ave = np.zeros((ver+2, wid+2))
d2_ave = np.zeros((ver+2, wid+2))

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

#p_1 = 0.0
p_1 = np.random.rand(ver+2, wid+2)

#p_23 = 0.0
p_23 = np.random.rand(ver+2, wid+2)

#p_b = 0.0
p_b = np.random.rand(ver+2, wid+2)

#アニメーションの準備
#function
#activation and inhibition of Hes-1 by NICD
def hes(x1, x2):
    return nu0 + (nu2*(x1**2)/(nu1 + x1**2))*(1 - ((x2**2)/(nu3 + x2**2)))

#ODEs
def Notchm1(x, d_ave1, d_ave2, nc1):
    return -alpha1*d_ave1*x - alpha2*d_ave2*x - mu1*x + (1-p_1)*gamma1*nc1
def Notchm2(x, d_ave1, d_ave2, nc2):
    return -alpha5*d_ave1*x - alpha6*d_ave2*x - mu1*x + (1-p_1)*gamma1*nc2
    
def Deltam1(x, n_ave1, n_ave2, dc1):
    return -alpha1*n_ave1*x -alpha5*n_ave2*x - mu2*x + gamma2*dc1
def Deltam2(x, n_ave1, n_ave2, dc2):
    return -alpha2*n_ave1*x - alpha6*n_ave2*x - mu2*x + gamma2*dc2
    
def NICD1(x, d_ave1, d_ave2, nm1):
    return (1-p_23)*(alpha1*d_ave1*nm1 + alpha2*d_ave2*nm1) - mu4*x    
def NICD2(x, d_ave1, d_ave2, nm2):
    return (1-p_23)*(alpha5*d_ave1*nm2 + alpha6*d_ave2*nm2)  - mu4*x
    
def Notchc1(x, i1):
    return (1-p_b)*beta2_1*(i1**2)/(beta1 + i1**2) - (mu5 + (1-p_1)*gamma1)*x    
def Notchc2(x, i2):
    return (1-p_b)*beta2_2*(i2**2)/(beta1 + i2**2) - (mu5 + (1-p_1)*gamma1)*x
    
def Deltac1(x, hh):
    return beta4_1/(beta3 + hh**2) - (mu6 + gamma2)*x    
def Deltac2(x, hh):
    return beta4_2/(beta3 + hh**2) - (mu6 + gamma2)*x

def LCHes1(x, i1, i2):
    return hes(i1, i2)-mu0*x
def BTHes1(x, i1, i2):
    return hes(i2, i1)-mu0*x

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
    
    r61 = Notchm2(Nm2, d1_ave, d2_ave, Nc2)
    r62 = Notchm2(Nm2+(s/2)*r61, d1_ave, d2_ave, Nc2)
    r63 = Notchm2(Nm2+(s/2)*r62, d1_ave, d2_ave, Nc2)
    r64 = Notchm2(Nm2+s*r63, d1_ave, d2_ave, Nc2)
#Delta in membrane
    r21 = Deltam1(Dm1, n1_ave, n2_ave, Dc1)
    r22 = Deltam1(Dm1+(s/2)*r21, n1_ave, n2_ave, Dc1)
    r23 = Deltam1(Dm1+(s/2)*r22, n1_ave, n2_ave, Dc1)
    r24 = Deltam1(Dm1+s*r23, n1_ave, n2_ave, Dc1)
    
    r71 = Deltam2(Dm2, n1_ave, n2_ave, Dc2)
    r72 = Deltam2(Dm2+(s/2)*r71, n1_ave, n2_ave, Dc2)
    r73 = Deltam2(Dm2+(s/2)*r72, n1_ave, n2_ave, Dc2)
    r74 = Deltam2(Dm2+s*r73, n1_ave, n2_ave, Dc2)
#NICD    
    r31 = NICD1(I1, d1_ave, d2_ave, Nm1)
    r32 = NICD1(I1+(s/2)*r31, d1_ave, d2_ave, Nm1)
    r33 = NICD1(I1+(s/2)*r32, d1_ave, d2_ave, Nm1)
    r34 = NICD1(I1+s*r33, d1_ave, d2_ave, Nm1)
    
    r81 = NICD2(I2, d1_ave, d2_ave, Nm2)
    r82 = NICD2(I2+(s/2)*r81, d1_ave, d2_ave, Nm2)
    r83 = NICD2(I2+(s/2)*r82, d1_ave, d2_ave, Nm2)
    r84 = NICD2(I2+s*r83, d1_ave, d2_ave, Nm2)
#Notch in cytosol   
    r41 = Notchc1(Nc1, I1)
    r42 = Notchc1(Nc1+(s/2)*r41, I1)
    r43 = Notchc1(Nc1+(s/2)*r42, I1)
    r44 = Notchc1(Nc1+s*r43, I1)
    
    r91 = Notchc2(Nc2, I2)
    r92 = Notchc2(Nc2+(s/2)*r91, I2)
    r93 = Notchc2(Nc2+(s/2)*r92, I2)
    r94 = Notchc2(Nc2+s*r93, I2)
#Delta in cytosol
    r51 = Deltac1(Dc1, H1)
    r52 = Deltac1(Dc1+(s/2)*r51, H1)
    r53 = Deltac1(Dc1+(s/2)*r52, H1)
    r54 = Deltac1(Dc1+s*r53, H1)
    
    r101 = Deltac2(Dc2, H1)
    r102 = Deltac2(Dc2+(s/2)*r101, H1)
    r103 = Deltac2(Dc2+(s/2)*r102, H1)
    r104 = Deltac2(Dc2+s*r103, H1)
#Hes1
    r111 = LCHes1(H1, I1, I2)
    r112 = LCHes1(H1+(s/2)*r111, I1, I2)
    r113 = LCHes1(H1+(s/2)*r112, I1, I2)
    r114 = LCHes1(H1+s*r113, I1, I2)
    Nm1[1:-1, 1:-1] = Nm1[1:-1, 1:-1] + (s/6)*(r11[1:-1, 1:-1] + 2*r12[1:-1, 1:-1] + 2*r13[1:-1, 1:-1] + r14[1:-1, 1:-1])
    Nm2[1:-1, 1:-1] = Nm2[1:-1, 1:-1] + (s/6)*(r61[1:-1, 1:-1] + 2*r62[1:-1, 1:-1] + 2*r63[1:-1, 1:-1] + r64[1:-1, 1:-1])
    Dm1[1:-1, 1:-1] = Dm1[1:-1, 1:-1] + (s/6)*(r21[1:-1, 1:-1] + 2*r22[1:-1, 1:-1] + 2*r23[1:-1, 1:-1] + r24[1:-1, 1:-1])
    Dm2[1:-1, 1:-1] = Dm2[1:-1, 1:-1] + (s/6)*(r71[1:-1, 1:-1] + 2*r72[1:-1, 1:-1] + 2*r73[1:-1, 1:-1] + r74[1:-1, 1:-1])
    I1[1:-1, 1:-1] = I1[1:-1, 1:-1] + (s/6)*(r31[1:-1, 1:-1] + 2*r32[1:-1, 1:-1] + 2*r33[1:-1, 1:-1] + r34[1:-1, 1:-1])
    I2[1:-1, 1:-1] = I2[1:-1, 1:-1] + (s/6)*(r81[1:-1, 1:-1] + 2*r82[1:-1, 1:-1] + 2*r83[1:-1, 1:-1] + r84[1:-1, 1:-1])
    Nc1[1:-1, 1:-1] = Nc1[1:-1, 1:-1] + (s/6)*(r41[1:-1, 1:-1] + 2*r42[1:-1, 1:-1] + 2*r43[1:-1, 1:-1] + r44[1:-1, 1:-1])
    Nc2[1:-1, 1:-1] = Nc2[1:-1, 1:-1] + (s/6)*(r91[1:-1, 1:-1] + 2*r92[1:-1, 1:-1] + 2*r93[1:-1, 1:-1] + r94[1:-1, 1:-1])
    Dc1[1:-1, 1:-1] = Dc1[1:-1, 1:-1] + (s/6)*(r51[1:-1, 1:-1] + 2*r52[1:-1, 1:-1] + 2*r53[1:-1, 1:-1] + r54[1:-1, 1:-1])
    Dc2[1:-1, 1:-1] = Dc2[1:-1, 1:-1] + (s/6)*(r101[1:-1, 1:-1] + 2*r102[1:-1, 1:-1] + 2*r103[1:-1, 1:-1] + r104[1:-1, 1:-1])
    H1[1:-1, 1:-1] = H1[1:-1, 1:-1] + (s/6)*(r111[1:-1, 1:-1] + 2*r112[1:-1, 1:-1] + 2*r113[1:-1, 1:-1] + r114[1:-1, 1:-1])

    if (l+1)%ITVL == 0:
        print(100*(l+1)/N, "%")
h_lv = np.ravel(H1[1:-1, 1:-1])

fig1 = plt.figure(figsize=(10, 5))
ax1 = fig1.add_subplot(121)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.imshow(H1[1:-1, 1:-1], interpolation='nearest', vmin=0, vmax=55, cmap='jet')
ax2 = fig1.add_subplot(122)
ax2.hist(h_lv, bins=11, range=(0, 55), rwidth=0.9)
ax2.set_ylim([0, 9800])
#fig1.savefig("im_inh_p1p23pbLC.png")

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

    r61 = Notchm2(Nm2, d1_ave, d2_ave, Nc2)
    r62 = Notchm2(Nm2+(s/2)*r61, d1_ave, d2_ave, Nc2)
    r63 = Notchm2(Nm2+(s/2)*r62, d1_ave, d2_ave, Nc2)
    r64 = Notchm2(Nm2+s*r63, d1_ave, d2_ave, Nc2)
#Delta in membrane
    r21 = Deltam1(Dm1, n1_ave, n2_ave, Dc1)
    r22 = Deltam1(Dm1+(s/2)*r21, n1_ave, n2_ave, Dc1)
    r23 = Deltam1(Dm1+(s/2)*r22, n1_ave, n2_ave, Dc1)
    r24 = Deltam1(Dm1+s*r23, n1_ave, n2_ave, Dc1)

    r71 = Deltam2(Dm2, n1_ave, n2_ave, Dc2)
    r72 = Deltam2(Dm2+(s/2)*r71, n1_ave, n2_ave, Dc2)
    r73 = Deltam2(Dm2+(s/2)*r72, n1_ave, n2_ave, Dc2)
    r74 = Deltam2(Dm2+s*r73, n1_ave, n2_ave, Dc2)
#NICD    
    r31 = NICD1(I1, d1_ave, d2_ave, Nm1)
    r32 = NICD1(I1+(s/2)*r31, d1_ave, d2_ave, Nm1)
    r33 = NICD1(I1+(s/2)*r32, d1_ave, d2_ave, Nm1)
    r34 = NICD1(I1+s*r33, d1_ave, d2_ave, Nm1)

    r81 = NICD2(I2, d1_ave, d2_ave, Nm2)
    r82 = NICD2(I2+(s/2)*r81, d1_ave, d2_ave, Nm2)
    r83 = NICD2(I2+(s/2)*r82, d1_ave, d2_ave, Nm2)
    r84 = NICD2(I2+s*r83, d1_ave, d2_ave, Nm2)
#Notch in cytosol   
    r41 = Notchc1(Nc1, I1)
    r42 = Notchc1(Nc1+(s/2)*r41, I1)
    r43 = Notchc1(Nc1+(s/2)*r42, I1)
    r44 = Notchc1(Nc1+s*r43, I1)

    r91 = Notchc2(Nc2, I2)
    r92 = Notchc2(Nc2+(s/2)*r91, I2)
    r93 = Notchc2(Nc2+(s/2)*r92, I2)
    r94 = Notchc2(Nc2+s*r93, I2)
#Delta in cytosol
    r51 = Deltac1(Dc1, H1)
    r52 = Deltac1(Dc1+(s/2)*r51, H1)
    r53 = Deltac1(Dc1+(s/2)*r52, H1)
    r54 = Deltac1(Dc1+s*r53, H1)

    r101 = Deltac2(Dc2, H1)
    r102 = Deltac2(Dc2+(s/2)*r101, H1)
    r103 = Deltac2(Dc2+(s/2)*r102, H1)
    r104 = Deltac2(Dc2+s*r103, H1)
#Hes1
    r111 = BTHes1(H1, I1, I2)
    r112 = BTHes1(H1+(s/2)*r111, I1, I2)
    r113 = BTHes1(H1+(s/2)*r112, I1, I2)
    r114 = BTHes1(H1+s*r113, I1, I2)
    Nm1[1:-1, 1:-1] = Nm1[1:-1, 1:-1] + (s/6)*(r11[1:-1, 1:-1] + 2*r12[1:-1, 1:-1] + 2*r13[1:-1, 1:-1] + r14[1:-1, 1:-1])
    Nm2[1:-1, 1:-1] = Nm2[1:-1, 1:-1] + (s/6)*(r61[1:-1, 1:-1] + 2*r62[1:-1, 1:-1] + 2*r63[1:-1, 1:-1] + r64[1:-1, 1:-1])
    Dm1[1:-1, 1:-1] = Dm1[1:-1, 1:-1] + (s/6)*(r21[1:-1, 1:-1] + 2*r22[1:-1, 1:-1] + 2*r23[1:-1, 1:-1] + r24[1:-1, 1:-1])
    Dm2[1:-1, 1:-1] = Dm2[1:-1, 1:-1] + (s/6)*(r71[1:-1, 1:-1] + 2*r72[1:-1, 1:-1] + 2*r73[1:-1, 1:-1] + r74[1:-1, 1:-1])
    I1[1:-1, 1:-1] = I1[1:-1, 1:-1] + (s/6)*(r31[1:-1, 1:-1] + 2*r32[1:-1, 1:-1] + 2*r33[1:-1, 1:-1] + r34[1:-1, 1:-1])
    I2[1:-1, 1:-1] = I2[1:-1, 1:-1] + (s/6)*(r81[1:-1, 1:-1] + 2*r82[1:-1, 1:-1] + 2*r83[1:-1, 1:-1] + r84[1:-1, 1:-1])
    Nc1[1:-1, 1:-1] = Nc1[1:-1, 1:-1] + (s/6)*(r41[1:-1, 1:-1] + 2*r42[1:-1, 1:-1] + 2*r43[1:-1, 1:-1] + r44[1:-1, 1:-1])
    Nc2[1:-1, 1:-1] = Nc2[1:-1, 1:-1] + (s/6)*(r91[1:-1, 1:-1] + 2*r92[1:-1, 1:-1] + 2*r93[1:-1, 1:-1] + r94[1:-1, 1:-1])
    Dc1[1:-1, 1:-1] = Dc1[1:-1, 1:-1] + (s/6)*(r51[1:-1, 1:-1] + 2*r52[1:-1, 1:-1] + 2*r53[1:-1, 1:-1] + r54[1:-1, 1:-1])
    Dc2[1:-1, 1:-1] = Dc2[1:-1, 1:-1] + (s/6)*(r101[1:-1, 1:-1] + 2*r102[1:-1, 1:-1] + 2*r103[1:-1, 1:-1] + r104[1:-1, 1:-1])
    H1[1:-1, 1:-1] = H1[1:-1, 1:-1] + (s/6)*(r111[1:-1, 1:-1] + 2*r112[1:-1, 1:-1] + 2*r113[1:-1, 1:-1] + r114[1:-1, 1:-1])
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
ax4.set_ylim([0, 9800])
fig2.savefig("im_inh_p1p23pbBT.png")

plt.show()
