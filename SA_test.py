#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:32:56 2022

@author: wakamototamaki
"""

#Sensitivity Analysis
#first order sensitivity index

import matplotlib.pyplot as plt
import numpy as np

N = 32000 #時間刻み数
T = 800 #最終時刻
s = T/N #時間の刻み幅
ver = 10 #細胞の数（たて）
wid = 10 #細胞の数（よこ）
ITVL = int(N/10)

#ready of eFAST
freq1 = np.asarray([11, 39, 21])
freq2 = np.asarray([39, 11, 21])
freq1 = np.stack([freq1, freq2], axis=0)
M = 4 #interference factor
Nr = 2 #resampling
fmax = max(max(freq1, key=max))
Ns = 2*M*fmax + 1
ds = np.linspace(-np.pi*(Ns-1)/Ns, np.pi*(Ns-1)/Ns, Ns)
def make_par(p_min, p_max, freq, x, psi):
    return p_min + (p_max-p_min)*(1/2 + (1/np.pi)*np.arcsin(np.sin(freq*x + psi)))
psi = 2*np.pi*np.random.rand(2, 3)
y_lc = np.zeros((2, Ns))
A_total = np.zeros((2, M*fmax))
B_total = np.zeros((2, M*fmax))
A_i = np.zeros((2, 3, M))
B_i = np.zeros((2, 3, M))
V_i = np.zeros((2, 3))
V_total = np.zeros(2)

#sequence
Nm1 = np.zeros((2, Ns, ver+2, wid+2)) #Notch1 in membrane
Dm1 = np.zeros((2, Ns, ver+2, wid+2)) #DLL1 in membrane
Nc1 = np.zeros((2, Ns, ver+2, wid+2)) #Notch1 in cytosol
Dc1 = np.zeros((2, Ns, ver+2, wid+2)) #DLL1 in cytosol
I1 = np.zeros((2, Ns, ver+2, wid+2)) #NICD1
Nm2 = np.zeros((2, Ns, ver+2, wid+2)) #Notch2 in membrane
Dm2 = np.zeros((2, Ns, ver+2, wid+2)) #DLL2 in membrane
Nc2 = np.zeros((2, Ns, ver+2, wid+2)) #Notch2 in cytosol
Dc2 = np.zeros((2, Ns, ver+2, wid+2)) #DLL2 in cytosol
I2 = np.zeros((2, Ns, ver+2, wid+2)) #NICD2
H1 = np.zeros((2, Ns, ver+2, wid+2)) #Hes1

Nm1_new = np.zeros((2, Ns, ver+2, wid+2)) #Notch1 in membrane
Dm1_new = np.zeros((2, Ns, ver+2, wid+2)) #DLL1 in membrane
Nc1_new = np.zeros((2, Ns, ver+2, wid+2)) #Notch1 in cytosol
Dc1_new = np.zeros((2, Ns, ver+2, wid+2)) #DLL1 in cytosol
I1_new = np.zeros((2, Ns, ver+2, wid+2)) #NICD1
Nm2_new = np.zeros((2, Ns, ver+2, wid+2)) #Notch2 in membrane
Dm2_new = np.zeros((2, Ns, ver+2, wid+2)) #DLL2 in membrane
Nc2_new = np.zeros((2, Ns, ver+2, wid+2)) #Notch2 in cytosol
Dc2_new = np.zeros((2, Ns, ver+2, wid+2)) #DLL2 in cytosol
I2_new = np.zeros((2, Ns, ver+2, wid+2)) #NICD2
H1_new = np.zeros((2, Ns, ver+2, wid+2)) #Hes1

#initial condition
e = 0.01
Nm1_0 = e*np.random.rand(ver, wid) #Notch1 in membrane
Dm1_0 = e*np.random.rand(ver, wid) #DLL1 in membrane
Nc1_0 = e*np.random.rand(ver, wid) #Notch1 in cytosol
Dc1_0 = e*np.random.rand(ver, wid) #DLL1 in cytosol
I1_0 = e*np.random.rand(ver, wid) #NICD1
Nm2_0 = e*np.random.rand(ver, wid) #Notch2 in membrane
Dm2_0 = e*np.random.rand(ver, wid) #DLL2 in membrane
Nc2_0 = e*np.random.rand(ver, wid) #Notch2 in cytosol
Dc2_0 = e*np.random.rand(ver, wid) #DLL2 in cytosol
I2_0 = e*np.random.rand(ver, wid) #NICD2
H1_0 = e*np.random.rand(ver, wid) #Hes1
Nm1_ini = np.zeros((2, Ns, ver+2, wid+2))
Nm2_ini = np.zeros((2, Ns, ver+2, wid+2))
Dm1_ini = np.zeros((2, Ns, ver+2, wid+2))
Dm2_ini = np.zeros((2, Ns, ver+2, wid+2))
Nc1_ini = np.zeros((2, Ns, ver+2, wid+2))
Nc2_ini = np.zeros((2, Ns, ver+2, wid+2))
Dc1_ini = np.zeros((2, Ns, ver+2, wid+2))
Dc2_ini = np.zeros((2, Ns, ver+2, wid+2))
I1_ini = np.zeros((2, Ns, ver+2, wid+2))
I2_ini = np.zeros((2, Ns, ver+2, wid+2))
H1_ini = np.zeros((2, Ns, ver+2, wid+2))
for i in range(2):
    for j in range(Ns):
        Nm1_ini[i, j, 1:-1 ,1:-1] = Nm1_0
        Nm2_ini[i, j, 1:-1 ,1:-1] = Nm2_0
        Dm1_ini[i, j, 1:-1 ,1:-1] = Dm1_0
        Dm2_ini[i, j, 1:-1 ,1:-1] = Dm2_0
        Nc1_ini[i, j, 1:-1 ,1:-1] = Nc1_0
        Nc2_ini[i, j, 1:-1 ,1:-1] = Nc2_0
        Dc1_ini[i, j, 1:-1 ,1:-1] = Dc1_0
        Dc2_ini[i, j, 1:-1 ,1:-1] = Dc2_0
        I1_ini[i, j, 1:-1 ,1:-1] = I1_0
        I2_ini[i, j, 1:-1 ,1:-1] = I2_0
        H1_ini[i, j, 1:-1 ,1:-1] = H1_0

#effect of neighboring cells
n1_ave = np.zeros((2, Ns, ver+2, wid+2))
n2_ave = np.zeros((2, Ns, ver+2, wid+2))
d1_ave = np.zeros((2, Ns, ver+2, wid+2))
d2_ave = np.zeros((2, Ns, ver+2, wid+2))

#parameter
#binding rate
alpha1 = 6.0
alpha2 = 10.0
alpha5 = 8.0
alpha6 = 6.0
#move to membrane
gamma1 = np.full((2, Ns, ver+2, wid+2), 2.0)
gamma2 = np.full((2, Ns, ver+2, wid+2), 1.0)
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
#basal decay
mu0 = 0.5
mu1 = 1.0
mu2 = 1.0
mu4 = 0.1
mu5 = 1.0
mu6 = 0.5
dummy = np.full((2, Ns, ver+2, wid+2), 1.0)
for tate in range(ver+2):
    for yoko in range(wid+2):
        gamma1[0,:, tate, yoko] = make_par(1.0, 3.0, freq1[0, 0], ds, psi[0, 0])
        gamma1[1,:, tate, yoko] = make_par(1.0, 3.0, freq1[1, 0], ds, psi[1, 0])
        gamma2[0,:, tate, yoko] = make_par(0.5, 1.5, freq1[0, 1], ds, psi[0, 1])
        gamma2[1,:, tate, yoko] = make_par(0.5, 1.5, freq1[1, 1], ds, psi[1, 1])
        dummy[0,:, tate, yoko] = make_par(0.5, 1.5, freq1[0, 2], ds, psi[0, 2])
        dummy[1,:, tate, yoko] = make_par(0.5, 1.5, freq1[1, 2], ds, psi[1, 2])
#function
#activation and inhibition of Hes-1 by NICD
def hes(x1, x2):
    return nu0 + (nu2*(x1**2)/(nu1 + x1**2))*(1 - ((x2**2)/(nu3 + x2**2)))

#ODEs
def Notchm1(x, d_ave1, d_ave2, nc1):
    return -alpha1*d_ave1*x - alpha2*d_ave2*x - mu1*x + gamma1*nc1
def Notchm2(x, d_ave1, d_ave2, nc2):
    return -alpha5*d_ave1*x - alpha6*d_ave2*x - mu1*x + gamma1*nc2
    
def Deltam1(x, n_ave1, n_ave2, dc1):
    return -alpha1*n_ave1*x -alpha5*n_ave2*x - mu2*x + gamma2*dc1
def Deltam2(x, n_ave1, n_ave2, dc2):
    return -alpha2*n_ave1*x - alpha6*n_ave2*x - mu2*x + gamma2*dc2
    
def NICD1(x, d_ave1, d_ave2, nm1):
    return alpha1*d_ave1*nm1 + alpha2*d_ave2*nm1 - mu4*x    
def NICD2(x, d_ave1, d_ave2, nm2):
    return alpha5*d_ave1*nm2 + alpha6*d_ave2*nm2  - mu4*x
    
def Notchc1(x, i1):
    return beta2_1*(i1**2)/(beta1 + i1**2) - (mu5 + gamma1)*x    
def Notchc2(x, i2):
    return beta2_2*(i2**2)/(beta1 + i2**2) - (mu5 + gamma1)*x
    
def Deltac1(x, hh):
    return beta4_1/(beta3 + hh**2) - (mu6 + gamma2)*x    
def Deltac2(x, hh):
    return beta4_2/(beta3 + hh**2) - (mu6 + gamma2)*x

def LCHes1(x, i1, i2):
    return hes(i1, i2)-mu0*x
def BTHes1(x, i1, i2):
    return hes(i2, i1)-mu0*x

for l in range(N-1):
    n1_ave[:,:, 1:-1, 1:-1] = (Nm1[:,:, 0:-2, 1:-1] + Nm1[:,:, 2:, 1:-1] + Nm1[:,:, 1:-1, 0:-2] + Nm1[:,:, 1:-1, 2:])/4
    n2_ave[:,:, 1:-1, 1:-1] = (Nm2[:,:, 0:-2, 1:-1] + Nm2[:,:, 2:, 1:-1] + Nm2[:,:, 1:-1, 0:-2] + Nm2[:,:, 1:-1, 2:])/4
    d1_ave[:,:, 1:-1, 1:-1] = (Dm1[:,:, 0:-2, 1:-1] + Dm1[:,:, 2:, 1:-1] + Dm1[:,:, 1:-1, 0:-2] + Dm1[:,:, 1:-1, 2:])/4
    d2_ave[:,:, 1:-1, 1:-1] = (Dm2[:,:, 0:-2, 1:-1] + Dm2[:,:, 2:, 1:-1] + Dm2[:,:, 1:-1, 0:-2] + Dm2[:,:, 1:-1, 2:])/4
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
    r111 = LCHes1(H1, I1, I2)
    r112 = LCHes1(H1+(s/2)*r111, I1, I2)
    r113 = LCHes1(H1+(s/2)*r112, I1, I2)
    r114 = LCHes1(H1+s*r113, I1, I2)
    H1_new = H1 + (s/6)*(r111 + 2*r112 + 2*r113 + r114)
    Nm1[:,:,1:-1, 1:-1] = Nm1_new[:,:,1:-1, 1:-1]
    Nm2[:,:,1:-1, 1:-1] = Nm2_new [:,:,1:-1, 1:-1]
    Dm1[:,:,1:-1, 1:-1] = Dm1_new[:,:,1:-1, 1:-1]
    Dm2[:,:,1:-1, 1:-1] = Dm2_new[:,:,1:-1, 1:-1]
    Nc1[:,:,1:-1, 1:-1] = Nc1_new[:,:,1:-1, 1:-1]
    Nc2[:,:,1:-1, 1:-1] = Nc2_new[:,:,1:-1, 1:-1]
    Dc1[:,:,1:-1, 1:-1] = Dc1_new[:,:,1:-1, 1:-1]
    Dc2[:,:,1:-1, 1:-1] = Dc2_new[:,:,1:-1, 1:-1]
    I1[:,:,1:-1, 1:-1] = I1_new[:,:,1:-1, 1:-1]
    I2[:,:,1:-1, 1:-1] = I2_new[:,:,1:-1, 1:-1] 
    H1[:,:,1:-1, 1:-1] = H1_new[:,:,1:-1, 1:-1]
    if (l+1)%ITVL == 0:
        print(100*(l+1)/N, "%")
h1 = H1[:,:, 1:-1, 1:-1]-H1_0
for i in range(2):
    for k in range(Ns):
        hmax = max(max(H1[i, k,:,:], key = max))
        h = h1[i, k,:,:]/hmax
        y_lc[i, k] = np.sum(h)
    for k2 in range(M*fmax):
        A_total[i, k2] = (1/Ns)*np.sum(y_lc[i, :]*np.cos((k2+1)*ds))
        B_total[i, k2] = (1/Ns)*np.sum(y_lc[i, :]*np.sin((k2+1)*ds))
    for j in range(3):
        for k3 in range(M):    
            A_i[i, j, k3] = (1/Ns)*np.sum(y_lc[i, :]*np.cos((k3+1)*freq1[i, j]*ds))
            B_i[i, j, k3] = (1/Ns)*np.sum(y_lc[i, :]*np.sin((k3+1)*freq1[i, j]*ds))
        V_i[i, j] = 2*np.sum((A_i[i, j, :]**2)+(B_i[i, j, :]**2))
    V_total[i] = 2*(np.sum(A_total[i, :]**2)+np.sum(B_total[i, :]**2))

print(V_i, V_total)    
Si = (1/2)*((V_i[0,:]/V_total[0])+(V_i[1,:]/V_total[1]))
print(Si)
fig1 = plt.figure(figsize=(10,5))
ax1 = fig1.add_subplot(111)
label = [r"$\gamma_1$", r"$\gamma_2$", "dummy"]
left = np.arange(3)
ax1.bar(left, Si, tick_label=label)
ax1.axhline(y=Si[2], linestyle='dashed', color='black')
plt.show()