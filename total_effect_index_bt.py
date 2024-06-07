#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:41:19 2022

@author: tamaki
"""

#Sensitivity Analysis

import matplotlib.pyplot as plt
import numpy as np

N = 40000 #時間刻み数
T = 1000 #最終時刻
s = T/N #時間の刻み幅
ver = 10 #細胞の数（たて）
wid = 10 #細胞の数（よこ）
ITVL = int(N/10)

#ready of eFAST
#make sequence of frequency
freq1 = np.array([234, 1, 3, 5, 7, 11, 13, 17, 19, 23, 27, 29])
freq = np.empty((0, 12))
for i in range(12):
    freq2 = np.roll(freq1, i)
    freq = np.vstack((freq, freq2))

M = 4 #interference factor
Nr = 2 #resampling
fmax = max(freq1)
Ns = 2*M*fmax + 1

s1 = np.linspace(-np.pi*(Ns-1)/Ns, np.pi*(Ns-1)/Ns, Ns)
ds = np.empty((0, ver+2))
for k in range(Ns):
    s2 = np.full((ver+2, wid+2), s1[k])
    ds = np.vstack((ds, s2))
ds = np.reshape(ds,(Ns, ver+2, wid+2))

def make_par(p_min, p_max, freqs, x, psi):
    return p_min + (p_max-p_min)*(1/2 + (1/np.pi)*np.arcsin(np.sin(freqs*x + psi)))
psi = 2*np.pi*np.random.rand(2, 12)
#output of sensitivity function
y = np.zeros((12, Ns))
#Fourier sequence
A_total = np.zeros((12, M*fmax))
B_total = np.zeros((12, M*fmax))
A_i = np.zeros((12, int(fmax/2)))
B_i = np.zeros((12, int(fmax/2)))
V_i = np.zeros((12, 2))
V_total = np.zeros((12, 2))
#sequence
Nm1 = np.zeros((12, Ns, ver+2, wid+2)) #Notch1 in membrane
Dm1 = np.zeros((12, Ns, ver+2, wid+2)) #DLL1 in membrane
Nc1 = np.zeros((12, Ns, ver+2, wid+2)) #Notch1 in cytosol
Dc1 = np.zeros((12, Ns, ver+2, wid+2)) #DLL1 in cytosol
I1 = np.zeros((12, Ns, ver+2, wid+2)) #NICD1
Nm2 = np.zeros((12, Ns, ver+2, wid+2)) #Notch2 in membrane
Dm2 = np.zeros((12, Ns, ver+2, wid+2)) #DLL2 in membrane
Nc2 = np.zeros((12, Ns, ver+2, wid+2)) #Notch2 in cytosol
Dc2 = np.zeros((12, Ns, ver+2, wid+2)) #DLL2 in cytosol
I2 = np.zeros((12, Ns, ver+2, wid+2)) #NICD2
H1 = np.zeros((12, Ns, ver+2, wid+2)) #Hes1

Nm1_new = np.zeros((12, Ns, ver+2, wid+2)) #DLL1 in membrane
Nc1_new = np.zeros((12, Ns, ver+2, wid+2)) #Notch1 in cytosol
Dc1_new = np.zeros((12, Ns, ver+2, wid+2)) #DLL1 in cytosol
I1_new = np.zeros((12, Ns, ver+2, wid+2)) #NICD1
Nm2_new = np.zeros((12, Ns, ver+2, wid+2)) #Notch2 in membrane
Dm2_new = np.zeros((12, Ns, ver+2, wid+2)) #DLL2 in membrane
Nc2_new = np.zeros((12, Ns, ver+2, wid+2)) #Notch2 in cytosol
Dc2_new = np.zeros((12, Ns, ver+2, wid+2)) #DLL2 in cytosol
I2_new = np.zeros((12, Ns, ver+2, wid+2)) #NICD2
H1_new = np.zeros((12, Ns, ver+2, wid+2)) #Hes1

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

Nm1_ini = np.zeros((12, Ns, ver+2, wid+2))
Nm2_ini = np.zeros((12, Ns, ver+2, wid+2))
Dm1_ini = np.zeros((12, Ns, ver+2, wid+2))
Dm2_ini = np.zeros((12, Ns, ver+2, wid+2))
Nc1_ini = np.zeros((12, Ns, ver+2, wid+2))
Nc2_ini = np.zeros((12, Ns, ver+2, wid+2))
Dc1_ini = np.zeros((12, Ns, ver+2, wid+2))
Dc2_ini = np.zeros((12, Ns, ver+2, wid+2))
I1_ini = np.zeros((12, Ns, ver+2, wid+2))
I2_ini = np.zeros((12, Ns, ver+2, wid+2))
H1_ini = np.zeros((12, Ns, ver+2, wid+2))
for i in range(12):
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
Nm1 = Nm1 + Nm1_ini
Nm2 = Nm2 + Nm2_ini
Dm1 = Dm1 + Dm1_ini
Dm2 = Dm2 + Dm2_ini
Nc1 = Nc1 + Nc1_ini
Nc2 = Nc2 + Nc2_ini
Dc1 = Dc1 + Dc1_ini
Dc2 = Dc2 + Dc2_ini
I1 = I1 + I1_ini
I2 = I2 + I2_ini
H1 = H1 + H1_ini
#effect of neighboring cells
n1_ave = np.zeros((12, Ns, ver+2, wid+2))
n2_ave = np.zeros((12, Ns, ver+2, wid+2))
d1_ave = np.zeros((12, Ns, ver+2, wid+2))
d2_ave = np.zeros((12, Ns, ver+2, wid+2))

#parameter
beta1 = 0.1
beta3 = 2.0
nu0 = 0.5
nu1 = 5.0
nu3 = 5.0
mu0 = 0.5
mu1 = 1.0
mu2 = 1.0
mu4 = 0.1
mu5 = 1.0
mu6 = 0.5

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

for j in range(2):
    alpha1 = np.empty(0)
    alpha2 = np.empty(0)
    alpha5 = np.empty(0)
    alpha6 = np.empty(0)
    gamma1 = np.empty(0)
    gamma2 = np.empty(0)
    beta2_1 = np.empty(0)
    beta2_2 = np.empty(0)
    beta4_1 = np.empty(0)
    beta4_2 = np.empty(0)
    nu2 = np.empty(0)
    dummy = np.empty(0)
    for i in range(12):
        a1 = make_par(4.0, 14.0, freq[i, 0], ds, psi[j, 0])
        alpha1 = np.append(alpha1, a1)
        a2 = make_par(4.0, 14.0, freq[i, 1], ds, psi[j, 1])
        alpha2 = np.append(alpha2, a2)
        a5 = make_par(4.0, 14.0, freq[i, 2], ds, psi[j, 2])
        alpha5 = np.append(alpha5, a5)
        a6 = make_par(4.0, 14.0, freq[i, 3], ds, psi[j, 3])
        alpha6 = np.append(alpha6, a6)
        g1 = make_par(0.01, 3.0, freq[i, 4], ds, psi[j, 4])
        gamma1 = np.append(gamma1, g1)
        g2 = make_par(0.01, 3.0, freq[i, 5], ds, psi[j, 5])
        gamma2 = np.append(gamma2, g2)
        b2_1 = make_par(0.01, 6.0, freq[i, 6], ds, psi[j, 6])
        beta2_1 = np.append(beta2_1, b2_1)
        b2_2 = make_par(0.01, 6.0, freq[i, 7], ds, psi[j, 7])
        beta2_2 = np.append(beta2_2, b2_2)
        b4_1 = make_par(0.01, 10.0, freq[i, 8], ds, psi[j, 8])
        beta4_1 = np.append(beta4_1, b4_1)
        b4_2 = make_par(0.01, 10.0, freq[i, 9], ds, psi[j, 9])
        beta4_2 = np.append(beta4_2, b4_2)
        n2 = make_par(0.1, 40.0, freq[i, 10], ds, psi[j, 10])
        nu2 = np.append(nu2, n2)
        dum = make_par(0.0, 2.0, freq[i, 11], ds, psi[j, 11])
        dummy = np.append(dummy, dum)
    alpha1 = np.reshape(alpha1,(12, Ns, ver+2, wid+2))
    alpha2 = np.reshape(alpha2,(12, Ns, ver+2, wid+2))
    alpha5 = np.reshape(alpha5,(12, Ns, ver+2, wid+2))
    alpha6 = np.reshape(alpha6,(12, Ns, ver+2, wid+2))
    gamma1 = np.reshape(gamma1,(12, Ns, ver+2, wid+2))
    gamma2 = np.reshape(gamma2,(12, Ns, ver+2, wid+2))
    beta2_1 = np.reshape(beta2_1,(12, Ns, ver+2, wid+2))
    beta2_2 = np.reshape(beta2_2,(12, Ns, ver+2, wid+2))
    beta4_1 = np.reshape(beta4_1,(12, Ns, ver+2, wid+2))
    beta4_2 = np.reshape(beta4_2,(12, Ns, ver+2, wid+2))
    nu2 = np.reshape(nu2,(12, Ns, ver+2, wid+2))
    dummy = np.reshape(dummy,(12, Ns, ver+2, wid+2))
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
        r111 = BTHes1(H1, I1, I2)
        r112 = BTHes1(H1+(s/2)*r111, I1, I2)
        r113 = BTHes1(H1+(s/2)*r112, I1, I2)
        r114 = BTHes1(H1+s*r113, I1, I2)
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
    h1 = H1[:,:, 1:-1, 1:-1]-H1_ini[:,:, 1:-1, 1:-1]
    for k0 in range(12):
        for k in range(Ns):
            hmax = max(max(H1[k0, k,:,:], key = max))
            h = h1[k0, k,:,:]/hmax
            y[k0, k] = np.sum(h)
        for k2 in range(M*fmax):
            A_total[k0, k2] = (1/Ns)*np.sum(y[k0, :]*np.cos((k2+1)*s1))
            B_total[k0, k2] = (1/Ns)*np.sum(y[k0, :]*np.sin((k2+1)*s1))
        for p in range(int(fmax/2)):
            A_i[k0, p] = (1/Ns)*np.sum(y[k0, :]*np.cos((p+1)*s1))
            B_i[k0, p] = (1/Ns)*np.sum(y[k0, :]*np.sin((p+1)*s1))
        V_i[k0, j] = 2*np.sum((A_i[k0, :]**2)+(B_i[k0, :]**2))
        V_total[k0, j] = 2*(np.sum(A_total[k0, :]**2)+np.sum(B_total[k0, :]**2))
    Nm1 = np.zeros((12, Ns, ver+2, wid+2)) #Notch1 in membrane
    Dm1 = np.zeros((12, Ns, ver+2, wid+2)) #DLL1 in membrane
    Nc1 = np.zeros((12, Ns, ver+2, wid+2)) #Notch1 in cytosol
    Dc1 = np.zeros((12, Ns, ver+2, wid+2)) #DLL1 in cytosol
    I1 = np.zeros((12, Ns, ver+2, wid+2)) #NICD1
    Nm2 = np.zeros((12, Ns, ver+2, wid+2)) #Notch2 in membrane
    Dm2 = np.zeros((12, Ns, ver+2, wid+2)) #DLL2 in membrane
    Nc2 = np.zeros((12, Ns, ver+2, wid+2)) #Notch2 in cytosol
    Dc2 = np.zeros((12, Ns, ver+2, wid+2)) #DLL2 in cytosol
    I2 = np.zeros((12, Ns, ver+2, wid+2)) #NICD2
    H1 = np.zeros((12, Ns, ver+2, wid+2)) #Hes1
    Nm1 = Nm1 + Nm1_ini
    Nm2 = Nm2 + Nm2_ini
    Dm1 = Dm1 + Dm1_ini
    Dm2 = Dm2 + Dm2_ini
    Nc1 = Nc1 + Nc1_ini
    Nc2 = Nc2 + Nc2_ini
    Dc1 = Dc1 + Dc1_ini
    Dc2 = Dc2 + Dc2_ini
    I1 = I1 + I1_ini
    I2 = I2 + I2_ini
    H1 = H1 + H1_ini
Si = 1 - ((1/2)*(V_i[:, 0]+V_i[:, 1])/((1/2)*(V_total[:, 0]+V_total[:,1])))
print(Si)

#make a figure
fig2 = plt.figure(figsize=(10,5))
ax2 = fig2.add_subplot(111)
left = np.arange(12)
label = [r"$\alpha_1$", r"$\alpha_2$", r"$\alpha_5$", r"$\alpha_6$", r"$\gamma_N$", r"$\gamma_D$", r"$\beta_{21}$", r"$\beta_{22}$", r"$\beta_{41}$", r"$\beta_{42}$", r"$\nu_2$", "dummy"]
ax2.bar(left, Si, tick_label=label)
ax2.axhline(y=Si[11], linestyle='dashed', color='black')
fig2.savefig("SA_total-effect_index_bt.png")
plt.show()
