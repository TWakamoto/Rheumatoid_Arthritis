#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:24:31 2022

@author: wakamototamaki
"""

#Sensitivity Analysis
# %%
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

# %%
N = 50000 #時間刻み数
T = 500 #最終時刻
s = T/N #時間の刻み幅
ver = 10 #細胞の数（たて）
wid = 10 #細胞の数（よこ）
ITVL = int(N/10)
param = 13 #the number of parameter

# %%
#ready of eFAST
#make sequence of frequency
freq1 = np.array([234, 1, 3, 5, 7, 11, 13, 17, 19, 23, 25, 27, 29])
freq = np.empty((0, param))
for i in range(param):
    freq2 = np.roll(freq1, i)
    freq = np.vstack((freq, freq2))

# %%
M = 4 #interference factor
Nr = 2 #resampling
fmax = max(freq1)
Ns = 2*M*fmax + 1

# %%
s1 = np.linspace(-np.pi*(Ns-1)/Ns, np.pi*(Ns-1)/Ns, Ns)
ds = np.empty((0, ver+2))
for k in range(Ns):
    s2 = np.full((ver+2, wid+2), s1[k])
    ds = np.vstack((ds, s2))
ds = np.reshape(ds,(Ns, ver+2, wid+2))

# %%
def make_par(p_min, p_max, freqs, x, psi):
    return p_min + (p_max-p_min)*(1/2 + (1/np.pi)*np.arcsin(np.sin(freqs*x + psi)))
psi = 2*np.pi*np.random.rand(2, param)
#output of sensitivity function
y = np.zeros((param, Ns))
#Fourier sequence
A_total = np.zeros((param, M*fmax))
B_total = np.zeros((param, M*fmax))
A_i = np.zeros((param, int(fmax/2)))
B_i = np.zeros((param, int(fmax/2)))
V_i = np.zeros((param, 2))
V_total = np.zeros((param, 2))

# %%
#sequence
Nm1 = np.zeros((param, Ns, ver+2, wid+2)) #Notch1 in membrane
Dm1 = np.zeros((param, Ns, ver+2, wid+2)) #DLL1 in membrane
Nc1 = np.zeros((param, Ns, ver+2, wid+2)) #Notch1 in cytosol
Dc1 = np.zeros((param, Ns, ver+2, wid+2)) #DLL1 in cytosol
I1 = np.zeros((param, Ns, ver+2, wid+2)) #NICD1
Nm2 = np.zeros((param, Ns, ver+2, wid+2)) #Notch2 in membrane
Dm2 = np.zeros((param, Ns, ver+2, wid+2)) #DLL2 in membrane
Nc2 = np.zeros((param, Ns, ver+2, wid+2)) #Notch2 in cytosol
Dc2 = np.zeros((param, Ns, ver+2, wid+2)) #DLL2 in cytosol
I2 = np.zeros((param, Ns, ver+2, wid+2)) #NICD2
H1 = np.zeros((param, Ns, ver+2, wid+2)) #Hes1

# %%
#initial condition
e = 0.01
y_init = e*np.random.rand(11, ver, wid)

for i in range(12):
    for j in range(Ns):
        Nm1[i, j, 1:-1 ,1:-1] = y_init[0, :, :]
        Nm2[i, j, 1:-1 ,1:-1] = y_init[1, :, :]
        Dm1[i, j, 1:-1 ,1:-1] = y_init[2, :, :]
        Dm2[i, j, 1:-1 ,1:-1] = y_init[3, :, :]
        Nc1[i, j, 1:-1 ,1:-1] = y_init[4, :, :]
        Nc2[i, j, 1:-1 ,1:-1] = y_init[5, :, :]
        Dc1[i, j, 1:-1 ,1:-1] = y_init[6, :, :]
        Dc2[i, j, 1:-1 ,1:-1] = y_init[7, :, :]
        I1[i, j, 1:-1 ,1:-1] = y_init[8, :, :]
        I2[i, j, 1:-1 ,1:-1] = y_init[9, :, :]
        H1[i, j, 1:-1 ,1:-1] = y_init[10, :, :]

H1_ini = np.copy(H1)

# %%
#parameter
beta1 = 0.1
beta3 = 2.0
nu0 = 0.5
nu1 = 5.0
mu0 = 0.5
mu1 = 1.0
mu2 = 1.0
mu4 = 0.1
mu5 = 1.0
mu6 = 0.5

# %%
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
#effect of neighboring cells
n1_ave = np.zeros((param, Ns, ver+2, wid+2))
n2_ave = np.zeros((param, Ns, ver+2, wid+2))
d1_ave = np.zeros((param, Ns, ver+2, wid+2))
d2_ave = np.zeros((param, Ns, ver+2, wid+2))

# %%
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
    nu3 = np.empty(0)
    dummy = np.empty(0)
    for i in range(param):
        a1 = make_par(1.0, 15.0, freq[i, 0], ds, psi[j, 0])
        alpha1 = np.append(alpha1, a1)
        a2 = make_par(1.0, 15.0, freq[i, 1], ds, psi[j, 1])
        alpha2 = np.append(alpha2, a2)
        a5 = make_par(1.0, 15.0, freq[i, 2], ds, psi[j, 2])
        alpha5 = np.append(alpha5, a5)
        a6 = make_par(1.0, 15.0, freq[i, 3], ds, psi[j, 3])
        alpha6 = np.append(alpha6, a6)
        g1 = make_par(0.01, 5.0, freq[i, 4], ds, psi[j, 4])
        gamma1 = np.append(gamma1, g1)
        g2 = make_par(0.01, 5.0, freq[i, 5], ds, psi[j, 5])
        gamma2 = np.append(gamma2, g2)
        b2_1 = make_par(0.1, 10.0, freq[i, 6], ds, psi[j, 6])
        beta2_1 = np.append(beta2_1, b2_1)
        b2_2 = make_par(0.1, 10.0, freq[i, 7], ds, psi[j, 7])
        beta2_2 = np.append(beta2_2, b2_2)
        b4_1 = make_par(0.1, 15.0, freq[i, 8], ds, psi[j, 8])
        beta4_1 = np.append(beta4_1, b4_1)
        b4_2 = make_par(0.1, 15.0, freq[i, 9], ds, psi[j, 9])
        beta4_2 = np.append(beta4_2, b4_2)
        n2 = make_par(0.1, 40.0, freq[i, 10], ds, psi[j, 10])
        nu2 = np.append(nu2, n2)
        n3 = make_par(5.0, 100.0, freq[i, 11], ds, psi[j, 11])
        nu3 = np.append(nu3, n3)
        dum = make_par(0.0, 2.0, freq[i, 12], ds, psi[j, 12])
        dummy = np.append(dummy, dum)
    alpha1 = np.reshape(alpha1,(param, Ns, ver+2, wid+2))
    alpha2 = np.reshape(alpha2,(param, Ns, ver+2, wid+2))
    alpha5 = np.reshape(alpha5,(param, Ns, ver+2, wid+2))
    alpha6 = np.reshape(alpha6,(param, Ns, ver+2, wid+2))
    gamma1 = np.reshape(gamma1,(param, Ns, ver+2, wid+2))
    gamma2 = np.reshape(gamma2,(param, Ns, ver+2, wid+2))
    beta2_1 = np.reshape(beta2_1,(param, Ns, ver+2, wid+2))
    beta2_2 = np.reshape(beta2_2,(param, Ns, ver+2, wid+2))
    beta4_1 = np.reshape(beta4_1,(param, Ns, ver+2, wid+2))
    beta4_2 = np.reshape(beta4_2,(param, Ns, ver+2, wid+2))
    nu2 = np.reshape(nu2,(param, Ns, ver+2, wid+2))
    nu3 = np.reshape(nu3,(param, Ns, ver+2, wid+2))
    dummy = np.reshape(dummy,(param, Ns, ver+2, wid+2))
    for l in trange(N-1):
        n1_ave[:, :, 1:-1, 1:-1] = (Nm1[:, :, 0:-2, 1:-1] + Nm1[:, :, 2:,1:-1] + Nm1[:, :, 1:-1, 0:-2] + Nm1[:, :, 1:-1, 2:])/4
        n2_ave[:, :, 1:-1, 1:-1] = (Nm2[:, :, 0:-2, 1:-1] + Nm2[:, :, 2:,1:-1] + Nm2[:, :, 1:-1, 0:-2] + Nm2[:, :, 1:-1, 2:])/4
        d1_ave[:, :, 1:-1, 1:-1] = (Dm1[:, :, 0:-2, 1:-1] + Dm1[:, :, 2:,1:-1] + Dm1[:, :, 1:-1, 0:-2] + Dm1[:, :, 1:-1, 2:])/4
        d2_ave[:, :, 1:-1, 1:-1] = (Dm2[:, :, 0:-2, 1:-1] + Dm2[:, :, 2:,1:-1] + Dm2[:, :, 1:-1, 0:-2] + Dm2[:, :, 1:-1, 2:])/4
    #Delta in membrane
        Dm1_new = ud_m(Deltam1, Dm1, n1_ave, n2_ave, Dc1)
        Dm2_new = ud_m(Deltam2, Dm2, n1_ave, n2_ave, Dc2)
    #Delta in cytosol
        Dc1 = ud_c(Deltac1, Dc1, H1)
        Dc2 = ud_c(Deltac2, Dc2, H1)
    #Hes1
        H1 = ud_h(LCHes1, H1, I1, I2)
    #Notch in membrane
        Nm1_new = ud_m(Notchm1, Nm1, d1_ave, d2_ave, Nc1)
        Nm2_new = ud_m(Notchm2, Nm2, d1_ave, d2_ave, Nc2)
    #Notch in cytosol 
        Nc1 = ud_c(Notchc1, Nc1, I1)
        Nc2 = ud_c(Notchc2, Nc2, I2)
    #NICD
        I1 = ud_i(NICD1, I1, d1_ave, d2_ave, Nm1)
        I2 = ud_i(NICD2, I2, d1_ave, d2_ave, Nm2)
        Nm1[:, :, 1:-1, 1:-1] = Nm1_new[:, :, 1:-1, 1:-1]
        Nm2[:, :, 1:-1, 1:-1] = Nm2_new[:, :, 1:-1, 1:-1]
        Dm1[:, :, 1:-1, 1:-1] = Dm1_new[:, :, 1:-1, 1:-1]
        Dm2[:, :, 1:-1, 1:-1] = Dm2_new[:, :, 1:-1, 1:-1]
    h1 = H1[:,:, 1:-1, 1:-1]-H1_ini[:,:, 1:-1, 1:-1]
    for k0 in range(param):
        for k in range(Ns):
            hmax = np.max(H1[k0, k,:,:])
            h = h1[k0, k,:,:]/hmax
            y[k0, k] = np.sum(h[h >= 0.75])
        for k2 in range(M*fmax):
            A_total[k0, k2] = (1/Ns)*np.sum(y[k0, :]*np.cos((k2+1)*s1))
            B_total[k0, k2] = (1/Ns)*np.sum(y[k0, :]*np.sin((k2+1)*s1))
        for p in range(int(fmax/2)):
            A_i[k0, p] = (1/Ns)*np.sum(y[k0, :]*np.cos((p+1)*s1))
            B_i[k0, p] = (1/Ns)*np.sum(y[k0, :]*np.sin((p+1)*s1))
        V_i[k0, j] = 2*np.sum((A_i[k0, :]**2)+(B_i[k0, :]**2))
        V_total[k0, j] = 2*np.sum((A_total[k0, :]**2)+(B_total[k0, :]**2))
    for i in range(param):
        for j in range(Ns):
            Nm1[i, j, 1:-1 ,1:-1] = y_init[0, :, :]
            Nm2[i, j, 1:-1 ,1:-1] = y_init[1, :, :]
            Dm1[i, j, 1:-1 ,1:-1] = y_init[2, :, :]
            Dm2[i, j, 1:-1 ,1:-1] = y_init[3, :, :]
            Nc1[i, j, 1:-1 ,1:-1] = y_init[4, :, :]
            Nc2[i, j, 1:-1 ,1:-1] = y_init[5, :, :]
            Dc1[i, j, 1:-1 ,1:-1] = y_init[6, :, :]
            Dc2[i, j, 1:-1 ,1:-1] = y_init[7, :, :]
            I1[i, j, 1:-1 ,1:-1] = y_init[8, :, :]
            I2[i, j, 1:-1 ,1:-1] = y_init[9, :, :]
            H1[i, j, 1:-1 ,1:-1] = y_init[10, :, :]
            
# %%
Si_lc = 1 - ((1/2)*(V_i[:, 0]+V_i[:, 1])/((1/2)*(V_total[:, 0]+V_total[:,1])))
print(Si_lc)

# %%
#make a figure
fig1 = plt.figure(figsize=(10,5))
ax1 = fig1.add_subplot(111)
label = [r"$\alpha_1$", r"$\alpha_2$", r"$\alpha_5$", r"$\alpha_6$", r"$\gamma_N$", r"$\gamma_D$", r"$\beta_{21}$", r"$\beta_{22}$", r"$\beta_{41}$", r"$\beta_{42}$", r"$\nu_2$", r"$\nu_3$", "dummy"]
left = np.arange(param)
ax1.bar(left, Si_lc, tick_label=label)
ax1.axhline(y=Si_lc[param-1], linestyle='dashed', color='black')
fig1.savefig("0.75-SA_total-effect_index_lc.png")

# %%
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
    nu3 = np.empty(0)
    dummy = np.empty(0)
    for i in range(param):
        a1 = make_par(1.0, 15.0, freq[i, 0], ds, psi[j, 0])
        alpha1 = np.append(alpha1, a1)
        a2 = make_par(1.0, 15.0, freq[i, 1], ds, psi[j, 1])
        alpha2 = np.append(alpha2, a2)
        a5 = make_par(1.0, 15.0, freq[i, 2], ds, psi[j, 2])
        alpha5 = np.append(alpha5, a5)
        a6 = make_par(1.0, 15.0, freq[i, 3], ds, psi[j, 3])
        alpha6 = np.append(alpha6, a6)
        g1 = make_par(0.01, 5.0, freq[i, 4], ds, psi[j, 4])
        gamma1 = np.append(gamma1, g1)
        g2 = make_par(0.01, 5.0, freq[i, 5], ds, psi[j, 5])
        gamma2 = np.append(gamma2, g2)
        b2_1 = make_par(0.1, 10.0, freq[i, 6], ds, psi[j, 6])
        beta2_1 = np.append(beta2_1, b2_1)
        b2_2 = make_par(0.1, 10.0, freq[i, 7], ds, psi[j, 7])
        beta2_2 = np.append(beta2_2, b2_2)
        b4_1 = make_par(0.1, 15.0, freq[i, 8], ds, psi[j, 8])
        beta4_1 = np.append(beta4_1, b4_1)
        b4_2 = make_par(0.1, 15.0, freq[i, 9], ds, psi[j, 9])
        beta4_2 = np.append(beta4_2, b4_2)
        n2 = make_par(0.1, 40.0, freq[i, 10], ds, psi[j, 10])
        nu2 = np.append(nu2, n2)
        n3 = make_par(5.0, 100.0, freq[i, 11], ds, psi[j, 11])
        nu3 = np.append(nu3, n3)
        dum = make_par(0.0, 2.0, freq[i, 12], ds, psi[j, 12])
        dummy = np.append(dummy, dum)
    alpha1 = np.reshape(alpha1,(param, Ns, ver+2, wid+2))
    alpha2 = np.reshape(alpha2,(param, Ns, ver+2, wid+2))
    alpha5 = np.reshape(alpha5,(param, Ns, ver+2, wid+2))
    alpha6 = np.reshape(alpha6,(param, Ns, ver+2, wid+2))
    gamma1 = np.reshape(gamma1,(param, Ns, ver+2, wid+2))
    gamma2 = np.reshape(gamma2,(param, Ns, ver+2, wid+2))
    beta2_1 = np.reshape(beta2_1,(param, Ns, ver+2, wid+2))
    beta2_2 = np.reshape(beta2_2,(param, Ns, ver+2, wid+2))
    beta4_1 = np.reshape(beta4_1,(param, Ns, ver+2, wid+2))
    beta4_2 = np.reshape(beta4_2,(param, Ns, ver+2, wid+2))
    nu2 = np.reshape(nu2,(param, Ns, ver+2, wid+2))
    nu3 = np.reshape(nu3,(param, Ns, ver+2, wid+2))
    dummy = np.reshape(dummy,(param, Ns, ver+2, wid+2))
    for l in trange(N-1):
        n1_ave[:, :, 1:-1, 1:-1] = (Nm1[:, :, 0:-2, 1:-1] + Nm1[:, :, 2:,1:-1] + Nm1[:, :, 1:-1, 0:-2] + Nm1[:, :, 1:-1, 2:])/4
        n2_ave[:, :, 1:-1, 1:-1] = (Nm2[:, :, 0:-2, 1:-1] + Nm2[:, :, 2:,1:-1] + Nm2[:, :, 1:-1, 0:-2] + Nm2[:, :, 1:-1, 2:])/4
        d1_ave[:, :, 1:-1, 1:-1] = (Dm1[:, :, 0:-2, 1:-1] + Dm1[:, :, 2:,1:-1] + Dm1[:, :, 1:-1, 0:-2] + Dm1[:, :, 1:-1, 2:])/4
        d2_ave[:, :, 1:-1, 1:-1] = (Dm2[:, :, 0:-2, 1:-1] + Dm2[:, :, 2:,1:-1] + Dm2[:, :, 1:-1, 0:-2] + Dm2[:, :, 1:-1, 2:])/4
    #Delta in membrane
        Dm1_new = ud_m(Deltam1, Dm1, n1_ave, n2_ave, Dc1)
        Dm2_new = ud_m(Deltam2, Dm2, n1_ave, n2_ave, Dc2)
    #Delta in cytosol
        Dc1 = ud_c(Deltac1, Dc1, H1)
        Dc2 = ud_c(Deltac2, Dc2, H1)
    #Hes1
        H1 = ud_h(BTHes1, H1, I1, I2)
    #Notch in membrane
        Nm1_new = ud_m(Notchm1, Nm1, d1_ave, d2_ave, Nc1)
        Nm2_new = ud_m(Notchm2, Nm2, d1_ave, d2_ave, Nc2)
    #Notch in cytosol 
        Nc1 = ud_c(Notchc1, Nc1, I1)
        Nc2 = ud_c(Notchc2, Nc2, I2)
    #NICD
        I1 = ud_i(NICD1, I1, d1_ave, d2_ave, Nm1)
        I2 = ud_i(NICD2, I2, d1_ave, d2_ave, Nm2)
        Nm1[:, :, 1:-1, 1:-1] = Nm1_new[:, :, 1:-1, 1:-1]
        Nm2[:, :, 1:-1, 1:-1] = Nm2_new[:, :, 1:-1, 1:-1]
        Dm1[:, :, 1:-1, 1:-1] = Dm1_new[:, :, 1:-1, 1:-1]
        Dm2[:, :, 1:-1, 1:-1] = Dm2_new[:, :, 1:-1, 1:-1]
    h1 = H1[:,:, 1:-1, 1:-1]-H1_ini[:,:, 1:-1, 1:-1]
    for k0 in range(param):
        for k in range(Ns):
            hmax = max(max(H1[k0, k,:,:], key = max))
            h = h1[k0, k,:,:]/hmax
            y[k0, k] = np.sum(h[h >= 0.75])
        for k2 in range(M*fmax):
            A_total[k0, k2] = (1/Ns)*np.sum(y[k0, :]*np.cos((k2+1)*s1))
            B_total[k0, k2] = (1/Ns)*np.sum(y[k0, :]*np.sin((k2+1)*s1))
        for p in range(int(fmax/2)):
            A_i[k0, p] = (1/Ns)*np.sum(y[k0, :]*np.cos((p+1)*s1))
            B_i[k0, p] = (1/Ns)*np.sum(y[k0, :]*np.sin((p+1)*s1))
        V_i[k0, j] = 2*np.sum((A_i[k0, :]**2)+(B_i[k0, :]**2))
        V_total[k0, j] = 2*(np.sum((A_total[k0, :]**2) + (B_total[k0, :]**2)))
    for i in range(param):
        for k in range(Ns):
            Nm1[i, k, 1:-1 ,1:-1] = y_init[0, :, :]
            Nm2[i, k, 1:-1 ,1:-1] = y_init[1, :, :]
            Dm1[i, k, 1:-1 ,1:-1] = y_init[2, :, :]
            Dm2[i, k, 1:-1 ,1:-1] = y_init[3, :, :]
            Nc1[i, k, 1:-1 ,1:-1] = y_init[4, :, :]
            Nc2[i, k, 1:-1 ,1:-1] = y_init[5, :, :]
            Dc1[i, k, 1:-1 ,1:-1] = y_init[6, :, :]
            Dc2[i, k, 1:-1 ,1:-1] = y_init[7, :, :]
            I1[i, k, 1:-1 ,1:-1] = y_init[8, :, :]
            I2[i, k, 1:-1 ,1:-1] = y_init[9, :, :]
            H1[i, k, 1:-1 ,1:-1] = y_init[10, :, :]
# %%
Si_bt = 1 - ((1/2)*(V_i[:, 0]+V_i[:, 1])/((1/2)*(V_total[:, 0]+V_total[:,1])))
print(Si_bt)

# %%
#make a figure
fig2 = plt.figure(figsize=(10,5))
ax2 = fig2.add_subplot(111)
ax2.bar(left, Si_bt, tick_label=label)
ax2.axhline(y=Si_bt[param-1], linestyle='dashed', color='black')
fig2.savefig("0.75-SA_total-effect_index_bt.png")
plt.show()
