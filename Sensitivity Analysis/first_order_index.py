#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:20:42 2022

@author: wakamototamaki
"""

#Sensitivity Analysis
#first order sensitivity index

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
param = 13

# %%
#ready of eFAST
freq1 = np.asarray([41, 67, 105, 145, 177, 203, 223, 229, 235, 241, 247, 249, 59])
M = 4 #interference factor
Nr = 2 #resampling
fmax = np.max(freq1)
Ns = 2*M*fmax + 1
ds = np.linspace(-np.pi*(Ns-1)/Ns, np.pi*(Ns-1)/Ns, Ns)
def make_par(p_min, p_max, freq, x, psi):
    return p_min + (p_max-p_min)*(1/2 + (1/np.pi)*np.arcsin(np.sin(freq*x + psi)))
psi = 2*np.pi*np.random.rand(2, param)
#output of sensitivity function
y = np.zeros((2, Ns))
#Fourier sequence
A_total = np.zeros((2, M*fmax))
B_total = np.zeros((2, M*fmax))
A_i = np.zeros((2, param, M))
B_i = np.zeros((2, param, M))
V_i = np.zeros((2, param))
V_total = np.zeros(2)

# %%
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

# %%
#initial condition
e = 0.01
y_init = e*np.random.rand(11, ver, wid)

for i in range(2):
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
H1_ini = np.copy(H1)

# %%
#effect of neighboring cells
n1_ave = np.zeros((2, Ns, ver+2, wid+2))
n2_ave = np.zeros((2, Ns, ver+2, wid+2))
d1_ave = np.zeros((2, Ns, ver+2, wid+2))
d2_ave = np.zeros((2, Ns, ver+2, wid+2))

# %%
#parameter
#binding rate
alpha1 = np.full((2, Ns, ver+2, wid+2), 6.0)
alpha2 = np.full((2, Ns, ver+2, wid+2), 10.0)
alpha5 = np.full((2, Ns, ver+2, wid+2), 8.0)
alpha6 = np.full((2, Ns, ver+2, wid+2), 6.0)
#move to membrane
gamma1 = np.full((2, Ns, ver+2, wid+2), 2.0)
gamma2 = np.full((2, Ns, ver+2, wid+2), 1.0)
#parameter of function
beta1 = 0.1
beta2_1 = np.full((2, Ns, ver+2, wid+2), 8.0)
beta2_2 = np.full((2, Ns, ver+2, wid+2), 5.0)
beta3 = 2.0
beta4_1 = np.full((2, Ns, ver+2, wid+2), 8.0)
beta4_2 = np.full((2, Ns, ver+2, wid+2), 8.0)
#parameter of Hes-1
nu0 = 0.5
nu1 = 5.0
nu2 = np.full((2, Ns, ver+2, wid+2), 25.0)
nu3 = np.full((2, Ns, ver+2, wid+2), 5.0)
#basal decay
mu0 = 0.5
mu1 = 1.0
mu2 = 1.0
mu4 = 0.1
mu5 = 1.0
mu6 = 0.5
dummy = np.full((2, Ns, ver+2, wid+2), 1.0)
for tate in trange(ver+2):
    for yoko in range(wid+2):
        alpha1[0,:, tate, yoko] = make_par(1.0, 15.0, freq1[0], ds, psi[0, 0])
        alpha1[1,:, tate, yoko] = make_par(1.0, 15.0, freq1[0], ds, psi[1, 0])
        alpha2[0,:, tate, yoko] = make_par(1.0, 15.0, freq1[1], ds, psi[0, 1])
        alpha2[1,:, tate, yoko] = make_par(1.0, 15.0, freq1[1], ds, psi[1, 1])
        alpha5[0,:, tate, yoko] = make_par(1.0, 15.0, freq1[2], ds, psi[0, 2])
        alpha5[1,:, tate, yoko] = make_par(1.0, 15.0, freq1[2], ds, psi[1, 2])
        alpha6[0,:, tate, yoko] = make_par(1.0, 15.0, freq1[3], ds, psi[0, 3])
        alpha6[1,:, tate, yoko] = make_par(1.0, 15.0, freq1[3], ds, psi[1, 3])
        gamma1[0,:, tate, yoko] = make_par(0.01, 5.0, freq1[4], ds, psi[0, 4])
        gamma1[1,:, tate, yoko] = make_par(0.01, 5.0, freq1[4], ds, psi[1, 4])
        gamma2[0,:, tate, yoko] = make_par(0.01, 5.0, freq1[5], ds, psi[0, 5])
        gamma2[1,:, tate, yoko] = make_par(0.01, 5.0, freq1[5], ds, psi[1, 5])
        beta2_1[0,:, tate, yoko] = make_par(0.1, 10.0, freq1[6], ds, psi[0, 6])
        beta2_1[1,:, tate, yoko] = make_par(0.1, 10.0, freq1[6], ds, psi[1, 6])
        beta2_2[0,:, tate, yoko] = make_par(0.1, 10.0, freq1[7], ds, psi[0, 7])
        beta2_2[1,:, tate, yoko] = make_par(0.1, 10.0, freq1[7], ds, psi[1, 7])
        beta4_1[0,:, tate, yoko] = make_par(0.1, 15.0, freq1[8], ds, psi[0, 8])
        beta4_1[1,:, tate, yoko] = make_par(0.1, 15.0, freq1[8], ds, psi[1, 8])
        beta4_2[0,:, tate, yoko] = make_par(0.1, 15.0, freq1[9], ds, psi[0, 9])
        beta4_2[1,:, tate, yoko] = make_par(0.1, 15.0, freq1[9], ds, psi[1, 9])
        nu2[0,:, tate, yoko] = make_par(0.1, 40.0, freq1[10], ds, psi[0, 10])
        nu2[1,:, tate, yoko] = make_par(0.1, 40.0, freq1[10], ds, psi[1, 10])
        nu3[0,:, tate, yoko] = make_par(5.0, 100.0, freq1[11], ds, psi[0, 11])
        nu3[1,:, tate, yoko] = make_par(5.0, 100.0, freq1[11], ds, psi[1, 11])
        dummy[0,:, tate, yoko] = make_par(0.0, 2.0, freq1[12], ds, psi[0, 12])
        dummy[1,:, tate, yoko] = make_par(0.0, 2.0, freq1[12], ds, psi[1, 12])
        
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

# %%
for i in range(2):
    for k in range(Ns):
        hmax = np.max(H1[i, k,:,:])
        h = h1[i, k,:,:]/hmax
        y[i, k] = np.sum(h[h >= 0.75])
    for k2 in range(M*fmax):
        A_total[i, k2] = (1/Ns)*np.sum(y[i, :]*np.cos((k2+1)*ds))
        B_total[i, k2] = (1/Ns)*np.sum(y[i, :]*np.sin((k2+1)*ds))
    for j in range(param):
        for k3 in range(M):    
            A_i[i, j, k3] = (1/Ns)*np.sum(y[i, :]*np.cos((k3+1)*freq1[j]*ds))
            B_i[i, j, k3] = (1/Ns)*np.sum(y[i, :]*np.sin((k3+1)*freq1[j]*ds))
        V_i[i, j] = 2*np.sum((A_i[i, j, :]**2)+(B_i[i, j, :]**2))
    V_total[i] = 2*np.sum((A_total[i, :]**2)+(B_total[i, :]**2))

Si_lc = (1/2)*(V_i[0,:]+V_i[1,:])/((1/2)*(V_total[0]+V_total[1]))
print(Si_lc)

# %%
#make a figure
fig1 = plt.figure(figsize=(10,5))
ax1 = fig1.add_subplot(111)
label = [r"$\alpha_1$", r"$\alpha_2$", r"$\alpha_5$", r"$\alpha_6$", r"$\gamma_N$", r"$\gamma_D$", r"$\beta_{21}$", r"$\beta_{22}$", r"$\beta_{41}$", r"$\beta_{42}$", r"$\nu_2$", r"$\nu_3$", "dummy"]
left = np.arange(param)
ax1.bar(left, Si_lc, tick_label=label)
ax1.axhline(y=Si_lc[param-1], linestyle='dashed', color='black')
fig1.savefig("0.75-SA_1st_order_index_lc.png")

# %%
#Model-BT
for i in range(2):
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
beta2_1 = np.full((2, Ns, ver+2, wid+2), 5.0)
beta2_2 = np.full((2, Ns, ver+2, wid+2), 8.0)
for tate in range(ver+2):
    for yoko in range(wid+2):
        beta2_1[0,:, tate, yoko] = make_par(0.1, 10.0, freq1[6], ds, psi[0, 6])
        beta2_1[1,:, tate, yoko] = make_par(0.1, 10.0, freq1[6], ds, psi[1, 6])
        beta2_2[0,:, tate, yoko] = make_par(0.1, 10.0, freq1[7], ds, psi[0, 7])
        beta2_2[1,:, tate, yoko] = make_par(0.1, 10.0, freq1[7], ds, psi[1, 7])

# %%
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

# %%
h1 = H1[:,:, 1:-1, 1:-1]-H1_ini[:,:, 1:-1, 1:-1]
for i in range(2):
    for k in range(Ns):
        hmax = max(max(H1[i, k,:,:], key = max))
        h = h1[i, k,:,:]/hmax
        y[i, k] = np.sum(h[h >= 0.75])
    for k2 in range(M*fmax):
        A_total[i, k2] = (1/Ns)*np.sum(y[i, :]*np.cos((k2+1)*ds))
        B_total[i, k2] = (1/Ns)*np.sum(y[i, :]*np.sin((k2+1)*ds))
    for j in range(param):
        for k3 in range(M):    
            A_i[i, j, k3] = (1/Ns)*np.sum(y[i, :]*np.cos((k3+1)*freq1[j]*ds))
            B_i[i, j, k3] = (1/Ns)*np.sum(y[i, :]*np.sin((k3+1)*freq1[j]*ds))
        V_i[i, j] = 2*np.sum((A_i[i, j, :]**2)+(B_i[i, j, :]**2))
    V_total[i] = 2*np.sum((A_total[i, :]**2)+(B_total[i, :]**2))
Si_bt = (1/2)*(V_i[0,:]+V_i[1,:])/((1/2)*(V_total[0]+V_total[1]))
print(Si_bt) 

# %%
#make a figure
fig2 = plt.figure(figsize=(10,5))
ax2 = fig2.add_subplot(111)
ax2.bar(left, Si_bt, tick_label=label)
ax2.axhline(y=Si_bt[param-1], linestyle='dashed', color='black')
fig2.savefig("0.75-SA_1st_order_index_bt.png")
plt.show()
