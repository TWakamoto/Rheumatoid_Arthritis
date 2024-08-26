#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:47:47 2021

@author: tamachan
"""

#Notch-Delta式[5]-[7] n-cells
import matplotlib.pyplot as plt
import numpy as np

side = 10  # 1つのセルの一辺ピクセル数
N = 150 #刻み数
T = 50 #時間
s = T/N #時間の刻み幅
ver = 4 #細胞の数（たて）
wid = 4 #細胞の数（よこ）

#配列の準備
t = np.zeros(N)
no = np.zeros((ver+2, wid+2, N)) #N_ij
de = np.zeros((ver+2, wid+2, N)) #D_ij
ni = np.zeros((ver+2, wid+2, N)) #I_ij

gam = 0.1 #notch, delta, jaggedの分解速度
gamI = 0.5 #NICDの分解速度
kc = 0.0005 #cis-inhibitionの強さ
kt = 0.00005 #trans-activationの強さ
#notch, delta, jaggedの生産速度
N0 = 500
D0 = 1000
J0 = 1200
#外部の細胞からの影響
lambN = 2.0 
lambD = 0.0
n = 3 #Hill係数


#関数の定義
def H(lamb, I):
    return 1/(1+(I**n)) + lamb*(I**n/(1+I**n))
def dN(t,x): #d(N_ij)/dt
    return N0*H(lambN, ni[i, j, l]) - kc*x*de[i, j, l] - kt*x*Dext - gam*x
def dD(t,x): #d(d_ij)/dt
    return D0*H(lambD, ni[i, j, l]) - kc*x*no[i, j, l] - kt*x*Next - gam*x
def dI(t,x):
    return kt*no[i, j, l]*Dext - gamI*x

for l in range(N-1):
    for i in range(1, ver+1):
        for j in range(1, wid+1):
            Next = (no[i-1, j, l] + no[i+1, j, l] + no[i, j-1, l] + no[i, j+1, l])/4
            Dext = (de[i-1, j, l] + de[i+1, j, l] + de[i, j-1, l] + de[i, j+1, l])/4
#N_ijの計算
            t[l+1] = (l+1) * s
            r11 = dN(t[l],no[i, j, l])
            r12 = dN(t[l] + s/2, no[i, j, l] + (s/2)*r11)
            r13 = dN(t[l] + s/2, no[i, j, l] + (s/2)*r12)
            r14 = dN(t[l] + s, no[i, j, l] + s*r13)
            no[i, j, l+1] = no[i, j, l] + (s/6)*(r11 + 2*r12 + 2*r13 + r14)
#D_ijの計算
            r21 = dD(t[l],de[i, j, l])
            r22 = dD(t[l] + s/2, de[i, j, l] + (s/2)*r21)
            r23 = dD(t[l] + s/2, de[i, j, l] + (s/2)*r22)
            r24 = dD(t[l] + s, de[i, j, l] + s*r23)
            de[i, j, l+1] = de[i, j, l] + (s/6)*(r21 + 2*r22 + 2*r23 + r24)
#I_ijの計算            
            r41 = dI(t[l],ni[i, j, l])
            r42 = dI(t[l] + s/2, ni[i, j, l] + (s/2)*r41)
            r43 = dI(t[l] + s/2, ni[i, j, l] + (s/2)*r42)
            r44 = dI(t[l] + s, ni[i, j, l] + s*r43)
            ni[i, j, l+1] = ni[i, j, l] + (s/6)*(r41 + 2*r42 + 2*r43 + r44)
for j in range(1,ver+1):
    for i in range(1, wid+1):
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.plot(t, no[i, j]/1000, linestyle = "-", label="Notch")
        ax.plot(t, de[i, j]/1000, linestyle = "-", label="Delta")
        ax.plot(t, ni[i, j]/1000, linestyle = "-", label="NICD")
        ax.set_xlabel("t", fontsize=20)
        ax.set_ylabel("", fontsize=20)
        plt.ylim(0, 10)
        plt.legend(prop={"size":20}, loc="best")
        print(de[i, j, N-2], ni[i, j, N-2])

plt.show()    