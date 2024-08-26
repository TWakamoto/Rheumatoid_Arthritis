#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:58:46 2022

@author: tamaki
"""

#境界条件０
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

N = 900 #刻み数
T = 30 #最終時刻
s = T/N #時間の刻み幅
ver = 9 #細胞の数（たて）
wid = 9 #細胞の数（よこ）
ITVL = 18 #描画間隔

#配列の準備
t = np.linspace(0, T, N+1)
n = np.zeros((ver+2, wid+2)) #n_ij
d = np.zeros((ver+2, wid+2)) #d_ij
n_new = np.zeros((ver+2, wid+2)) #n_ij
d_new = np.zeros((ver+2, wid+2)) #d_ij
d_ave = np.zeros((ver+2, wid+2))
#パラメータの設定
v = 1.0
k = 2.0
h = 2.0
a = 0.01
b = 100.0

#関数定義
def n_ij(z,ave): #d(n_ij)/dt
    return (ave ** k)/(a + (ave ** k)) - z
def d_ij(z,notch): #d(d_ij)/dt
    return v*(1/(1 + b * (notch ** h)) - z)

#初期値
n_0 = np.random.rand(ver, wid)
d_0 = np.random.rand(ver, wid)
n[1:-1, 1:-1] = n_0
d[1:-1, 1:-1] = d_0
#アニメーションの準備
fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])
ims1 = []
im11 = plt.imshow(n[1:-1, 1:-1], interpolation='nearest', animated=True, vmin=0, vmax=1, cmap='jet')
im12 = fig1.colorbar(im11, ax=ax1)
txt11 = ax1.text(0.1, -0.03, f't={t[0]:.2f}', transform=ax1.transAxes)
ims1.append([im11]+[txt11])

#Runge-Kutta法による数値計算
for l in range(0, N-1):
    d_ave[1:-1, 1:-1] = (d[0:-2, 1:-1] + d[2:,1:-1] + d[1:-1, 0:-2] + d[1:-1, 2:])/4
#n_ijの計算
    r11 = n_ij(n, d_ave)
    r12 = n_ij(n+(s/2)*r11, d_ave)
    r13 = n_ij(n+(s/2)*r12, d_ave)
    r14 = n_ij(n+s*r13, d_ave)
    n_new = n + (s/6)*(r11 + 2*r12 + 2*r13 + r14)
#d_ijの計算
    r21 = d_ij(d, n)
    r22 = d_ij(d+(s/2)*r21, n)
    r23 = d_ij(d+(s/2)*r22, n)
    r24 = d_ij(d+s*r23, n)
    d_new = d + (s/6)*(r21 + 2*r22 + 2*r23 + r24)
    n[1:-1, 1:-1] = n_new[1:-1, 1:-1]
    d[1:-1, 1:-1] = d_new[1:-1, 1:-1]
    if (l+1)%ITVL == 0:
        im11 = plt.imshow(n[1:-1, 1:-1], interpolation='nearest', vmin=0, vmax=1, animated=True, cmap='jet')
        txt11 = ax1.text(0.1, -0.03, f't={t[l+1]:.2f}', transform=ax1.transAxes)
        ims1.append([im11]+[txt11])

anim = animation.ArtistAnimation(fig1, ims1, interval=150)
anim.save('N-D.gif', writer="pillow")
plt.show()