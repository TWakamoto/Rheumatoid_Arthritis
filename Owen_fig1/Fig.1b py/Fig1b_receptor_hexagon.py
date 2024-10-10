#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 13:24:26 2022

@author: kyaby
"""
#六角形
#周期境界条件
#Fig.1-b(Receptor)
#matplotlibのArtistAnimationを使ったシミュレーション
import matplotlib.pyplot as plt
import random
import numpy as np
from matplotlib import animation

N = 10000 #時間刻み数
T = 5000 #最終時刻
s = T/N #時間の刻み幅
Iy = 30 #細胞の数（たて）
Ix = 60 #細胞の数（よこ）
ITVL = 100 #描画間隔

#配列の準備
t = np.zeros(N)
a = np.full(3000, (Iy+2, Ix+2)) #the number of ligand
b = np.full(1000, (Iy+2, Ix+2)) #the number of bound receptor-ligand complex
f = np.full(5000, (Iy+2, Ix+2)) #the number of free receptor
f1 = np.zeros((Iy, Ix))

#関数Pの定義
C1 = 61
C2 = 1000
C3 = 30
C4 = 7.7
C5 = 3600
def Pa(x):
    return (C1*x)/(C2 + x)
def Pf(x):
    return C3 + ((C4*x)**3)/(C5**3 + x**3)

#パラメータの設定
ka = 0.0003
kd = 0.12
ki = 0.019
da = 0.006
df = 0.03
e = 0.01

#初期値
for i in range(1, 6):
    for j in range(1, Iy+1):
        if j == int(Ix/2):
            a[i, j] = 3000+e*random.random()
            f[i, j] = 5000+e*random.random()
            b[i, j] = 1000+e*random.random()
        elif j == int(Ix/2)+1:
            a[i, j] = a[i, int(Ix/2), 0]
            f[i, j] = f[i, int(Ix/2), 0]
            b[i, j] = b[i, int(Ix/2), 0]
        for k in range(1, 6):
            a[i+5*k, j] = a[i, j]
            f[i+5*k, j] = f[i, j]
            b[i+5*k, j] = b[i, j]
#最大値・最小値を調べる
Mf = np.max(f)
mf = np.min(f)
f1 = (f[1:-1, 1:-1]-mf)/(Mf-mf)
        
#関数の定義
def A(t, a, b, f):
    return -ka*a*fext + kd*bext - da*a + Pa(b)
def F(t, a, b, f):
    return -ka*aext*f + kd*b - df*f + Pf(b)
def B(t, a, b, f):
    return ka*aext*f - kd*b - ki*b

#アニメーションの準備
fig, ax = plt.subplots() #fig,axオブジェクトを作成
ax.set_xticks([])
ax.set_yticks([])
ims = []
im1 = plt.imshow(f1, interpolation='nearest', animated=True, vmin=0, vmax=1, cmap='jet')
im2 = fig.colorbar(im1, ax=ax)
ims.append([im1])

#Runge-Kutta法による数値計算
for l in range(N-1):
#周期境界条件
    a[0, :] = a[-1, :]
    a[Iy+1, :] = a[1, :]
    a[:, 0] = a[:, -1]
    a[:, Ix+1] = a[:, 1]
    b[0, :] = b[-1, :]
    b[Iy+1, :] = b[1, :]
    b[:, 0] = b[:, -1]
    b[:, Ix+1] = b[:, 1]
    f[0, :] = f[-1, :]
    f[Iy+1, :] = f[1, :]
    f[:, 0] = f[:, -1]
    f[:, Ix+1] = f[:, 1]
    aext = (a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] + a[:-2, :-2] + a[:-2, 2:])/6
    bext = (b[1:-1, :-2] + b[1:-1, 2:] + b[:-2, 1:-1] + b[2:, 1:-1] + b[:-2, :-2] + b[:-2, 2:])/6
    fext = (f[1:-1, :-2] + f[1:-1, 2:] + f[:-2, 1:-1] + f[2:, 1:-1] + f[:-2, :-2] + f[:-2, 2:])/6
    t[l+1] = (l+1) * s
    r11 = A(t[l],a[1:-1, 1:-1], b[1:-1, 1:-1], f[1:-1, 1:-1])
    r21 = F(t[l],f[1:-1, 1:-1], b[1:-1, 1:-1], f[1:-1, 1:-1])
    r31 = B(t[l],b[1:-1, 1:-1], b[1:-1, 1:-1], f[1:-1, 1:-1])

    r12 = A(t[l]+s/2, a[1:-1, 1:-1]+(s/2)*r11, b[1:-1, 1:-1]+(s/2)*r21, f[1:-1, 1:-1]+(s/2)*r31)
    r22 = F(t[l]+s/2, a[1:-1, 1:-1]+(s/2)*r11, b[1:-1, 1:-1]+(s/2)*r21, f[1:-1, 1:-1]+(s/2)*r31)
    r32 = B(t[l]+s/2, a[1:-1, 1:-1]+(s/2)*r11, b[1:-1, 1:-1]+(s/2)*r21, f[1:-1, 1:-1]+(s/2)*r31)

    r13 = A(t[l]+s/2, a[1:-1, 1:-1]+(s/2)*r12, b[1:-1, 1:-1]+(s/2)*r22, f[1:-1, 1:-1]+(s/2)*r32)
    r23 = F(t[l]+s/2, a[1:-1, 1:-1]+(s/2)*r12, b[1:-1, 1:-1]+(s/2)*r22, f[1:-1, 1:-1]+(s/2)*r32)
    r33 = B(t[l]+s/2, a[1:-1, 1:-1]+(s/2)*r12, b[1:-1, 1:-1]+(s/2)*r22, f[1:-1, 1:-1]+(s/2)*r32)

    r14 = A(t[l]+s, a[1:-1, 1:-1]+s*r13, b[1:-1, 1:-1]+s*r23, f[1:-1, 1:-1]+s*r33)
    r24 = F(t[l]+s, a[1:-1, 1:-1]+s*r13, b[1:-1, 1:-1]+s*r23, f[1:-1, 1:-1]+s*r33)
    r34 = B(t[l]+s, a[1:-1, 1:-1]+s*r13, b[1:-1, 1:-1]+s*r23, f[1:-1, 1:-1]+s*r33)
    a[1:-1, 1:-1] = a[1:-1, 1:-1] + (s/6)*(r11 + 2*r12 + 2*r13 + r14)
    f[1:-1, 1:-1] = f[1:-1, 1:-1] + (s/6)*(r21 + 2*r22 + 2*r23 + r24)
    b[1:-1, 1:-1] = b[1:-1, 1:-1] + (s/6)*(r31 + 2*r32 + 2*r33 + r34)
    f1 = (f[1:-1, 1:-1]-mf)/(Mf-mf)
#アニメーションの作成            
    if (l+1) % ITVL == 0:
        im1 = plt.imshow(f1, interpolation='nearest', animated=True, vmin=0, vmax=1, cmap='jet')
        ims.append([im1])

anim = animation.ArtistAnimation(fig, ims, interval=150)
anim.save('Fig1b_receptor_hexagon.gif', writer="pillow")
plt.show()