#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 13:07:12 2022

@author: kyaby
"""

#境界条件０
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import animation
from matplotlib import patches

N = 100 #刻み数
T = 40 #最終時刻
s = T/N #時間の刻み幅
ver = 8 #細胞の数（たて）
wid = 8 #細胞の数（よこ）
ITVL = 3 #描画間隔
radius= 1 / np.sqrt(3) #np.sqrt(3)/2 #1 #六角形の一辺の長さ
apex=6 #頂点の数


#配列の準備
t = 0
n = np.zeros((ver+2, wid+2)) #n_ij
d = np.zeros((ver+2, wid+2)) #d_ij
i_box = np.arange(1, ver+1) #縦の配列
j_box = np.arange(1, wid+1) #横の配列
#六角形の中心の座標を入れる箱
x_box=[] 
y_box=[]
p_box=[]#六角形の頂点を入れる箱
#中心の初期値
x=0
y=0

hanicums=[] #図形を入れる箱

#パラメータの設定
v = 1.0
k = 2.0
h = 2.0
a = 0.01
b = 100.0

#関数の定義
def f(z):
    return (z ** k)/(a + (z ** k))
def g(z):
    return 1/(1 + b * (z ** h))
def n_ij(t,z): #d(n_ij)/dt
    return f(d_ave) - z
def d_ij(t,z): #d(d_ij)/dt
    return v*(g(n[1:ver+1, 1:wid+1]) - z)

#初期値
n[1:ver+1, 1:wid+1] = np.random.uniform(0.95, 1.0, (ver, wid))
d[1:ver+1, 1:wid+1] = np.random.uniform(0.95, 1.0, (ver, wid))
        
#頂点作るやつ#
def make_hexagon(center, length, apex):
    theta = np.arange(0, apex) * 2 * np.pi / apex + np.pi / 2 #座標がpi/3ずつ回っていく
    x = np.cos(theta) * length #頂点のx座標
    y = np.sin(theta) * length #頂点のy座標
    xy = np.stack([x, y], axis=-1) #頂点の座標の配列
    xy = xy + np.array(center) #頂点が中心に合わせて平行移動する
    return xy

#アニメーションの準備
fig, ax = plt.subplots()
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal")
ax.set_xlim((-1, np.sqrt(3)*(wid-1)))
ax.set_ylim((0, (2/np.sqrt(3))*(ver-1)))

for i in i_box:
    x = np.sqrt(3) /2 * radius * i #列の先頭の六角形の中心(x)
    y = 1.5*radius *i #六角形の中心(y)
    for j in j_box:
        x_box.append(x) #中心の座標を追加
        y_box.append(y)
        p = make_hexagon([x,y], radius, apex) #頂点を作って
        p_box.append(p) #配列に追加
        x = x + np.sqrt(3)*radius #次の中心を横にずらす
        patch = patches.Polygon(p, facecolor=(n[i, j], 0.2, 1.0-n[i, j]))#六角形を作る
        hanicums.append(patch) #図形を箱に追加
        ax.add_patch(patch) #図形をfig.に追加
        
def animate(k):
    global d_ave
#Runge-Kutta法による数値計算
    for l in range(0, ITVL):
        d_ave = (d[:ver, 1:wid+1] + d[:ver, 2:] + d[1:ver+1, :wid] + d[1:ver+1, 2:] + d[2:, :wid] + d[2:, 1:wid+1])/6
#notch
        t = (k*ITVL + l) * s
        r11 = n_ij(t,n[1:ver+1, 1:wid+1])
        r12 = n_ij(t + s/2, n[1:ver+1, 1:wid+1] + (s/2)*r11)
        r13 = n_ij(t + s/2, n[1:ver+1, 1:wid+1] + (s/2)*r12)
        r14 = n_ij(t + s, n[1:ver+1, 1:wid+1] + s*r13)
        n_new = n[1:ver+1, 1:wid+1] + (s/6)*(r11 + 2*r12 + 2*r13 + r14)
#delta
        r21 = d_ij(t,d[1:ver+1, 1:wid+1])
        r22 = d_ij(t + s/2, d[1:ver+1, 1:wid+1] + (s/2)*r21)
        r23 = d_ij(t + s/2, d[1:ver+1, 1:wid+1] + (s/2)*r22)
        r24 = d_ij(t + s, d[1:ver+1, 1:wid+1] + s*r23)
        d_new = d[1:ver+1, 1:wid+1] + (s/6)*(r21 + 2*r22 + 2*r23 + r24)
        n[1:ver+1, 1:wid+1] = n_new
        d[1:ver+1, 1:wid+1] = d_new
        tau = 0
        for i in i_box:
            for j in j_box:
                hanicums[tau].set_facecolor(((n[i, j], 0.2, 1.0-n[i, j])))
                tau = tau+1

anim = animation.FuncAnimation(fig, animate, interval=50)
anim.save('N-D.gif', writer="pillow")
plt.show()