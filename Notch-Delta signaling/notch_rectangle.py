#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:30:40 2021

@author: tamachan
"""
#周期境界条件
import numpy as np
import random
from PIL import Image, ImageDraw

side = 10  # 1つのセルの一辺ピクセル数
N = 500 #刻み数
T = 100 #時間
s = T/N #時間の刻み幅
ver = 7 #細胞の数（たて）
wid = 7 #細胞の数（よこ）

#配列の準備
t = np.zeros(N)
x = np.zeros((ver+2, wid+2, N)) #n_ij
y = np.zeros((ver+2, wid+2, N)) #d_ij

#パラメータの設定
v = 1.0
k = 2.0
h = 2.0
a = 0.01
b = 100.0

#関数の定義
def f(x):
    return (x ** k)/(a + (x ** k))
def g(x):
    return 1/(1 + b * (x ** h))
def n_ij(t,x): #d(n_ij)/dt
    return f(d_ave) - x
def d_ij(t,y): #d(d_ij)/dt
    return v*(g(x[i, j, l]) - y)

#初期値
for j in range(1, wid+1):
    for i in range(1, ver+1):
        x[i, j, 0] = random.uniform(0.95, 1.0)
        y[i, j, 0] = random.uniform(0.95, 1.0)
       
#Runge-Kutta法による数値計算
for l in range(N-1):
    for i in range(1, ver+1):
        for j in range(1, wid+1):
            #境界条件
            x[0, j, l] = x[ver, j, l]
            y[0, j, l] = y[ver, j, l]
            x[ver+1, j, l] = x[1, j, l]
            y[ver+1, j, l] = y[1, j, l]
            x[i, 0, l] = x[i, wid, l]
            y[i, 0, l] = y[i, wid, l]
            x[i, wid+1, l] = x[i, 1, l]
            y[i, wid+1, l] = y[i, 1, l]
            x[0, 0, l] = x[0, wid, l]
            y[0, 0, l] = y[0, wid, l]
            x[ver+1, 0, l] = x[1, 0, l]
            y[ver+1, 0, l] = y[1, 0, l]
            x[0, wid+1, l] = x[ver, wid+1, l]
            y[0, wid+1, l] = y[ver, wid+1, l]
            x[ver+1, wid+1, l] = x[ver+1, 1, l]
            y[ver+1, wid+1, l] = y[ver+1, 1, l]
            d_ave = (y[i-1, j, l] + y[i-1, j+1, l] + y[i, j-1, l] + y[i, j+1, l] + y[i+1, j-1, l] + y[i+1, j, l])/6
#n_ijの計算
            t[l+1] = (l+1) * s
            r11 = n_ij(t[l],x[i, j, l])
            r12 = n_ij(t[l] + s/2, x[i, j, l] + (s/2)*r11)
            r13 = n_ij(t[l] + s/2, x[i, j, l] + (s/2)*r12)
            r14 = n_ij(t[l] + s, x[i, j, l] + s*r13)
            x[i, j, l+1] = x[i, j, l] + (s/6)*(r11 + 2*r12 + 2*r13 + r14)
#d_ijの計算
            r21 = d_ij(t[l],y[i, j, l])
            r22 = d_ij(t[l] + s/2, y[i, j, l] + (s/2)*r21)
            r23 = d_ij(t[l] + s/2, y[i, j, l] + (s/2)*r22)
            r24 = d_ij(t[l] + s, y[i, j, l] + s*r23)
            y[i, j, l+1] = y[i, j, l] + (s/6)*(r21 + 2*r22 + 2*r23 + r24)

def image():
    im = Image.new('RGB', ((2*wid+ver)*20, 60+(ver-1)*3*side), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    for i in range(1, ver+1):
        for j in range(1, wid+1):
            if x[i, j, N-2] > 0.6:  # xが大きい場合(二次運命)は白
                color = (255, 255, 255)
            if x[i, j, N-2] <= 0.3:  # xが小さい場合(一次運命)は黒
                color = (0, 0, 0)
            draw.polygon(((10+40*(j-1)+20*(i-1),20+30*(i-1)), (10+40*(j-1)+20*(i-1),40+30*(i-1)), (30+40*(j-1)+20*(i-1), 50+30*(i-1)), (50+40*(j-1)+20*(i-1), 40+30*(i-1)), (50+40*(j-1)+20*(i-1), 20+30*(i-1)), (30+40*(j-1)+20*(i-1), 10+30*(i-1))), fill=color, outline = (0,0,0))        
    image_list.append(im)

image_list = []  # 各時間ごとのセルの状態画像を保存するリスト　
image()  # 各セルの状態を描画する関数を呼び出し
image_list[0].save('notch_77.gif', save_all=True, append_images=image_list[1:],
                   optimize=False, duation=200, loop=0)        