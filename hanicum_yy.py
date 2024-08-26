# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 11:27:11 2022

@author: yasuf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

#頂点作るやつ#
def make_hexagon(center, length, apex):
    theta = np.arange(0, apex) * 2 * np.pi / apex + np.pi / 2 #座標がpi/3ずつ回っていく
    x = np.cos(theta) * length #頂点のx座標
    y = np.sin(theta) * length #頂点のy座標
    xy = np.stack([x, y], axis=-1) #頂点の座標の配列
    xy = xy + np.array(center) #頂点が中心に合わせて平行移動する
    return xy

i_max=5 #横(x軸)の六角形の数
j_max=3 #縦(y軸)の六角形の数
radius= 1 / np.sqrt(3) #np.sqrt(3)/2 #1 #六角形の一辺の長さ
apex=6 #頂点の数

i_box = np.arange(i_max) #横の配列
j_box = np.arange(j_max) #縦の配列
#六角形の中心の座標を入れる箱
x_box=[] 
y_box=[]
p_box=[]#六角形の頂点を入れる箱
#中心の初期値
x=0
y=0

hanicums=[] #図形を入れる箱

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_xlim((-1, 10))
ax.set_ylim((-1, 10))


for j in j_box:
    x = np.sqrt(3) /2 * radius * j #列の先頭の六角形の中心(x)
    y = 1.5*radius *j #六角形の中心(y)
    for i in i_box:
        x_box.append(x) #中心の座標を追加
        y_box.append(y)
        p = make_hexagon([x,y], radius, apex) #頂点を作って
        p_box.append(p) #配列に追加
        x = x + np.sqrt(3)*radius #次の中心を横にずらす
        patch = patches.Polygon(p, edgecolor="black", facecolor=(0, 1, 0))#六角形を作る
        hanicums.append(patch) #図形を箱に追加
        ax.add_patch(patch) #図形をfig.に追加

hanicums[8].set_facecolor(((0,0,1))) #(0,0,1)はRGBを表す．ここにa_ijのaに相当する値を反映させる工夫をすれば色が変わる．
#hanicumsは一次元ベクトルで[]内の数値を変えると六角形の場所が変わります．色々数値を変えて並び方を見てみてね．
#hanicums[0].set_facecolor('b')

plt.show()
