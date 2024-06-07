#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 12:32:30 2022

@author: wakamototamaki
"""

#カラーバーを作るやつ
import matplotlib as mpl
import matplotlib.pyplot as plt

#カラーバーの範囲
vmin = 0 #最小値
vmax = 1 #最大値

norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(figsize=(1, 8)) #（横：縦）で比を決める
cmap = plt.get_cmap("jet") #カラーバーの色を決める
cbar = mpl.colorbar.ColorbarBase(
    ax=ax,
    cmap=cmap,
    norm=norm,
    orientation="vertical", #縦のグラデーション→vertical、横のグラデーション→horizontal
)

plt.savefig("colormap_vertical.png", bbox_inches="tight")