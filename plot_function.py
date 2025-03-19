# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D

# %%
x = np.arange(0, 60, 1)


# %%
n1=1
n2=3
n3=8
n5=10
betaS=10.0
y1 = 1 - x**n1/(betaS**n1 + x**n1)
y2 = 1 - x**n2/(betaS**n2 + x**n2)
y3 = 1 - x**n3/(betaS**n3 + x**n3)
y4 = 1 - x**n5/(betaS**n5 + x**n5)

fig1 = plt.figure(figsize=(10, 5))
ax1 = fig1.add_subplot(111)
#ax1.set_ylim(0, 1)
ax1.plot(x, y1, label="k=1")
ax1.plot(x, y2, label="k=3")
ax1.plot(x, y3, label="k=8")
#ax1.plot(x, y4, label="k=5")
ax1.set_xlim(0, 60)

#ax1.legend(loc="best")
#fig1.savefig("sigmaS_function2.png")
plt.show()


# %%
a1 = 1.0
b1 = 34.0
c1 = 15.0

y = a1*np.exp(-((x-b1)**2)/(2*c1*c1))

fig1 = plt.figure(figsize=(10, 5))
ax1 = fig1.add_subplot(111)
#ax1.set_ylim(0, 1)
ax1.plot(x, y, label="exp")
ax1.set_xlim(0, 60)

plt.show()

# %%
a1 = 0.95
b1 = 0.0
c1 = 10

a2 = 0.5
b2 = 60
c2 = 20
y = a1*np.exp(-((x-b1)**2)/(2*c1*c1)) + a2*np.exp(-((x-b2)**2)/(2*c2*c2))

fig1 = plt.figure(figsize=(10, 5))
ax1 = fig1.add_subplot(111)
#ax1.set_ylim(0, 1)
ax1.plot(x, y, label="exp")
ax1.set_xlim(0, 60)

#ax1.legend(loc="best")
#fig1.savefig("sigmaS_function2.png")
plt.show()


# %%
color = cm.inferno
# %%
x = np.arange(0, 60, 1)
y = np.arange(0, 60, 1)
X, Y = np.meshgrid(x, y)
x1 = 20
y1 = 40
x2 = 40
y2 = 20
s = np.array([[30, 0.1], [0.1, 30]])
s_ = np.array([[s[1,1], -s[1,0]], [-s[0,1], s[0,0]]])/(s[0,0]*s[1,1] - s[0,1]*s[1,0])
a = 1
z1 = a*np.exp((-1/4)*((s_[0,0]*((X-x1)**2) + (s_[0,1]+s_[1,0])*(X-x1)*(Y-y1) + s_[1,1]*((Y-y1)**2))*(1/4)*(s_[0,0]*((X-x2)**2) + (s_[0,1]+s_[1,0])*(X-x2)*(Y-y2) + s_[1,1]*((Y-y2)**2))))
#z2 = a*np.exp(-(s_[0,0]*((X-x2)**2) + (s_[0,1]+s_[1,0])*(X-x2)*(Y-y2) + s_[1,1]*((Y-y2)**2))/2)
#z3 = z1+z2
fig2 = plt.figure(figsize=(10, 5))
ax2 = fig2.add_subplot(projection='3d')
ax2.plot_surface(X, Y, z1, cmap=color, vmax=1.0, vmin=0)
#ax2.view_init(elev=0, azim=45)
#ax2.set_xlim(0, 60)
#ax2.set_ylim(0, 30)
#fig2.savefig("sigmaS_NL_2p_25.png")
plt.show()

# %%
x1 = np.arange(0.0, 10.0, 0.1)
b1 = 1.0
b2 = 5.0
y = b1*x1/(b2 + x1)

fig3 = plt.figure(figsize=(10, 5))
ax3 = fig3.add_subplot(111)
ax3.plot(x1, y)
# %%
