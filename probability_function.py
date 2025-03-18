# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
x = np.arange(0, 60, 1)

# %%
#parameter
a1 = 0.25
a2 = 0.5
a3 = 1.0

y1 = 1/(1 + np.exp(a1*(50*0.75 - x)))
y2 = 1/(1 + np.exp(a2*(50*0.75 - x)))
y3 = 1/(1 + np.exp(a3*(50*0.75 - x)))

fig1 = plt.figure(figsize=(10, 5))
ax1 = fig1.add_subplot(111)
ax1.plot(x, y1, label="a=0.25")
ax1.plot(x, y2, label="a=0.5")
ax1.plot(x, y3, label="a=1.0")

# %%
n=3
betaS=10.0**n
a = 30
y = 1 - (x**n)/(betaS + (x**n))
y2 = 1 - ((2*a-x)**n)/(betaS + ((2*a-x)**n))
y3 = y + y2
fig1 = plt.figure(figsize=(10, 5))
ax1 = fig1.add_subplot(111)
#ax1.set_ylim(0, 1)
#ax1.plot(x, y)
#ax1.plot(x, y2)
ax1.plot(x, y3)
# %%
#ax1.legend(loc='best')
fig1.savefig("sigmaS_function.png")
plt.show()

# %%
x = np.arange(0, 100, 1)
a1 = 0.1
y1 = 1 - 1/(1 + np.exp(a1*(50 - x)))

# %%
fig1 = plt.figure(figsize=(10, 5))
ax1 = fig1.add_subplot(111)
ax1.plot(x, y1)

fig1.savefig("sigma_s.png")
plt.show()