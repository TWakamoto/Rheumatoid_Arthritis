# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
param = 13
Si = np.array([])
# %%
fig1 = plt.figure(figsize=(10,5))
ax1 = fig1.add_subplot(111)
label = [r"$\alpha_1$", r"$\alpha_2$", r"$\alpha_5$", r"$\alpha_6$", r"$\gamma_N$", r"$\gamma_D$", r"$\beta_{21}$", r"$\beta_{22}$", r"$\beta_{41}$", r"$\beta_{42}$", r"$\nu_2$", r"$\nu_3$", "dummy"]
left = np.arange(param)
ax1.bar(left, Si, tick_label=label)
ax1.axhline(y=Si[param-1], linestyle='dashed', color='black')
fig1.savefig("CellDivision_SA_1st_lc_2000.png")
