# %%
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from torchdiffeq import odeint
import torch
import matplotlib.animation as animation
import matplotlib.cm as cm

# %%
# definition of the ODE system
@dataclass
class Model():

    alpha1: float = 6.0
    alpha2: float = 10.0
    alpha5: float = 8.0
    alpha6: float = 6.0

    mu0: float = 0.5
    mu1: float = 1.0
    mu2: float = 1.0
    mu4: float = 0.1
    mu5: float = 1.0
    mu6: float = 0.5

    gamma1: float = 2.0
    gamma2: float = 1.0

    beta1: float = 0.1
    beta2_1: float = 3.0
    beta2_2: float = 10.0
    beta3: float = 2.0
    beta4_1: float = 10.0
    beta4_2: float = 10.0

    nu0: float = 0.5
    nu1: float = 5.0
    nu2: float = 25.0
    nu3: float = 5.0

    omega_n: float = 0.01
    omega_d: float = 0.01

    p_1: float = 0.0
    p_23: float = 0.0
    p_b: float = 0.0

    n_grid: int = 11

    canc: int = 0
    hmax: float = 12.0
    e: float = 0.01

    device: ... = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype: ... = torch.float32


    def __post_init__(self):
        self.ver = 1
        self.wid = self.n_grid

        self.n1_avg = torch.zeros((self.ver, self.wid), device=self.device, dtype=self.dtype)
        self.n2_avg = torch.zeros((self.ver, self.wid), device=self.device, dtype=self.dtype)
        self.d1_avg = torch.zeros((self.ver, self.wid), device=self.device, dtype=self.dtype)
        self.d2_avg = torch.zeros((self.ver, self.wid), device=self.device, dtype=self.dtype)

        self.rhs = torch.zeros((11, self.ver + 2, self.wid + 2), device=self.device, dtype=self.dtype)


    #initial conditions===============================================================================================
    def random_init(self):
        (ver, wid, canc) = (self.ver, self.wid, self.canc)
        y_init = torch.zeros((11, ver+2, wid+2), device=self.device, dtype=self.dtype)
        if canc == 0:
            y_init[:, 1:-1, 1:-1] = self.e*torch.rand((11, ver, wid), device=self.device, dtype=self.dtype)
        else:
            y_init [0:2, 1:-1, 1:-1] = self.gamma1*self.omega_n/(self.mu1*(self.mu5 + self.gamma1))
            y_init[4:6, 1:-1, 1:-1] = self.omega_n/(self.mu5 + self.gamma1)
            y_init[6, 1:-1, 1:-1] = (1/(self.mu6 + self.gamma2))*(self.omega_d + self.beta4_1/(self.beta3 + (self.nu0/self.mu0)**2))
            y_init[7, 1:-1, 1:-1] = (1/(self.mu6 + self.gamma2))*(self.omega_d + self.beta4_2/(self.beta3 + (self.nu0/self.mu0)**2))
            y_init[-1, 1:-1, 1:-1] = self.nu0/self.mu0
            y_init[-1,int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = self.hmax

        return y_init

    #ODE models=======================================================================================================
    def __call__(self, t, y):
        Nm1, Nm2, Dm1, Dm2, Nc1, Nc2, Dc1, Dc2, I1, I2, H1 = torch.unbind(y)

        # define equations
        self.n1_ave = (Nm1[1:-1, 0:-2] + Nm1[1:-1, 2:])/2
        self.n2_ave = (Nm2[1:-1, 0:-2] + Nm2[1:-1, 2:])/2
        self.d1_ave = (Dm1[1:-1, 0:-2] + Dm1[1:-1, 2:])/2
        self.d2_ave = (Dm2[1:-1, 0:-2] + Dm2[1:-1, 2:])/2

        dNm1, dNm2, dDm1, dDm2, dNc1, dNc2, dDc1, dDc2, dI1, dI2, dH1 = torch.unbind(self.rhs)

        # take the inner part of the concentrations (note: no data is copied, these are just views)
        Nm1_ = Nm1[1:-1,1:-1]
        Nm2_ = Nm2[1:-1,1:-1]
        Dm1_ = Dm1[1:-1,1:-1]
        Dm2_ = Dm2[1:-1,1:-1]
        Dc1_ = Dc1[1:-1,1:-1]
        Dc2_ = Dc2[1:-1,1:-1]
        (p_1, p_23, p_b) = (self.p_1, self.p_23, self.p_b)
        
        dNm1[1:-1,1:-1] = -self.alpha1 * self.d1_ave * Nm1_ - self.alpha2 * self.d2_ave * Nm1_ - self.mu1 * Nm1_ + (1 - p_1) * self.gamma1 * Nc1[1:-1,1:-1]
        dNm2[1:-1,1:-1] = -self.alpha5 * self.d1_ave * Nm2_ - self.alpha6 * self.d2_ave * Nm2_ - self.mu1 * Nm2_ + (1 - p_1) * self.gamma1 * Nc2[1:-1,1:-1]
        dDm1[1:-1,1:-1] = -self.alpha1 * self.n1_ave * Dm1_ - self.alpha5 * self.n2_ave * Dm1_ - self.mu2 * Dm1_ + self.gamma2 * Dc1[1:-1,1:-1]
        dDm2[1:-1,1:-1] = -self.alpha2 * self.n1_ave * Dm2_ - self.alpha6 * self.n2_ave * Dm2_ - self.mu2 * Dm2_ + self.gamma2 * Dc2[1:-1,1:-1]

        dI1[1:-1,1:-1] = (1 - p_23) * (self.alpha1 * self.d1_ave * Nm1_ + self.alpha2 * self.d2_ave * Nm1_) - self.mu4 * I1[1:-1,1:-1]  
        dI2[1:-1,1:-1] = (1 - p_23) * (self.alpha5 * self.d1_ave * Nm2_ + self.alpha6 * self.d2_ave * Nm2_) - self.mu4 * I2[1:-1,1:-1]

        dH1 = self.nu0 + (self.nu2 * I2**2) / (self.nu1 + I2**2) * (1.0 - I1**2 / (self.nu3 + I1**2)) - self.mu0 * H1 
        dNc1[1:-1,1:-1] = self.omega_n + (1 - p_b) * self.beta2_1 * ( I1[1:-1,1:-1]**2 / (self.beta1 + I1[1:-1,1:-1]**2) ) - (self.mu5 + (1 - p_1) * self.gamma1) * Nc1[1:-1,1:-1]
        dNc2[1:-1,1:-1] = self.omega_n + (1 - p_b) * self.beta2_2 * ( I2[1:-1,1:-1]**2 / (self.beta1 + I2[1:-1,1:-1]**2) ) - (self.mu5 + (1 - p_1) * self.gamma1) * Nc2[1:-1,1:-1]

        dDc1 = self.omega_d + self.beta4_1 / (self.beta3 + H1**2) - (self.mu6 + self.gamma2) * Dc1
        dDc2 = self.omega_d + self.beta4_2 / (self.beta3 + H1**2) - (self.mu6 + self.gamma2) * Dc2

        (canc, ver, wid) = (self.canc, self.ver, self.wid)

        if( canc > 0 ):
            #get index of cancer cell===============================================
            index1 = torch.where(H1[1, 2:-2] > 0.75*self.hmax)
            index1_right = index1[0]+1
            index1_left = index1[0]-1
            index_1 = torch.cat((index1_left, index1_right), dim=0)
            index1_ = torch.unique(index_1, dim=0)

            dDm1[1:-1, 1:-1] = torch.zeros((ver, wid), device=self.device, dtype=self.dtype)
            dDm2[1:-1, 1:-1] = torch.zeros((ver, wid), device=self.device, dtype=self.dtype)
            #cells next to cancer cell
            dDm1[1:-1, index1_+2] = -self.alpha1 * self.n1_ave[:, index1_+1] * Dm1_[:, index1_+1] - self.alpha5 * self.n2_ave[:, index1_+1] * Dm1_[:, index1_+1] - self.mu2 * Dm1_[:, index1_+1] + self.gamma2 * Dc1_[:, index1_+1]
            dDm2[1:-1, index1_+2] = -self.alpha2 * self.n1_ave[:, index1_+1] * Dm2_[:, index1_+1] - self.alpha6 * self.n2_ave[:, index1_+1] * Dm2_[:, index1_+1] - self.mu2 * Dm2_[:, index1_+1] + self.gamma2 * Dc2_[:, index1_+1]
            
            #condition of cancer======================================================================================
            dNm1[1, int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            dNm2[1, int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            dDm1[1, int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            dDm2[1, int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            dNc1[1, int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            dNc2[1, int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            dDc1[1, int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            dDc2[1, int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            dI1[1, int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            dI2[1, int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            dH1[1, int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        return torch.stack( (dNm1, dNm2, dDm1, dDm2, dNc1, dNc2, dDc1, dDc2, dI1, dI2, dH1) )

# %% set up pytorch==========

dtype = torch.float32
device = torch.device("cpu")
# %%============================================
# simulation without treatment

T = 200.0 #最終時
model = Model(n_grid=11, e = 0.01, device=device)
y_init = model.random_init()

# %%====================================================
rtol = torch.tensor( 1e-6, device=device, dtype = dtype)
atol = torch.tensor( 1e-7, device=device, dtype = dtype)
t = torch.tensor([0, T], device=device, dtype=dtype)

# %% solve ODE ============================================================================================
%%time
sol = odeint(model, y_init, t, atol=atol, rtol=rtol, method="bosh3", options= dict(dtype=dtype))


# %%================================
# extract components of the solution
def get_H1(sol, k):
    return sol[:, k, :, :]

# %% get highest level of HES-1 ================================
sol_H1 = get_H1(sol, 10)
hmax = torch.max( sol_H1[-1, :, :] )
print(f"hmax = {hmax}")

# %% get the level of protein in cell which has highest level of HES-1 =======================
position = torch.argmax( sol_H1[-1, :, :] )

nm1_max = torch.ravel(sol[-1, 0, :, :])[position]
nm2_max = torch.ravel(sol[-1, 1, :, :])[position]
dm1_max = torch.ravel(sol[-1, 2, :, :])[position]
dm2_max = torch.ravel(sol[-1, 3, :, :])[position]
nc1_max = torch.ravel(sol[-1, 4, :, :])[position]
nc2_max = torch.ravel(sol[-1, 5, :, :])[position]
dc1_max = torch.ravel(sol[-1, 6, :, :])[position]
dc2_max = torch.ravel(sol[-1, 7, :, :])[position]
i1_max = torch.ravel(sol[-1, 8, :, :])[position]
i2_max = torch.ravel(sol[-1, 9, :, :])[position]

print(nm1_max, nm2_max, dm1_max, dm2_max, nc1_max, nc2_max, dc1_max, dc2_max, i1_max, i2_max)

# %% run experiment ========
(ver, wid) = (1, model.wid)

# %% animation function====================================================================================================================
def plot_animation(t, sol_component, fn):
    fig = plt.figure()

    vmin = 0.0
    vmax = np.max(sol_component[-1, :, :])

    # init plot
    im = plt.imshow(sol_component[0, :,:], vmin = vmin, vmax = vmax, cmap='jet')
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
    plt.colorbar()
        
    def animate_func(i):
        im.set_array(sol_component[i, :,:])
        plt.title(f"t = {t[i]:.{2}f}")
        return [im]

    anim = animation.FuncAnimation(fig, animate_func, frames=len(t), interval=100, blit=False)
    anim.save(fn)
    return anim

# %%ratio of treatment effect=================================
def p_rand():
    return torch.rand( (ver, wid), device=device, dtype=dtype)
p_0 = 0.0


# %%=======================================================
model.canc = 1  # number of cancer
canc = model.canc

model.hmax = hmax # max value of H1 (used for initial data)

# %%===============================================================
t = torch.tensor(np.linspace(0, T, 50), device=device, dtype=dtype)
t_anim = t.to('cpu').detach().numpy().copy()

# %% initial condition for cancer cell===============================================================
e = model.e
y_init = model.random_init()
y_init[0, int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = nm1_max
y_init[1, int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = nm2_max
y_init[2, int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = dm1_max
y_init[3, int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = dm2_max
y_init[4, int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = nc1_max
y_init[5, int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = nc2_max
y_init[6, int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = dc1_max
y_init[7, int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = dc2_max
y_init[8, int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = i1_max
y_init[9, int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = i2_max

# %% solve ODEs==================================================================================
%%time
model.p_1 = p_0
model.p_23 = p_0
model.p_b = p_0
sol = odeint( model, y_init, t, atol=atol, rtol=rtol, method="bosh3", options= dict(dtype=dtype))

# %%======================================================================================================
# get color map of HES-1 (animation)
sol_H1 = get_H1(sol, 10)
sol_H1anim = sol_H1.to('cpu').detach().numpy().copy()
plot_animation(t_anim, sol_H1anim[:, 1:-1, 1:-1], "test_beta-1d_bt.gif")

# %%=======================================================================================================
# get color map of protein (or gene) A (animation)
sol_anim = sol[:, 1, 1:-1, 1:-1].to('cpu').detach().numpy().copy()
plot_animation(t_anim, sol_anim, "test_beta-1d_nm2_bt.gif")

# %%========================================
sol_ = sol.to('cpu').detach().numpy().copy()

# %%=======================================================================================================
# figure showing dynamics of protein (or gene) A
fig1 = plt.figure(figsize=(8,5))
ax1 = fig1.add_subplot(111)
protein1 = 10
ax1.plot(t_anim, sol_[:, protein1, 1, 0], label="cell2", color=cm.viridis(0/10))
ax1.plot(t_anim, sol_[:, protein1, 1, 1], label="cell1", color=cm.viridis(1/10))
ax1.plot(t_anim, sol_[:, protein1, 1, 2], label="cell3", color=cm.viridis(2/10))
ax1.plot(t_anim, sol_[:, protein1, 1, 3], label="cell4", color=cm.viridis(3/10))
ax1.plot(t_anim, sol_[:, protein1, 1, 4], label="cell5", color=cm.viridis(4/10))
ax1.plot(t_anim, sol_[:, protein1, 1, 5], label="cell6", color=cm.viridis(5/10))
ax1.plot(t_anim, sol_[:, protein1, 1, 6], label="cell7", color=cm.viridis(6/10))
ax1.plot(t_anim, sol_[:, protein1, 1, 7], label="cell8", color=cm.viridis(7/10))
ax1.plot(t_anim, sol_[:, protein1, 1, 8], label="cell9", color=cm.viridis(8/10))
ax1.plot(t_anim, sol_[:, protein1, 1, 9], label="cell10", color=cm.viridis(9/10))
ax1.plot(t_anim, sol_[:, protein1, 1, 10], label="cell11", color=cm.viridis(10/10))

ax1.legend(loc="best", fontsize=10)

# %%=================================================================================================
# animation function
# graph showing dynamics of each cell
def plot_graph(t, sol_component, fn):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cell_ = np.arange(wid)
    y_max = np.max(sol_component[:, 1, :])
    y_max_ = y_max+0.1*y_max
    # init plot
    im, = ax.plot(cell_, sol_component[0, 1, :])
    ax.set_ylim([0, y_max_])

    title = ax.set_title(f"t = {t[0]:.{2}f}")
        
    def animate_func(i):
        im.set_data(cell_, sol_component[i, 1,:])
        title.set_text(f"t = {t[i]:.{2}f}")
        return [im]

    anim = animation.FuncAnimation(fig, animate_func, frames=len(t), interval=100, blit=False)
    anim.save(fn)
    return anim

# %%================================================================================================================
# animation
# x_axis: cell position, y_axis: level of protein or gene
plot_graph(t_anim, sol_[:, 10, :, 1:-1], "beta4-10_graph_h_bt.gif")

# %%===================================================================================================
# figure (except for cancer cell)
# x_axis: cell, y_axis: level of protein or gene
fig2 = plt.figure(figsize=(8,5))
ax2 = fig2.add_subplot(111)
cell = np.arange(int(wid/2))

protein = -1
ax2.plot(cell, sol_[0, protein, 1, 1:int(wid/2)+1], color=cm.viridis(0/3))
ax2.plot(cell+8, sol_[0, protein, 1, int(wid/2)+2:-1], label=f"t={t[0]:.{2}f}", color=cm.viridis(0/3))
ax2.plot(cell, sol_[4, protein, 1, 1:int(wid/2)+1], color=cm.viridis(1/3))
ax2.plot(cell+8, sol_[4, protein, 1, int(wid/2)+2:-1], label=f"t={t[4]:.{2}f}", color=cm.viridis(1/3))
ax2.plot(cell, sol_[35, protein, 1, 1:int(wid/2)+1], color=cm.viridis(2/3))
ax2.plot(cell+8, sol_[35, protein, 1, int(wid/2)+2:-1], label=f"t={t[35]:.{2}f}", color=cm.viridis(2/3))
ax2.plot(cell, sol_[-1, protein, 1, 1:int(wid/2)+1], color=cm.viridis(3/3))
ax2.plot(cell+8, sol_[-1, protein, 1, int(wid/2)+2:-1], label=f"t={t[-1]:.{2}f}", color=cm.viridis(3/3))
ax2.legend(loc="best", fontsize=10)


















# %%=================================================================================================================
check = sol_[10, 2, 1, 7]
if check>0:
    print("positive")
if check == 0:
    print("zero")

# %%
count = np.zeros(11)
for i in range(11):
    count[i] = (np.count_nonzero((sol_H1[-1, :, :]>=(i*5)) & (sol_H1[-1,:,:] < (i+1)*5)))/(ver*wid)
# %%
fig1 = plt.figure(figsize=(10,5))
ax1 = fig1.add_subplot(111)
label1 = np.array([0, 5, 10, 15, 20, 25, 30])
label2 = np.array([35, 40, 45, 50])
ax1.bar(label1+2.5, count[:7], width=4.8)
ax1.bar(label2+2.5, count[7:], width=4.8)
ax1.set_xlim([0, 55])
fig1.savefig("p1p23pb_inhBT.png")

plt.show()