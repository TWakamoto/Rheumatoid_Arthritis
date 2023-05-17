# %%
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from torchdiffeq import odeint
import torch
import matplotlib.animation as animation

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
    beta2_1: float = 2.0
    beta2_2: float = 2.0
    beta3: float = 2.0
    beta4_1: float = 8.0
    beta4_2: float = 8.0

    nu0: float = 0.5
    nu1: float = 5.0
    nu2: float = 25.0
    nu3: float = 5.0

    omega_n: float = 0.01 
    omega_d: float = 0.01

    p_a: float = 0.0
    p_b: float = 0.0
    p_g: float = 0.0

    n_grid: int = 100

    canc: int = 0
    hmax: float = 12.0
    e: float = 0.01

    device: ... = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype: ... = torch.float32


    def __post_init__(self):
        self.ver = self.n_grid
        self.wid = self.n_grid

        self.n1_avg = torch.zeros((self.ver, self.wid), device=self.device, dtype=self.dtype)
        self.n2_avg = torch.zeros((self.ver, self.wid), device=self.device, dtype=self.dtype)
        self.d1_avg = torch.zeros((self.ver, self.wid), device=self.device, dtype=self.dtype)
        self.d2_avg = torch.zeros((self.ver, self.wid), device=self.device, dtype=self.dtype)

        self.rhs = torch.zeros((11, self.ver + 2, self.wid + 2), device=self.device, dtype=self.dtype)

    def random_init(self):
        (ver, wid, canc) = (self.ver, self.wid, self.canc)

        y_init = torch.zeros((11, ver+2, wid+2), device=self.device, dtype=self.dtype)
        if(canc==0):
            y_init[:, 1:-1, 1:-1] = self.e*torch.rand((11, ver, wid), device=self.device, dtype=self.dtype)
        else:
            y_init [0:2, 1:-1, 1:-1] = self.gamma1*self.omega_n/(self.mu1*(self.mu5 + self.gamma1))
            y_init[4:6, 1:-1, 1:-1] = self.omega_n/(self.mu5 + self.gamma1)
            y_init[6, 1:-1, 1:-1] = (1/(self.mu6 + self.gamma2))*(self.omega_d + self.beta4_1/(self.beta3 + (self.nu0/self.mu0)**2))
            y_init[7, 1:-1, 1:-1] = (1/(self.mu6 + self.gamma2))*(self.omega_d + self.beta4_2/(self.beta3 + (self.nu0/self.mu0)**2))
            y_init[-1, 1:-1, 1:-1] = self.nu0/self.mu0
            y_init[-1,int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = self.hmax

        return y_init

    def __call__(self, t, y):
        Nm1, Nm2, Dm1, Dm2, Nc1, Nc2, Dc1, Dc2, I1, I2, H1 = torch.unbind(y)

        # define equations
        self.n1_ave = (Nm1[0:-2, 1:-1] + Nm1[2:,1:-1] + Nm1[1:-1, 0:-2] + Nm1[1:-1, 2:])/4
        self.n2_ave = (Nm2[0:-2, 1:-1] + Nm2[2:,1:-1] + Nm2[1:-1, 0:-2] + Nm2[1:-1, 2:])/4
        self.d1_ave = (Dm1[0:-2, 1:-1] + Dm1[2:,1:-1] + Dm1[1:-1, 0:-2] + Dm1[1:-1, 2:])/4
        self.d2_ave = (Dm2[0:-2, 1:-1] + Dm2[2:,1:-1] + Dm2[1:-1, 0:-2] + Dm2[1:-1, 2:])/4

        dNm1, dNm2, dDm1, dDm2, dNc1, dNc2, dDc1, dDc2, dI1, dI2, dH1 = torch.unbind(self.rhs)

        # take the inner part of the concentrations (note: no data is copied, these are just views)
        Nm1_ = Nm1[1:-1,1:-1]
        Nm2_ = Nm2[1:-1,1:-1]
        Dm1_ = Dm1[1:-1,1:-1]
        Dm2_ = Dm2[1:-1,1:-1]
        Dc1_ = Dc1[1:-1,1:-1]
        Dc2_ = Dc2[1:-1,1:-1]
        (p_a, p_b, p_g) = (self.p_a, self.p_b, self.p_g)

        dNm1[1:-1,1:-1] = (1 + p_a) * (-self.alpha1 * self.d1_ave * Nm1_ - self.alpha2 * self.d2_ave * Nm1_) - self.mu1 * Nm1_ + (1 + p_g) * self.gamma1 * Nc1[1:-1,1:-1]
        dNm2[1:-1,1:-1] = (1 + p_a) * (-self.alpha5 * self.d1_ave * Nm2_ - self.alpha6 * self.d2_ave * Nm2_) - self.mu1 * Nm2_ + (1 + p_g) * self.gamma1 * Nc2[1:-1,1:-1]
        dDm1[1:-1,1:-1] = (1 + p_a) * (-self.alpha1 * self.n1_ave * Dm1_ - self.alpha5 * self.n2_ave * Dm1_) - self.mu2 * Dm1_ + self.gamma2 * Dc1[1:-1,1:-1]
        dDm2[1:-1,1:-1] = (1 + p_a) * (-self.alpha2 * self.n1_ave * Dm2_ - self.alpha6 * self.n2_ave * Dm2_) - self.mu2 * Dm2_ + self.gamma2 * Dc2[1:-1,1:-1]

        dI1[1:-1,1:-1] = (1 + p_a) * (self.alpha1 * self.d1_ave * Nm1_ + self.alpha2 * self.d2_ave * Nm1_) - self.mu4 * I1[1:-1,1:-1]  
        dI2[1:-1,1:-1] = (1 + p_a) * (self.alpha5 * self.d1_ave * Nm2_ + self.alpha6 * self.d2_ave * Nm2_) - self.mu4 * I2[1:-1,1:-1]

        dH1 = self.nu0 + (self.nu2 * I1**2) / (self.nu1 + I1**2) * (1.0 - I2**2 / (self.nu3 + I2**2)) - self.mu0 * H1 
        dNc1[1:-1,1:-1] = self.omega_n + (1 + p_b) * self.beta2_1 * ( I1[1:-1,1:-1]**2 / (self.beta1 + I1[1:-1,1:-1]**2) ) - (self.mu5 + (1 + p_g) * self.gamma1) * Nc1[1:-1,1:-1]
        dNc2[1:-1,1:-1] = self.omega_n + (1 + p_b) * self.beta2_2 * ( I2[1:-1,1:-1]**2 / (self.beta1 + I2[1:-1,1:-1]**2) ) - (self.mu5 + (1 + p_g) * self.gamma1) * Nc2[1:-1,1:-1]

        dDc1 = self.omega_d + self.beta4_1 / (self.beta3 + H1**2) - (self.mu6 + self.gamma2) * Dc1
        dDc2 = self.omega_d + self.beta4_2 / (self.beta3 + H1**2) - (self.mu6 + self.gamma2) * Dc2

        (canc, ver, wid) = (self.canc, self.ver, self.wid)

        if( canc > 0 ):
        #get index of cancer cell===============================================
            index = torch.where(H1[2:-2, 2:-2] > 0.75*self.hmax)
            index_up = index[0]+1
            index_down = index[0]-1
            index_right = index[1]+1
            index_left = index[1]-1
            index_ver = torch.cat((torch.cat((torch.cat((index_up, index_down), dim=0), index[0]), dim=0), index[0]), dim=0)
            index_wid = torch.cat((torch.cat((torch.cat((index[1], index[1]), dim=0), index_left), dim=0), index_right), dim=0)
            index_new = torch.stack([index_ver, index_wid])
            index_new = index_new.T
            index_ = torch.unique(index_new, dim=0)
            index_ = index_.T

            dDm1[1:-1, 1:-1] = torch.zeros((ver, wid), device=self.device, dtype=self.dtype)
            dDm2[1:-1, 1:-1] = torch.zeros((ver, wid), device=self.device, dtype=self.dtype)
            #cells next to cancer cell
            dDm1[index_[0]+2, index_[1]+2] = -self.alpha1 * self.n1_ave[index_[0]+1, index_[1]+1] * Dm1_[index_[0]+1, index_[1]+1] - self.alpha5 * self.n2_ave[index_[0]+1, index_[1]+1] * Dm1_[index_[0]+1, index_[1]+1] - self.mu2 * Dm1_[index_[0]+1, index_[1]+1] + self.gamma2 * Dc1_[index_[0]+1, index_[1]+1]
            dDm2[index_[0]+2, index_[1]+2] = -self.alpha2 * self.n1_ave[index_[0]+1, index_[1]+1] * Dm2_[index_[0]+1, index_[1]+1] - self.alpha6 * self.n2_ave[index_[0]+1, index_[1]+1] * Dm2_[index_[0]+1, index_[1]+1] - self.mu2 * Dm2_[index_[0]+1, index_[1]+1] + self.gamma2 * Dc2_[index_[0]+1, index_[1]+1]
            
            #condition of cancer=========================================================================================================================================================================================================================================================
            dNm1[int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            dNm2[int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            dDm1[int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            dDm2[int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            dNc1[int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            dNc2[int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            dDc1[int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            dDc2[int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            dI1[int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            dI2[int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            dH1[int((ver+2-canc)/2):int((ver+2+canc)/2), int((wid+2-canc)/2):int((wid+2+canc)/2)] = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        return torch.stack( (dNm1, dNm2, dDm1, dDm2, dNc1, dNc2, dDc1, dDc2, dI1, dI2, dH1) )

# %% set up pytorch==========

dtype = torch.float32
device = torch.device("cpu")
# %%==============================================
# simulation without treatment

T = 250.0 #最終時刻
model = Model(n_grid=100, e = 0.01, device=device)
y_init = model.random_init()

# %%====================================================
rtol = torch.tensor( 1e-6, device=device, dtype = dtype)
atol = torch.tensor( 1e-7, device=device, dtype = dtype)
t = torch.tensor([0, T], device=device, dtype=dtype)

# %% solve ODEs=================================================================================
%%time
sol = odeint(model, y_init, t, atol=atol, rtol=rtol, method="bosh3", options= dict(dtype=dtype))


# %%================================
# extract components of the solution
def get_H1(sol):
    return sol[:, 10, :, :]


# %%=======================
sol_H1 = get_H1(sol)
hmax = torch.max( sol_H1 )
print(f"hmax = {hmax}")

# %% get the position of cell which has the max value of HES-1===============================
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
# %% run experiment ===============
(ver, wid) = (model.ver, model.wid)

# %% animation function========================================================================
def plot_animation(t, sol_component, fn):
    fig = plt.figure()

    vmin = 0.0
    vmax = np.max(sol_component)

    # init plot
    im = plt.imshow(sol_component[0, :,:], vmin = vmin, vmax = vmax, cmap='jet')
    plt.colorbar()
        
    def animate_func(i):
        im.set_array(sol_component[i, :,:])
        plt.title(f"t = {t[i]:.{2}f}")
        return [im]

    anim = animation.FuncAnimation(fig, animate_func, frames=len(t), interval=100, blit=False)
    anim.save(fn)
    return anim

# %% tratment effect===========================================
def p_rand():
    return torch.rand( (ver, wid), device=device, dtype=dtype)
p_0 = 0.0

# %%=======================================================
model.canc = 30  # number of cancer
model.hmax = hmax # max value of H1 (used for initial data)

#%%================================================================
canc = model.canc
T = 1000
t = torch.tensor(np.linspace(0, T, 50), device=device, dtype=dtype)
t_anim = t.to('cpu').detach().numpy().copy()

# %% initial condition of cancer cell================================================================
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


# %% solve ODEs=================================================================================
%%time
model.p_a = p_0
model.p_b = p_0
model.p_g = p_0
sol = odeint( model, y_init, t, atol=atol, rtol=rtol, method="bosh3", options= dict(dtype=dtype))

# %%==============================================================
# get color map of HES-1 (animation)
sol_H1 = get_H1(sol)
sol_H1anim = sol_H1.to('cpu').detach().numpy().copy()
plot_animation(t_anim, sol_H1anim[:, 1:-1, 1:-1], "cancer-spread_2d_h_lc.gif")

# %%=======================================================================================================
# get color map of protein (or gene) A (animation)
sol_anim = sol[:, 10, 1:-1, 1:-1].to('cpu').detach().numpy().copy()
plot_animation(t_anim, sol_anim, "cancer-spread_2d_h_lc.gif")

# %%
sol_ = sol.to('cpu').detach().numpy().copy()

# %%=================================================================================================
# animation function
# graph showing dynamics of each cell
def plot_graph(t, sol_component, fn, y): #(time, sol, file_name, y_axis)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cell_ = np.arange(wid)
    y_max = np.max(sol_component[:, y, :])
    y_max_ = y_max+0.1*y_max
    # init plot
    im, = ax.plot(cell_, sol_component[0, y, :])
    ax.set_ylim([0, y_max_])

    title = ax.set_title(f"t = {t[0]:.{2}f}")
        
    def animate_func(i):
        im.set_data(cell_, sol_component[i, y,:])
        title.set_text(f"t = {t[i]:.{2}f}")
        return [im]

    anim = animation.FuncAnimation(fig, animate_func, frames=len(t), interval=100, blit=False)
    anim.save(fn)
    return anim

# %%================================================================================================================
# animation
# x_axis: cell position, y_axis: level of protein or gene
plot_graph(t_anim, sol_[:, 10, 1:-1, 1:-1], "test_graph_h_lc.gif", 95)








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
fig1.savefig("papbpg_actLC.png")

plt.show()

# %%
