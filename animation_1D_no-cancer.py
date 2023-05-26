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
    beta2_1: float = 8.0
    beta2_2: float = 5.0
    beta3: float = 2.0
    beta4_1: float = 8.0
    beta4_2: float = 8.0

    nu0: float = 0.5
    nu1: float = 5.0
    nu2: float = 25.0
    nu3: float = 5.0

    omega_n: float = 0.0
    omega_d: float = 0.0

    n_grid: int = 11

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
        (ver, wid) = (self.ver, self.wid)
        y_init = torch.zeros((11, ver+2, wid+2), device=self.device, dtype=self.dtype)
        y_init[:, 1:-1, 1:-1] = self.e*torch.rand((11, ver, wid), device=self.device, dtype=self.dtype)
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
        
        dNm1[1:-1,1:-1] = -self.alpha1 * self.d1_ave * Nm1_ - self.alpha2 * self.d2_ave * Nm1_ - self.mu1 * Nm1_ + self.gamma1 * Nc1[1:-1,1:-1]
        dNm2[1:-1,1:-1] = -self.alpha5 * self.d1_ave * Nm2_ - self.alpha6 * self.d2_ave * Nm2_ - self.mu1 * Nm2_ + self.gamma1 * Nc2[1:-1,1:-1]
        dDm1[1:-1,1:-1] = -self.alpha1 * self.n1_ave * Dm1_ - self.alpha5 * self.n2_ave * Dm1_ - self.mu2 * Dm1_ + self.gamma2 * Dc1[1:-1,1:-1]
        dDm2[1:-1,1:-1] = -self.alpha2 * self.n1_ave * Dm2_ - self.alpha6 * self.n2_ave * Dm2_ - self.mu2 * Dm2_ + self.gamma2 * Dc2[1:-1,1:-1]

        dI1[1:-1,1:-1] = self.alpha1 * self.d1_ave * Nm1_ + self.alpha2 * self.d2_ave * Nm1_ - self.mu4 * I1[1:-1,1:-1]  
        dI2[1:-1,1:-1] = self.alpha5 * self.d1_ave * Nm2_ + self.alpha6 * self.d2_ave * Nm2_ - self.mu4 * I2[1:-1,1:-1]

        dH1 = self.nu0 + (self.nu2 * I1**2) / (self.nu1 + I1**2) * (1.0 - I2**2 / (self.nu3 + I2**2)) - self.mu0 * H1 
        dNc1[1:-1,1:-1] = self.omega_n + self.beta2_1 * ( I1[1:-1,1:-1]**2 / (self.beta1 + I1[1:-1,1:-1]**2) ) - (self.mu5 + self.gamma1) * Nc1[1:-1,1:-1]
        dNc2[1:-1,1:-1] = self.omega_n + self.beta2_2 * ( I2[1:-1,1:-1]**2 / (self.beta1 + I2[1:-1,1:-1]**2) ) - (self.mu5 + self.gamma1) * Nc2[1:-1,1:-1]

        dDc1 = self.omega_d + self.beta4_1 / (self.beta3 + H1**2) - (self.mu6 + self.gamma2) * Dc1
        dDc2 = self.omega_d + self.beta4_2 / (self.beta3 + H1**2) - (self.mu6 + self.gamma2) * Dc2

        return torch.stack( (dNm1, dNm2, dDm1, dDm2, dNc1, dNc2, dDc1, dDc2, dI1, dI2, dH1) )

# %% set up pytorch==========

dtype = torch.float32
device = torch.device("cpu")
# %%=============================================
# simulation without treatment

T = 500.0 #最終時
model = Model(n_grid=11, e = 0.01, device=device)
y_init = model.random_init()

# %%====================================================
rtol = torch.tensor( 1e-6, device=device, dtype = dtype)
atol = torch.tensor( 1e-7, device=device, dtype = dtype)
t = torch.tensor(np.linspace(0, T, 50), device=device, dtype=dtype)

# %% solve ODEs ================================================================================
%%time
sol = odeint(model, y_init, t, atol=atol, rtol=rtol, method="bosh3", options= dict(dtype=dtype))

# %% run experiment ====================
(ver, wid) = (1, model.wid)

# %% animaiton function==================================================================================================================
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

# %%
t_anim = t.to('cpu').detach().numpy().copy()

# %%=======================================================================================================
# get color map of protein (or gene) A (animation)
sol_anim = sol[:, 10, 1:-1, 1:-1].to('cpu').detach().numpy().copy()
plot_animation(t_anim, sol_anim, "test_beta_1d_lc.gif")

# %%
sol_ = sol.to('cpu').detach().numpy().copy()

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
plot_graph(t_anim, sol_[:, 10, :, 1:-1], "test_beta_h_lc.gif")

# %%
