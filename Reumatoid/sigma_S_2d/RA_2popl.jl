using OrdinaryDiffEq, DiffEqCallbacks
using Random
using Plots
using ColorSchemes

#======================CPU======================#
using ParallelStencil
#@init_parallel_stencil(CUDA, Float32, 2) #for GPU
@init_parallel_stencil(Threads, Float64, 2) #for CPU

#=======================RA=======================#
Lx = 60 #number of cells along x_axis
Ly = 60 #number of cells along y-axis

T_end = 50

#=======================Function (ODEs)=======================#
function notch_func!(du, u, p, t)
#parameters
    (;
        α1, α2, μ1, μ2, μ3, μ4, μ5, μ6, μ7,
        β1, β2, β3, β4, β6, β7,
        γ1, γ2, γ3,
        σs, σv, ECx1, ECy1, ECx2, ECy2,
        n_ave, d_ave, j_ave,
    ) = p
#area except for boundary
    Ix = 2:(Lx-1)
    Iy = 2:(Ly-1)
    
#periodic boundary condition
    @views Nm = u[:, :, 1]
    @views Dm = u[:, :, 2]
    @views Jm = u[:, :, 3]
    @views Nc = u[:, :, 4]
    @views Dc = u[:, :, 5]
    @views Jc = u[:, :, 6]
    @views I  = u[:, :, 7]

#mean of Notch/Delta in surrounding cells
    @views begin
        
        #corner
        n_ave[1, Lx]  = (Nm[1, Lx-1]  + Nm[2, Lx])/2
        n_ave[Ly, Lx] = (Nm[Ly-1, Lx] + Nm[Ly, Lx-1])/2
        n_ave[1, 1] = (Nm[1, 2] + Nm[2, 1])/2
        n_ave[Ly, 1] = (Nm[Ly-1, 1] + Nm[Ly, 2])/2

        d_ave[1, Lx]  = (Dm[1, Lx-1]  + Dm[2, Lx])/2
        d_ave[Ly, Lx] = (Dm[Ly-1, Lx] + Dm[Ly, Lx-1])/2
        d_ave[1, 1] = (Dm[1, 2] + Dm[2, 1])/2
        d_ave[Ly, 1] = (Dm[Ly-1, 1] + Dm[Ly, 2])/2

        j_ave[1, Lx]  = (Jm[1, Lx-1]  + Jm[2, Lx])/2
        j_ave[Ly, Lx] = (Jm[Ly-1, Lx] + Jm[Ly, Lx-1])/2
        j_ave[1, 1] = (Jm[1, 2] + Jm[2, 1])/2
        j_ave[Ly, 1] = (Jm[Ly-1, 1] + Jm[Ly, 2])/2
        
        #edge
        @. n_ave[1, Ix]  = (Nm[2, Ix]    + Nm[1, Ix-1]  + Nm[1, Ix+1])/3
        @. n_ave[Ly, Ix] = (Nm[Ly-1, Ix] + Nm[Ly, Ix-1] + Nm[Ly, Ix+1])/3
        @. n_ave[Iy, Lx] = (Nm[Iy, Lx-1] + Nm[Iy-1, Lx] + Nm[Iy+1, Lx])/3
        @. n_ave[Iy, 1]  = (Nm[Iy, 2]    + Nm[Iy-1, 1]  + Nm[Iy+1, 1])/3

        @. d_ave[1, Ix]  = (Dm[2, Ix]    + Dm[1, Ix-1]  + Dm[1, Ix+1])/3
        @. d_ave[Ly, Ix] = (Dm[Ly-1, Ix] + Dm[Ly, Ix-1] + Dm[Ly, Ix+1])/3
        @. d_ave[Iy, Lx] = (Dm[Iy, Lx-1] + Dm[Iy-1, Lx] + Dm[Iy+1, Lx])/3
        @. d_ave[Iy, 1]  = (Dm[Iy, 2]    + Dm[Iy-1, 1]  + Dm[Iy+1, 1])/3

        @. j_ave[1, Ix]  = (Jm[2, Ix]    + Jm[1, Ix-1]  + Jm[1, Ix+1])/3
        @. j_ave[Ly, Ix] = (Jm[Ly-1, Ix] + Jm[Ly, Ix-1] + Jm[Ly, Ix+1])/3
        @. j_ave[Iy, Lx] = (Jm[Iy, Lx-1] + Jm[Iy-1, Lx] + Jm[Iy+1, Lx])/3
        @. j_ave[Iy, 1]  = (Jm[Iy, 2]    + Jm[Iy-1, 1]  + Jm[Iy+1, 1])/3

        @. n_ave[Iy, Ix] = (Nm[Iy-1, Ix] + Nm[Iy+1, Ix] + Nm[Iy, Ix-1] + Nm[Iy, Ix+1])/4
        @. d_ave[Iy, Ix] = (Dm[Iy-1, Ix] + Dm[Iy+1, Ix] + Dm[Iy, Ix-1] + Dm[Iy, Ix+1])/4
        @. j_ave[Iy, Ix] = (Jm[Iy-1, Ix] + Jm[Iy+1, Ix] + Jm[Iy, Ix-1] + Jm[Iy, Ix+1])/4
        

#ODEs
        @. du[:, :, 1] = -α1 * σs * d_ave * Nm - μ1 * Nm + γ1 * Nc - α2 * σs * j_ave * Nm #NOTCH in cell membrane
        @. du[:, :, 2] = -α1 * σs * n_ave * Dm - μ2 * Dm + γ2 * Dc #DELTA in cell membrane
        @. du[:, :, 3] = -α1 * σs * n_ave * Jm - μ6 * Jm + γ3 * Jc #JAGGED in cell membrane
        @. du[:, :, 4] = β2 * (I^2 / (β1 + I^2)) - (μ4 + γ1*σv) * Nc #NOTCH in cytosol
        @. du[:, :, 5] = β4 / (1 + β3*(I^2))     - (μ5 + γ2*σv) * Dc #DELTA in cytosol
        @. du[:, :, 6] = β6 * (I^2 / (β7 + I^2)) - (μ7 + γ3*σv) * Jc #JAGGED in cytosol
        @. du[:, :, 7] = α1 * (σs/σv) * d_ave * Nm + α2 * (σs/σv) * j_ave * Nm - μ3 * I #NICD in cytosol
        
        @. du[ECy1, ECx1, 4] = 0 #Notch in cytosol
        @. du[ECy1, ECx1, 5] = 0 #Delta in cytosol
        @. du[ECy2, ECx2, 4] = 0 #Notch in cytosol
        @. du[ECy2, ECx2, 5] = 0 #Delta in cytosol
        @. u[u<0] = 0
    end
    nothing
end

#=======================SIMULATION=======================#
#cell density
σs = zeros(Ly, Lx)
σx = Array(collect(1:1:Lx))
σy = Array(collect(1:1:Ly))
ecx1 = 2
ecy1 = 2
ECx1 = 40-Int(floor(ecx1/2)):40-1+Int(floor((ecx1+1)/2))
ECy1 = 20-Int(floor(ecy1/2)):20-1+Int(floor((ecy1+1)/2))
x1 = ECx1[1] + (ecx1-1)/2
y1 = ECy1[1] + (ecy1-1)/2

ecx2 = 2
ecy2 = 2
ECx2 = 20-Int(floor(ecx2/2)):20-1+Int(floor((ecx2+1)/2))
ECy2 = 40-Int(floor(ecy2/2)):40-1+Int(floor((ecy2+1)/2))
x2 = ECx2[1] + (ecx2-1)/2
y2 = ECy2[1] + (ecy2-1)/2

#non-linear
a = 1.0
M1 = [[10.0 0.1]
[0.1 10.0]]
M1_ = inv(M1)
M2 = [[50.0 0.1]
[0.1 50.0]]
M2_ = inv(M2)

for i in 1:Ly
    for j in 1:Lx
        σs[i, j] = a*exp(-(M1_[1,1]*(σx[j]-x1)^2 + (M1_[1,2]+M1_[2,1])*(σx[j]-x1)*(σy[i]-y1) + M1_[2,2]*(σy[i]-y1)^2)*(M2_[1,1]*(σx[j]-x2)^2 + (M2_[1,2]+M2_[2,1])*(σx[j]-x2)*(σy[i]-y2) + M2_[2,2]*(σy[i]-y2)^2)/16)
    end
end
#σs=1.0
begin
    #timespan, parameters------------------------------------------------
    tspan = (0.0, T_end)

    cache = (
        n_ave = zeros(Ly,Lx),
        d_ave = zeros(Ly,Lx),
        j_ave = zeros(Ly, Lx),
    )
    p = (
        α1=1.0, α2=1.0, μ1=1.0, μ2=1.0, μ3=0.01, μ4=1.0, μ5=0.5, μ6=1.0, μ7=0.5,
        β1=1.1, β2=20.0, β3=100.0, β4=1.0, β6=1.0, β7=1.1,
        γ1=2.0, γ2=1.0,γ3=1.0,
        σs=σs, σv=1.0, ECx1=ECx1, ECy1=ECy1, ECx2=ECx2, ECy2=ECy2, cache...
        )
    
    #initial data--------------------------------------------------------
    u0 = rand(Ly, Lx, 7)*0.1
    #@. u0[:, :, 1] #NOTCH in cell membrane
    #@. u0[:, :, 4]  #NOTCH in cytosol
    @. u0[:, :, 2] = u0[:, :, 2] + 3.0 #DELTA in cell membrane
    @. u0[:, :, 5] = u0[:, :, 5] + 3.0 #DELTA in cytosol
    @. u0[:, :, 3] = u0[:, :, 3] #JAGGED in cell membrane
    @. u0[:, :, 6] = u0[:, :, 6] #JAGGED in cytosol
    @. u0[:, :, 7] = u0[:, :, 7] + 0.5 #NICD in cytosol

    prob = ODEProblem(notch_func!, u0, tspan, p)
end

function plot_heatmap(u, t, integrator)
    heatmap(Array(u[:, :, 4]), title="t=$(round(t, digits=1))", c=:inferno, clim=(0, 7), aspect_ratio=1)
    frame(anim)
    return nothing
end

cb1 = FunctionCallingCallback(plot_heatmap, funcat = LinRange(0, T_end, 50)) #plot at intervals
cb2 = TerminateSteadyState(5e-3) #If all biochemicals in cells reach steady state, terminate the simulation
cbs = CallbackSet(cb1, cb2)

anim = Animation() #array of image for animation

#calculation
@time sol = solve(prob, Heun(), progress = true, progress_steps = 1.0, saveat=[T_end], callback=cbs);
#calculation methods: BS3, Tsit5, Heun, ROCK2

fig1 = heatmap(Array(sol[end][:, :, 4]), title="Notch in cytosol",  c=:inferno, clim=(0, 7),   aspect_ratio=1)
fig2 = heatmap(Array(sol[end][:, :, 5]), title="Delta in cytosol",  c=:inferno, clim=(0, 0.1), aspect_ratio=1)
fig3 = heatmap(Array(sol[end][:, :, 6]), title="Jagged in cytosol", c=:inferno, clim=(0, 0.7), aspect_ratio=1)

savefig(fig1, "2d_nl_2pop_s1-25_notch.png")
savefig(fig2, "2d_nl_2pop_s1-25_Delta.png")
savefig(fig3, "2d_nl_2pop_s1-25_Jag.png")
gif(anim,     "2d_nl_2pop_s1-25_t200_notch.gif", fps=120) #create gif animation