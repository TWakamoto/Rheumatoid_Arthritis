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
        β1, β2, β3, β4, β5, β6,
        γ1, γ2, γ3,
        σs, σv, ECx, ECy,
        n_ave, d_ave, j_ave,
    ) = p
#area except for boundary
    Ix = 2:(Lx-1)
    Iy = 2:(Ly-1)
    
#periodic boundary condition
    @views Nm = u[:, :, 1]
    @views Dm = u[:, :, 2]
    @views Jm = u[:, :, 3]
    @views I  = u[:, :, 4]
    @views Nc = u[:, :, 5]
    @views Dc = u[:, :, 6]
    @views Jc = u[:, :, 7]
    

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

        #j_ave[1, Lx]  = (Jm[1, Lx-1]  + Jm[2, Lx])/2
        #j_ave[Ly, Lx] = (Jm[Ly-1, Lx] + Jm[Ly, Lx-1])/2
        #j_ave[1, 1] = (Jm[1, 2] + Jm[2, 1])/2
        #j_ave[Ly, 1] = (Jm[Ly-1, 1] + Jm[Ly, 2])/2
        
        #edge
        @. n_ave[1, Ix]  = (Nm[2, Ix]    + Nm[1, Ix-1]  + Nm[1, Ix+1])/3
        @. n_ave[Ly, Ix] = (Nm[Ly-1, Ix] + Nm[Ly, Ix-1] + Nm[Ly, Ix+1])/3
        @. n_ave[Iy, Lx] = (Nm[Iy, Lx-1] + Nm[Iy-1, Lx] + Nm[Iy+1, Lx])/3
        @. n_ave[Iy, 1]  = (Nm[Iy, 2]    + Nm[Iy-1, 1]  + Nm[Iy+1, 1])/3

        @. d_ave[1, Ix]  = (Dm[2, Ix]    + Dm[1, Ix-1]  + Dm[1, Ix+1])/3
        @. d_ave[Ly, Ix] = (Dm[Ly-1, Ix] + Dm[Ly, Ix-1] + Dm[Ly, Ix+1])/3
        @. d_ave[Iy, Lx] = (Dm[Iy, Lx-1] + Dm[Iy-1, Lx] + Dm[Iy+1, Lx])/3
        @. d_ave[Iy, 1]  = (Dm[Iy, 2]    + Dm[Iy-1, 1]  + Dm[Iy+1, 1])/3

        #@. j_ave[1, Ix]  = (Jm[2, Ix]    + Jm[1, Ix-1]  + Jm[1, Ix+1])/3
        #@. j_ave[Ly, Ix] = (Jm[Ly-1, Ix] + Jm[Ly, Ix-1] + Jm[Ly, Ix+1])/3
        #@. j_ave[Iy, Lx] = (Jm[Iy, Lx-1] + Jm[Iy-1, Lx] + Jm[Iy+1, Lx])/3
        #@. j_ave[Iy, 1]  = (Jm[Iy, 2]    + Jm[Iy-1, 1]  + Jm[Iy+1, 1])/3

        @. n_ave[Iy, Ix] = (Nm[Iy-1, Ix] + Nm[Iy+1, Ix] + Nm[Iy, Ix-1] + Nm[Iy, Ix+1])/4
        @. d_ave[Iy, Ix] = (Dm[Iy-1, Ix] + Dm[Iy+1, Ix] + Dm[Iy, Ix-1] + Dm[Iy, Ix+1])/4
        #@. j_ave[Iy, Ix] = (Jm[Iy-1, Ix] + Jm[Iy+1, Ix] + Jm[Iy, Ix-1] + Jm[Iy, Ix+1])/4
        

#ODEs
        @. du[:, :, 1] = -α1 * σs * d_ave * Nm - μ1 * Nm + γ1 * Nc - α2 * σs * j_ave * Nm #NOTCH in cell membrane
        @. du[:, :, 2] = -α1 * σs * n_ave * Dm - μ2 * Dm + γ2 * Dc #DELTA in cell membrane
        @. du[:, :, 3] = -α1 * σs * n_ave * Jm - μ3 * Jm + γ3 * Jc #JAGGED in cell membrane
        @. du[:, :, 4] = α1 * (σs/σv) * d_ave * Nm - μ4 * I + α2 * (σs/σv) * j_ave * Nm  #NICD in cytosol
        @. du[:, :, 5] = β2 * (I^2 / (β1 + I^2)) - (μ5 + γ1*σv) * Nc #NOTCH in cytosol
        @. du[:, :, 6] = β4 / (1 + β3*(I^2))     - (μ6 + γ2*σv) * Dc #DELTA in cytosol
        @. du[:, :, 7] = β6 * (I^2 / (β5 + I^2)) - (μ7 + γ3*σv) * Jc #JAGGED in cytosol
        
        
        @. du[ECy, ECx, 4] = 0 #Notch in cytosol
        @. du[ECy, ECx, 5] = 0 #Delta in cytosol
        @. u[u<0] = 0
    end
    nothing
end

#=======================SIMULATION=======================#
#cell density
σs = zeros(Ly, Lx)
σx = Array(collect(1:1:Lx))
σy = Array(collect(1:1:Ly))
ecx = 6
ecy = 2
ECx = Int(floor((Lx-ecx)/2))+1:Int(floor((Lx+ecx)/2))
ECy = Int(floor((Ly-ecy)/2))+1:Int(floor((Ly+ecy)/2))
x1 = ECx[1] + (ecx-1)/2
y1 = ECy[1] + (ecy-1)/2

#non-linear
#k = 3
#βs = 1.0
#for i in 1:Lx
#    for j in 1:Ly
#        σs[j, i] = 1 - sqrt((σx[i]-ECx)^2+(σy[j]-ECy)^2)^k / (βs^k + sqrt((σx[i]-ECx)^2+(σy[j]-ECy)^2)^k)
#    end
#end

#linear
a = 1.0
M = [[30.0 0.1]
[0.1 30.0]]
M_ = inv(M)
for i in 1:Ly
    for j in 1:Lx
        σs[i, j] = a*exp(-(M_[1,1]*(σx[j]-x1)^2 + (M_[1,2]+M_[2,1])*(σx[j]-x1)*(σy[i]-y1) + M_[2,2]*(σy[i]-y1)^2)/2)
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
        α1=1.0, α2=1.0, μ1=1.0, μ2=1.0, μ3=1.0, μ4=0.01, μ5=1.0, μ6=0.5, μ7=0.5,
        β1=1.1, β2=20.0, β3=100.0, β4=1.0, β5=1.1, β6=1.0,
        γ1=2.0, γ2=1.0,γ3=1.0,
        σs=σs, σv=1.0, ECx=ECx, ECy=ECy, cache...
        )
    
    #initial data--------------------------------------------------------
    u0 = rand(Ly, Lx, 7)*0.1
    @. u0[:, :, 2] = u0[:, :, 2] + 3.0 #DELTA in cell membrane
    @. u0[:, :, 6] = u0[:, :, 6] + 3.0 #DELTA in cytosol
    @. u0[:, :, 4] = u0[:, :, 4] + 0.5 #NICD in cytosol

    prob = ODEProblem(notch_func!, u0, tspan, p)
end

function plot_heatmap(u, t, integrator)
    heatmap(Array(u[:, :, 5]), title="t=$(round(t, digits=1))", c=:inferno, clim=(0, 7), aspect_ratio=1)
    frame(anim)
    return nothing
end

function plot_line(u, t, integrator)
    plot(x, Array(u[:, 30, 5]), title="t=$(round(t, digits=1))", ylim=(0, 7))
    frame(anim2)
    return nothing
end

cb1 = FunctionCallingCallback(plot_heatmap, funcat = LinRange(0, T_end, 50)) #plot at intervals
cb2 = TerminateSteadyState(5e-3) #If all biochemicals in cells reach steady state, terminate the simulation
cb3 = FunctionCallingCallback(plot_line, funcat = LinRange(0, T_end, 50)) #plot at intervals
cbs = CallbackSet(cb1, cb2, cb3)

anim = Animation() #array of image for animation
anim2 = Animation()
x = Array(0:1:Lx-1)

#calculation
@time sol = solve(prob, Heun(), progress = true, progress_steps = 1.0, saveat=[T_end], callback=cbs);
#calculation methods: BS3, Tsit5, Heun, ROCK2

fig1 = heatmap(Array(sol[end][:, :, 5]), title="Notch in cytosol",  c=:inferno, clim=(0, 7),   aspect_ratio=1)
fig2 = heatmap(Array(sol[end][:, :, 6]), title="Delta in cytosol",  c=:inferno, clim=(0, 0.1), aspect_ratio=1)
fig3 = heatmap(Array(sol[end][:, :, 7]), title="Jagged in cytosol", c=:inferno, clim=(0, 0.7), aspect_ratio=1)


fig4 = plot(Array(sol[end][:, 30, 5]), title="Notch in cytosol", xlabel="position", ylabel="Notch expression")
savefig(fig1, "2d_ND_ec-6-2_notch.png")
savefig(fig2, "2d_ND_ec-6-2_Delta.png")
savefig(fig3, "2d_ND_ec-6-2_Jag.png")
savefig(fig4, "2d_ND_ec-6-2_notch_l.png")
gif(anim,     "2d_ND_ec-6-2_notch.gif", fps=120) #create gif animation
gif(anim2,    "2d_ND_ec-6-2_notch_l.gif", fps=120) #create gif animation