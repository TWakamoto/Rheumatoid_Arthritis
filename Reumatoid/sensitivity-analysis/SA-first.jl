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
Ly = 30 #number of cells along y-axis

T_end = 50

#=======================Function (ODEs)=======================#
function notch_func!(du, u, p, t)
#parameters
    (;
        α1, α2, μ1, μ2, μ3, μ4, μ5, μ6, μ7,
        β1, β2, β3, β4, β5, β6,
        γ1, γ2, γ3, dummy,
        σs, σv,
        n_ave, d_ave, j_ave,
    ) = p
#area except for boundary
    Ix = 2:(Lx-1)
    Iy = 2:(Ly-1)
    
#periodic boundary condition
    #@views Nm = u[:, :, 1]
    #@views Dm = u[:, :, 2]
    #@views I  = u[:, :, 3]
    #@views Nc = u[:, :, 4]
    #@views Dc = u[:, :, 5]

    @views Dm = u[:, :, 2]
    @views Nm = u[:, :, 1]
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
        #@. du[:, :, 1] = -α1 * σs * d_ave * Nm - μ1 * Nm + γ1 * Nc #NOTCH in cell membrane
        #@. du[:, :, 2] = -α1 * σs * n_ave * Dm - μ2 * Dm + γ2 * Dc #DELTA in cell membrane
        #@. du[:, :, 3] = α1 * (σs/σv) * d_ave * Nm - μ4 * I #NICD in cytosol
        #@. du[:, :, 4] = β2 * (I^2 / (β1 + I^2)) - (μ5 + γ1*σv) * Nc #NOTCH in cytosol
        #@. du[:, :, 5] = β4 / (1 + β3*(I^2))     - (μ6 + γ2*σv) * Dc #DELTA in cytosol
        
        @. du[:, :, 1] = -α1 * σs * d_ave * Nm - μ1 * Nm + γ1 * Nc - α2 * σs * j_ave * Nm #NOTCH in cell membrane
        @. du[:, :, 2] = -α1 * σs * n_ave * Dm - μ2 * Dm + γ2 * Dc #DELTA in cell membrane
        @. du[:, :, 3] = -α1 * σs * n_ave * Jm - μ3 * Jm + γ3 * Jc #JAGGED in cell membrane
        @. du[:, :, 4] = α1 * (σs/σv) * d_ave * Nm + α2 * (σs/σv) * j_ave * Nm - μ4 * I #NICD in cytosol
        @. du[:, :, 5] = β2 * (I^2 / (β1 + I^2)) - (μ5 + γ1*σv) * Nc #NOTCH in cytosol
        @. du[:, :, 6] = β4 / (1 + β3*(I^2))     - (μ6 + γ2*σv) * Dc #DELTA in cytosol
        @. du[:, :, 7] = β6 * (I^2 / (β5 + I^2)) - (μ7 + γ3*σv) * Jc #JAGGED in cytosol
        
        #@. du[:, 1, 4] = 0 #Notch in cytosol
        #@. du[:, 1, 5] = 0 #Delta in cytosol

        @. du[:, 1, 5] = 0 #Notch in cytosol
        @. du[:, 1, 6] = 0 #Delta in cytosol
        @. u[u<0] = 0
    end
    nothing
end

#=======================SIMULATION=======================#
#cell density

σs = zeros(Ly, Lx)
σ = Array(collect(0:1:Lx-1))
a = 1.0
b = 0.0
c = 15.0

#non-linear
for i in 1:Ly
    #left
    σs[i, :] .= a*exp.(-((σ.-b).^2)/(2*c*c))
end
#======================Sensitivity analysis======================#

#freq = Array([1, 21, 31, 37, 45, 49]) #Notch-Delta
freq = Array([19, 59, 91, 113, 133, 143, 149, 157, 161]) #Notch-Delta-Jagged
param = 9
M = 4 #interface factor
Nr = 2 #resumpling
fmax = maximum(freq)
Ns = 2*M*fmax + 1

function make_param(p_min, p_max, freq, Ds, psi)
    return p_min + (p_max-p_min)*((1/2) + (1/pi)*asin(sin(freq*Ds + psi)))
end

begin
    A_t1 = zeros(M*fmax)
    B_t1 = zeros(M*fmax)
    A_i1 = zeros(M, param)
    B_i1 = zeros(M, param)
    V_i1 = zeros(param)
    V_t1 = 0.0

    A_t2 = zeros(M*fmax)
    B_t2 = zeros(M*fmax)
    A_i2 = zeros(M, param)
    B_i2 = zeros(M, param) 
    V_i2 = zeros(param)
    V_t2 = 0.0
end

begin
    #initial data--------------------------------------------------------
    #u0 = rand(Ly, Lx, 5)*0.1
    #@. u0[:, :, 2] = u0[:, :, 2] + 3.0 #DELTA in cell membrane
    #@. u0[:, :, 5] = u0[:, :, 5] + 3.0 #DELTA in cytosol
    #@. u0[:, :, 3] = u0[:, :, 3] + 0.5 #NICD in cytosol

    u0 = rand(Ly, Lx, 7)*0.1
    @. u0[:, :, 2] = u0[:, :, 2] + 3.0 #DELTA in cell membrane
    @. u0[:, :, 6] = u0[:, :, 6] + 3.0 #DELTA in cytosol
    @. u0[:, :, 4] = u0[:, :, 4] + 0.5 #NICD in cytosol

#timespan, parameters------------------------------------------------
    tspan = (0.0, T_end)

    cache = (
        n_ave = zeros(Ly,Lx),
        d_ave = zeros(Ly,Lx),
        j_ave = zeros(Ly, Lx),
    )
end

for k in 1:1
    psi = 2*pi*rand(param)
    for ns in 1:Ns
        s = pi*(2*ns-Ns-1)/Ns
        #p = (
        #    α1=make_param(0.01, 50.0, freq[1], s, psi[1]), α2=1.0, μ1=1.0, μ2=1.0, μ3=1.0, μ4=0.01, μ5=1.0, μ6=0.5, μ7=0.5,
        #    β1=1.1, β2=make_param(0.01, 100.0, freq[2], s, psi[2]), β3=100.0, β4=make_param(0.01, 50.0, freq[3], s, psi[3]), β5=1.1, β6=1.0, 
        #    γ1=make_param(0.01, 50.0, freq[4], s, psi[4]), γ2=make_param(0.01, 50.0, freq[5], s, psi[5]), γ3=1.0, dummy=make_param(0.0, 1.0, freq[6], s, psi[6]),
        #    σs=σs, σv=1.0, cache...
        #    )
        p = (
        α1=make_param(0.01, 50.0, freq[1], s, psi[1]), α2=make_param(0.01, 50.0, freq[2], s, psi[2]), μ1=1.0, μ2=1.0, μ3=1.0, μ4=0.01, μ5=1.0, μ6=0.5, μ7=0.5,
        β1=1.1, β2=make_param(0.01, 100.0, freq[3], s, psi[3]), β3=100.0, β4=make_param(0.01, 50.0, freq[4], s, psi[4]), β5=1.1, β6=make_param(0.01, 50.0, freq[5], s, psi[5]), 
        γ1=make_param(0.01, 50.0, freq[6], s, psi[6]), γ2=make_param(0.01, 50.0, freq[7], s, psi[7]), γ3=make_param(0.01, 50.0, freq[8], s, psi[8]), dummy=make_param(0.0, 1.0, freq[9], s, psi[9]),
        σs=σs, σv=1.0, cache...
        )
        prob = ODEProblem(notch_func!, u0, tspan, p)

        #calculation
        @time sol = solve(prob, Heun(), progress = true, progress_steps = 1.0, saveat=[T_end], callback=TerminateSteadyState(5e-3));
        y1 = sum(sol[end][:, 2:end-1, 5] .- sol[end][:, 3:end, 5])/(Ly*(Lx-2))
        Nc = sol[end][floor(Int, Ly/2), :, 5].-(maximum(sol[end][:, :, 5]) - minimum(sol[end][:, 2:end, 5]))/2
        #y1 = sum(sol[end][:, 2:end-1, 4] .- sol[end][:, 3:end, 4])/(Ly*(Lx-2))
        #Nc = sol[end][floor(Int, Ly/2), :, 4].-(maximum(sol[end][:, :, 4]) - minimum(sol[end][:, 2:end, 4]))/2
        y2 = argmin(abs.(Nc))[1]
        for mf in 1:M*fmax
            A_t1[mf] += (1/Ns)*(y1*cos(mf*s))
            B_t1[mf] += (1/Ns)*(y1*sin(mf*s))
            A_t2[mf] += (1/Ns)*(y2*cos(mf*s))
            B_t2[mf] += (1/Ns)*(y2*sin(mf*s))
        end
        for p in 1:param
            for m in 1:M
                A_i1[m, p] = A_i1[m, p]+(1/Ns)*(y1*cos(m*freq[p]*s))
                B_i1[m, p] = B_i1[m, p]+(1/Ns)*(y1*sin(m*freq[p]*s))
                A_i2[m, p] = A_i2[m, p]+(1/Ns)*(y2*cos(m*freq[p]*s))
                B_i2[m, p] = B_i2[m, p]+(1/Ns)*(y2*sin(m*freq[p]*s))
            end
        end
        print("Ns=", ns, " k=", k)
    end
    for p in 1:param
        V_i1[p] += 2*sum((A_i1[:, p].^2) .+ (B_i1[:, p].^2))
        V_i2[p] += 2*sum((A_i2[:, p].^2) .+ (B_i2[:, p].^2))
    end
    V_t1 += 2*sum((A_t1.^2) .+ (B_t1.^2))
    V_t2 += 2*sum((A_t2.^2) .+ (B_t2.^2))
end

Si1 = V_i1/V_t1
Si2 = V_i2/V_t2

#calculation methods: BS3, Tsit5, Heun, ROCK2
x = Array(collect(1:1:param))
fig1 = bar(x, Si1)
fig2 = bar(x, Si2)

savefig(fig1, "SA_first_NDJ_difference.png")
savefig(fig2, "SA_first_NDJ_interface.png")
