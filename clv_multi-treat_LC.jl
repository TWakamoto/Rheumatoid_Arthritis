using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

using GLMakie
using OrdinaryDiffEq, DiffEqCallbacks
using ComponentArrays
using Random

#======================CPU======================#
using ParallelStencil
#@init_parallel_stencil(CUDA, Float32, 2)
@init_parallel_stencil(Threads, Float64, 2)

#=======================Lung Cancer=======================#
const tate = 100
const yoko = 100
T_end = 70000

#=======================Function (ODEs) for LUNG CANCER=======================#
function lung_func!(du, u, p, t)
    (;
        α1, α2, α5, α6, 
        γN, γD, 
        μ0, μ1, μ2, μ4, μ5, μ6,
        β1, β21, β22, β3, β41, β42,
        ν0, ν1, ν2, ν3, p1, p23, pβ,
        n1_ave, n2_ave, d1_ave, d2_ave,
    ) = p


    (; Nm1, Nm2, Dm1, Dm2, Nc1, Nc2, Dc1, Dc2, I1, I2, H) = u

    Ix = 2:(tate+1)
    Iy = 2:(yoko+1)

    @views begin
        @. n1_ave = (Nm1[1:tate, Iy] + Nm1[3:tate+2, Iy] + Nm1[Ix, 1:yoko] + Nm1[Ix, 3:yoko+2])/4
        @. n2_ave = (Nm2[1:tate, Iy] + Nm2[3:tate+2, Iy] + Nm2[Ix, 1:yoko] + Nm2[Ix, 3:yoko+2])/4
        @. d1_ave = (Dm1[1:tate, Iy] + Dm1[3:tate+2, Iy] + Dm1[Ix, 1:yoko] + Dm1[Ix, 3:yoko+2])/4
        @. d2_ave = (Dm2[1:tate, Iy] + Dm2[3:tate+2, Iy] + Dm2[Ix, 1:yoko] + Dm2[Ix, 3:yoko+2])/4

        @. du.Nm1[Ix, Iy]  = -α1 * d1_ave * Nm1[Ix, Iy] - α2 * d2_ave * Nm1[Ix, Iy] - μ1 * Nm1[Ix, Iy] + (1 - p1)*γN * Nc1[Ix, Iy]
        @. du.Nm2[Ix, Iy]  = -α5 * d1_ave * Nm2[Ix, Iy] - α6 * d2_ave * Nm2[Ix, Iy] - μ1 * Nm2[Ix, Iy] + (1 - p1)*γN * Nc2[Ix, Iy]
        @. du.Dm1[Ix, Iy]   = -α1 * n1_ave * Dm1[Ix, Iy] - α5 * n2_ave * Dm1[Ix, Iy] - μ2 * Dm1[Ix, Iy] + γD * Dc1[Ix, Iy]
        @. du.Dm2[Ix, Iy]   = -α2 * n1_ave * Dm2[Ix, Iy] - α6 * n2_ave * Dm2[Ix, Iy] - μ2 * Dm2[Ix, Iy] + γD * Dc2[Ix, Iy]
        @. du.Nc1[Ix, Iy]  = (1 - pβ) * β21 * ( I1[Ix, Iy]^2 / (β1 + I1[Ix, Iy]^2) ) - (μ5 + (1 - p1)*γN) * Nc1[Ix, Iy]
        @. du.Nc2[Ix, Iy]  = (1 - pβ) * β22 * ( I2[Ix, Iy]^2 / (β1 + I2[Ix, Iy]^2) ) - (μ5 + (1 - p1)*γN) * Nc2[Ix, Iy]
        @. du.Dc1[Ix, Iy]  = β41 / (β3 + H[Ix, Iy]^2) - (μ6 + γD) * Dc1[Ix, Iy]
        @. du.Dc2[Ix, Iy]  = β42 / (β3 + H[Ix, Iy]^2) - (μ6 + γD) * Dc2[Ix, Iy]
        @. du.I1[Ix, Iy]  = (1 - p23)*(α1 * d1_ave * Nm1[Ix, Iy] + α2 * d2_ave * Nm1[Ix, Iy]) - μ4 * I1[Ix, Iy]
        @. du.I2[Ix, Iy] = (1 - p23)*(α5 * d1_ave * Nm2[Ix, Iy] + α6 * d2_ave * Nm2[Ix, Iy]) - μ4 * I2[Ix, Iy]
        @. du.H[Ix, Iy] = ν0 + (ν2 * I1[Ix, Iy]^2) / (ν1 + I1[Ix, Iy]^2) * (1.0 - I2[Ix, Iy]^2 / (ν3 + I2[Ix, Iy]^2)) - μ0 * H[Ix, Iy]
        
    end
    nothing
end

#-----------------------initial condition-----------------------
function init_random(tate, yoko)
    ar = zeros(tate+2, yoko+2)
    ar[2:end-1, 2:end-1] .= 0.01*rand(tate, yoko)
    return ar 
end 

canc = zeros(4, 6)

#=======================SIMULATION (untreated)=======================#
begin
    #initial data--------------------------------------------------------
    u0 = ComponentArray(Nm1 = init_random(tate, yoko),
                        Nm2 = init_random(tate, yoko),
                        Dm1 = init_random(tate, yoko),
                        Dm2 = init_random(tate, yoko),
                        Nc1 = init_random(tate, yoko),
                        Nc2 = init_random(tate, yoko),
                        Dc1 = init_random(tate, yoko),
                        Dc2 = init_random(tate, yoko),
                        I1  = init_random(tate, yoko),
                        I2  = init_random(tate, yoko),
                        H   = init_random(tate, yoko),
                        )
    #timespan, parameters-------------------------------------------------------
    tspan = (0.0, T_end)

    cache = (
        n1_ave = zeros(tate,yoko),
        d1_ave = zeros(tate,yoko),
        n2_ave = zeros(tate,yoko),
        d2_ave = zeros(tate,yoko),
    )
    p = (
            α1=6.0, α2=10.0, α5=8.0, α6=6.0, 
            γN=2.0, γD=1.0, 
            μ0=0.5, μ1=1.0, μ2=1.0, μ4=0.1, μ5=1.0, μ6=0.5,
            β1=0.1, β21=8.0, β22=5.0, β3=2.0, β41=8.0, β42=8.0,
            ν0=0.5, ν1=5.0, ν2=25.0, ν3=5.0, p1=0, p23=0, pβ=0,
            cache...
        )
    prob_lc = ODEProblem(lung_func!, u0, tspan, p)
    @time lc_sol = solve(prob_lc, BS5(), progress=true, progress_steps=1.0, callback=TerminateSteadyState(1e-3), saveat=[T_end]);
    # BS3, Tsit5, Heun, 
    hes   = lc_sol[end].H[2:end-1, 2:end-1] .- u0.H[2:end-1, 2:end-1]
    h_max = maximum(hes)
end
canc[1, :] .= count(hes .> 0.75*h_max)

#=======================SIMULATION (with first treatment)=======================#
p1 = rand(tate, yoko)
p23 = rand(tate, yoko)
pβ = rand(tate, yoko)
begin
    #parameters-------------------------------------------------------
    p = (
            α1=6.0, α2=10.0, α5=8.0, α6=6.0, 
            γN=2.0, γD=1.0, 
            μ0=0.5, μ1=1.0, μ2=1.0, μ4=0.1, μ5=1.0, μ6=0.5,
            β1=0.1, β21=8.0, β22=5.0, β3=2.0, β41=8.0, β42=8.0,
            ν0=0.5, ν1=5.0, ν2=25.0, ν3=5.0, p1=0, p23=0, pβ=pβ,
            cache...
        )
    prob_lc = ODEProblem(lung_func!, u0, tspan, p)
    @time lc_sol = solve(prob_lc, Heun(), progress=true, progress_steps=1.0, callback=TerminateSteadyState(1e-3), saveat=[T_end]);
    # BS3, Tsit5, Heun, 
    hes   = lc_sol[end].H[2:end-1, 2:end-1] .- u0.H[2:end-1, 2:end-1]
end
canc[2, 5:6] .= count(hes .> 0.75*h_max)

@views begin
    Nm1_c = lc_sol[1].Nm1
    Nm2_c = lc_sol[1].Nm2
    Dm1_c = lc_sol[1].Dm1
    Dm2_c = lc_sol[1].Dm2
    Nc1_c = lc_sol[1].Nc1
    Nc2_c = lc_sol[1].Nc2
    Dc1_c = lc_sol[1].Dc1
    Dc2_c = lc_sol[1].Dc2
    I1_c  = lc_sol[1].I1
    I2_c  = lc_sol[1].I2
    H_c   = lc_sol[1].H
end
T_endc = lc_sol.t[end]


#=======================SIMULATION (with second treatment)=======================#
T_end2 = lc_sol.t[end] + 30000
begin
    u1 = ComponentArray(Nm1 = lc_sol[end].Nm1,
                        Nm2 = lc_sol[end].Nm2,
                        Dm1 = lc_sol[end].Dm1,
                        Dm2 = lc_sol[end].Dm2,
                        Nc1 = lc_sol[end].Nc1,
                        Nc2 = lc_sol[end].Nc2,
                        Dc1 = lc_sol[end].Dc1,
                        Dc2 = lc_sol[end].Dc2,
                        I1  = lc_sol[end].I1,
                        I2  = lc_sol[end].I2,
                        H   = lc_sol[end].H,
                        )
    tspan2 = (lc_sol.t[end], T_end2)
    #parameters-------------------------------------------------------
    p = (
            α1=6.0, α2=10.0, α5=8.0, α6=6.0, 
            γN=2.0, γD=1.0, 
            μ0=0.5, μ1=1.0, μ2=1.0, μ4=0.1, μ5=1.0, μ6=0.5,
            β1=0.1, β21=8.0, β22=5.0, β3=2.0, β41=8.0, β42=8.0,
            ν0=0.5, ν1=5.0, ν2=25.0, ν3=5.0, p1=p1, p23=0, pβ=0,
            cache...
        )
    prob_lc = ODEProblem(lung_func!, u1, tspan2, p)
    @time lc_sol = solve(prob_lc, Heun(), progress=true, progress_steps=1.0, callback=TerminateSteadyState(1e-3), saveat=[T_end2]);
    hes   = lc_sol[end].H[2:end-1, 2:end-1] .- u0.H[2:end-1, 2:end-1]
end
canc[3, 5] = count(hes .> 0.75*h_max)

#=======================SIMULATION (with third treatment)=======================#
T_end3 = lc_sol.t[end] + 30000
begin
    u1 = ComponentArray(Nm1 = lc_sol[end].Nm1,
                        Nm2 = lc_sol[end].Nm2,
                        Dm1 = lc_sol[end].Dm1,
                        Dm2 = lc_sol[end].Dm2,
                        Nc1 = lc_sol[end].Nc1,
                        Nc2 = lc_sol[end].Nc2,
                        Dc1 = lc_sol[end].Dc1,
                        Dc2 = lc_sol[end].Dc2,
                        I1  = lc_sol[end].I1,
                        I2  = lc_sol[end].I2,
                        H   = lc_sol[end].H,
                        )
    tspan2 = (lc_sol.t[end], T_end3)
    #parameters-------------------------------------------------------
    p = (
            α1=6.0, α2=10.0, α5=8.0, α6=6.0, 
            γN=2.0, γD=1.0, 
            μ0=0.5, μ1=1.0, μ2=1.0, μ4=0.1, μ5=1.0, μ6=0.5,
            β1=0.1, β21=8.0, β22=5.0, β3=2.0, β41=8.0, β42=8.0,
            ν0=0.5, ν1=5.0, ν2=25.0, ν3=5.0, p1=0, p23=p23, pβ=0,
            cache...
        )
    prob_lc = ODEProblem(lung_func!, u1, tspan2, p)
    @time lc_sol = solve(prob_lc, Heun(), progress=true, progress_steps=1.0, callback=TerminateSteadyState(1e-3), saveat=[T_end3]);
    # BS3, Tsit5, Heun, 
    hes   = lc_sol[end].H[2:end-1, 2:end-1] .- u0.H[2:end-1, 2:end-1]
end
canc[4, 5] = count(hes .> 0.75*h_max)


#=======================SIMULATION (with second treatment)=======================#
T_end4 = T_endc + 30000
begin
    u1 = ComponentArray(Nm1 = Nm1_c,
                        Nm2 = Nm2_c,
                        Dm1 = Dm1_c,
                        Dm2 = Dm2_c,
                        Nc1 = Nc1_c,
                        Nc2 = Nc2_c,
                        Dc1 = Dc1_c,
                        Dc2 = Dc2_c,
                        I1  = I1_c,
                        I2  = I2_c,
                        H   = H_c,
                        )
    tspan4 = (T_endc, T_end4)
    #parameters-------------------------------------------------------
    p = (
            α1=6.0, α2=10.0, α5=8.0, α6=6.0, 
            γN=2.0, γD=1.0, 
            μ0=0.5, μ1=1.0, μ2=1.0, μ4=0.1, μ5=1.0, μ6=0.5,
            β1=0.1, β21=8.0, β22=5.0, β3=2.0, β41=8.0, β42=8.0,
            ν0=0.5, ν1=5.0, ν2=25.0, ν3=5.0, p1=0, p23=p23, pβ=0,
            cache...
        )
    prob_lc = ODEProblem(lung_func!, u1, tspan4, p)
    @time lc_sol = solve(prob_lc, Heun(), progress=true, progress_steps=1.0, callback=TerminateSteadyState(1e-3), saveat=[T_end4]);
    # BS3, Tsit5, Heun, 
    hes   = lc_sol[end].H[2:end-1, 2:end-1] .- u0.H[2:end-1, 2:end-1]
end
canc[3, 6] = count(hes .> 0.75*h_max)


#=======================SIMULATION (with third treatment)=======================#
T_end5 = lc_sol.t[end] + 30000
begin
    u1 = ComponentArray(Nm1 = lc_sol[end].Nm1,
                        Nm2 = lc_sol[end].Nm2,
                        Dm1 = lc_sol[end].Dm1,
                        Dm2 = lc_sol[end].Dm2,
                        Nc1 = lc_sol[end].Nc1,
                        Nc2 = lc_sol[end].Nc2,
                        Dc1 = lc_sol[end].Dc1,
                        Dc2 = lc_sol[end].Dc2,
                        I1  = lc_sol[end].I1,
                        I2  = lc_sol[end].I2,
                        H   = lc_sol[end].H,
                        )
    tspan5 = (lc_sol.t[end], T_end5)
    #parameters-------------------------------------------------------
    p = (
            α1=6.0, α2=10.0, α5=8.0, α6=6.0, 
            γN=2.0, γD=1.0, 
            μ0=0.5, μ1=1.0, μ2=1.0, μ4=0.1, μ5=1.0, μ6=0.5,
            β1=0.1, β21=8.0, β22=5.0, β3=2.0, β41=8.0, β42=8.0,
            ν0=0.5, ν1=5.0, ν2=25.0, ν3=5.0, p1=p1, p23=0, pβ=0,
            cache...
        )
    prob_lc = ODEProblem(lung_func!, u1, tspan5, p)
    @time lc_sol = solve(prob_lc, Heun(), progress=true, progress_steps=1.0, callback=TerminateSteadyState(1e-3), saveat=[T_end5]);
    # BS3, Tsit5, Heun, 
    hes   = lc_sol[end].H[2:end-1, 2:end-1] .- u0.H[2:end-1, 2:end-1]
end
canc[4, 6] = count(hes .> 0.75*h_max)



#=======================MAKE FIGURE=======================#
using PyPlot
@.canc = canc/canc[1, 1]
#-------------------------------------------------------------
fig1 = plt.figure(figsize=(10,5))
ax1 = fig1.add_subplot(111)
label=["untreated", "first", "second", "third"]
ax1.set_ylim([0.5, 1.3])
ax1.plot(label, canc[:, 1], marker="o", markersize=10, label="S1-S2-S3", color="b")
ax1.plot(label, canc[:, 2], marker="o", markerfacecolor="w", markersize=10, label="S1-S3-S2", color="b")
ax1.plot(label, canc[:, 3], marker="o", markersize=10, label="S2-S1-S3", color="g")
ax1.plot(label, canc[:, 4], marker="o", markerfacecolor="w", markersize=10, label="S2-S3-S1", color="g")
ax1.plot(label, canc[:, 5], marker="o", markersize=10, label="S3-S1-S2", color="r")
ax1.plot(label, canc[:, 6], marker="o", markerfacecolor="w", markersize=10, label="S3-S2-S1", color="r")
ax1.legend(loc="best", fontsize=8)
fig1.savefig("clv_change_LC.png")
fig1