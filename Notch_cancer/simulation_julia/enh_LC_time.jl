include("src/utils.jl")
include("src/initial_condition.jl")
include("src/notch_interaction.jl")

using OrdinaryDiffEq, DiffEqCallbacks
using ComponentArrays
using Random
using CUDA

#======================GPU======================#
using ParallelStencil
@init_parallel_stencil(CUDA, Float32, 2)
#@init_parallel_stencil(Threads, Float64, 2)

#=======================constants=======================#
const tate = 1000
const yoko = 1000
T_end = 30000

#=======================Function (ODEs of LUNG CANCER)=======================#
function lung_func!(du, u, p, t)
    (;
        α1, α2, α5, α6, 
        γN, γD, 
        μ0, μ1, μ2, μ4, μ5, μ6,
        β1, β21, β22, β3, β41, β42,
        ν0, ν1, ν2, ν3, ωN, ωD, pα, pβ, pγ, canc, h_max,
        n1_ave, n2_ave, d1_ave, d2_ave,
    ) = p

    Ix = 2:(tate+1)
    Iy = 2:(yoko+1)
    Cx = Int(floor(1+(tate-canc)/2)):Int(floor((tate+canc)/2))
    Cy = Int(floor(1+(yoko-canc)/2)):Int(floor((yoko+canc)/2))

    @views Nm1 = u[:, :, 1]
    @views Nm2 = u[:, :, 2]
    @views Dm1 = u[:, :, 3]
    @views Dm2 = u[:, :, 4]
    @views Nc1 = u[:, :, 5]
    @views Nc2 = u[:, :, 6]
    @views Dc1 = u[:, :, 7]
    @views Dc2 = u[:, :, 8]
    @views I1  = u[:, :, 9]
    @views I2  = u[:, :, 10]
    @views H   = u[:, :, 11]
    @views dDm1 = du[:, :, 3]
    @views dDm2 = du[:, :, 4]

    @views begin
        @. n1_ave = (Nm1[1:tate, Iy] + Nm1[3:tate+2, Iy] + Nm1[Ix, 1:yoko] + Nm1[Ix, 3:yoko+2])/4
        @. n2_ave = (Nm2[1:tate, Iy] + Nm2[3:tate+2, Iy] + Nm2[Ix, 1:yoko] + Nm2[Ix, 3:yoko+2])/4
        @. d1_ave = (Dm1[1:tate, Iy] + Dm1[3:tate+2, Iy] + Dm1[Ix, 1:yoko] + Dm1[Ix, 3:yoko+2])/4
        @. d2_ave = (Dm2[1:tate, Iy] + Dm2[3:tate+2, Iy] + Dm2[Ix, 1:yoko] + Dm2[Ix, 3:yoko+2])/4

        @. du[Ix, Iy, 1]  = (1 + pα)*(-α1 * d1_ave * Nm1[Ix, Iy] - α2 * d2_ave * Nm1[Ix, Iy]) - μ1 * Nm1[Ix, Iy] + (1 + pγ)*γN * Nc1[Ix, Iy]
        @. du[Ix, Iy, 2]  = (1 + pα)*(-α5 * d1_ave * Nm2[Ix, Iy] - α6 * d2_ave * Nm2[Ix, Iy]) - μ1 * Nm2[Ix, Iy] + (1 + pγ)*γN * Nc2[Ix, Iy]
        @. dDm1[Ix, Iy]   = (1 + pα)*(-α1 * n1_ave * Dm1[Ix, Iy] - α5 * n2_ave * Dm1[Ix, Iy]) - μ2 * Dm1[Ix, Iy] + γD * Dc1[Ix, Iy]
        @. dDm2[Ix, Iy]   = (1 + pα)*(-α2 * n1_ave * Dm2[Ix, Iy] - α6 * n2_ave * Dm2[Ix, Iy]) - μ2 * Dm2[Ix, Iy] + γD * Dc2[Ix, Iy]
        @. du[Ix, Iy, 5]  = ωN + (1 + pβ)β21 * ( I1[Ix, Iy]^2 / (β1 + I1[Ix, Iy]^2) ) - (μ5 + (1 + pγ)*γN) * Nc1[Ix, Iy]
        @. du[Ix, Iy, 6]  = ωN + (1 + pβ)β22 * ( I2[Ix, Iy]^2 / (β1 + I2[Ix, Iy]^2) ) - (μ5 + (1 + pγ)*γN) * Nc2[Ix, Iy]
        @. du[Ix, Iy, 7]  = ωD + β41 / (β3 + H[Ix, Iy]^2) - (μ6 + γD) * Dc1[Ix, Iy]
        @. du[Ix, Iy, 8]  = ωD + β42 / (β3 + H[Ix, Iy]^2) - (μ6 + γD) * Dc2[Ix, Iy]
        @. du[Ix, Iy, 9]  = (1 + pα)*(α1 * d1_ave * Nm1[Ix, Iy] + α2 * d2_ave * Nm1[Ix, Iy]) - μ4 * I1[Ix, Iy]
        @. du[Ix, Iy, 10] = (1 + pα)*(α5 * d1_ave * Nm2[Ix, Iy] + α6 * d2_ave * Nm2[Ix, Iy]) - μ4 * I2[Ix, Iy]
        @. du[Ix, Iy, 11] = ν0 + (ν2 * I1[Ix, Iy]^2) / (ν1 + I1[Ix, Iy]^2) * (1.0 - I2[Ix, Iy]^2 / (ν3 + I2[Ix, Iy]^2)) - μ0 * H[Ix, Iy]
        if canc > 0

            @parallel notch!(dDm1, Dm1, H, h_max)
            @parallel notch!(dDm2, Dm2, H, h_max)

            @. du[Cx, Cy, :] = 0.0
        end
    end
    nothing
end

#=======================SIMULATION WITHOUT CANCER CELLS=======================#
begin
    #initial data--------------------------------------------------------
    u0 = init_random(tate, yoko)

    #timespan, parameters-------------------------------------------------
    tspan = (0.0, T_end)

    cache = (
        n1_ave = CUDA.zeros(tate,yoko),
        d1_ave = CUDA.zeros(tate,yoko),
        n2_ave = CUDA.zeros(tate,yoko),
        d2_ave = CUDA.zeros(tate,yoko),
    )
    p = (
            α1=6.0, α2=10.0, α5=8.0, α6=6.0, 
            γN=2.0, γD=1.0, 
            μ0=0.5, μ1=1.0, μ2=1.0, μ4=0.1, μ5=1.0, μ6=0.5,
            β1=0.1, β21=8.0, β22=5.0, β3=2.0, β41=8.0, β42=8.0,
            ν0=0.5, ν1=5.0, ν2=25.0, ν3=5.0, ωN=0.01, ωD=0.01,
            pα=0.0, pβ=0.0, pγ=0.0, canc=0, h_max=0.0,
            cache...
        )

    prob_lc = ODEProblem(lung_func!, u0, tspan, p)
end

@time lc_sol = solve(prob_lc, Heun(), progress = true, progress_steps = 1.0, callback=TerminateSteadyState(1e-3), saveat=[T_end]);
# BS3, Tsit5, Heun, 
lc_sol.t

#-----------------------get maximum level of HES-1-----------------------
Hes = lc_sol[:, :, end, end] .- Array(u0[:, :, 11])
h_max = maximum(Hes)
index_max = argmax(Hes)

clean_up_GPU()

#=======================SIMULATION WITH CANCER CELLS=======================#
canc = 332
begin
    #make initial data-----------------------------------------------------
    lc_sol = lc_sol[:, :, :, end]
    maxs = (
        Nm1_max = lc_sol[index_max, 1],
        Nm2_max = lc_sol[index_max, 2],
        Dm1_max = lc_sol[index_max, 3],
        Dm2_max = lc_sol[index_max, 4],
        Nc1_max = lc_sol[index_max, 5],
        Nc2_max = lc_sol[index_max, 6],
        Dc1_max = lc_sol[index_max, 7],
        Dc2_max = lc_sol[index_max, 8],
        I1_max  = lc_sol[index_max, 9],
        I2_max  = lc_sol[index_max, 10],
        H_max   = lc_sol[index_max, 11],
    )

    p_ = (
        γN=2.0, γD=1.0, 
        μ0=0.5, μ1=1.0, μ5=1.0, μ6=0.5,
        β3=2.0, β41=8.0, β42=8.0,
        ν0=0.5, ωN=0.01, ωD=0.01, canc=canc,
    )
    u0 .= init_cancer(tate, yoko, p_, maxs)

    #parameters----------------------------------------------------------
    p_ut = (
            α1=6.0, α2=10.0, α5=8.0, α6=6.0, 
            γN=2.0, γD=1.0, 
            μ0=0.5, μ1=1.0, μ2=1.0, μ4=0.1, μ5=1.0, μ6=0.5,
            β1=0.1, β21=8.0, β22=5.0, β3=2.0, β41=8.0, β42=8.0,
            ν0=0.5, ν1=5.0, ν2=25.0, ν3=5.0, ωN=0.01, ωD=0.01,
            pα=0.0, pβ=0.0, pγ=0.0, canc=canc, h_max=h_max,
            cache...
    )

    prob_clc = ODEProblem(lung_func!, u0, tspan, p_ut)
end

@time lc_sol = solve(prob_clc, Heun(), progress=true, progress_steps=1.0, callback=TerminateSteadyState(1e-3), saveat=[T_end]);
lc_sol.t
Hes =  lc_sol[:, :, end, end] .- Array(u0[:, :, end])

time_05 = zeros(8)
time_08 = zeros(8)
count_05 = zeros(8)
count_08 = zeros(8)
time_05[1] = lc_sol.t[1]
time_08[1] = lc_sol.t[1]
count_05[1] = count(Hes .> 0.75*h_max)
count_08[1] = count(Hes .> 0.75*h_max)

#=======================SIMULATION WITH CANCER CELLS (TERATMENT)=======================#
pα = CUDA.zeros(2, 7)
pβ = CUDA.zeros(2, 7)
pγ = CUDA.zeros(2, 7)
a = CuArray([0.5, 0.8])

for k=1:7
    if k == 1 || k == 4 || k == 5 || k == 7
        pα[:, k] .= a
    end
    if  k == 2 || k == 4 || k == 6 || k == 7
        pβ[:, k] .= a
    end
    if k == 3 || k == 5 || k == 6 || k == 7
        pγ[:, k] .= a
    end
end

clean_up_GPU()
#----------------------------------------------------------------------------------------
begin
    for treat = 1:7
        for effect = 1:2
            p = (
                α1=6.0, α2=10.0, α5=8.0, α6=6.0, 
                γN=2.0, γD=1.0, 
                μ0=0.5, μ1=1.0, μ2=1.0, μ4=0.1, μ5=1.0, μ6=0.5,
                β1=0.1, β21=8.0, β22=5.0, β3=2.0, β41=8.0, β42=8.0,
                ν0=0.5, ν1=5.0, ν2=25.0, ν3=5.0, ωN=0.01, ωD=0.01,
                pα=pα[effect, treat], pβ=pβ[effect, treat], pγ=pγ[effect, treat],
                canc=canc, h_max=h_max, cache...
            )
            cprob_lc = ODEProblem(lung_func!, u0, tspan, p)
            @time @views lc_sol = solve(cprob_lc, Heun(), progress=true, progress_steps=1.0, saveat=[T_end], callback=TerminateSteadyState(1e-3));
            Hes =  lc_sol[:, :, end, end] .- Array(u0[:, :, end])

            #obtain the time when steady state is reached--------------------------------
            if effect == 1
                time_05[treat+1] = lc_sol.t[1]
                count_05[treat+1] = count(Hes .> 0.75*h_max)
            end
            if effect == 2
                time_08[treat+1] = lc_sol.t[1]
                count_08[treat+1] = count(Hes .> 0.75*h_max)
            end
            println("treatment = ", treat, " intensity of effect = ", effect, "  0.5:", time_05[treat+1], "  0.8:", time_08[treat+1])
        end 
    end
end
#treatment effectiveness of cancer spreading
@. time_05 = time_05[1]/time_05
@. time_08 = time_08[1]/time_08
#treatment effectiveness of the number of cancer cell
@. count_05 = count_05/count_05[1]
@. count_08 = count_08/count_08[1]
#optimal treatment index
opt_05 = zeros(7)
opt_08 = zeros(7)
for i = 2:8
    opt_05[i-1] = max(0, 1-time_05[i]) * max(0, 1-count_05[i])
    opt_08[i-1] = max(0, 1-time_08[i]) * max(0, 1-count_08[i])
end
#=======================MAKE FIGURE=======================#
using PyPlot

fig1 = plt.figure(figsize=(10, 5))
ax1 = fig1.add_subplot(111)
label = ["untreated", L"$p_{\alpha}$", L"$p_{\beta}$", L"$p_{\gamma}$",
        L"$p_{\alpha}+p_{\beta}$", L"$p_{\alpha}+p_{\gamma}$", L"$p_{\beta}+p_{\gamma}$",
        L"$p_{\alpha}+p_{\beta}+p_{\gamma}$"]
x = Array(1:8)
width = 0.4
ax1.bar(x, time_05, width=width, capsize=5, align="center", label="0.5")
ax1.bar(x.+width, time_08, width=width, capsize=5, align="center", label="0.8")
ax1.axhline(y=1.0, linestyle="dashed", color="black")
ax1.set_xticks(x.+width/2, label)
ax1.legend(loc="best", fontsize=10)
fig1.savefig("enh_spr-time_LC.png")
fig1

fig2 = plt.figure(figsize=(10, 5))
ax2 = fig2.add_subplot(111)
ax2.bar(x, count_05, width=width, capsize=5, align="center", label="0.5")
ax2.bar(x.+width, count_08, width=width, capsize=5, align="center", label="0.8")
ax2.axhline(y=1.0, linestyle="dashed", color="black")
ax2.set_xticks(x.+width/2, label)
ax2.legend(loc="best", fontsize=10)
fig2.savefig("enh_spr-count_LC.png")
fig2

clean_up_GPU()