using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

using GLMakie
using OrdinaryDiffEq
using ComponentArrays
using Random
using CUDA

#======================CPU======================#
using ParallelStencil
#@init_parallel_stencil(CUDA, Float32, 2)
@init_parallel_stencil(Threads, Float64, 2)

#=======================Lung Cancer=======================#
const tate = 1000
const yoko = 1000
const T_end = 100

#=======================Notch signal=======================#
@parallel_indices (ix, iy) function notch!(dDm, Dm, H, H_max)
if ix > 1 && iy > 1 && ix < size(dDm,1) && iy < size(dDm,2)
        if H[ix-1,iy] < 0.1 * H_max && H[ix+1,iy] < 0.1 * H_max && H[ix,iy-1] < 0.1 * H_max && H[ix,iy+1] < 0.1 * H_max && Dm[ix, iy]==0
            dDm[ix, iy] = 0.0
        end
    end 
    return nothing
end

#=======================Function (ODEs)=======================#
function lung_func!(du, u, p, t)
    (;
        α1, α2, α5, α6, 
        γN, γD, 
        μ0, μ1, μ2, μ4, μ5, μ6,
        β1, β21, β22, β3, β41, β42,
        ν0, ν1, ν2, ν3, ωN, ωD, pα, pβ, pγ, canc, h_max,
        n1_ave, n2_ave, d1_ave, d2_ave,
    ) = p


    (; Nm1, Nm2, Dm1, Dm2, Nc1, Nc2, Dc1, Dc2, I1, I2, H) = u

    Ix = 2:(tate+1)
    Iy = 2:(yoko+1)
    Cx = Int(floor(1+(tate-canc)/2)):Int(floor((tate+canc)/2))
    Cy = Int(floor(1+(yoko-canc)/2)):Int(floor((yoko+canc)/2))

    @views begin
        @. n1_ave = (Nm1[1:tate, Iy] + Nm1[3:tate+2, Iy] + Nm1[Ix, 1:yoko] + Nm1[Ix, 3:yoko+2])/4
        @. n2_ave = (Nm2[1:tate, Iy] + Nm2[3:tate+2, Iy] + Nm2[Ix, 1:yoko] + Nm2[Ix, 3:yoko+2])/4
        @. d1_ave = (Dm1[1:tate, Iy] + Dm1[3:tate+2, Iy] + Dm1[Ix, 1:yoko] + Dm1[Ix, 3:yoko+2])/4
        @. d2_ave = (Dm2[1:tate, Iy] + Dm2[3:tate+2, Iy] + Dm2[Ix, 1:yoko] + Dm2[Ix, 3:yoko+2])/4

        @. du.Nm1[Ix, Iy] = (1 + pα)*(-α1 * d1_ave * Nm1[Ix, Iy] - α2 * d2_ave * Nm1[Ix, Iy]) - μ1 * Nm1[Ix, Iy] + (1 + pγ)*γN * Nc1[Ix, Iy]
        @. du.Nm2[Ix, Iy] = (1 + pα)*(-α5 * d1_ave * Nm2[Ix, Iy] - α6 * d2_ave * Nm2[Ix, Iy]) - μ1 * Nm2[Ix, Iy] + (1 + pγ)*γN * Nc2[Ix, Iy]
        @. du.Dm1[Ix, Iy] = (1 + pα)*(-α1 * n1_ave * Dm1[Ix, Iy] - α5 * n2_ave * Dm1[Ix, Iy]) - μ2 * Dm1[Ix, Iy] + γD * Dc1[Ix, Iy]
        @. du.Dm2[Ix, Iy] = (1 + pα)*(-α2 * n1_ave * Dm2[Ix, Iy] - α6 * n2_ave * Dm2[Ix, Iy]) - μ2 * Dm2[Ix, Iy] + γD * Dc2[Ix, Iy]
        @. du.Nc1[Ix, Iy] = ωN + (1 + pβ)β21 * ( I1[Ix, Iy]^2 / (β1 + I1[Ix, Iy]^2) ) - (μ5 + (1 + pγ)*γN) * Nc1[Ix, Iy]
        @. du.Nc2[Ix, Iy] = ωN + (1 + pβ)β22 * ( I2[Ix, Iy]^2 / (β1 + I2[Ix, Iy]^2) ) - (μ5 + (1 + pγ)*γN) * Nc2[Ix, Iy]
        @. du.Dc1[Ix, Iy] = ωD + β41 / (β3 + H[Ix, Iy]^2) - (μ6 + γD) * Dc1[Ix, Iy]
        @. du.Dc2[Ix, Iy] = ωD + β42 / (β3 + H[Ix, Iy]^2) - (μ6 + γD) * Dc2[Ix, Iy]
        @. du.I1[Ix, Iy] = (1 + pα)*(α1 * d1_ave * Nm1[Ix, Iy] + α2 * d2_ave * Nm1[Ix, Iy]) - μ4 * I1[Ix, Iy]
        @. du.I2[Ix, Iy] = (1 + pα)*(α5 * d1_ave * Nm2[Ix, Iy] + α6 * d2_ave * Nm2[Ix, Iy]) - μ4 * I2[Ix, Iy]
        @. du.H[Ix, Iy] = ν0 + (ν2 * I1[Ix, Iy]^2) / (ν1 + I1[Ix, Iy]^2) * (1.0 - I2[Ix, Iy]^2 / (ν3 + I2[Ix, Iy]^2)) - μ0 * H[Ix, Iy]
        if canc > 0

            @parallel notch!(du.Dm1, u.Dm1, u.H, h_max)
            @parallel notch!(du.Dm2, u.Dm1, u.H, h_max)

            @. du.Nm1[Cx, Cy] = 0.0
            @. du.Nm2[Cx, Cy] = 0.0
            @. du.Dm1[Cx, Cy] = 0.0
            @. du.Dm2[Cx, Cy] = 0.0
            @. du.Nc1[Cx, Cy] = 0.0
            @. du.Nc2[Cx, Cy] = 0.0
            @. du.Dc1[Cx, Cy] = 0.0
            @. du.Dc2[Cx, Cy] = 0.0
            @. du.I1[Cx, Cy] = 0.0
            @. du.I2[Cx, Cy] = 0.0
            @. du.H[Cx, Cy] = 0.0            
        end
    end
    nothing
end

#-----------------------initial condition-----------------------
function init_random(tate, yoko)
    ar = zeros(tate+2, yoko+2)
    ar[2:end-1, 2:end-1] .= 0.01*rand(tate, yoko)
    return ar 
end 

#=======================SIMULATION WITHOUT CANCER CELLS=======================#
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
                        I1 = init_random(tate, yoko),
                        I2 = init_random(tate, yoko),
                        H = init_random(tate, yoko),
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
            ν0=0.5, ν1=5.0, ν2=25.0, ν3=5.0, ωN=0.01, ωD=0.01,
            pα=0.0, pβ=0.0, pγ=0.0, canc=0, h_max=0.0,
            cache...
        )

    prob_lc = ODEProblem(lung_func!, u0, tspan, p)
end

@time lc_sol = solve(prob_lc, BS5(), progress = true, progress_steps = 1.0, saveat = [T_end]);
# BS3, Tsit5, Heun, 

#-----------------------get maximum level of HES-1-----------------------
h_max     = maximum(lc_sol[end].H .- u0.H)
index_max = argmax(lc_sol[end].H .- u0.H)

#-----------------------initial condition for simulation of cancer spread-----------------------
function init_cancer(tate, yoko, p_, maxs)
    (; Nm1_max, Nm2_max, Dm1_max, Dm2_max, Nc1_max, Nc2_max, Dc1_max, Dc2_max, I1_max, I2_max, H_max) = maxs
    (; γN, γD, μ0, μ1, μ5, μ6, β3, β41, β42, ν0, ωN, ωD, canc) = p_

    ar = zeros(11, tate+2, yoko+2)
    Ix = 2:(tate+1)
    Iy = 2:(yoko+1)
    Cx = Int(floor(1+(tate-canc)/2)):Int(floor((tate+canc)/2))
    Cy = Int(floor(1+(yoko-canc)/2)):Int(floor((yoko+canc)/2))
    begin
        ar[1:2, Ix, Iy] .= ωN*γN / (μ1*(μ5+γN))
        ar[5:6, Ix, Iy] .= ωN/(μ5 + γN)
        ar[7, Ix, Iy]   .= (1/(μ6 + γD))*(ωD + β41/(β3 + (ν0/μ0)^2))
        ar[8, Ix, Iy]   .= (1/(μ6 + γD))*(ωD + β42/(β3 + (ν0/μ0)^2))
        ar[11, Ix, Iy]  .= ν0/μ0

        ar[1, Cx, Cy]  .= Nm1_max
        ar[2, Cx, Cy]  .= Nm2_max
        ar[3, Cx, Cy]  .= Dm1_max
        ar[4, Cx, Cy]  .= Dm2_max
        ar[5, Cx, Cy]  .= Nc1_max
        ar[6, Cx, Cy]  .= Nc2_max
        ar[7, Cx, Cy]  .= Dc1_max
        ar[8, Cx, Cy]  .= Dc2_max
        ar[9, Cx, Cy]  .= I1_max
        ar[10, Cx, Cy] .= I2_max
        ar[11, Cx, Cy] .= H_max
    end
    return ar 
end 

#=======================SIMULATION WITH CANCER CELLS=======================#
begin
    #make initial data-----------------------------------------------------
    maxs = (
        Nm1_max = lc_sol[end].Nm1[index_max],
        Nm2_max = lc_sol[end].Nm2[index_max],
        Dm1_max = lc_sol[end].Dm1[index_max],
        Dm2_max = lc_sol[end].Dm2[index_max],
        Nc1_max = lc_sol[end].Nc1[index_max],
        Nc2_max = lc_sol[end].Nc2[index_max],
        Dc1_max = lc_sol[end].Dc1[index_max],
        Dc2_max = lc_sol[end].Dc2[index_max],
        I1_max  = lc_sol[end].I1[index_max],
        I2_max  = lc_sol[end].I2[index_max],
        H_max   = lc_sol[end].H[index_max],
    )

    p_ = (
        γN=2.0, γD=1.0, 
        μ0=0.5, μ1=1.0, μ5=1.0, μ6=0.5,
        β3=2.0, β41=8.0, β42=8.0,
        ν0=0.5, ωN=0.01, ωD=0.01, canc=10,
    )
    u0_canc = ComponentArray(Nm1 = init_cancer(tate, yoko, p_, maxs)[1, :, :],
                             Nm2 = init_cancer(tate, yoko, p_, maxs)[2, :, :],
                             Dm1 = init_cancer(tate, yoko, p_, maxs)[3, :, :],
                             Dm2 = init_cancer(tate, yoko, p_, maxs)[4, :, :],
                             Nc1 = init_cancer(tate, yoko, p_, maxs)[5, :, :],
                             Nc2 = init_cancer(tate, yoko, p_, maxs)[6, :, :],
                             Dc1 = init_cancer(tate, yoko, p_, maxs)[7, :, :],
                             Dc2 = init_cancer(tate, yoko, p_, maxs)[8, :, :],
                             I1  = init_cancer(tate, yoko, p_, maxs)[9, :, :],
                             I2  = init_cancer(tate, yoko, p_, maxs)[10, :, :],
                             H   = init_cancer(tate, yoko, p_, maxs)[11, :, :],
                            )
    #parameters----------------------------------------------------------
    p_ut = (
            α1=6.0, α2=10.0, α5=8.0, α6=6.0, 
            γN=2.0, γD=1.0, 
            μ0=0.5, μ1=1.0, μ2=1.0, μ4=0.1, μ5=1.0, μ6=0.5,
            β1=0.1, β21=8.0, β22=5.0, β3=2.0, β41=8.0, β42=8.0,
            ν0=0.5, ν1=5.0, ν2=25.0, ν3=5.0, ωN=0.01, ωD=0.01,
            pα=0.0, pβ=0.0, pγ=0.0, canc=10, h_max=h_max,
            cache...
    )

    prob_clc = ODEProblem(lung_func!, u0_canc, tspan, p_ut)
end

@time clc_sol = solve(prob_clc, BS5(), progress = true, progress_steps = 1.0);

#-----------------------get # of cells with high expression of HES-1-----------------------
hes =  clc_sol[end].H[2:end-1, 2:end-1] .- u0_canc.H[2:end-1, 2:end-1]
un_05  = count(hes .> 0.5*h_max)
un_075 = count(hes .> 0.75*h_max)

#------------------------Animation------------------------
using CairoMakie
using Makie.Colors

time = Observable(1)
data = @lift clc_sol[$time].H[2:tate+1, 2:yoko+1]
fig, ax, hm = heatmap(data, axis = (title = @lift("t= $(round(clc_sol.t[$time], digits=1))"),))
Colorbar(fig[:, end+1], hm)
len_lc = size(clc_sol)[2]
framerate = Int(floor(len_lc/10))
timestamps = range(1, len_lc)

Makie.record(fig, "HES-1_LC.mp4", timestamps;
        framerate=framerate) do t
    time[] = t
end


#=======================SIMULATION WITH CANCER CELLS (TERATMENT)=======================#
#treatment effect----------------------------------------------------------------------
function p_rand()
    return rand(tate, yoko, 10)
end

pα = zeros(tate, yoko, 7, 10)
pβ = zeros(tate, yoko, 7, 10)
pγ = zeros(tate, yoko, 7, 10)
for k=1:7
    if k == 1 || k == 4 || k == 5 || k == 7
        pα[:, :, k, :] .= p_rand()
    elseif  k == 2 || k == 4 || k == 6 || k == 7
        pβ[:, :, k, :] .= p_rand()
    elseif k == 3 || k == 5 || k == 6 || k == 7
        pγ[:, :, k, :] .= p_rand()
    end
end
#----------------------------------------------------------------------------------------

begin
    cell_05 = zeros(7, 10)
    cell_075 = zeros(7, 10)
    for treat = 1:7
        for patient = 1:10
            p = (
                α1=6.0, α2=10.0, α5=8.0, α6=6.0, 
                γN=2.0, γD=1.0, 
                μ0=0.5, μ1=1.0, μ2=1.0, μ4=0.1, μ5=1.0, μ6=0.5,
                β1=0.1, β21=8.0, β22=5.0, β3=2.0, β41=8.0, β42=8.0,
                ν0=0.5, ν1=5.0, ν2=25.0, ν3=5.0, ωN=0.01, ωD=0.01,
                pα=pα[:, :, treat, patient], pβ=pβ[:, :, treat, patient], pγ=pγ[:, :, treat, patient],
                canc=10, h_max=h_max, cache...
            )
            cprob_lc = ODEProblem(lung_func!, u0_canc, tspan, p)
            @time clc_sol = solve(cprob_lc, BS5(), progress = true, progress_steps = 1.0);

            #count # of cells with high expression of HES-1--------------------------------
            @. hes = clc_sol[end].H[2:end-1, 2:end-1] - u0_canc.H[2:end-1, 2:end-1]
            cell_05[treat, patient] = count(hes .> 0.5*h_max)/un_05
            cell_075[treat, patient] = count(hes .> 0.75*h_max)/un_075
            println("treatment =", treat, "patient =", patient)
        end 
    end
end

mean_05  = ones(8) 
mean_075 = ones(8) 
std_05  = zeros(8)
std_075 = zeros(8)

using StatsBase
for i = 2:8
    mean_05[i] = mean(cell_05[i-1, :])
    mean_075[i] = mean(cell_075[i-1, :]) 
    std_05[i] = std(cell_05[i-1, :])
    std_075[i] = std(cell_075[i-1, :])
end

mean_05  
mean_075 
std_05  
std_075 

#=======================MAKE FIGURE=======================#
using PyPlot

fig1 = plt.figure(figsize=(10, 5))
ax1 = fig1.add_subplot(111)
label = ["untreated", L"$p_{\alpha}$", L"$p_{\beta}$", L"$p_{\gamma}$",
        L"$p_{\alpha}+p_{\beta}$", L"$p_{\alpha}+p_{\gamma}$", L"$p_{\beta}+p_{\gamma}$",
        L"$p_{\alpha}+p_{\beta}+p_{\gamma}$"]
x = Array(1:8)
ax1.bar(x, mean_075, yerr=std_075, capsize=5, tick_label=label)
ax1.axhline(y=1.0, linestyle="dashed", color="black")

fig1.savefig("cancer_spr_LC_paper.png")

#-------------------------------------------------------------
fig2 = plt.figure(figsize=(10, 5))
ax2 = fig2.add_subplot(111)
width = 0.4
ax2.bar(x, mean_05, yerr=std_05, width=width, capsize=5, align="center")
ax2.bar(x.+width, mean_075, yerr=std_075, width=width, capsize=5, align="center")
ax2.axhline(y=1.0, linestyle="dashed", color="black")
ax2.set_xticks(x.+width/2, label)

fig2
fig2.savefig("cancer_spr_LC.png")