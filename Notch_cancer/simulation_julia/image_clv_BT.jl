include("src/utils.jl")
include("src/initial_condition.jl")
include("src/notch_interaction.jl")

using OrdinaryDiffEq, DiffEqCallbacks
using Random
using CUDA
using Plots
using ColorSchemes

#======================CPU======================#
using ParallelStencil
@init_parallel_stencil(CUDA, Float32, 2)
#@init_parallel_stencil(Threads, Float64, 2)

#=======================Brain tumor=======================#
const tate = 1000
const yoko = 1000
T_end = 10000

#=======================Function (ODEs)=======================#
function brain_func!(du, u, p, t)
    (;
        α1, α2, α5, α6, 
        γN, γD, 
        μ0, μ1, μ2, μ4, μ5, μ6,
        β1, β21, β22, β3, β41, β42,
        ν0, ν1, ν2, ν3, ωN, ωD, p1, p23, pβ, canc, h_max,
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

        @. du[Ix, Iy, 1]  = -α1 * d1_ave * Nm1[Ix, Iy] - α2 * d2_ave * Nm1[Ix, Iy] - μ1 * Nm1[Ix, Iy] + (1 - p1)*γN * Nc1[Ix, Iy]
        @. du[Ix, Iy, 2]  = -α5 * d1_ave * Nm2[Ix, Iy] - α6 * d2_ave * Nm2[Ix, Iy] - μ1 * Nm2[Ix, Iy] + (1 - p1)*γN * Nc2[Ix, Iy]
        @. dDm1[Ix, Iy]   = -α1 * n1_ave * Dm1[Ix, Iy] - α5 * n2_ave * Dm1[Ix, Iy] - μ2 * Dm1[Ix, Iy] + γD * Dc1[Ix, Iy]
        @. dDm2[Ix, Iy]   = -α2 * n1_ave * Dm2[Ix, Iy] - α6 * n2_ave * Dm2[Ix, Iy] - μ2 * Dm2[Ix, Iy] + γD * Dc2[Ix, Iy]
        @. du[Ix, Iy, 5]  = ωN + (1 - pβ) * β21 * ( I1[Ix, Iy]^2 / (β1 + I1[Ix, Iy]^2) ) - (μ5 + (1 - p1)*γN) * Nc1[Ix, Iy]
        @. du[Ix, Iy, 6]  = ωN + (1 - pβ) * β22 * ( I2[Ix, Iy]^2 / (β1 + I2[Ix, Iy]^2) ) - (μ5 + (1 - p1)*γN) * Nc2[Ix, Iy]
        @. du[Ix, Iy, 7]  = ωD + β41 / (β3 + H[Ix, Iy]^2) - (μ6 + γD) * Dc1[Ix, Iy]
        @. du[Ix, Iy, 8]  = ωD + β42 / (β3 + H[Ix, Iy]^2) - (μ6 + γD) * Dc2[Ix, Iy]
        @. du[Ix, Iy, 9]  = (1 - p23)*(α1 * d1_ave * Nm1[Ix, Iy] + α2 * d2_ave * Nm1[Ix, Iy]) - μ4 * I1[Ix, Iy]
        @. du[Ix, Iy, 10] = (1 - p23)*(α5 * d1_ave * Nm2[Ix, Iy] + α6 * d2_ave * Nm2[Ix, Iy]) - μ4 * I2[Ix, Iy]
        @. du[Ix, Iy, 11] = ν0 + (ν2 * I2[Ix, Iy]^2) / (ν1 + I2[Ix, Iy]^2) * (1.0 - I1[Ix, Iy]^2 / (ν3 + I1[Ix, Iy]^2)) - μ0 * H[Ix, Iy]
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
    #timespan, parameters------------------------------------------------
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
            β1=0.1, β21=5.0, β22=8.0, β3=2.0, β41=8.0, β42=8.0,
            ν0=0.5, ν1=5.0, ν2=25.0, ν3=5.0, ωN=0.01, ωD=0.01,
            p1=0.0, p23=0.0, pβ =0.0, canc=0, h_max=0.0,
            cache...
        )

    prob_bt = ODEProblem(brain_func!, u0, tspan, p)
end

#------------ arrange of animation ------------

# function condition(u, t, integrator)
#     return integrator.p.k_ref[1] - t < 1.0
# end
# function affect!(integrator)
#     heatmap(Array(integrator.u[2:tate+1, 2:yoko+1, 11]), c=:thermal, clim=(25, 50))
#     frame(integrator.p.anim)
#     integrator.p.k_ref[1] += 200
#     return nothing
# end
#cb1 = DiscreteCallback(condition, affect!)

#----------------------------------------------



@time bt_sol = solve(prob_bt, ROCK2(), progress = true, progress_steps = 1.0, saveat=[T_end], callback=TerminateSteadyState(1e-3));
# BS3, Tsit5, Heun, ROCK2

clean_up_GPU()

#-----------------------get maximum level of HES-1-----------------------
Hes = bt_sol[:, :, end, end] .- Array(u0[:, :, 11])
h_max = maximum(Hes)
index_max = argmax(Hes)

#=======================SIMULATION WITH CANCER CELLS=======================#
begin
    #make initial data-----------------------------------------------------
    bt_sol = Array(bt_sol[:, :, :, end])
    maxs = (
        Nm1_max = bt_sol[index_max, 1],
        Nm2_max = bt_sol[index_max, 2],
        Dm1_max = bt_sol[index_max, 3],
        Dm2_max = bt_sol[index_max, 4],
        Nc1_max = bt_sol[index_max, 5],
        Nc2_max = bt_sol[index_max, 6],
        Dc1_max = bt_sol[index_max, 7],
        Dc2_max = bt_sol[index_max, 8],
        I1_max  = bt_sol[index_max, 9],
        I2_max  = bt_sol[index_max, 10],
        H_max   = bt_sol[index_max, 11]
    )

    p_ = (
        γN=2.0, γD=1.0, 
        μ0=0.5, μ1=1.0, μ5=1.0, μ6=0.5,
        β3=2.0, β41=8.0, β42=8.0,
        ν0=0.5, ωN=0.01, ωD=0.01, canc=332,
    )
    u0 .= init_cancer(tate, yoko, p_, maxs)

    #parameters----------------------------------------------------------
    p_ut = (
            α1=6.0, α2=10.0, α5=8.0, α6=6.0, 
            γN=2.0, γD=1.0, 
            μ0=0.5, μ1=1.0, μ2=1.0, μ4=0.1, μ5=1.0, μ6=0.5,
            β1=0.1, β21=5.0, β22=8.0, β3=2.0, β41=8.0, β42=8.0,
            ν0=0.5, ν1=5.0, ν2=25.0, ν3=5.0, ωN=0.01, ωD=0.01,
            p1=0.0, p23=0.0, pβ=0.0, canc=332, h_max=h_max,
            cache...
    )

    prob_cbt = ODEProblem(brain_func!, u0, tspan, p_ut)
end

function plot_heatmap(u, t, integrator)
    heatmap(Array(u[2:tate+1, 2:yoko+1, 11]), title="t=$(round(t, digits=1))", c=:jet1, clim=(25, 50), aspect_ratio=1)
    frame(anim)
    return nothing
end
cb1 = FunctionCallingCallback(plot_heatmap, funcat = LinRange(0, T_end, 30))
cb2 = TerminateSteadyState(1e-3)
cbs = CallbackSet(cb1, cb2)
anim = Animation()
@time bt_sol = solve(prob_cbt, ROCK2(), progress = true, progress_steps = 1.0, callback=cbs, saveat=[T_end]);

clean_up_GPU()
gif(anim, "cancer_spr_ut_BT.gif", fps=60)

#=======================SIMULATION WITH CANCER CELLS (TERATMENT)=======================#
begin
    p = (
        α1=6.0, α2=10.0, α5=8.0, α6=6.0, 
        γN=2.0, γD=1.0, 
        μ0=0.5, μ1=1.0, μ2=1.0, μ4=0.1, μ5=1.0, μ6=0.5,
        β1=0.1, β21=5.0, β22=8.0, β3=2.0, β41=8.0, β42=8.0,
        ν0=0.5, ν1=5.0, ν2=25.0, ν3=5.0, ωN=0.01, ωD=0.01,
        p1=0.5, p23=0, pβ=0,
        canc=332, h_max=h_max, cache...
    )
    cprob_bt = ODEProblem(brain_func!, u0, tspan, p)
    anim = Animation()
    @time bt_sol = solve(cprob_bt, ROCK2(), progress = true, progress_steps = 1.0, callback=cbs, saveat=[T_end]);
end
gif(anim, "cancer_spr_pa-05_clvBT.gif", fps=60)
clean_up_GPU()