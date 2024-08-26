include("src/utils.jl") #clean up GPU
include("function/initial_condition.jl") #make array for initial data
include("function/notch_interaction.jl") #no interaction in cells surrounded only by healthy cells
include("function/shift.jl") #shift cells if cell division occured
include("function/division_random.jl") #make array of cell division probabilities
include("function/ode_zeros.jl")


using OrdinaryDiffEq, DiffEqCallbacks
using Random
using CUDA
using Plots
using ColorSchemes

#======================CPU======================#
using ParallelStencil
@init_parallel_stencil(CUDA, Float32, 2) #for GPU
#@init_parallel_stencil(Threads, Float64, 2) #for CPU

#=======================Brain tumor=======================#
const tate = 50 #number of cells in each column
const yoko = 50 #number of cells in each row
T_end = 20000

#=======================Function (ODEs)=======================#
function lung_func!(du, u, p, t)
#parameters
    (;
        α1, α2, α5, α6, 
        γN, γD, 
        μ0, μ1, μ2, μ4, μ5, μ6,
        β1, β21, β22, β3, β41, β42,
        ν0, ν1, ν2, ν3, ωN, ωD, pα, pβ, pγ, canc, h_max,
        n1_ave, n2_ave, d1_ave, d2_ave,
    ) = p
#area except for boundary
    Ix = 2:(tate+1)
    Iy = 2:(yoko+1)

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

#mean of Notch/Delta in surrounding cells
    @views begin
        @. n1_ave = (Nm1[1:tate, Iy] + Nm1[3:tate+2, Iy] + Nm1[Ix, 1:yoko] + Nm1[Ix, 3:yoko+2])/4
        @. n2_ave = (Nm2[1:tate, Iy] + Nm2[3:tate+2, Iy] + Nm2[Ix, 1:yoko] + Nm2[Ix, 3:yoko+2])/4
        @. d1_ave = (Dm1[1:tate, Iy] + Dm1[3:tate+2, Iy] + Dm1[Ix, 1:yoko] + Dm1[Ix, 3:yoko+2])/4
        @. d2_ave = (Dm2[1:tate, Iy] + Dm2[3:tate+2, Iy] + Dm2[Ix, 1:yoko] + Dm2[Ix, 3:yoko+2])/4

#ODEs
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
            #the state of cancer cells are steady
            @parallel cancer!(du, u0[:, :, end], h_max)
            
            #signaling does not take pace if all surrounding cells are healthy
            @parallel notch_d!(dDm1, Dm1, u0[:, :, end], h_max)
            @parallel notch_d!(dDm2, Dm2, u0[:, :, end], h_max)
            
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
            β1=0.1, β21=8.0, β22=5.0, β3=2.0, β41=8.0, β42=8.0,
            ν0=0.5, ν1=5.0, ν2=25.0, ν3=5.0, ωN=0.01, ωD=0.01,
            pα=0.0, pβ=0.0, pγ=0.0, canc=0, h_max=0.0,
            cache...
        )

    prob_lc = ODEProblem(lung_func!, u0, tspan, p)
end

#=------------ arrange of animation ------------

function condition(u, t, integrator)
    return integrator.p.k_ref[1] - t < 1.0
 end
function affect!(integrator)
    heatmap(Array(integrator.u[2:tate+1, 2:yoko+1, 11]), c=:thermal, clim=(25, 50))
    frame(integrator.p.anim)
    integrator.p.k_ref[1] += 200
    return nothing
end
cb1 = DiscreteCallback(condition, affect!)

----------------------------------------------=#

#calculation in the absence of cancer
@time lc_sol = solve(prob_lc, ROCK2(), progress = true, progress_steps = 1.0, saveat=[T_end], callback=TerminateSteadyState(1e-3));
#calculation methods: BS3, Tsit5, Heun, ROCK2

clean_up_GPU()

#-----------------------get maximum level of HES-1-----------------------
Hes = lc_sol[:, :, end, end] .- Array(u0[:, :, 11])
h_max = maximum(Hes) #obtain the highest level of HES1
index_max = argmax(Hes) #obtain an index of a cell that express the highest level of HES1

canc = 10
#=======================SIMULATION WITH CANCER CELLS=======================#
begin
    #make initial data-----------------------------------------------------
    lc_sol = Array(lc_sol[:, :, :, end])
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
        H_max   = lc_sol[index_max, 11]
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
#create image of initial state
function plot_heatmap(u, t, integrator)
    heatmap(Array(u[2:tate+1, 2:yoko+1, 11]), title="t=$(round(t, digits=1))", c=:jet1, clim=(0, h_max), aspect_ratio=1)
    frame(anim)
    return nothing
end
cb1 = FunctionCallingCallback(plot_heatmap, funcat = LinRange(0, T_end, 30)) #create image at regular intervals
cb2 = TerminateSteadyState(1e-3) #if all cells reach steady state, terminate the simulation
cbs = CallbackSet(cb1, cb2)
anim = Animation() #array of image for animation

while maximum(Array(u0[2, 2:yoko+1, end]))<h_max*0.75 && maximum(Array(u0[tate+1, 2:yoko+1, end]))<h_max*0.75 && maximum(Array(u0[2:tate+1, 2, end]))<h_max*0.75 && maximum(Array(u0[2:tate+1, yoko+1, end]))<h_max*0.75
    #cellular interaction
    @time lc_sol = solve(prob_clc, ROCK2(), progress = true, progress_steps = 1.0, callback=cbs, saveat=[T_end]);
    u0 = lc_sol[end]

    #cell division
    r = CUDA.zeros(tate, yoko)
    @parallel division_random!(r, u0[2:tate+1, 2:yoko+1, end], h_max)
    u0 = shift!(r, u0, tate, yoko)
    heatmap(Array(u0[2:tate+1, 2:yoko+1, end]), title="t=$(round(lc_sol.t[end], digits=1))", c=:jet1, clim=(0, h_max), aspect_ratio=1)
    frame(anim)
    #updata
    T_end = lc_sol.t[end] + 50000
    tspan = (lc_sol.t[end], T_end)
    prob_clc = ODEProblem(lung_func!, u0, tspan, p_ut)
    cb1 = FunctionCallingCallback(plot_heatmap, funcat = LinRange(tspan[1], T_end, 30)) #create image at regular intervals
    cbs = CallbackSet(cb1, cb2)
    clean_up_GPU()
end

clean_up_GPU()
gif(anim, "CellDivision_LC.gif", fps=90) #create gif animation
heatmap(Array(u0[2:tate+1, 2:yoko+1, 11]), c=:jet1, clim=(h_max*0.5, h_max), aspect_ratio=1)


#=======================SIMULATION WITH CANCER CELLS (TERATMENT)=======================#
begin
    p = (
        α1=6.0, α2=10.0, α5=8.0, α6=6.0, 
        γN=2.0, γD=1.0, 
        μ0=0.5, μ1=1.0, μ2=1.0, μ4=0.1, μ5=1.0, μ6=0.5,
        β1=0.1, β21=8.0, β22=5.0, β3=2.0, β41=8.0, β42=8.0,
        ν0=0.5, ν1=5.0, ν2=25.0, ν3=5.0, ωN=0.01, ωD=0.01,
        pα=0.0, pβ=0.0, pγ=0.0,
        canc=canc, h_max=h_max, cache...
    )
    T_end = 20000
    tspan = (0.0, T_end)
    cprob_bt = ODEProblem(lung_func!, u0, tspan, p)
    cb1 = FunctionCallingCallback(plot_heatmap, funcat = LinRange(0, T_end, 30)) #create image at regular intervals
    cbs = CallbackSet(cb1, cb2)
    anim = Animation()
    while maximum(Array(u0[2, 2:yoko+1, end]))<h_max*0.75 && maximum(Array(u0[tate+1, 2:yoko+1, end]))<h_max*0.75 && maximum(Array(u0[2:tate+1, 2, end]))<h_max*0.75 && maximum(Array(u0[2:tate+1, yoko+1, end]))<h_max*0.75
        #cellular interaction
        @time lc_sol = solve(cprob_lc, ROCK2(), progress = true, progress_steps = 1.0, callback=cbs, saveat=[T_end]);
        u0 = lc_sol[end]

        #cell division
        r = CUDA.zeros(tate, yoko)
        @parallel division_random!(r, u0[2:tate+1, 2:yoko+1, end], h_max)
        u0 = shift!(r, u0, tate, yoko)
        heatmap(Array(u0[2:tate+1, 2:yoko+1, 11]), title="t=$(round(T_end, digits=1))", c=:jet1, clim=(0, h_max), aspect_ratio=1)
        frame(anim)

        #updata
        T_end = lc_sol.t[end] + 50000
        tspan = (lc_sol.t[end], T_end)
        cprob_bt = ODEProblem(lung_func!, u0, tspan, p)
        cb1 = FunctionCallingCallback(plot_heatmap, funcat = LinRange(tspan[1], T_end, 30)) #create image at regular intervals
        cbs = CallbackSet(cb1, cb2)
        clean_up_GPU()
    end

end
gif(anim, "CellDivision_clv_LC.gif", fps=90)
clean_up_GPU()