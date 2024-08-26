using GLMakie
using OrdinaryDiffEq
using ComponentArrays
using Random

const tate = 100
const yoko = 100
T = 200.0

#=======================Lung Cancer=======================#

function lung_func!(du, u, p, t)
    (
        α1, α2, α5, α6, 
        γN, γD, 
        μ0, μ1, μ2, μ4, μ5, μ6,
        β1, β21, β22, β3, β41, β42,
        ν0, ν1, ν2, ν3 
    ) = p
    Nm1 = u[1, :, :]
    Nm2 = u[2, :, :] 
    Dm1 = u[3, :, :] 
    Dm2 = u[4, :, :] 
    Nc1 = u[5, :, :] 
    Nc2 = u[6, :, :] 
    Dc1 = u[7, :, :] 
    Dc2 = u[8, :, :]
    I1 = u[9, :, :]
    I2 = u[10, :, :]
    H = u[11, :, :]
    Ix = 2:(tate+1)
    Iy = 2:(yoko+1)

    n1_ave = (Nm1[1:tate, Iy] + Nm1[3:tate+2, Iy] + Nm1[Ix, 1:yoko] + Nm1[Ix, 3:yoko+2])/4
    n2_ave = (Nm2[1:tate, Iy] + Nm2[3:tate+2, Iy] + Nm2[Ix, 1:yoko] + Nm2[Ix, 3:yoko+2])/4
    d1_ave = (Dm1[1:tate, Iy] + Dm1[3:tate+2, Iy] + Dm1[Ix, 1:yoko] + Dm1[Ix, 3:yoko+2])/4
    d2_ave = (Dm2[1:tate, Iy] + Dm2[3:tate+2, Iy] + Dm2[Ix, 1:yoko] + Dm2[Ix, 3:yoko+2])/4

    @. du[1, Ix, Iy] = -α1 * d1_ave * Nm1[Ix, Iy] - α2 * d2_ave * Nm1[Ix, Iy] - μ1 * Nm1[Ix, Iy] + γN * Nc1[Ix, Iy]
    @. du[2, Ix, Iy] = -α5 * d1_ave * Nm2[Ix, Iy] - α6 * d2_ave * Nm2[Ix, Iy] - μ1 * Nm2[Ix, Iy] + γN * Nc2[Ix, Iy]
    @. du[3, Ix, Iy] = -α1 * n1_ave * Dm1[Ix, Iy] - α5 * n2_ave * Dm1[Ix, Iy] - μ2 * Dm1[Ix, Iy] + γD * Dc1[Ix, Iy]
    @. du[4, Ix, Iy] = -α2 * n1_ave * Dm2[Ix, Iy] - α6 * n2_ave * Dm2[Ix, Iy] - μ2 * Dm2[Ix, Iy] + γD * Dc2[Ix, Iy]
    @. du[5, Ix, Iy] = β21 * ( I1[Ix, Iy]^2 / (β1 + I1[Ix, Iy]^2) ) - (μ5 + γN) * Nc1[Ix, Iy]
    @. du[6, Ix, Iy] = β22 * ( I2[Ix, Iy]^2 / (β1 + I2[Ix, Iy]^2) ) - (μ5 + γN) * Nc2[Ix, Iy]
    @. du[7, Ix, Iy] = β41 / (β3 + H[Ix, Iy]^2) - (μ6 + γD) * Dc1[Ix, Iy]
    @. du[8, Ix, Iy] = β42 / (β3 + H[Ix, Iy]^2) - (μ6 + γD) * Dc2[Ix, Iy]
    @. du[9, Ix, Iy] = α1 * d1_ave * Nm1[Ix, Iy] + α2 * d2_ave * Nm1[Ix, Iy] - μ4 * I1[Ix, Iy]
    @. du[10, Ix, Iy] = α5 * d1_ave * Nm2[Ix, Iy] + α6 * d2_ave * Nm2[Ix, Iy] - μ4 * I2[Ix, Iy]
    @. du[11, Ix, Iy] = ν0 + (ν2 * I1[Ix, Iy]^2) / (ν1 + I1[Ix, Iy]^2) * (1.0 - I2[Ix, Iy]^2 / (ν3 + I2[Ix, Iy]^2)) - μ0 * H[Ix, Iy]
    return du
end

param = Vector(0:1.0:10.0)

for i in param
    a=[]
    begin
        u0 = zeros(11, tate+2, yoko+2)
        u0[:, 2:tate+1, 2:yoko+1] .= rand(11, tate, yoko)
        tspan = (0.0, T)

        p = (
                α1=i, α2=10.0, α5=8.0, α6=6.0, 
                γN=2.0, γD=1.0, 
                μ0=0.5, μ1=1.0, μ2=1.0, μ4=0.1, μ5=1.0, μ6=0.5,
                β1=0.1, β21=8.0, β22=5.0, β3=2.0, β41=8.0, β42=8.0,
                ν0=0.5, ν1=5.0, ν2=25.0, ν3=5.0
        )

        prob_lc = ODEProblem(lung_func!, u0, tspan, p)
    end
    
    @time lc_sol = solve(prob_lc, BS3(), saveat=[T]);
    
    hmax = maximum(lc_sol[11, 2:end-1, 2:end-1])
    a_new = (count(lc_sol[11, 2:end-1, 2:end-1] .< hmax*0.75))/(tate*yoko)
    push!(a, a_new)
end

#------------------------------Plot------------------------------
using CairoMakie
using Makie.Colors

fig = figure()
Axis(fig[1, 1])
len = size(a)
xs = LinRange(0, 1, len)

scatterlines(xs, a, markersize=20)

fig

