using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())
using Random

function init_random(tate, yoko)
    ar = CUDA.zeros(tate+2, yoko+2, 11)
    ar[2:end-1, 2:end-1, :] .= 0.01*CUDA.rand(tate, yoko, 11)
    return ar 
end

function init_cancer(tate, yoko, p_, maxs)
    (; Nm1_max, Nm2_max, Dm1_max, Dm2_max, Nc1_max, Nc2_max, Dc1_max, Dc2_max, I1_max, I2_max, H_max) = maxs
    (; γN, γD, μ0, μ1, μ5, μ6, β3, β41, β42, ν0, ωN, ωD, canc) = p_

    ar = CUDA.zeros(tate+2, yoko+2, 11)
    Ix = 2:(tate+1)
    Iy = 2:(yoko+1)
    Cx = Int(floor(3+(tate-canc)/2)):Int(floor((tate+canc)/2))
    Cy = Int(floor(3+(yoko-canc)/2)):Int(floor((yoko+canc)/2))

    @. ar[Ix, Iy, 1:2] = ωN*γN / (μ1*(μ5+γN))
    @. ar[Ix, Iy, 5:6] = ωN/(μ5 + γN)
    @. ar[Ix, Iy, 7]   = (1/(μ6 + γD))*(ωD + β41/(β3 + (ν0/μ0)^2))
    @. ar[Ix, Iy, 8]   = (1/(μ6 + γD))*(ωD + β42/(β3 + (ν0/μ0)^2))
    @. ar[Ix, Iy, 11]  = ν0/μ0

    @. ar[Cx, Cy, 1]  = Nm1_max
    @. ar[Cx, Cy, 2]  = Nm2_max
    @. ar[Cx, Cy, 3]  = Dm1_max
    @. ar[Cx, Cy, 4]  = Dm2_max
    @. ar[Cx, Cy, 5]  = Nc1_max
    @. ar[Cx, Cy, 6]  = Nc2_max
    @. ar[Cx, Cy, 7]  = Dc1_max
    @. ar[Cx, Cy, 8]  = Dc2_max
    @. ar[Cx, Cy, 9]  = I1_max
    @. ar[Cx, Cy, 10] = I2_max
    @. ar[Cx, Cy, 11] = H_max
    return ar 
end