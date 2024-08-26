using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())
using OrdinaryDiffEq

@parallel_indices (ix, iy) function notch!(dDm, Dm, H, H_max)
if ix > 1 && iy > 1 && ix < size(dDm,1) && iy < size(dDm,2)
        if H[ix-1,iy] < 0.1 * H_max && H[ix+1,iy] < 0.1 * H_max && H[ix,iy-1] < 0.1 * H_max && H[ix,iy+1] < 0.1 * H_max && Dm[ix, iy]==0
            dDm[ix, iy] = 0.0
        end
    end 
    return nothing
end

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

        @. du[Ix, Iy, 1] = (1 + pα)*(-α1 * d1_ave * Nm1[Ix, Iy] - α2 * d2_ave * Nm1[Ix, Iy]) - μ1 * Nm1[Ix, Iy] + (1 + pγ)*γN * Nc1[Ix, Iy]
        @. du[Ix, Iy, 2] = (1 + pα)*(-α5 * d1_ave * Nm2[Ix, Iy] - α6 * d2_ave * Nm2[Ix, Iy]) - μ1 * Nm2[Ix, Iy] + (1 + pγ)*γN * Nc2[Ix, Iy]
        @. du[Ix, Iy, 3] = (1 + pα)*(-α1 * n1_ave * Dm1[Ix, Iy] - α5 * n2_ave * Dm1[Ix, Iy]) - μ2 * Dm1[Ix, Iy] + γD * Dc1[Ix, Iy]
        @. du[Ix, Iy, 4] = (1 + pα)*(-α2 * n1_ave * Dm2[Ix, Iy] - α6 * n2_ave * Dm2[Ix, Iy]) - μ2 * Dm2[Ix, Iy] + γD * Dc2[Ix, Iy]
        @. du[Ix, Iy, 5] = ωN + (1 + pβ)β21 * ( I1[Ix, Iy]^2 / (β1 + I1[Ix, Iy]^2) ) - (μ5 + (1 + pγ)*γN) * Nc1[Ix, Iy]
        @. du[Ix, Iy, 6] = ωN + (1 + pβ)β22 * ( I2[Ix, Iy]^2 / (β1 + I2[Ix, Iy]^2) ) - (μ5 + (1 + pγ)*γN) * Nc2[Ix, Iy]
        @. du[Ix, Iy, 7] = ωD + β41 / (β3 + H[Ix, Iy]^2) - (μ6 + γD) * Dc1[Ix, Iy]
        @. du[Ix, Iy, 8] = ωD + β42 / (β3 + H[Ix, Iy]^2) - (μ6 + γD) * Dc2[Ix, Iy]
        @. du[Ix, Iy, 9] = (1 + pα)*(α1 * d1_ave * Nm1[Ix, Iy] + α2 * d2_ave * Nm1[Ix, Iy]) - μ4 * I1[Ix, Iy]
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