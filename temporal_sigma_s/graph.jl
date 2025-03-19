using Random
using Plots
using ColorSchemes

Lx = 60 #number of cells along x_axis

#----------hill function----------
σ = Array(collect(0:1:Lx+1))
f = 0.1 .+ 0.9*(σ.^2)./(2.0^2 .+ σ.^2)
fig1 = plot(σ, f, ylims=(0, 1))
savefig(fig1, "sigmaS_notch.png")


#--------linear (one-oder) graph--------
λ1 = 1.0
λ2 = 0.5
λ3 = 0.1
y1 = -λ1*σ/(Lx+1) .+ λ1
y2 = -λ2*σ/(Lx+1) .+ λ2
y3 = -λ3*σ/(Lx+1) .+ λ3

fig2 = plot(σ, y1, xlims=(0, Lx+1), ylims=(0, 1))
plot!(σ, y2)
plot!(σ, y3)
savefig(fig2, "linear_graph.png")

#----------hill function----------
β1 = 30.0
β2 = 20.0
β3 = 10.0
ya = 1 .-(σ.^2)./(β1^2 .+ σ.^2)
yb = 1 .-(σ.^2)./(β2^2 .+ σ.^2)
yc = 1 .-(σ.^2)./(β3^2 .+ σ.^2)

fig3 = plot(σ, ya, ylims=(0, 1))
plot!(σ, yb)
plot!(σ, yc)
savefig(fig3, "nonlinear_graph_b.png")


k1 = 4.0
k2 = 2.0
k3 = 1.0
ya = 1 .-(σ.^k1)./(β2^k1 .+ σ.^k1)
yb = 1 .-(σ.^k2)./(β2^k2 .+ σ.^k2)
yc = 1 .-(σ.^k3)./(β2^k3 .+ σ.^k3)

fig4 = plot(σ, ya, ylims=(0, 1))
plot!(σ, yb)
plot!(σ, yc)
savefig(fig4, "nonlinear_graph_k.png")

#-----------tada no sen-----------
d1 = 1.0
d2 = 0.5
d3 = 0.1
y4 = zeros(62)
y5 = zeros(62)
y6 = zeros(62)

y4 .= d1
y5 .= d2
y6 .= d3

fig5 = plot(σ, y4, ylims=(0, 1))
plot!(σ, y5)
plot!(σ, y6)
savefig(fig4, "const.png")