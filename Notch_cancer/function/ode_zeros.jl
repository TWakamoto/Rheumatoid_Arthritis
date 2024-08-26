using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())
using ParallelStencil
@init_parallel_stencil(CUDA, Float32, 2)

@parallel_indices (ix, iy) function cancer!(du, H, h_max)
p = 0.75 #Criterion value for a cell to be a cancer cell
if ix > 1 && iy > 1 && ix < size(H,1) && iy < size(H,2)
        if H[ix,iy] > p * h_max #check a cell to be cancer or not
            @. du[ix, iy, :] = 0.0 #if a cell is cancer, its dynamics does not vary
        end
    end 
    return nothing
end