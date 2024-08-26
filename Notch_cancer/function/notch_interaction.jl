using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())
using ParallelStencil
@init_parallel_stencil(CUDA, Float32, 2)

@parallel_indices (ix, iy) function notch_d!(dDm, Dm, H, H_max)
p = 0.75 #Criterion value for a cell to be a cancer cell
if ix > 1 && iy > 1 && ix < size(dDm,1) && iy < size(dDm,2)
        if H[ix-1,iy] < p * H_max && H[ix+1,iy] < p * H_max && H[ix,iy-1] < p * H_max && H[ix,iy+1] < p * H_max #check the neighboring cell to be cancer or not
            dDm[ix, iy] = 0.0 #if all surrounding cells are healthy, signaling does not take place
        end
    end 
    return nothing
end