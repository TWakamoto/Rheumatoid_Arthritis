using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())
using Random

@parallel_indices (ix, iy) function division_random!(r, H, h_max)
    if H[ix, iy] >= 0.75*h_max
        r[ix, iy] = rand()
    end
    return nothing
end