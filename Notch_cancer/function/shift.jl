using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())
using ParallelStencil
@init_parallel_stencil(CUDA, Float32, 2)

function shift!(r, sol, tate, yoko)
    for j in 3:yoko
        for i in 3:tate
            #shift cells upwards
            if 0.2 < r[i-1, j-1] <= 0.4
                sol[2:i ,j, :] = circshift(sol[2:i ,j, :], -1)
                sol[i, j, :] = sol[i-1, j, :]
            end
            #shift cells downwards
            if 0.4 < r[i-1, j-1] <= 0.6
                sol[i:end-1 ,j, :] = circshift(sol[i:end-1 ,j, :], 1)
                r[i-1:end ,j-1] = circshift(r[i-1:end ,j-1], 1)
                sol[i, j, :] = sol[i+1, j, :]
                if i < tate+1
                    r[i, j-1] = 0
                end
            end
            #shift cells to the right
            if 0.6 < r[i-1, j-1] <= 0.8
                sol[i, j:end-1, :] = circshift(sol[i, j:end-1, :], 1)
                r[i-1 ,j-1:end] = circshift(r[i-1 ,j-1:end], 1)
                sol[i, j, :] = sol[i, j+1, :]
                if j < yoko+1
                    r[i-1, j] = 0
                end
            end
            #shift cells to the left
            if 0.8 < r[i-1, j-1] <= 1.0
                sol[i, 2:j, :] = circshift(sol[i, 2:j, :], -1)
                sol[i, j, :] = sol[i, j-1, :]
            end

        end 
        
    end
    return sol
end