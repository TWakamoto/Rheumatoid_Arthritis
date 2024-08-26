using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

function clean_up_GPU()
    CUDA.memory_status()  # optional
    GC.gc(true)
    CUDA.reclaim()
    CUDA.memory_status()
end