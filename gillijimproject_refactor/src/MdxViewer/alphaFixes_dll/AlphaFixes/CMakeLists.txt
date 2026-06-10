cmake_minimum_required(VERSION 3.10)
project(AlphaFixes VERSION 1.0.0)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/dlls)

add_definitions(-D_WIN32_WINNT=0x0501)
add_definitions(-DNOMINMAX)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

# Subdirectories
add_subdirectory(lib/minhook)

# Main DLL
add_library(alphafixes SHARED
    dll/alpha_main.c
    dll/alpha_tsc.c
    dll/alpha_snapshot.c
    dll/alpha_cpufreq.c
    dll/alpha_helpers.c
)

target_include_directories(alphafixes PRIVATE
    dll/
    lib/minhook/include
)

target_link_libraries(alphafixes minhook)
