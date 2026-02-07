# MinHook - Minimal x86/x64 API Hooking Library
# Included for AlphaFixes project

cmake_minimum_required(VERSION 3.10)
project(minhook VERSION 1.0.0)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

add_library(minhook STATIC
    source/buffer.c
    source/hook.c
    source/trampoline.c
)

target_include_directories(minhook PUBLIC include)
