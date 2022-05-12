cmake_minimum_required(VERSION 3.18...3.23)
cmake_policy(VERSION 3.18)

project(sdf_renderer LANGUAGES CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_executable(sdf_renderer main.cu)
target_compile_options(sdf_renderer PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -restrict -extra-device-vectorization -dw>)
