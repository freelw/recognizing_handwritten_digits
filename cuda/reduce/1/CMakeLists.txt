add_executable(reduce1 main.cu)
target_link_libraries(reduce1 PRIVATE CUDA::cudart)