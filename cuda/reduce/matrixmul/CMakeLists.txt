add_executable(matrix_baseline baseline.cu)
target_link_libraries(matrix_baseline PRIVATE CUDA::cudart)
target_compile_options(matrix_baseline PRIVATE -fno-omit-frame-pointer)
target_link_options(matrix_baseline PRIVATE)