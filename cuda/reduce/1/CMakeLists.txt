add_executable(reduce1 main.cu)
target_link_libraries(reduce1 PRIVATE CUDA::cudart)
target_compile_options(reduce1 PRIVATE -fno-omit-frame-pointer)
target_link_options(reduce1 PRIVATE )

add_executable(reduce2 main1.cu)
target_link_libraries(reduce1 PRIVATE CUDA::cudart)
target_compile_options(reduce1 PRIVATE -fno-omit-frame-pointer)
target_link_options(reduce1 PRIVATE )