add_executable(reduce1 main.cu)
target_link_libraries(reduce1 PRIVATE CUDA::cudart)
target_compile_options(reduce1 PRIVATE -fno-omit-frame-pointer)
target_link_options(reduce1 PRIVATE)

add_executable(reduce2 main1.cu)
target_link_libraries(reduce2 PRIVATE CUDA::cudart)
target_compile_options(reduce2 PRIVATE -fno-omit-frame-pointer)
target_link_options(reduce2 PRIVATE)

add_executable(test test.cu)
target_link_libraries(test PRIVATE CUDA::cudart)
target_compile_options(test PRIVATE -fno-omit-frame-pointer)
target_link_options(test PRIVATE)

add_executable(baseline baseline.cu)
target_link_libraries(baseline PRIVATE CUDA::cudart)
target_compile_options(baseline PRIVATE -fno-omit-frame-pointer)
target_link_options(baseline PRIVATE)


add_executable(opt1 opt1.cu)
target_link_libraries(opt1 PRIVATE CUDA::cudart)
target_compile_options(opt1 PRIVATE -fno-omit-frame-pointer)
target_link_options(opt1 PRIVATE)