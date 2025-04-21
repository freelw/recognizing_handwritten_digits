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

add_executable(opt2 opt2.cu)
target_link_libraries(opt2 PRIVATE CUDA::cudart)
target_compile_options(opt2 PRIVATE -fno-omit-frame-pointer)
target_link_options(opt2 PRIVATE)

add_executable(opt3 opt3.cu)
target_link_libraries(opt3 PRIVATE CUDA::cudart)
target_compile_options(opt3 PRIVATE -fno-omit-frame-pointer)
target_link_options(opt3 PRIVATE)

add_executable(opt4 opt4.cu)
target_link_libraries(opt4 PRIVATE CUDA::cudart)
target_compile_options(opt4 PRIVATE -fno-omit-frame-pointer)
target_link_options(opt4 PRIVATE)

add_executable(opt5 opt5.cu)
target_link_libraries(opt5 PRIVATE CUDA::cudart)
target_compile_options(opt5 PRIVATE -fno-omit-frame-pointer)
target_link_options(opt5 PRIVATE)

add_executable(opt6 opt6.cu)
target_link_libraries(opt6 PRIVATE CUDA::cudart)
target_compile_options(opt6 PRIVATE -fno-omit-frame-pointer)
target_link_options(opt6 PRIVATE)