# 为 C 和 C++ 编译器设置 AddressSanitizer 选项
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")

add_executable(reduce1 main.cu)
target_link_libraries(reduce1 PRIVATE CUDA::cudart)
target_compile_options(reduce1 PRIVATE -fno-omit-frame-pointer)
target_link_options(reduce1 PRIVATE )

add_executable(reduce2 main1.cu)
target_link_libraries(reduce1 PRIVATE CUDA::cudart)
target_compile_options(reduce1 PRIVATE -fno-omit-frame-pointer)
target_link_options(reduce1 PRIVATE )