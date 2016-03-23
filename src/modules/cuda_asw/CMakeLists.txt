# Set name of the module
SET (MODULE cuda_asw)

# Set all source files module uses
SET (SRC CudaASW.cpp
		 CudaASW.h
            asw.cu)


 SET(CUDA_PROPAGATE_HOST_FLAGS OFF) 
list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-O3")
CUDA_INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/src )
cuda_add_module(${MODULE} ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${LIBNAME_HELPER_CUDA})

YURI_INSTALL_MODULE(${MODULE})
