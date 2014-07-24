# Set name of the module
SET (MODULE gpujpeg_module)

# Set all source files module uses
SET (SRC GPUJpegBase.cpp
         GPUJpegBase.h
		 GPUJpegDecoder.cpp
		 GPUJpegDecoder.h
		 GPUJpegEncoder.cpp
		 GPUJpegEncoder.h
		 register.cpp)



# This is hack, but necessary (?)
link_directories("/opt/cuda/lib64") 
add_library(${MODULE} MODULE ${SRC})
include_directories(${GPUJPEG_INCLUDE_DIR})



target_link_libraries(${MODULE} ${LIBNAME} ${GPUJPEG_LIBRARIES})

YURI_INSTALL_MODULE(${MODULE})
