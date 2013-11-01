# Set name of the module
SET (MODULE jpeg_module)

# Set all source files module uses
SET (SRC jpeg_common.cpp
		 jpeg_common.h 
		 JpegDecoder.cpp
		 JpegDecoder.h
		 JpegEncoder.cpp
		 JpegEncoder.h
		 register.cpp)


 
add_library(${MODULE} MODULE ${SRC})
include_directories(${JPEG_INCLUDE_DIR})
target_link_libraries(${MODULE} ${LIBNAME} ${JPEG_LIBRARIES})

YURI_INSTALL_MODULE(${MODULE})
