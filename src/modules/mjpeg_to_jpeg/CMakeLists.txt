# Set name of the module
SET (MODULE mjpeg_to_jpeg)

# Set all source files module uses
SET (SRC MjpegToJpeg.cpp
		 MjpegToJpeg.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
