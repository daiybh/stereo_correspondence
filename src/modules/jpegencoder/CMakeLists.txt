# Set name of the module
SET (MODULE jpegencoder)

# Set all source files module uses
SET (SRC JPEGEncoder.cpp
		 JPEGEncoder.h)



# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
include_directories(${JPEG_INCLUDE_DIR})
target_link_libraries(${MODULE} ${LIBNAME} ${JPEG_LIBRARIES})

YURI_INSTALL_MODULE(${MODULE})