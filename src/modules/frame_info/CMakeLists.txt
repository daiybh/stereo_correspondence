# Set name of the module
SET (MODULE frame_info)

# Set all source files module uses
SET (SRC FrameInfo.cpp
		 FrameInfo.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
