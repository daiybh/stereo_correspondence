# Set name of the module
SET (MODULE framebuffer)

# Set all source files module uses
SET (SRC FrameBuffer.cpp
		 FrameBuffer.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
