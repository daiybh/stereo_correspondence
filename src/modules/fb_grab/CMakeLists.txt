# Set name of the module
SET (MODULE fb_grab)

# Set all source files module uses
SET (SRC FrameBufferGrabber.cpp
		 FrameBufferGrabber.h)



# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${LIBNAME_HELPER_X11})

YURI_INSTALL_MODULE(${MODULE})