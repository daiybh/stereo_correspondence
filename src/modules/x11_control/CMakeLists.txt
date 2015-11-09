# Set name of the module
SET (MODULE x11_control)

# Set all source files module uses
SET (SRC X11Control.cpp
		 X11Control.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${XCB_XTEST_LIBRARIES})

YURI_INSTALL_MODULE(${MODULE})
