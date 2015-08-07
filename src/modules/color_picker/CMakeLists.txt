# Set name of the module
SET (MODULE color_picker)

# Set all source files module uses
SET (SRC ColorPicker.cpp
		 ColorPicker.h
		 color_kernels.impl)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
