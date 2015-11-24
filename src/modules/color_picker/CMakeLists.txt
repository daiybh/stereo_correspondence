# Set name of the module
SET (MODULE color_picker)

# Set all source files module uses
SET (SRC color_kernels.impl
		 common.h
		 common.cpp 
		 ColorPicker.cpp
		 ColorPicker.h
		 )


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
