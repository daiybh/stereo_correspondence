# Set name of the module
SET (MODULE event_device)

# Set all source files module uses
SET (SRC EventDevice.cpp
		 EventDevice.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
