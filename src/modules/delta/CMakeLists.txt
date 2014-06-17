# Set name of the module
SET (MODULE delta_module)

# Set all source files module uses
SET (SRC DeltaInput.cpp
		 DeltaInput.h)



include_directories(${VIDEOMASTER_INCLUDE_DIRS})
 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${VIDEOMASTER_LIBRARIES})

YURI_INSTALL_MODULE(${MODULE})
