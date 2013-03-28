# Set name of the module
SET (MODULE delta_input)

# Set all source files module uses
SET (SRC DeltaInput.cpp
		 DeltaInput.h)



include_directories(${VIDEOMASTER_INCLUDE_DIR})
 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} videomasterhd)

YURI_INSTALL_MODULE(${MODULE})
