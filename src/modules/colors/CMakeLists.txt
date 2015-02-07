# Set name of the module
SET (MODULE colors)

# Set all source files module uses
SET (SRC Saturate.cpp
		 Saturate.h
		 Contrast.h
		 Contrast.cpp
		 manipulate_colors.h
		 register.cpp)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
