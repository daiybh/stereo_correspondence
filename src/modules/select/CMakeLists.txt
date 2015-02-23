# Set name of the module
SET (MODULE select)

# Set all source files module uses
SET (SRC Select.cpp
		 Select.h
		 Unselect.cpp
		 Unselect.h
		 register.cpp)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
