# Set name of the module
SET (MODULE cuda_asw)

# Set all source files module uses
SET (SRC CudaASW.cpp
		 CudaASW.h)



# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
