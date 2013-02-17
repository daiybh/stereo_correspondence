# Set name of the module
SET (MODULE avscaler)

# Set all source files module uses
SET (SRC AVScaler.cpp
		 AVScaler.h)



# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME}  ${LIBNAME_HELPER_LIBAV} ${libswscale_LIBRARIES})

YURI_INSTALL_MODULE(${MODULE})