# Set name of the module
SET (MODULE dvsource)

# Set all source files module uses
SET (SRC DVSource.cpp
		 DVSource.h)

# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
include_directories(${ieee1394_INCLUDE_DIR})
target_link_libraries(${MODULE} ${LIBNAME} ${LIBNAME_HELPER_IEEE1394})

YURI_INSTALL_MODULE(${MODULE})
