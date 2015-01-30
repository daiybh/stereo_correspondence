# Set name of the module
SET (MODULE gphoto)

# Set all source files module uses
SET (SRC GPhoto.cpp
		 GPhoto.h)


 include_directories(${GPHOTO2_INCLUDE_DIRS})
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${GPHOTO2_LIBRARIES})

YURI_INSTALL_MODULE(${MODULE})
