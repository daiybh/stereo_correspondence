SET(MODULE "rotate")

SET(SRC Rotate.cpp
		Rotate.h)
		
 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})