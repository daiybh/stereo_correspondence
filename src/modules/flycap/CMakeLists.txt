# Set name of the module
SET (MODULE flycap)

# Set all source files module uses
SET (SRC FlyCap.cpp
		 FlyCap.h
		 flycap_c_helpers.h
		 flycap_defs.h
		 flycap_enumerate.cpp
		 register.cpp)


include_directories(${FLYCAP_INCLUDE_DIR})
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${FLYCAP_LIBRARY})

YURI_INSTALL_MODULE(${MODULE})
