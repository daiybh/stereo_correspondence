# Set name of the module
SET (MODULE openni)

# Set all source files module uses
SET (SRC OpenNIKinect.cpp
		 OpenNIKinect.h
		 openni_wrapper.h)

#SET(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-rpath ./")

add_library(${MODULE} MODULE ${SRC})
include_directories(${OpenNI_INCLUDE_DIRS})
link_directories(${OpenNI_LIBRARY_DIRECTORIES})
#add_definitions(${PNG_DEFINITIONS})
target_link_libraries(${MODULE} ${LIBNAME} ${OpenNI_LIBRARIES})

YURI_INSTALL_MODULE(${MODULE})