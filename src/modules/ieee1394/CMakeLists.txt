# Set name of the module
SET (MODULE ieee1394)

# Set all source files module uses
SET (SRC IEEE1394SourceBase.h
         IEEE1394SourceBase.cpp
         DVSource.cpp
		 DVSource.h
		 HDVSource.cpp
		 HDVSource.h
		 register.cpp)

# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
include_directories(${ieee1394_INCLUDE_DIR}  ${ieee1394_INCLUDE_DIRS})
target_link_libraries(${MODULE} ${LIBNAME} ${ieee1394_LIBRARIES})
link_directories( ${ieee1394_LIBRARY_DIRS})
	
	 

YURI_INSTALL_MODULE(${MODULE})
