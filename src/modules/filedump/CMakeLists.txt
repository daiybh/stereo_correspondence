# Set name of the module
SET (MODULE filedump)

find_package( Boost 1.46 COMPONENTS regex)
IF (Boost_REGEX_FOUND)
	include_directories( ${Boost_INCLUDE_DIRS})
    link_directories (${Boost_LIBRARY_DIRS})
    add_definitions("-DHAVE_BOOST_REGEX")
ENDIF()

# Set all source files module uses
SET (SRC FileDump.cpp
		 FileDump.h)



# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${Boost_REGEX_LIBRARY})

YURI_INSTALL_MODULE(${MODULE})