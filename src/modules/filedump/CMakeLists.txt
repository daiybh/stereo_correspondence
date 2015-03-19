# Set name of the module
SET (MODULE filedump)

find_package( Boost 1.46 COMPONENTS regex)
IF (Boost_REGEX_FOUND)
    add_definitions("-DHAVE_BOOST_REGEX")
ENDIF()

# Set all source files module uses
SET (SRC FileDump.cpp
		 FileDump.h)



# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})