# Set name of the module
SET (MODULE webserver)

# Set all source files module uses
SET (SRC WebServer.cpp
		 WebServer.h
		 WebPageGenerator.cpp
		 WebPageGenerator.h
		 WebResource.cpp
		 WebResource.h
		 WebStaticResource.cpp
		 WebStaticResource.h
		 register.cpp
		 )


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${Boost_REGEX_LIBRARY})

YURI_INSTALL_MODULE(${MODULE})
