# Set name of the module
SET (MODULE irc_client)

# Set all source files module uses
SET (SRC IrcClient.cpp
		 IrcClient.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${LIBNAME_HELPER_ASIO})

YURI_INSTALL_MODULE(${MODULE})
