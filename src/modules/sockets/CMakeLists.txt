# Set name of the module
SET (MODULE yuri_net)

# Set all source files module uses
SET (SRC YuriNetSocket.cpp
		 YuriNetSocket.h
		 YuriDatagram.cpp
		 YuriDatagram.h 
		 YuriStreamSocket.cpp
		 YuriStreamSocket.h
		 YuriUdp.cpp
		 YuriUdp.h
		 YuriTcp.cpp
		 YuriTcp.h
		 UnixDatagramSocket.cpp
		 UnixDatagramSocket.h
		 UnixStreamSocket.cpp
		 UnixStreamSocket.h
		 register.cpp
		 )


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
