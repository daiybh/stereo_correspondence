# Set name of the module
SET (MODULE osc_module)

# Set all source files module uses
SET (SRC OSCReceiver.cpp
		 OSCReceiver.h
		 OSCSender.cpp
		 OSCSender.h
		 OSC.h
		 register.cpp)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
