# Set name of the module
SET (MODULE synchronization)
 
# Set all source files module uses
SET (SRC 
    DelayEstimation.cpp
    DelayEstimation.h
    OnepcProtocolCohort.cpp
    OnepcProtocolCohort.h
    OnepcProtocolCoordinator.cpp
    OnepcProtocolCoordinator.h
    TimestampObserver.cpp
	TimestampObserver.h
    	
    register.cpp)
 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})
 
YURI_INSTALL_MODULE(${MODULE})

