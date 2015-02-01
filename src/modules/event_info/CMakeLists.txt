# Set name of the module
SET (MODULE event_info)

# Set all source files module uses
SET (SRC register.cpp 
         EventInfo.cpp
		 EventInfo.h
		 EventTimer.cpp
		 EventTimer.h
		 PassEvent.cpp
		 PassEvent.h
		 )

IF(UNIX AND NOT APPLE)
SET(SRC ${SRC}
		 EventDevice.cpp
		 EventDevice.h)
ENDIF()

 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})
