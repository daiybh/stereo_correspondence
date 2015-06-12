# Set name of the module
SET (MODULE decklink)

# Set all source files module uses
SET (SRC DeckLinkAPIDispatch.cpp 
	     DeckLinkBase.cpp
	     DeckLinkBase.h
	     DeckLink3DVideoFrame.cpp
	     DeckLink3DVideoFrame.h 
         DeckLinkInput.cpp
		 DeckLinkInput.h
		 DeckLinkOutput.cpp
		 DeckLinkOutput.h
		 DeckLinkAPI_wrapper.h 
		 register.cpp)

add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/DeckLinkAPIDispatch.cpp"
	    	COMMAND ${CMAKE_COMMAND} -E copy_if_different
	    	"${DECKLINK_INCLUDE_DIR}/DeckLinkAPIDispatch.cpp"
	    	"${CMAKE_CURRENT_BINARY_DIR}/DeckLinkAPIDispatch.cpp"
	    )
IF (UNIX OR MINGW OR CYGWIN)
    SET_SOURCE_FILES_PROPERTIES(DeckLinkAPIDispatch.cpp PROPERTIES COMPILE_FLAGS "-w")
ENDIF()	    
# You shouldn't need to edit anything below this line
include_directories(${DECKLINK_INCLUDE_DIRS}) 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${DECKLINK_LIBRARIES})

YURI_INSTALL_MODULE(${MODULE})