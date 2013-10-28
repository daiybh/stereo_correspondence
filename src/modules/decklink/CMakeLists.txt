# Set name of the module
SET (MODULE decklink)

# Set all source files module uses
SET (SRC DeckLinkInput.cpp
		 DeckLinkInput.h
		 DeckLinkOutput.cpp
		 DeckLinkOutput.h 
		 register.cpp)



# You shouldn't need to edit anything below this line
include_directories(${DECKLINK_INCLUDE_DIRS}) 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME}  ${LIBNAME_HELPER_DECKLINK})

YURI_INSTALL_MODULE(${MODULE})