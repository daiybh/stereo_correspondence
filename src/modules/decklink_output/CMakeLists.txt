# Set name of the module
SET (MODULE decklink_output)

# Set all source files module uses
SET (SRC DeckLinkOutput.cpp
		 DeckLinkOutput.h)



# You shouldn't need to edit anything below this line
include_directories(${DECKLINK_INCLUDE_DIRS}) 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME}  ${LIBNAME_HELPER_DECKLINK})

YURI_INSTALL_MODULE(${MODULE})