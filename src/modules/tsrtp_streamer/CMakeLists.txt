# Set name of the module
SET (MODULE tsrtp_streamer)

# Set all source files module uses
SET (SRC TSRtpStreamer.cpp
		 TSRtpStreamer.h)



# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(${MODULE} ${LIBNAME} ${LIBNAME_HELPER_ASIO})

YURI_INSTALL_MODULE(${MODULE})