SET (MODULE v4l2)
CHECK_INCLUDE_FILES (linux/videodev2.h HAVE_VIDEODEV2_H)

SET (SRC V4l2Source.cpp
		 V4l2Source.h)


# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})