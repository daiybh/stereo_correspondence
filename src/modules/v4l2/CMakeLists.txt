SET (MODULE v4l2_source)
CHECK_INCLUDE_FILES (linux/videodev2.h HAVE_VIDEODEV2_H)

SET (SRC V4l2Source.cpp
		 V4l2Source.h
		 v4l2_controls.cpp
		 v4l2_controls.h)


# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})