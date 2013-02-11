add_subdirectory(blank)
add_subdirectory(crop)
add_subdirectory(dummy)
add_subdirectory(dup)
add_subdirectory(filedump)
add_subdirectory(flip)
add_subdirectory(null)
add_subdirectory(rawfilesource)
add_subdirectory(split)


find_package(JPEG)
IF(JPEG_FOUND)
	add_subdirectory(jpegdecoder)
	add_subdirectory(jpegencoder)
ENDIF()


CHECK_INCLUDE_FILES (linux/videodev2.h HAVE_VIDEODEV2_H)
IF(HAVE_VIDEODEV2_H)
	add_subdirectory(v4l2)
ENDIF()
