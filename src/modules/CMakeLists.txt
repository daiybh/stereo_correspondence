#################################################################
# Add modules that are compile unconditionally
#################################################################
add_subdirectory(anaglyph)
add_subdirectory(blank)
add_subdirectory(crop)
add_subdirectory(dummy)
add_subdirectory(dup)
add_subdirectory(filedump)
add_subdirectory(flip)
add_subdirectory(null)
add_subdirectory(rawfilesource)
add_subdirectory(split)

#################################################################
# Let's check for libraries required 
# for optional modules, respecting YURI_DISABLE_*
#################################################################
find_package(JPEG)
find_package(PNG)
find_package(CURL)
IF(NOT YURI_DISABLE_SAGE)
	find_package(SAGE)
ENDIF()
IF(NOT YURI_DISABLE_CUDA)
	find_package(CUDA)
ENDIF()
CHECK_INCLUDE_FILES (linux/videodev2.h HAVE_VIDEODEV2_H)
IF(NOT YURI_DISABLE_LIBAV)
	pkg_check_modules(libavformat libavformat) 
	pkg_check_modules(libavutil libavutil)
	pkg_check_modules(libswscale libswscale)
ENDIF()

#################################################################
# Add optional modules based on results of the detection
#################################################################
IF(JPEG_FOUND)
	add_subdirectory(jpegdecoder)
	add_subdirectory(jpegencoder)
ENDIF()

IF(HAVE_VIDEODEV2_H)
	add_subdirectory(v4l2)
ENDIF()

IF(PNG_FOUND)
	add_subdirectory(pngencoder)
	add_subdirectory(pngdecoder)
ENDIF()

IF(CURL_FOUND)
	add_subdirectory(fetcher)
ENDIF()

IF(CORE_LIBAV)
	IF (${libavutil_FOUND})
		add_subdirectory(avencoder)
	ENDIF()
	IF (${libavformat_FOUND})
		add_subdirectory(tsmuxer)
	ENDIF()
	IF (${libswscale_FOUND})
		add_subdirectory(avscaler)
		add_subdirectory(avdecoder)
	ENDIF()
ENDIF()

IF(${CORE_CUDA})
	add_definitions("-DYURI_HAVE_CUDA")
ENDIF()
add_subdirectory(yuriconvert)

IF(${CORE_X11})
	add_subdirectory(simplerenderer)
ENDIF()
IF(${SAGE_FOUND})
	add_subdirectory(sage)
ENDIF()