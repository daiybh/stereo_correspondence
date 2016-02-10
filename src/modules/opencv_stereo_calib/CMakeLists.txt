# Set name of the module
SET (MODULE opencv_stereo_calib)

# Set all source files module uses
SET (SRC OpenCVStereoCalib.cpp
		 OpenCVStereoCalib.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${OpenCV_LIBRARIES} ${OpenCV_LDFLAGS})

YURI_INSTALL_MODULE(${MODULE})
