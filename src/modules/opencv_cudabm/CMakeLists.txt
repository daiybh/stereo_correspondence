# Set name of the module
SET (MODULE opencv_cudabm)

# Set all source files module uses
SET (SRC OpenCVCudaBM.cpp
		 OpenCVCudaBM.h)


 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${OpenCV_LIBRARIES} ${OpenCV_LDFLAGS})

YURI_INSTALL_MODULE(${MODULE})
