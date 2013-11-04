# Set name of the module
SET (MODULE opencv)

# Set all source files module uses
SET (SRC OpenCVConvert.cpp
		 OpenCVConvert.h
		 OpenCVFaceDetect.cpp
		 OpenCVFaceDetect.h
		 register.cpp)



# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME} ${OpenCV_LIBRARIES} ${OpenCV_LDFLAGS})

YURI_INSTALL_MODULE(${MODULE})
