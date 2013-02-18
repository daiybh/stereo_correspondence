# Set name of the module
SET (MODULE dxt_compress)

# Set all source files module uses
SET (SRC DXTCompress.cpp
		 DXTCompress.h)

SET (SRC ${SRC} alpha.h clusterfit.h colourblock.h colourfit.h colourset.h 
				config.h maths.h rangefit.h simd_float.h simd.h simd_sse.h 
				simd_ve.h singlecolourfit.h squish.h 
				alpha.cpp clusterfit.cpp colourblock.cpp colourfit.cpp 
				colourset.cpp maths.cpp rangefit.cpp singlecolourfit.cpp 
				squish.cpp)

# You shouldn't need to edit anything below this line 
add_library(${MODULE} MODULE ${SRC})
target_link_libraries(${MODULE} ${LIBNAME})

YURI_INSTALL_MODULE(${MODULE})

