#include <cuda.h>


namespace yuri {
namespace cuda {


void* map_array(cudaArray *array)
{
	//texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRef;
	//cudaBindTextureToArray(&texRef,array);
	return 0;
}

}
}