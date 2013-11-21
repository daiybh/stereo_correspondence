/*!
 * @file 		CudaHelpers.cu
 * @author 		Zdenek Travnicek
 * @date 		15.2.2012
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2012 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */
 
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