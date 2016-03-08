/*!
 * @file 		IOThread.cpp
 * @author 		Zdenek Travnicek
 * @date 		15.02.2012
 * @date		08.03.2016
 * @copyright	Institute of Intermedia, CTU in Prague, 2012 - 2016
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "Cuda.h"

namespace yuri {
namespace cuda {


namespace detail {

void *cuda_alloc_impl(size_t size)
{
	void *x;
	cudaMalloc((void **) &x, size);
	return x;
}
void cuda_dealloc_impl(void *mem)
{
	cudaFree(mem);
}
}

} /* namespace cuda */
} /* namespace yuri */
