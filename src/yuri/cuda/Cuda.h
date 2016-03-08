/*!
 * @file 		Cuda.h
 * @author 		Zdenek Travnicek
 * @date 		15.2.2012
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2012 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef CUDA_H_
#define CUDA_H_
#include "yuri/log/Log.h"
// cuda_runtime has some warnings about usage of long long, let's disable them for the moment (before upstream fixes them)
#if defined __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++11-long-long"
#elif defined __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wlong-long"
#endif
#include <cuda_runtime.h>
#if defined __clang__
#pragma clang diagnostic pop
#elif defined __GNUC__
#pragma GCC  diagnostic pop
#endif

#include <memory>
#include <cstdint>
#include <vector>
#include "yuri/core/utils/uvector.h"
namespace yuri {
namespace cuda {


namespace detail {
void *cuda_alloc_impl(size_t size);
void cuda_dealloc_impl(void *mem);
}

template<typename T>
std::shared_ptr<T> cuda_alloc(size_t size)
{
	using ptr_type = typename std::add_pointer<T>::type;
	return {	reinterpret_cast<ptr_type>(detail::cuda_alloc_impl(size * sizeof(T))),
				detail::cuda_dealloc_impl};
}



template<typename T>
bool copy_to_gpu(std::shared_ptr<T>& dest, const int *src, yuri::size_t size)
{
	return cudaMemcpy(dest.get(), src, size * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
}
template<typename T>
bool copy_to_cpu(void *dest, const std::shared_ptr<T>& src, yuri::size_t size)
{
	return cudaMemcpy(dest, src.get(), size * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
}
template<typename T>
bool copy_to_gpu(std::shared_ptr<T>& dest, const std::vector<T>& src)
{
	return copy_to_gpu(dest, &src[0], src.size());
}
template<typename T>
bool copy_to_cpu(std::vector<T>& dest, const std::shared_ptr<T>& src)
{
	return copy_to_cpu(&dest[0], src, dest.size());
}
template<typename T>
bool copy_to_gpu(std::shared_ptr<T>& dest, const uvector<T>& src)
{
	return copy_to_gpu(dest, &src[0], src.size());
}
template<typename T>
bool copy_to_cpu(uvector<T>& dest, const std::shared_ptr<T>& src)
{
	return copy_to_cpu(&dest[0], src, dest.size());
}

} /* namespace graphics */
} /* namespace yuri */
#endif /* CUDA_H_ */
