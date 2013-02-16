/*!
 * @file 		Cuda.h
 * @author 		Zdenek Travnicek
 * @date 		15.2.2012
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2012 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef CUDA_H_
#define CUDA_H_
#include "yuri/log/Log.h"
#include <boost/smart_ptr.hpp>
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



#ifdef YURI_HAVE_X11
#include "yuri/graphics/GL.h"
#endif

using yuri::log::Log;

#ifdef YURI_HAVE_X11
using yuri::graphics::GL;
#endif
namespace yuri {
namespace graphics {

class Cuda {
public:
	struct _tex_data {
		cudaGraphicsResource* resource;
		cudaArray *array;
	};
	Cuda(Log &log_);
	virtual ~Cuda();
	static shared_ptr<void> cuda_alloc(yuri::size_t size);
	static void *_cuda_alloc(yuri::size_t size);
	static void _cuda_dealloc(void *);
	static bool copy_to_gpu(void *dest, const void *src, yuri::size_t size);
	static bool copy_to_cpu(void *dest, const void *src, yuri::size_t size);
	bool lock_mem(void *mem, yuri::size_t size);
	bool unlock_mem(void *mem, yuri::size_t size);
	void *map_mem(void *mem, yuri::size_t size);
	void unmap_mem(void *mem);
	static void sync();
	static std::string uint_fs, uint_vs;
	bool set_device(yuri::uint_t id);
#ifdef YURI_HAVE_X11
	bool prepare_texture(GL &gl, yuri::uint_t tid, yuri::size_t width, yuri::size_t height);
	bool register_texture(GL &gl, uint_t tid);
	bool map_texture(GL &gl, uint_t tid);
	bool unmap_texture(GL &gl, uint_t tid);
	bool copy_to_texture(GL &gl, yuri::uint_t tid, shared_ptr<void>src, yuri::size_t size);
	std::map<yuri::uint_t, _tex_data> textures;
#endif
protected:
	Log log;
};

} /* namespace graphics */
} /* namespace yuri */
#endif /* CUDA_H_ */
