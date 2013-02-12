/*
 * Cuda.h
 *
 *  Created on: Feb 15, 2012
 *      Author: neneko
 */

#ifndef CUDA_H_
#define CUDA_H_
#include "yuri/log/Log.h"
#include <boost/smart_ptr.hpp>
#include <cuda_runtime.h>
#include "yuri/graphics/GL.h"

using boost::shared_ptr;
using yuri::log::Log;
using std::vector;
using yuri::graphics::GL;
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
	bool prepare_texture(GL &gl, yuri::uint_t tid, yuri::size_t width, yuri::size_t height);
	bool register_texture(GL &gl, uint_t tid);
	bool map_texture(GL &gl, uint_t tid);
	bool unmap_texture(GL &gl, uint_t tid);
	bool copy_to_texture(GL &gl, yuri::uint_t tid, shared_ptr<void>src, yuri::size_t size);
	std::map<yuri::uint_t, _tex_data> textures;

protected:
	Log log;
};

} /* namespace graphics */
} /* namespace yuri */
#endif /* CUDA_H_ */
