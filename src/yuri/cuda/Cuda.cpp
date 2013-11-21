/*!
 * @file 		IOThread.cpp
 * @author 		Zdenek Travnicek
 * @date 		15.2.2012
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2012 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "Cuda.h"
//include <cuda.h>
#ifdef YURI_HAVE_X11
#include "cuda_gl_interop.h"
#endif

namespace yuri {
namespace cuda {
	void* map_array(cudaArray *array);
}

namespace graphics {

std::string Cuda::uint_fs(
		"uniform usampler2D tex0;\n"
		"void main()\n"
		"{\n"
		"vec4 col = texture2D(tex0, gl_TexCoord[0].st);\n"
		"vec4 col2 = vec4(1.0, (float)col.g, (float)col.b/255, 1.0);\n"
		"gl_FragColor = col2;\n"
		"}\n");
std::string Cuda::uint_vs(
		"#version 130\n"
		"void main()\n"
		"{\n"
		"gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n"
		"gl_TexCoord[0] = gl_MultiTexCoord0;\n"
		"}\n");

Cuda::Cuda(Log &_log):log(_log)
{
	log.set_label("[CUDA] ");
}

Cuda::~Cuda()
{
}
shared_ptr<void> Cuda::cuda_alloc(yuri::size_t size)
{
	return shared_ptr<void> (_cuda_alloc(size), &_cuda_dealloc);
}


void *Cuda::_cuda_alloc(yuri::size_t size)
{
	void *x;
	cudaMalloc((void **) &x, size);
	return x;
}
void Cuda::_cuda_dealloc(void *mem)
{
	cudaFree(mem);
}
bool Cuda::copy_to_gpu(void *dest, const void *src, yuri::size_t size)
{
	cudaMemcpy(reinterpret_cast<char*>(dest),reinterpret_cast<const char*>(src),
				size,cudaMemcpyHostToDevice);
	return true;
}
bool Cuda::copy_to_cpu(void *dest, const void *src, yuri::size_t size)
{
	cudaMemcpy(reinterpret_cast<char*>(dest),reinterpret_cast<const char*>(src),
					size,cudaMemcpyDeviceToHost);
	return true;
}
bool Cuda::lock_mem(void *mem, yuri::size_t size)
{
	cudaError_t err;
	if ((err=cudaHostRegister(mem,size,cudaHostRegisterPortable))==cudaSuccess) {
		return true;
	}
	log[log::error] << "Failed to lock memory: "<< cudaGetErrorString(err) << std::endl;
	return false;
}
bool Cuda::unlock_mem(void *mem, yuri::size_t /*size*/)
{
	cudaError_t err;
	if ((err=cudaHostUnregister(mem))==cudaSuccess) {
		return true;
	}
	log[log::error] << "Failed to unlock memory: "<< cudaGetErrorString(err) << std::endl;
	return false;
}
void *Cuda::map_mem(void *mem, yuri::size_t size)
{
	cudaError_t err;
	if ((err=cudaHostRegister(mem,size,cudaHostRegisterMapped|cudaHostRegisterPortable))!=cudaSuccess) {
		log[log::error] << "Failed to lock memory: "<< cudaGetErrorString(err) << std::endl;
		return 0;
	}
	void *ptr;
	if ((err=cudaHostGetDevicePointer(&ptr,mem,0))!=cudaSuccess) {
		log[log::error] << "Failed to get device pointer: "<< cudaGetErrorString(err) << std::endl;
		unmap_mem(mem);
		return 0;
	}
	return ptr;
}
void Cuda::unmap_mem(void *mem)
{
	unlock_mem(mem,0);
}

void Cuda::sync()
{
	cudaThreadSynchronize();
}
bool Cuda::set_device(yuri::uint_t id)
{
	cudaError_t err;
#ifdef YURI_HAVE_X11
	yuri::lock_t l(GL::big_gpu_lock);
#endif
	if ((err=cudaSetDeviceFlags(cudaDeviceMapHost)) != cudaSuccess) {
		log[log::warning] << "Failed to set flags for device:"<<cudaGetErrorString(err) << std::endl;
		return true;
	}

	if ((err=cudaGLSetGLDevice(id)) == cudaSuccess) {
		log[log::info] << "Successfully set device: " <<  id << std::endl;
		cudaDeviceReset();
		return true;
	}
	log[log::error] << "Failed to set device " << id << ": " << cudaGetErrorString(err) << std::endl;
	return false;
}
#ifdef YURI_HAVE_X11
bool Cuda::prepare_texture(GL &gl, yuri::uint_t tid, yuri::size_t width, yuri::size_t height)
{
	gl.generate_empty_texture(tid,YURI_FMT_RGBA,width,height);
	/*if (!gl.prepare_texture(tid,0,0,width,height,GL_RGBA,GL_RGBA,false))
		return false;
	gl.textures[tid].tx = 1.0f;
	gl.textures[tid].ty = 1.0f;
	gl.textures[tid].keep_aspect=false;
	gl.textures[tid].finish_update(log,YURI_FMT_RGBA,uint_vs, uint_fs);*/
	gl.textures[tid].keep_aspect=false;
	return true;
}
bool Cuda::register_texture(GL &gl, yuri::uint_t tid)
{
	GLuint tex = gl.textures[tid].tid[0];
	assert(tex!=(GLuint)-1);
	//cudaGraphicsResource *res;
	cudaError_t err;
	log[log::info] << "Registering texture " << tex << " (" << tid << ")" << std::endl;
	if ((err=cudaGraphicsGLRegisterImage(&textures[tid].resource, tex,GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard))
			== cudaSuccess) {
		//textures[tid].resource=res;
		return true;
	}
	log[log::error] << "Failed to register gl texture "  <<tex<<": " << cudaGetErrorString(err) << std::endl;
	return false;
}
// Mapping texture to array
bool Cuda::map_texture(GL &/*gl*/, yuri::uint_t tid)
{
//	cudaGraphicsResource *res=textures[tid].resource;
	cudaError_t ret;
	if ((ret=cudaGraphicsMapResources (1, &(textures[tid].resource),0)) != cudaSuccess) {
		log[log::error] << "Failed to map GL texture to Cuda: " << cudaGetErrorString(ret) << std::endl;
		return false;
	}
//	void *ptr;
//	cudaArray *array;
	if ((ret=cudaGraphicsSubResourceGetMappedArray(&textures[tid].array, textures[tid].resource, 0, 0)) != cudaSuccess) {
		log[log::error] << "Failed to get array for GL texture: " << cudaGetErrorString(ret) << std::endl;
		return false;
	}
	//textures[tid].array = array;
	return true;
}
bool Cuda::unmap_texture(GL &/*gl*/, yuri::uint_t tid)
{
	//cudaGraphicsResource *res=textures[tid].resource;
	cudaError_t err;
	textures[tid].array = 0;
	if ((err=cudaGraphicsUnmapResources (1, &textures[tid].resource,0)) != cudaSuccess) {
		log[log::error] << "Failed to map GL texture to Cuda: " << cudaGetErrorString(err) << std::endl;
		return false;
	}
	return true;
}
bool Cuda::copy_to_texture(GL &gl, yuri::uint_t tid, shared_ptr<void> src, yuri::size_t size)
{
	if (!map_texture(gl,tid)) {
		return false;
	}
	cudaError_t err;
	assert(textures[tid].array);
	if ((err=cudaMemcpyToArray(textures[tid].array,0,0,src.get(),size,cudaMemcpyDeviceToDevice)) != cudaSuccess) {
		log[log::error] << "Failed to copy "<< size << "B to GL texture: " << cudaGetErrorString(err) << std::endl;
		unmap_texture(gl,tid);
		return false;
	}
	log[log::info] << "Copied " << size << "B to texture " << tid << std::endl;
	unmap_texture(gl,tid);
	return true;
}
#endif
} /* namespace graphics */
} /* namespace yuri */
