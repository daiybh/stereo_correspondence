/*!
 * @file 		FrameBuffer.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		05.02.2016
 * @copyright	Institute of Intermedia, CTU in Prague, 2016
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef FRAMEBUFFER_H_
#define FRAMEBUFFER_H_

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include <linux/fb.h>
#include <sys/mman.h>

namespace yuri {
namespace framebuffer {

template<typename T>
class mmap_handle_t {
public:
	mmap_handle_t():ptr_(nullptr),len_(0) {}
	mmap_handle_t(void *addr, size_t len, int prot, int flags, int fd, off_t offset):
		ptr_(nullptr), len_(len) {
		ptr_ = reinterpret_cast<T*>(mmap(addr, len_, prot, flags, fd, offset));
	}
	mmap_handle_t(const mmap_handle_t&) = delete;
	mmap_handle_t(mmap_handle_t&& rhs) {
		using std::swap;		
		swap(ptr_, rhs.ptr_);
		swap(len_, rhs.len_);
	}
	mmap_handle_t& operator=(const mmap_handle_t&) = delete;
	mmap_handle_t& operator=(mmap_handle_t&& rhs) {
		using std::swap;		
		swap(ptr_, rhs.ptr_);
		swap(len_, rhs.len_);
		return *this;
	}
	~mmap_handle_t() noexcept {
		if (ptr_) {
			munmap(ptr_, len_);
		}
	}
	
	T* get() { return ptr_; }
//	operator T*() { return get();}
	operator bool() const { return ptr_!=nullptr;}
	size_t size() const { return len_; }
private:
	T* ptr_;
	size_t len_;
};


class FrameBuffer: public core::SpecializedIOFilter<core::RawVideoFrame>
{
	using base_type = core::SpecializedIOFilter<core::RawVideoFrame>;
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	FrameBuffer(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~FrameBuffer() noexcept;
private:
	virtual core::pFrame do_special_single_step(core::pRawVideoFrame frame) override;
	virtual bool set_param(const core::Parameter& param) override;

	int handle_;
	format_t format_;
	resolution_t resolution_;
	mmap_handle_t<uint8_t> memory_;
//	uint8_t* memory_;
	bool flip_;
	bool clear_;
	fb_fix_screeninfo info_;
	fb_var_screeninfo vinfo_;
};

} /* namespace framebuffer */
} /* namespace yuri */
#endif /* FRAMEBUFFER_H_ */
