/*!
 * @file 		FrameBuffer.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		05.02.2016
 * @copyright	Institute of Intermedia, CTU in Prague, 2016
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "FrameBuffer.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/utils/irange.h"
#include "yuri/core/utils/frame_info.h"
#include "yuri/core/utils/environment.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unordered_map>
namespace yuri {
namespace framebuffer {


IOTHREAD_GENERATOR(FrameBuffer)

MODULE_REGISTRATION_BEGIN("framebuffer")
		REGISTER_IOTHREAD("framebuffer",FrameBuffer)
MODULE_REGISTRATION_END()

core::Parameters FrameBuffer::configure()
{
	core::Parameters p = base_type::configure();
	p.set_description("FrameBuffer");
	p["flip"]["Flip picture upside down"]=false;
	p["clear"]["Clear framebuffer on start"]=true;
	p["device"]["Path to framebuffer device. Empty value uses either env. variable FRAMEBUFFER or /dev/fb0"]="";

	return p;
}
namespace {

constexpr uint64_t mask_offset(uint64_t num, int mask, int offset)
{
	return (num & (0xffffffffffffffff >> mask)) << offset;
}
// Return proper mask, if num !=0 ,zero otherwise
// Zero width channels should return the same value independently on their position
constexpr uint64_t mask_offset_pair(uint64_t num, int mask, int offset, uint64_t num2, int mask2, int offset2)
{
	return 	num?mask_offset(num, mask, offset) |
				mask_offset(num2, mask2, offset2):
				0;
}

constexpr uint64_t pack_fmt(uint8_t bpp, uint8_t red_w, uint8_t red_o, uint8_t green_w, uint8_t green_o, uint8_t blue_w, uint8_t blue_o, uint8_t alpha_w, uint8_t alpha_o)
{
	return	mask_offset(bpp, 	6, 48) |
		mask_offset_pair(red_w, 	6, 42,
						red_o, 	6, 36) |
		mask_offset_pair(green_w, 	6, 30,
						green_o, 	6, 24) |
		mask_offset_pair(blue_w, 	6, 18,
						blue_o, 	6, 12) |
		mask_offset_pair(alpha_w, 	6,  6,
						alpha_o, 	6,  0);

}


std::unordered_map<uint64_t, format_t> fb_formats =  {
	{ pack_fmt(16, 5, 11, 6, 5, 5, 0, 0, 16), core::raw_format::rgb16},
	{ pack_fmt(16, 5, 0, 6, 5, 5, 11, 0, 16), core::raw_format::bgr16},
	{ pack_fmt(24, 8, 16, 8, 8, 8, 0, 0, 24), core::raw_format::rgb24},
	};

}
FrameBuffer::FrameBuffer(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("framebuffer")),
flip_{false}
{
	IOTHREAD_INIT(parameters)
	if (device_.empty()) {
		device_ = core::utils::get_environment_variable("FRAMEBUFFER", "/dev/fb0");
	}
	log[log::info] << "Using framebuffer device " << device_;
	handle_ = open(device_.c_str(),O_RDWR);
	if (ioctl(handle_, FBIOGET_FSCREENINFO, &info_)) {
		throw exception::InitializationFailed("Failed to query screen info");
	}
	log[log::info] << "Opened device " << info_.id << ", with " << info_.smem_len << " bytes memory";		
	if (info_.type != FB_TYPE_PACKED_PIXELS) {
		throw exception::InitializationFailed("Device doesn't use packed pixels!");
	}
	if (info_.visual != FB_VISUAL_TRUECOLOR) {
		throw exception::InitializationFailed("Device doesn't use truecolor!");
	}

	if (ioctl(handle_, FBIOGET_VSCREENINFO, &vinfo_)) {
		throw exception::InitializationFailed("Failed to query variable screen info");
	}
	resolution_ = resolution_t{vinfo_.xres, vinfo_.yres};
	log[log::info] << "Resolution " << resolution_ << " @ " << vinfo_.bits_per_pixel <<"bits per pixel";

	auto it = fb_formats.find(pack_fmt(vinfo_.bits_per_pixel, 
					vinfo_.red.length, vinfo_.red.offset,
					vinfo_.green.length, vinfo_.green.offset,
					vinfo_.blue.length, vinfo_.blue.offset,
					vinfo_.transp.length, vinfo_.transp.offset));
	if (it == fb_formats.end()) {
		throw exception::InitializationFailed("Unsupported pixel format");
	}
	format_ = it->second;
	log[log::info] << "Framebuffer using format " << core::utils::get_frame_type_name(format_, false);
	set_supported_formats({format_});

	memory_ = mmap_handle_t<uint8_t>(0, info_.smem_len,
		                         PROT_READ | PROT_WRITE,
                		         MAP_SHARED, 
					 handle_,
                		         0);
	if (!memory_) {
		throw exception::InitializationFailed("Failed to map memory");
	}
	log[log::info] << "Memory mapped";
	if (clear_) {
		std::fill(memory_.get(), memory_.get()+memory_.size(), 0);
	}

}

FrameBuffer::~FrameBuffer() noexcept
{
	close(handle_);
//	if (memory_) munmap(memory_, info_.smem_len);
}

core::pFrame FrameBuffer::do_special_single_step(core::pRawVideoFrame frame)
{
	const auto res = frame->get_resolution();
	for (auto y: irange(std::min(res.height, resolution_.height))) {
		const auto len = 2 * std::min(res.width, resolution_.width);
		auto start = memory_.get() + (flip_?res.height-y-1:y) * resolution_.width*2;
		auto s_start = PLANE_RAW_DATA(frame,0) + y * res.width*2;
		std::copy(s_start, s_start+len, start);
	}
	
	return frame;
}

bool FrameBuffer::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(flip_, "flip")
			(clear_, "clear")
			(device_, "device")
		) 
		return true;
	return base_type::set_param(param);
}

} /* namespace framebuffer */
} /* namespace yuri */
