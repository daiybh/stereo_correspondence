/*!
 * @file 		UVSdl.cpp
 * @author 		<Your name>
 * @date		16.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "UVSdl.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/ultragrid/YuriUltragrid.h"
extern "C" {
#include "video_display.h"
#include "video_display/sdl.h"
}
namespace yuri {
namespace uv_sdl {


IOTHREAD_GENERATOR(UVSdl)

MODULE_REGISTRATION_BEGIN("uv_sdl")
		REGISTER_IOTHREAD("uv_sdl",UVSdl)
MODULE_REGISTRATION_END()

core::Parameters UVSdl::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("UVSdl");
	return p;
}

namespace {
class SDLHelper: public core::ThreadBase
{
public:
	SDLHelper(const log::Log& log_, core::pwThreadBase parent, void* device):
		ThreadBase(log_,parent,"SDLHelper"),device_(device)
{}
	~SDLHelper() noexcept {}
private:
	virtual void run() override
{
	log[log::info] << "SDL start";
	try {
		display_sdl_run(device_);
	} catch (std::runtime_error& e) {
		log[log::warning] << "UV exit: " << e.what();
	}
	log[log::info] << "SDL quit";
}
	void* device_;
};
}


UVSdl::UVSdl(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOFilter(log_,parent,std::string("uv_sdl")),device_(nullptr),last_desc_{1,1,VIDEO_CODEC_NONE,1,PROGRESSIVE, 1}
{
	IOTHREAD_INIT(parameters)
	device_ = display_sdl_init(nullptr,0);
}

UVSdl::~UVSdl() noexcept
{
	display_sdl_done(device_);
}

void UVSdl::run()
{
	core::pThreadBase helper = make_shared<SDLHelper>(log,get_this_ptr(),device_);
	spawn_thread(helper);
	IOThread::run();
}
void UVSdl::child_ends_hook(core::pwThreadBase /*child*/, int /*code*/, size_t /*remaining_child_count*/)
{
	request_end(core::yuri_exit_interrupted);
}
core::pFrame UVSdl::do_simple_single_step(const core::pFrame& framex)
{
	core::pRawVideoFrame frame = dynamic_pointer_cast<core::RawVideoFrame>(framex);
	if (frame) {
		codec_t uv_fmt = ultragrid::yuri_to_uv(frame->get_format());
		if (!uv_fmt) return {};
		resolution_t res = frame->get_resolution();
		video_desc desc {static_cast<unsigned int>(res.width), static_cast<unsigned int>(res.height), uv_fmt, 1,PROGRESSIVE, 1};
		if (desc != last_desc_) {
			display_sdl_reconfigure(device_, desc);
			last_desc_ = desc;
		}
		//unique_ptr<video_frame, void(*)(video_frame*)> uv_frame (
		unique_ptr<video_frame, function<void(video_frame*)>> uv_frame (
				display_sdl_getf(device_),
				[this](video_frame* uvf)mutable{display_sdl_putf(this->device_, uvf, false);});

		if (!ultragrid::copy_to_uv_frame(frame, uv_frame.get())) {
			log[log::warning ] << "Failed to convert frame";
		}

	}
	return {};
}
bool UVSdl::set_param(const core::Parameter& param)
{
	return core::IOThread::set_param(param);
}

} /* namespace uv_sdl */
} /* namespace yuri */
