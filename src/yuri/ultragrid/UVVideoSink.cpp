/*
 * UVSink.cpp
 *
 *  Created on: 23.10.2013
 *      Author: neneko
 */

#include "UVVideoSink.h"
#include "yuri/core/Module.h"
#include "yuri/ultragrid/YuriUltragrid.h"
#include "yuri/core/frame/VideoFrame.h"
extern "C" {
#include "video_display.h"
}

namespace yuri {
namespace ultragrid {

namespace {
class UVSinkHelper: public core::ThreadBase
{
public:
	UVSinkHelper(const log::Log& log_, core::pwThreadBase parent, function<void()> run_func):
		ThreadBase(log_,parent,"SDLHelper"),run_func(run_func)
{}
	~UVSinkHelper() noexcept {}
private:

	virtual void run() override
{
	log[log::info] << "Sink helper start";
	try {
		run_func();
	//	display_sdl_run(device_);
	} catch (std::runtime_error& e) {
		log[log::warning] << "UV exit: " << e.what();
	}
	log[log::info] << "Sink helper quit";
}
	function<void()> run_func;
};
}



UVVideoSink::UVVideoSink(const log::Log &log_, core::pwThreadBase parent, const std::string& name, detail::uv_display_params sink_params)
:core::SpecializedIOFilter<core::VideoFrame>(log_,parent,name),
 device_(nullptr),
 last_desc_{1,1,VIDEO_CODEC_NONE,1,PROGRESSIVE, 1},
 sink_params_(sink_params)
{
}

UVVideoSink::~UVVideoSink() noexcept
{
	try {
		if (sink_params_.done_func && device_) {
			sink_params_.done_func(device_);
		}
	}
	catch (std::runtime_error&) {}
}

void UVVideoSink::run()
{
	if (sink_params_.get_property_func) {
		std::vector<format_t> supported_formats;
		log[log::info] << "Querying supported formats";
		std::vector<codec_t> scodecs(100);
		size_t len = scodecs.size()* sizeof(codec_t);
		if (sink_params_.get_property_func(device_, DISPLAY_PROPERTY_CODECS,
				scodecs.data(), &len)) {
			scodecs.resize(len/sizeof(codec_t));
			for (auto c: scodecs) {
				format_t yfmt = uv_to_yuri(c);
				if (!yfmt) {
					yfmt = uv_to_yuri_compressed(c);
				}
				if (yfmt) {
//					supported_formats_.insert(yfmt);
					supported_formats.push_back(yfmt);
				} else {
					log[log::info] << "Unsupported codec " << c ;
				}
			}
		}
		set_supported_formats(supported_formats);
		set_supported_priority(true);
	}

	if (sink_params_.run_func) {
		core::pThreadBase helper = make_shared<UVSinkHelper>(log,get_this_ptr(),[this](){sink_params_.run_func(device_);});
		spawn_thread(helper);
	}
	IOThread::run();
}
void UVVideoSink::child_ends_hook(core::pwThreadBase /*child*/, int /*code*/, size_t /*remaining_child_count*/)
{
	request_end(core::yuri_exit_interrupted);
}

core::pFrame UVVideoSink::do_special_single_step(const core::pVideoFrame& frame)
{
	if (!frame) return {};
	format_t yfmt = frame->get_format();
	codec_t uv_fmt = ultragrid::yuri_to_uv(yfmt);
	if (!uv_fmt) return {};
//	core::pVideoFrame frame = dynamic_pointer_cast<core::VideoFrame>(source_format);

	resolution_t res = frame->get_resolution();
	video_desc desc {static_cast<unsigned int>(res.width), static_cast<unsigned int>(res.height), uv_fmt, 1, PROGRESSIVE, 1};
	if (desc != last_desc_) {
		if (sink_params_.reconfigure_func) sink_params_.reconfigure_func(device_, desc);
		last_desc_ = desc;
	}


	unique_ptr<video_frame, function<void(video_frame*)>> uv_frame (
			sink_params_.getf_func(device_),
			[this](video_frame* uvf)mutable{sink_params_.putf_func(this->device_, uvf, false);});

	if (!ultragrid::copy_to_uv_frame(frame, uv_frame.get())) {
		log[log::warning ] << "Failed to convert frame";
	}
	return {};
}


bool UVVideoSink::init_sink(const std::string& format, int flags)
{
	log[log::info] << "Initializing " << sink_params_.name;
	// FIXME: Once ultragrid accepts const char* as format, we should gtet rid of the const_cast
	if (sink_params_.init_func) return (device_ = sink_params_.init_func(
			const_cast<char*>(format.c_str()),
			flags)) != nullptr;
	log[log::warning] << "No initializer specified for " << sink_params_.name;
	return false;
}

}
}



