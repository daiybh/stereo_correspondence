/*!
 * @file 		UVV4l2.cpp
 * @author 		<Your name>
 * @date		16.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "UVV4l2.h"
#include "yuri/core/Module.h"
#include "yuri/ultragrid/YuriUltragrid.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/RawVideoFrame.h"
extern "C" {
#include "video_capture/v4l2.h"
#include "video_capture.h"
}
namespace yuri {
namespace uv_v4l2 {


IOTHREAD_GENERATOR(UVV4l2)

MODULE_REGISTRATION_BEGIN("uv_v4l2")
		REGISTER_IOTHREAD("uv_v4l2",UVV4l2)
MODULE_REGISTRATION_END()

core::Parameters UVV4l2::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("UVV4l2");
	p["path"]="/dev/video0";
	p["fps"]=30;
	p["format"]="YUYV";
	p["resolution"]=resolution_t{800,600};
	return p;
}


UVV4l2::UVV4l2(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("uv_v4l2"))
{
	IOTHREAD_INIT(parameters)
	set_latency(5_ms);

	codec_t uv_fmt = ultragrid::yuri_to_uv(format_);
	if (uv_fmt == VIDEO_CODEC_NONE) {
		log[log::fatal] << "Specified format is not supported in ultragrid";
		throw exception::InitializationFailed("Unsupported format");
	}
	std::string uv_fmt_str = ultragrid::uv_to_string(uv_fmt);

	unique_ptr<vidcap_params, void(*)(vidcap_params*)> par (vidcap_params_allocate(),
			[](vidcap_params* par){vidcap_params_free_struct(par);});

	std::stringstream strs;
	// @ TODO : set format!
	strs << "v4l2:" << device_ << ":" << uv_fmt_str << ":" << resolution_.width << ":" << resolution_.height << ":1/" << fps_;// << ":"
	const std::string uv_params = strs.str();
	log[log::debug] << "Initializing UV v4l2 input with string: " << uv_params;
	vidcap_params_assign_device(par.get(), uv_params.c_str());

	log[log::debug] << "Format string is " << vidcap_params_get_fmt(par.get());
	state_ = vidcap_v4l2_init(par.get());


	if (!state_) {
		throw exception::InitializationFailed("Failed to initialize v4l2 device!");
	}
}

UVV4l2::~UVV4l2() noexcept
{
}

void UVV4l2::run()
{
	while(still_running()) {
		audio_frame* audio_frame=nullptr;
		video_frame* uv_frame = vidcap_v4l2_grab(state_,&audio_frame);
		if (!uv_frame) {
			sleep(get_latency());
			continue;
		}
		core::pFrame frame = ultragrid::copy_from_from_uv(uv_frame, log);
		if (frame) push_frame(0, frame);

	}

	vidcap_v4l2_done(state_);
	close_pipes();
}
bool UVV4l2::set_param(const core::Parameter& param)
{
	if (param.get_name() == "path") {
		device_ = param.get<std::string>();
	} else if (param.get_name() == "fps") {
		fps_ = param.get<int>();
	} else if (param.get_name() == "resolution") {
		resolution_ = param.get<resolution_t>();
	} else if (param.get_name() == "format") {
		format_ = core::raw_format::parse_format(param.get<std::string>());
	} else return core::IOThread::set_param(param);
	return true;
}

} /* namespace uv_v4l2 */
} /* namespace yuri */
