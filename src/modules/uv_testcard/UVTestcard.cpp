/*!
 * @file 		UVTestcard.cpp
 * @author 		<Your name>
 * @date		16.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "UVTestcard.h"
#include "yuri/core/Module.h"
#include "yuri/ultragrid/YuriUltragrid.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/RawVideoFrame.h"
extern "C" {
#include "video_capture/testcard.h"
#include "video_capture.h"
}
namespace yuri {
namespace uv_v4l2 {


IOTHREAD_GENERATOR(UVTestcard)

MODULE_REGISTRATION_BEGIN("uv_testcard")
		REGISTER_IOTHREAD("uv_testcard",UVTestcard)
MODULE_REGISTRATION_END()

core::Parameters UVTestcard::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("UVTestcard");
	p["fps"]=30;
	return p;
}


UVTestcard::UVTestcard(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("uv_testcard")),state_(nullptr),resolution_{800,600},
format_(core::raw_format::yuyv422),fps_(25)
{
	IOTHREAD_INIT(parameters)
	set_latency(5_ms);

	unique_ptr<vidcap_params, void(*)(vidcap_params*)> par (vidcap_params_allocate(),
			[](vidcap_params* par){vidcap_params_free_struct(par);});

	std::stringstream strs;

	std::string codec = ultragrid::yuri_to_uv_string(format_);
	strs << "testcard:" << resolution_.width << ":" << resolution_.height << ":" << fps_ <<":" << codec;
	const std::string uv_params = strs.str();
	log[log::debug] << "Initializing UV screen input with string: " << uv_params;
	vidcap_params_assign_device(par.get(), uv_params.c_str());

	log[log::debug] << "Format string is " << vidcap_params_get_fmt(par.get());
	state_ = vidcap_testcard_init(par.get());


	if (!state_) {
		throw exception::InitializationFailed("Failed to initialize v4l2 device!");
	}
}

UVTestcard::~UVTestcard() noexcept
{
}

void UVTestcard::run()
{
	while(still_running()) {
		audio_frame* audio_frame=nullptr;
		video_frame* uv_frame = vidcap_testcard_grab(state_,&audio_frame);
		if (!uv_frame) {
			sleep(get_latency());
			continue;
		}
		core::pFrame frame = ultragrid::copy_from_from_uv(uv_frame, log);
		if (frame) push_frame(0, frame);

	}

	vidcap_testcard_done(state_);
	close_pipes();
}
bool UVTestcard::set_param(const core::Parameter& param)
{
	if (param.get_name() == "fps") {
		fps_ = param.get<int>();
	} else if (param.get_name() == "format") {
		format_ = core::raw_format::parse_format(param.get<std::string>());
	} else if (param.get_name() == "resolution") {
		resolution_ = param.get<resolution_t>();
	} else return core::IOThread::set_param(param);
	return true;
}

} /* namespace uv_v4l2 */
} /* namespace yuri */
