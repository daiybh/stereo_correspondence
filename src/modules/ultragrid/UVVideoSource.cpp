/*!
 * @file 		UVVideoSource.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		23.10.2013
 * @date		21.11.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#include "UVVideoSource.h"
#include "yuri/core/Module.h"
#include "YuriUltragrid.h"
#include "video_capture.h"
//#include "yuri/core/frame/raw_frame_params.h"
//#include "yuri/core/frame/RawVideoFrame.h"

namespace yuri {
namespace ultragrid {

UVVideoSource::UVVideoSource(const log::Log &log_, core::pwThreadBase parent, const std::string& name, detail::capture_params capt_params)
:IOThread(log_, parent, 0, 1, name),state_(nullptr),capt_params_(capt_params)
{
	set_latency(5_ms);
	init_uv();
}

UVVideoSource::~UVVideoSource() noexcept
{

}
bool UVVideoSource::init_capture(const std::string& params)
{
	log[log::info] << "Params: '"<<params<<"'";
	if (capt_params_.init_func) {
                struct vidcap_params *uv_params = vidcap_params_allocate();
                vidcap_params_set_device(uv_params, params.c_str());
		state_ = capt_params_.init_func(uv_params);
                vidcap_params_free_struct(uv_params);
	}
	return state_ != nullptr;
}

void UVVideoSource::run()
{
	if (capt_params_.grab_func) {
		while(still_running()) {
			audio_frame* audio_frame=nullptr;
			video_frame* uv_frame = capt_params_.grab_func(state_,&audio_frame);
			if (!uv_frame) {
				sleep(get_latency());
				continue;
			}
			core::pFrame frame = ultragrid::copy_from_from_uv(uv_frame, log);
                        VIDEO_FRAME_DISPOSE(uv_frame);
			if (frame) push_frame(0, frame);
		}
	}
	if (capt_params_.done_func) {
		capt_params_.done_func(state_);
	}
	close_pipes();
}

}
}
