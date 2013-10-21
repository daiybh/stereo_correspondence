/*!
 * @file 		UVAlsaInput.cpp
 * @author 		<Your name>
 * @date		21.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "UVAlsaInput.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/RawAudioFrame.h"
#include "yuri/core/frame/raw_audio_frame_types.h"
#include "yuri/ultragrid/YuriUltragrid.h"
extern "C" {
#include "audio/audio.h"
#include "audio/capture/alsa.h"
}
namespace yuri {
namespace uv_alsa_input {


IOTHREAD_GENERATOR(UVAlsaInput)


core::Parameters UVAlsaInput::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("UVAlsaInput");
	return p;
}


UVAlsaInput::UVAlsaInput(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("uv_alsa_input")),device_(nullptr)
{
	IOTHREAD_INIT(parameters)
	set_latency(2_ms);
	device_ = audio_cap_alsa_init("alsa");
	if (!device_) throw exception::InitializationFailed("Failed to initialize ALSA device");
}

UVAlsaInput::~UVAlsaInput() noexcept
{
}

void UVAlsaInput::run()
{
	audio_frame* frame = nullptr;
	while(still_running()) {
		frame = audio_cap_alsa_read(device_);
		if (!frame || (!frame->data_len)) {
			sleep(get_latency());
			continue;
		}
		core::pRawAudioFrame out_frame = core::RawAudioFrame::create_empty(core::raw_audio_format::signed_16bit,
				frame->ch_count,
				frame->sample_rate,
				frame->data,
				frame->data_len);
		push_frame(0,out_frame);
	}

	audio_cap_alsa_finish(device_);
}
bool UVAlsaInput::set_param(const core::Parameter& param)
{
	return core::IOThread::set_param(param);
}

} /* namespace uv_alsa_input */
} /* namespace yuri */
