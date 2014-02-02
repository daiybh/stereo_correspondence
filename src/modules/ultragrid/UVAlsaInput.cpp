/*!
 * @file 		UVAlsaInput.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		21.10.2013
 * @copyright	CESNET, z.s.p.o, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "UVAlsaInput.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/RawAudioFrame.h"
#include "yuri/core/frame/raw_audio_frame_types.h"
#include "YuriUltragrid.h"
extern "C" {
#include "audio/audio.h"
#include "audio/capture/alsa.h"
#include "host.h"
}
namespace yuri {
namespace uv_alsa_input {


IOTHREAD_GENERATOR(UVAlsaInput)


core::Parameters UVAlsaInput::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("UVAlsaInput");
	p["device"]["Alsa device"]="default";
	p["channels"]["Number of channels to capture."]=2;
	return p;
}


UVAlsaInput::UVAlsaInput(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("uv_alsa_input")),device_(nullptr),device_name_("default"),capture_channels_(1)
{
	IOTHREAD_INIT(parameters)
	set_latency(2_ms);
	audio_capture_channels = capture_channels_;
	// TODO Ewwww!! Let's fix upstream to take const char*!!
	device_ = audio_cap_alsa_init(const_cast<char*>(device_name_.c_str()));
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
//		log[log::info] << "Pushing sample with " << frame->bps << " bytes per sample, "
//					<< frame->sample_rate << " samples per second and " << frame->ch_count
//					<< " channels";
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
	if (param.get_name() == "device") {
		device_name_ = param.get<std::string>();
	} else if (param.get_name() == "channels") {
		capture_channels_ = param.get<size_t>();
	} else return core::IOThread::set_param(param);
	return true;
}

} /* namespace uv_alsa_input */
} /* namespace yuri */
