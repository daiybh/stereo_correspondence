/*!
 * @file 		UVPortaudioInput.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		21.03.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "UVPortaudioInput.h"
#include "yuri/core/Module.h"

#include "yuri/core/frame/RawAudioFrame.h"
#include "yuri/core/frame/raw_audio_frame_types.h"
#include "YuriUltragrid.h"
extern "C" {
#include "audio/audio.h"
#include "audio/capture/portaudio.h"
#include "host.h"
}


namespace yuri {
namespace ultragrid {


IOTHREAD_GENERATOR(UVPortaudioInput)

core::Parameters UVPortaudioInput::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("UVPortaudioInput");
	p["index"]["Device index (-1 for default)"]=-1;
	p["channels"]["Number of capture channels"]=2;
	return p;
}


UVPortaudioInput::UVPortaudioInput(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("uv")),channels_(2)
{
	IOTHREAD_INIT(parameters)
	std::string param_str;
	if (device_idx_ >= 0) {
		param_str=std::to_string(device_idx_);
	}
	audio_capture_channels = channels_;
	device_ = portaudio_capture_init(const_cast<char*>(param_str.c_str()));
	if (!device_) throw exception::InitializationFailed("Failed to initialize ALSA device");
}

UVPortaudioInput::~UVPortaudioInput() noexcept
{
}

void UVPortaudioInput::run()
{
	audio_frame* frame = nullptr;
	while(still_running()) {
		frame = portaudio_read(device_);
		if (!frame || (!frame->data_len)) {
			sleep(get_latency());
			continue;
		}
		log[log::verbose_debug] << "Pushing sample with " << frame->bps << " bytes per sample, "
					<< frame->sample_rate << " samples per second and " << frame->ch_count
					<< " channels";
		core::pRawAudioFrame out_frame = core::RawAudioFrame::create_empty(core::raw_audio_format::signed_16bit,
				frame->ch_count,
				frame->sample_rate,
				frame->data,
				frame->data_len);
		push_frame(0,out_frame);
	}

	portaudio_capture_done(device_);
}
bool UVPortaudioInput::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(device_idx_, 	"index")
			(channels_, 	"channels"))
		return true;
	return core::IOThread::set_param(param);
}

} /* namespace ultragrid */
} /* namespace yuri */
