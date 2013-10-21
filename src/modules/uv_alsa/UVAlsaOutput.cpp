/*!
 * @file 		UVAlsaOutput.cpp
 * @author 		<Your name>
 * @date		21.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "UVAlsaOutput.h"
#include "yuri/core/Module.h"
extern "C" {
#include "audio/playback/alsa.h"
#include "audio/audio.h"
}

namespace yuri {
namespace uv_alsa_output {


IOTHREAD_GENERATOR(UVAlsaOutput)


core::Parameters UVAlsaOutput::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("UVAlsaOutput");
	return p;
}


UVAlsaOutput::UVAlsaOutput(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::SpecializedIOFilter<core::RawAudioFrame>(log_,parent,std::string("uv_alsa_output")),device_(nullptr)
{
	IOTHREAD_INIT(parameters)
	device_ = audio_play_alsa_init("alsa");
	if (!device_) throw exception::InitializationFailed("Failed to initialize ALSA device");

}

UVAlsaOutput::~UVAlsaOutput() noexcept
{
	audio_play_alsa_done(device_);
}

core::pFrame UVAlsaOutput::do_special_single_step(const core::pRawAudioFrame& frame)
{
	audio_frame * f = audio_play_alsa_get_frame(device_);

	// Fill in audio_frame
	f->data_len = std::min<int>(f->data_len, frame->size());
	std::copy(frame->data(), frame->data() + f->data_len, f->data);

	audio_play_alsa_put_frame(device_, f);

	return {};
}
bool UVAlsaOutput::set_param(const core::Parameter& param)
{
	return core::SpecializedIOFilter<core::RawAudioFrame>::set_param(param);
}

} /* namespace uv_alsa_output */
} /* namespace yuri */
