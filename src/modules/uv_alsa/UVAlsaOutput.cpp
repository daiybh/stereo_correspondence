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
#include "yuri/core/frame/raw_audio_frame_params.h"
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
	p["device"]["Alsa device"]="default";
	return p;
}


UVAlsaOutput::UVAlsaOutput(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::SpecializedIOFilter<core::RawAudioFrame>(log_,parent,std::string("uv_alsa_output")),device_(nullptr),
device_name_("default")
{
	IOTHREAD_INIT(parameters)
	//TODO Let's fix upstream to take const char*!!!!
	device_ = audio_play_alsa_init(const_cast<char*>(device_name_.c_str()));
	if (!device_) throw exception::InitializationFailed("Failed to initialize ALSA device");

}

UVAlsaOutput::~UVAlsaOutput() noexcept
{
	audio_play_alsa_done(device_);
}

core::pFrame UVAlsaOutput::do_special_single_step(const core::pRawAudioFrame& frame)
{


	// TODO: fill frame params!
	const auto& fi = core::raw_audio_format::get_format_info(frame->get_format());


	audio_play_alsa_reconfigure(device_, fi.bits_per_sample,frame->get_channel_count(), frame->get_sampling_frequency());
	audio_frame * f = audio_play_alsa_get_frame(device_);
//	f->bps = fi.bits_per_sample / 8;
//	f->sample_rate = frame->get_sampling_frequency();
//	f->ch_count = frame->get_channel_count();

	// Fill in audio_frame
	f->data_len = std::min<int>(f->data_len, frame->size());
	std::copy(frame->data(), frame->data() + f->data_len, f->data);


//	log[log::info] << "Pushing sample with " << f->bps << " bytes per sample, "
//			<< f->sample_rate << " samples per second and " << f->ch_count
//			<< " channels.";
	audio_play_alsa_put_frame(device_, f);

	return {};
}
bool UVAlsaOutput::set_param(const core::Parameter& param)
{
	if (param.get_name() == "device") {
		device_name_ = param.get<std::string>();
	} else return core::SpecializedIOFilter<core::RawAudioFrame>::set_param(param);
	return true;

}

} /* namespace uv_alsa_output */
} /* namespace yuri */
