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
device_name_("default"),format_(0),channels_(0),sampling_frequency_(0)
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
bool UVAlsaOutput::format_changed(const core::pRawAudioFrame& frame)
{
	return (format_ != frame->get_format()) ||
			(channels_ != frame->get_channel_count()) ||
			(sampling_frequency_ != frame->get_sampling_frequency());
}
void UVAlsaOutput::reconfigure(const core::pRawAudioFrame& frame)
{
	format_ = frame->get_format();
	channels_ = frame->get_channel_count();
	sampling_frequency_ = frame->get_sampling_frequency();
	const auto& fi = core::raw_audio_format::get_format_info(format_);
	audio_play_alsa_reconfigure(device_, fi.bits_per_sample, channels_, sampling_frequency_);
}

core::pFrame UVAlsaOutput::do_special_single_step(const core::pRawAudioFrame& frame)
{
	if (format_changed(frame)) reconfigure(frame);

	audio_frame * f = audio_play_alsa_get_frame(device_);

	f->data_len = std::min<int>(f->max_size, frame->size());
	std::copy(frame->data(), frame->data() + f->data_len, reinterpret_cast<uint8_t*>(f->data));

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
